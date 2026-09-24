"""Paper broker. Simulates buys and sells against the *live* bonding curve.

Realism that matters for P&L:

* Orders fill ``jev latency + exec latency`` after the decision, at whatever
  the curve is by then. Other bots front-running you is baked in.
* Fill price comes from constant-product math on the reserves at fill
  time, so size moves price exactly as it would on chain.
* pump.fun's fee is charged on both sides, plus per-tx network/priority
  cost, plus an extra adverse-slippage haircut.
* Stops trigger on a print but *fill later*, so gaps through a stop are
  paid in full.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .curve import Curve

log = logging.getLogger(__name__)


@dataclass
class Order:
    mint: str
    side: str  # "buy" | "sell"
    due: float
    sol: float = 0.0  # for buys
    reason: str = ""
    fraction: float = 1.0  # for sells: share of the position to sell
    meta: dict = field(default_factory=dict)


@dataclass
class Position:
    mint: str
    symbol: str
    tokens: float
    cost_sol: float
    entry_ts: float
    entry_price: float
    peak_price: float
    ev: float
    exiting: bool = False
    partial_pending: bool = False
    took_initials: bool = False
    proceeds_sol: float = 0.0  # banked from partial sells
    last_exit_check: float = -1e18
    exit_checks: int = 0
    last_exit: dict = field(default_factory=dict)  # Jev's latest exit read
    sell_streak: int = 0  # consecutive Jev sell reads


@dataclass
class ClosedTrade:
    mint: str
    symbol: str
    entry_ts: float
    exit_ts: float
    cost_sol: float
    proceeds_sol: float
    pnl_sol: float
    ret: float
    reason: str
    ev: float
    took_initials: bool = False
    exit_checks: int = 0
    last_exit: dict = field(default_factory=dict)


class PaperBroker:
    def __init__(self, cfg, log_path: Path | None = None) -> None:
        self.cfg = cfg
        self.balance = cfg.starting_sol
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.closed: list[ClosedTrade] = []
        self.fees_paid = 0.0
        self.fee_of: dict[str, float] = {}  # per-token venue fee, set by the engine
        self._fh = None
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = log_path.open("a", buffering=1)

    # ------------------------------------------------------------ state

    def busy(self, mint: str) -> bool:
        return mint in self.positions or any(o.mint == mint for o in self.orders)

    @property
    def open_count(self) -> int:
        return len(self.positions) + sum(o.side == "buy" for o in self.orders)

    def equity(self, curves: dict[str, Curve]) -> float:
        eq = self.balance + sum(o.sol for o in self.orders if o.side == "buy")
        for p in self.positions.values():
            c = curves.get(p.mint)
            if c is not None:
                sol, _ = c.sell(p.tokens, self._fee(p.mint))
                eq += max(sol - self.cfg.tx_cost_sol, 0.0)
        return eq

    # ------------------------------------------------------------ orders

    def submit_buy(self, mint: str, sol: float, due: float, meta: dict) -> None:
        cost = sol + self.cfg.tx_cost_sol
        if cost > self.balance:
            return
        self.balance -= cost  # reserve now so concurrent decisions cannot overspend
        self.fees_paid += self.cfg.tx_cost_sol
        self.orders.append(Order(mint, "buy", due, sol=sol, meta=meta))

    def submit_sell(self, mint: str, due: float, reason: str, fraction: float = 1.0) -> None:
        pos = self.positions.get(mint)
        if pos is None or pos.exiting:
            return
        if fraction < 1.0:
            if pos.partial_pending or pos.took_initials:
                return
            pos.partial_pending = True
        else:
            pos.exiting = True
        self.orders.append(Order(mint, "sell", due, reason=reason, fraction=fraction))

    def fill_due(self, mint: str, now: float, curve: Curve) -> Curve:
        """Fill every order for ``mint`` whose time has come, at ``curve``.

        Returns the curve after our own fills so later orders in the same
        instant see the impact we caused.
        """
        remaining = []
        for o in self.orders:
            if o.mint != mint or o.due > now:
                remaining.append(o)
                continue
            if o.side == "buy":
                curve = self._fill_buy(o, now, curve)
            else:
                curve = self._fill_sell(o, now, curve)
        self.orders = remaining
        return curve

    def _fee(self, mint: str) -> float:
        return self.fee_of.get(mint, self.cfg.fee_rate)

    def _fill_buy(self, o: Order, now: float, curve: Curve) -> Curve:
        fee = self._fee(o.mint)
        tokens, after = curve.buy(o.sol, fee)
        tokens *= 1.0 - self.cfg.extra_slippage
        self.fees_paid += o.sol * fee
        price = o.sol / tokens if tokens else 0.0
        self.positions[o.mint] = Position(
            mint=o.mint,
            symbol=o.meta.get("symbol", ""),
            tokens=tokens,
            cost_sol=o.sol + self.cfg.tx_cost_sol,
            entry_ts=now,
            # barriers are measured from the marginal price after our own buy,
            # so our impact never counts as profit
            entry_price=after.price,
            peak_price=after.price,
            ev=o.meta.get("ev", 0.0),
        )
        self._log("buy", mint=o.mint, sol=o.sol, tokens=tokens, avg_price=price, ts=now, **o.meta)
        return after

    def _fill_sell(self, o: Order, now: float, curve: Curve) -> Curve:
        pos = self.positions.get(o.mint)
        if pos is None:
            return curve
        partial = o.fraction < 1.0 and not pos.exiting
        qty = pos.tokens * (o.fraction if partial else 1.0)
        fee = self._fee(o.mint)
        sol, after = curve.sell(qty, fee)
        self.fees_paid += sol / (1 - fee) * fee
        sol *= 1.0 - self.cfg.extra_slippage
        proceeds = max(sol - self.cfg.tx_cost_sol, 0.0)
        self.fees_paid += self.cfg.tx_cost_sol
        self.balance += proceeds
        if partial:
            pos.tokens -= qty
            pos.proceeds_sol += proceeds
            pos.took_initials = True
            pos.partial_pending = False
            self._log("partial", mint=pos.mint, symbol=pos.symbol, ts=now, sol=proceeds, reason=o.reason)
            return after
        self.positions.pop(o.mint, None)
        total = proceeds + pos.proceeds_sol
        pnl = total - pos.cost_sol
        trade = ClosedTrade(
            pos.mint, pos.symbol, pos.entry_ts, now, pos.cost_sol, total, pnl,
            pnl / pos.cost_sol, o.reason, pos.ev, pos.took_initials, pos.exit_checks, pos.last_exit,
        )
        self.closed.append(trade)
        self._log("sell", **asdict(trade))
        return after

    # ------------------------------------------------------------- exits

    def check_exits(self, mint: str, now: float, price: float) -> None:
        pos = self.positions.get(mint)
        if pos is None or pos.exiting:
            return
        pos.peak_price = max(pos.peak_price, price)
        r = price / pos.entry_price - 1.0
        peak_r = pos.peak_price / pos.entry_price - 1.0
        due = now + self.cfg.exec_latency_s
        if self.cfg.exit_mode == "jev":
            # Jev makes the calls (see Engine._exit_check); only rails Jev
            # cannot override live here
            if r <= -self.cfg.hard_stop:
                self.submit_sell(mint, due, "hard_stop")
            elif now - pos.entry_ts >= self.cfg.position_max_hold_s:
                self.submit_sell(mint, due, "max_hold")
            elif r >= self.cfg.take_initials_at and not pos.took_initials:
                self.submit_sell(mint, due, "take_initials_2x", fraction=0.5)
            return
        reason = ""
        if r >= self.cfg.take_profit:
            reason = "take_profit"
        elif r <= -self.cfg.stop_loss:
            reason = "stop_loss"
        elif (
            self.cfg.trailing_stop > 0
            and peak_r >= self.cfg.trail_arm
            and price <= pos.peak_price * (1 - self.cfg.trailing_stop)
        ):
            reason = "trailing_stop"
        elif now - pos.entry_ts >= self.cfg.max_hold_s:
            reason = "time_stop"
        if reason:
            self.submit_sell(mint, now + self.cfg.exec_latency_s, reason)

    def _log(self, kind: str, **row) -> None:
        if self._fh:
            self._fh.write(json.dumps({"kind": kind, **row}, default=str) + "\n")
