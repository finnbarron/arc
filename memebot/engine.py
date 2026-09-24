"""Wires feed -> token state -> Jev -> policy -> paper broker.

The engine is driven only by event timestamps (``on_event``/``on_clock``),
so live trading and replay run the exact same code. Live mode spawns Jev
calls as tasks; replay awaits them inline and charges their latency on the
virtual clock.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter
from dataclasses import dataclass, field

from .brain import Brain
from .broker import PaperBroker
from .calibration import Calibrator, Labeler
from .events import Event, Migration, NewToken, Trade
from .policy import decide
from .token_state import TokenState

log = logging.getLogger(__name__)


@dataclass
class Stats:
    events: int = 0
    tokens_seen: int = 0
    evals: int = 0
    buys: int = 0
    skipped: Counter = field(default_factory=Counter)


class Engine:
    def __init__(self, cfg, brain: Brain, broker: PaperBroker, calibrator: Calibrator, feed=None, live: bool = False) -> None:
        self.cfg = cfg
        self.brain = brain
        self.broker = broker
        self.cal = calibrator
        self.labeler = Labeler(cfg)
        self.feed = feed
        self.live = live
        self.tokens: dict[str, TokenState] = {}
        self.stats = Stats()
        self.paused_until = 0.0
        self.pause_reason = ""
        self.peak_equity = cfg.starting_sol
        self.now = 0.0
        self._tasks: set[asyncio.Task] = set()
        self._inflight: set[str] = set()

    # ------------------------------------------------------------- events

    async def on_event(self, ev: Event) -> None:
        self.stats.events += 1
        await self.on_clock(ev.ts)
        if isinstance(ev, NewToken):
            if ev.mint not in self.tokens:
                self.tokens[ev.mint] = TokenState.from_event(ev)
                self.stats.tokens_seen += 1
                if self.feed:
                    await self.feed.watch([ev.mint])
                await self._evict(ev.ts)
            return
        st = self.tokens.get(ev.mint)
        if st is None:
            return
        if isinstance(ev, Migration):
            st.migrated = True
            self._settle_token(st, ev.ts, "migrated")
            return
        assert isinstance(ev, Trade)
        # orders due before this print fill at the pre-print curve
        st.curve = self.broker.fill_due(st.mint, ev.ts, st.curve)
        st.apply(ev)
        pos = self.broker.positions.get(st.mint)
        if pos is not None:
            st.curve = st.curve.with_holding(pos.tokens)
        price = st.curve.price
        for label in self.labeler.on_price(st.mint, ev.ts, price):
            self.cal.add(label)
        self.broker.check_exits(st.mint, ev.ts, price)
        await self._maybe_evaluate(st, ev.ts)

    async def on_clock(self, now: float) -> None:
        """Time passes even when a token stops trading: fill due orders,
        run time stops, and settle labels on quiet tokens."""
        self.now = max(self.now, now)
        for mint in {o.mint for o in self.broker.orders if o.due <= now}:
            st = self.tokens.get(mint)
            if st:
                st.curve = self.broker.fill_due(mint, now, st.curve)
        for mint in list(self.broker.positions):
            st = self.tokens.get(mint)
            if st:
                self.broker.check_exits(mint, now, st.curve.price)
        for mint in list(self.labeler.pending):
            st = self.tokens.get(mint)
            if st:
                for label in self.labeler.on_price(mint, now, st.curve.price):
                    self.cal.add(label)
        self._risk_check(now)

    # ------------------------------------------------------- evaluation

    def _candidate(self, st: TokenState, now: float) -> str:
        age = now - st.created_ts
        if st.migrated:
            return "migrated"
        if age < self.cfg.eval_min_age_s or age > self.cfg.eval_max_age_s:
            return "age"
        if st.evals >= self.cfg.max_evals_per_token:
            return "eval_cap"
        if now - st.last_eval_ts < self.cfg.reeval_every_s:
            return "cooldown"
        if len(st.trades) < self.cfg.eval_min_trades:
            return "few_trades"
        if len(st.buyers) < self.cfg.eval_min_unique_buyers:
            return "few_buyers"
        mcap = st.curve.price * 1e9
        if not self.cfg.eval_min_mcap_sol <= mcap <= self.cfg.eval_max_mcap_sol:
            return "mcap"
        if st.mint in self._inflight or self.broker.busy(st.mint):
            return "busy"
        return ""

    async def _maybe_evaluate(self, st: TokenState, now: float) -> None:
        if self._candidate(st, now):
            return
        st.evals += 1
        st.last_eval_ts = now
        self._inflight.add(st.mint)
        coro = self._evaluate(st, now)
        if self.live:
            task = asyncio.create_task(coro)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        else:
            await coro

    async def _evaluate(self, st: TokenState, now: float) -> None:
        try:
            features = st.features(now)
            meta = {"name": st.name, "symbol": st.symbol}
            j = await self.brain.judge(features, meta)
            if j is None:
                self.stats.skipped["jev_error"] += 1
                return
            self.stats.evals += 1
            self.labeler.add(st.mint, now, j)
            enabled, why = self.trading_enabled(now)
            if self.broker.open_count >= self.cfg.max_open_positions:
                enabled, why = False, "max positions"
            v = decide(self.cfg, j, self.cal, st.curve, self.broker.balance, enabled)
            if not v.buy:
                # "shadow mode" from the policy means every gate passed but
                # trading is off, so record *why* it is off instead
                key = why if v.reason == "shadow mode" else v.reason.split(" ")[0]
                self.stats.skipped[key] += 1
                return
            due = now + j.latency_s + self.cfg.exec_latency_s
            self.broker.submit_buy(
                st.mint, v.size_sol, due,
                {"symbol": st.symbol, "ev": round(v.ev, 4), "p_tp": round(v.p_tp, 3),
                 "p_sl": round(v.p_sl, 3), "rug": round(j.rug_risk, 3), "decided_ts": now},
            )
            self.stats.buys += 1
            log.info("BUY %s %.3f SOL ev=%+.3f p_tp=%.2f p_sl=%.2f", st.symbol, v.size_sol, v.ev, v.p_tp, v.p_sl)
        finally:
            self._inflight.discard(st.mint)

    # ------------------------------------------------------------ safety

    def trading_enabled(self, now: float) -> tuple[bool, str]:
        if now < self.paused_until:
            return False, f"paused: {self.pause_reason}"
        if not self.cal.warmed_up:
            return False, "shadow mode"
        if not self.cal.skill_ok():
            return False, "no measured edge"
        return True, ""

    def _risk_check(self, now: float) -> None:
        eq = self.broker.equity({m: t.curve for m, t in self.tokens.items()})
        self.peak_equity = max(self.peak_equity, eq)
        if now < self.paused_until:
            return
        if self.peak_equity - eq > self.cfg.kill_max_drawdown * self.cfg.starting_sol:
            self._pause(now, f"drawdown {self.peak_equity - eq:.2f} SOL")
            self.peak_equity = eq  # re-arm from here after the pause
            return
        recent = self.broker.closed[-self.cfg.kill_window :]
        if len(recent) == self.cfg.kill_window:
            rets = [t.ret for t in recent]
            mean = sum(rets) / len(rets)
            sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) or 1e-9
            if mean / (sd / math.sqrt(len(rets))) < -1.0:
                self._pause(now, f"last {len(rets)} trades losing (mean {mean:+.1%})")

    def _pause(self, now: float, reason: str) -> None:
        self.paused_until = now + self.cfg.kill_pause_s
        self.pause_reason = reason
        log.warning("KILL SWITCH: %s; no new entries for %.0fs", reason, self.cfg.kill_pause_s)

    # --------------------------------------------------------- lifecycle

    def _settle_token(self, st: TokenState, now: float, reason: str) -> None:
        price = st.curve.price
        for label in self.labeler.expire(st.mint, now, price):
            self.cal.add(label)
        if st.mint in self.broker.positions:
            # post-migration trades move to PumpSwap; exit on the last curve price
            self.broker.submit_sell(st.mint, now, reason)
            st.curve = self.broker.fill_due(st.mint, now + 1e-9, st.curve)

    async def _evict(self, now: float) -> None:
        horizon = self.cfg.eval_max_age_s + self.cfg.max_hold_s + 30
        idle = [
            st for st in self.tokens.values()
            if not self.broker.busy(st.mint)
            and not self.labeler.busy(st.mint)
            and st.mint not in self._inflight
        ]
        drop = [st for st in idle if now - st.created_ts > horizon]
        over = len(self.tokens) - len(drop) - self.cfg.max_tracked
        if over > 0:
            rest = sorted((st for st in idle if st not in drop), key=lambda s: s.last_ts)
            drop += rest[:over]
        for st in drop:
            self.tokens.pop(st.mint, None)
        if drop and self.feed:
            await self.feed.unwatch([st.mint for st in drop])

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def close_all(self, now: float) -> None:
        """Flush every pending order and exit every position at the current curve."""
        for mint in {o.mint for o in self.broker.orders} | set(self.broker.positions):
            st = self.tokens.get(mint)
            if st is None:
                continue
            st.curve = self.broker.fill_due(mint, float("inf"), st.curve)
            self.broker.submit_sell(mint, now, "shutdown")
            st.curve = self.broker.fill_due(mint, float("inf"), st.curve)

    def snapshot(self) -> dict:
        eq = self.broker.equity({m: t.curve for m, t in self.tokens.items()})
        closed = self.broker.closed
        wins = sum(t.pnl_sol > 0 for t in closed)
        return {
            "equity_sol": round(eq, 4),
            "pnl_sol": round(eq - self.cfg.starting_sol, 4),
            "balance_sol": round(self.broker.balance, 4),
            "open": len(self.broker.positions),
            "closed": len(closed),
            "win_rate": round(wins / len(closed), 3) if closed else None,
            "realised_pnl_sol": round(sum(t.pnl_sol for t in closed), 4),
            "fees_paid_sol": round(self.broker.fees_paid, 4),
            "tokens_seen": self.stats.tokens_seen,
            "tracked": len(self.tokens),
            "jev_evals": self.stats.evals,
            "jev_calls": self.brain.calls,
            "jev_errors": self.brain.errors,
            "jev_usd": round(self.brain.usd_spent, 5),
            "jev_p50_ms": round(sorted(self.brain.latencies)[len(self.brain.latencies) // 2] * 1000, 1) if self.brain.latencies else None,
            "trading": self.trading_enabled(self.now)[1] or "ENABLED",
            "calibration": self.cal.summary(),
            "skipped": dict(self.stats.skipped.most_common(8)),
        }
