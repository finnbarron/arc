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
from collections import deque
from collections import Counter
from dataclasses import dataclass, field, replace

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
        self._exit_inflight: set[str] = set()
        self._last_tick = -1e18
        self._last_evict = -1e18
        self._eval_times: deque = deque()  # entry evals in the last minute (spend cap)
        self._pool_names: dict[str, tuple[str, str]] = {}  # curve mint -> (name, symbol)

    # ------------------------------------------------------------- events

    async def on_event(self, ev: Event) -> None:
        self.stats.events += 1
        await self.on_clock(ev.ts)
        if isinstance(ev, NewToken):
            if ev.mint not in self.tokens:
                st = TokenState.from_event(ev)
                if ev.venue == "curve":
                    self._pool_names[ev.mint] = (ev.name, ev.symbol)
                elif ev.base_mint in self._pool_names:
                    # a curve token we watched just graduated into this pool
                    st.name, st.symbol = self._pool_names[ev.base_mint]
                    st.launch_seen = False
                self.tokens[ev.mint] = st
                self.stats.tokens_seen += 1
                if self.feed:
                    await self.feed.watch([ev.mint])
            return
        st = self.tokens.get(ev.mint)
        if st is None:
            if not (self.cfg.scan_all and isinstance(ev, Trade)):
                return
            if ev.venue == "amm" and not self.cfg.scan_amm:
                return
            st = self.tokens[ev.mint] = TokenState.from_trade(ev)
            self.stats.tokens_seen += 1
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
        await self._maybe_exit_check(st, ev.ts)
        await self._maybe_evaluate(st, ev.ts)

    async def on_clock(self, now: float) -> None:
        """Time passes even when a token stops trading: fill due orders,
        run time stops, and settle labels on quiet tokens."""
        self.now = max(self.now, now)
        if now - self._last_tick < 0.2 and not any(o.due <= now for o in self.broker.orders):
            return  # at hundreds of events a second, five sweeps a second is plenty
        self._last_tick = now
        if now - self._last_evict >= 15:
            self._last_evict = now
            await self._evict(now)
        for mint in {o.mint for o in self.broker.orders if o.due <= now}:
            st = self.tokens.get(mint)
            if st:
                st.curve = self.broker.fill_due(mint, now, st.curve)
        for mint in list(self.broker.positions):
            st = self.tokens.get(mint)
            if st:
                self.broker.check_exits(mint, now, st.curve.price)
                await self._maybe_exit_check(st, now)
        for mint in list(self.labeler.pending):
            st = self.tokens.get(mint)
            if st:
                for label in self.labeler.on_price(mint, now, st.curve.price):
                    self.cal.add(label)
        self._risk_check(now)

    # ------------------------------------------------------- evaluation

    def _candidate(self, st: TokenState, now: float) -> str:
        """Cheap code-side screen, run on every trade. Two ways in:
        a fresh launch in its first minutes, or a price spike on real
        volume in any tracked token (curve or PumpSwap, any age)."""
        if st.migrated:
            return "migrated"
        if now - st.last_eval_ts < self.cfg.reeval_every_s:
            return "cooldown"
        if st.evals >= self.cfg.max_evals_per_token:
            if now - st.last_eval_ts < 600:
                return "eval_cap"
            st.evals = 0  # a token can earn a fresh look after ten quiet minutes
        if st.mint in self._inflight or self.broker.busy(st.mint):
            return "busy"
        if st.venue == "curve":
            mcap = st.curve.price * 1e9
            if not self.cfg.eval_min_mcap_sol <= mcap <= self.cfg.eval_max_mcap_sol * 4:
                return "mcap"
        elif st.liquidity_sol < self.cfg.amm_min_liquidity_sol:
            return "thin_pool"
        age = now - st.created_ts
        fresh = (
            st.launch_seen
            and self.cfg.eval_min_age_s <= age <= self.cfg.eval_max_age_s
            and len(st.trades) >= self.cfg.eval_min_trades
            and len(st.buyers) >= self.cfg.eval_min_unique_buyers
        )
        if not fresh:
            rise, n, buyers = st.spike(now, self.cfg.spike_window_s)
            if not (rise >= self.cfg.spike_pct and n >= self.cfg.spike_min_trades
                    and buyers >= self.cfg.spike_min_buyers):
                return "no_trigger"
        # spend cap on entry looks
        while self._eval_times and self._eval_times[0] < now - 60:
            self._eval_times.popleft()
        if len(self._eval_times) >= self.cfg.max_evals_per_min:
            return "rate_cap"
        return ""

    async def _maybe_evaluate(self, st: TokenState, now: float) -> None:
        if self._candidate(st, now):
            return
        st.evals += 1
        st.last_eval_ts = now
        self._eval_times.append(now)
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
            v = decide(self.cfg, j, self.cal, st.curve, self.broker.balance, enabled,
                       fee_rate=st.fee_rate, liquidity_sol=st.liquidity_sol)
            if not v.buy:
                # "shadow mode" from the policy means every gate passed but
                # trading is off, so record *why* it is off instead
                key = why if v.reason == "shadow mode" else v.reason.split(" ")[0]
                self.stats.skipped[key] += 1
                return
            if self.cfg.exit_mode == "jev":
                # Jev's hold/sell read must agree with its buy read. Live data
                # showed entries it would sell one second later, which just
                # pays the round-trip cost twice.
                x = await self.brain.judge_exit(
                    features, meta, self._position_view(0.0, 0.0, 0.0, False),
                    st.exit_view(now, st.curve.price, now),
                )
                if x is None or x.p_hold <= max(x.p_sell, x.p_half) or x.dump_risk >= self.cfg.exit_dump_p:
                    self.stats.skipped["jev_would_not_hold"] += 1
                    return
                j = replace(j, latency_s=j.latency_s + x.latency_s)
            due = now + j.latency_s + self.cfg.exec_latency_s
            self.broker.fee_of[st.mint] = st.fee_rate
            self.broker.submit_buy(
                st.mint, v.size_sol, due,
                {"symbol": st.symbol, "ev": round(v.ev, 4), "p_tp": round(v.p_tp, 3),
                 "p_sl": round(v.p_sl, 3), "rug": round(j.rug_risk, 3), "decided_ts": now},
            )
            self.stats.buys += 1
            log.info("BUY %s %.3f SOL ev=%+.3f p_tp=%.2f p_sl=%.2f", st.symbol, v.size_sol, v.ev, v.p_tp, v.p_sl)
        finally:
            self._inflight.discard(st.mint)

    # -------------------------------------------------------- Jev exits

    async def _maybe_exit_check(self, st: TokenState, now: float) -> None:
        """Ask Jev, about once a second per position, whether to keep holding."""
        if self.cfg.exit_mode != "jev":
            return
        pos = self.broker.positions.get(st.mint)
        if pos is None or pos.exiting or now < pos.entry_ts or st.mint in self._exit_inflight:
            return
        if now - pos.last_exit_check < self.cfg.exit_check_every_s:
            return
        pos.last_exit_check = now
        self._exit_inflight.add(st.mint)
        coro = self._exit_check(st, pos, now)
        if self.live:
            task = asyncio.create_task(coro)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        else:
            await coro

    def _position_view(self, r: float, peak: float, held: float, took_half: bool, drawdown: float = 0.0) -> dict:
        return {
            "return_pct": round(r * 100, 1),
            "peak_return_pct": round(peak * 100, 1),
            "drawdown_from_peak_pct": round(drawdown * 100, 1),
            "seconds_held": round(held, 1),
            "max_hold_seconds": self.cfg.position_max_hold_s,
            "took_half_off_already": took_half,
            "round_trip_cost_pct": 5.5,
        }

    async def _exit_check(self, st: TokenState, pos, now: float) -> None:
        try:
            price = st.curve.price
            r = price / pos.entry_price - 1.0
            peak = pos.peak_price / pos.entry_price - 1.0
            position = self._position_view(r, peak, now - pos.entry_ts, pos.took_initials,
                                           drawdown=price / pos.peak_price - 1)
            j = await self.brain.judge_exit(
                st.features(now), {"name": st.name, "symbol": st.symbol}, position,
                st.exit_view(now, pos.entry_price, pos.entry_ts),
            )
            if j is None or self.broker.positions.get(st.mint) is not pos or pos.exiting:
                return
            pos.exit_checks += 1
            pos.last_exit = {"t": round(now - pos.entry_ts, 1), "r": round(r, 4), "hold": round(j.p_hold, 3),
                             "half": round(j.p_half, 3), "sell": round(j.p_sell, 3), "dump": round(j.dump_risk, 3)}
            due = now + j.latency_s + self.cfg.exec_latency_s
            wants_out = j.p_sell >= self.cfg.exit_sell_p or j.dump_risk >= self.cfg.exit_dump_p
            pos.sell_streak = pos.sell_streak + 1 if wants_out else 0
            if j.dump_risk >= self.cfg.exit_dump_now_p:
                self.broker.submit_sell(st.mint, due, "jev_dump_risk")
            elif wants_out and pos.sell_streak >= self.cfg.exit_confirm:
                # Jev's read flips within a second; act on a confirmed read
                self.broker.submit_sell(st.mint, due, "jev_sell" if j.p_sell >= self.cfg.exit_sell_p else "jev_dump_risk")
            elif j.p_half >= max(j.p_hold, j.p_sell) and not pos.took_initials and r > 0.03:
                self.broker.submit_sell(st.mint, due, "jev_take_half", fraction=0.5)
            else:
                return
            log.info("EXIT %s r=%+.1f%% after %.0fs: hold %.2f half %.2f sell %.2f dump %.2f",
                     st.symbol, r * 100, now - pos.entry_ts, j.p_hold, j.p_half, j.p_sell, j.dump_risk)
        finally:
            self._exit_inflight.discard(st.mint)

    # ------------------------------------------------------------ safety

    def trading_enabled(self, now: float) -> tuple[bool, str]:
        if now < self.paused_until:
            return False, f"paused: {self.pause_reason}"
        if not self.cal.warmed_up:
            return False, "shadow mode"
        if self.cfg.require_edge and not self.cal.skill_ok():
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
        idle = [
            st for st in self.tokens.values()
            if not self.broker.busy(st.mint)
            and not self.labeler.busy(st.mint)
            and st.mint not in self._inflight
        ]
        drop = [st for st in idle if now - st.last_ts > self.cfg.idle_evict_s]
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
            "pnl_sol": round(eq - self.cfg.starting_sol, 4),  # since the account opened
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
            "jev_exit_calls": self.brain.exit_calls,
            "exit_mode": self.cfg.exit_mode,
            "jev_errors": self.brain.errors,
            "jev_usd": round(self.brain.usd_spent, 5),
            "jev_p50_ms": round(sorted(self.brain.latencies)[len(self.brain.latencies) // 2] * 1000, 1) if self.brain.latencies else None,
            "trading": self.trading_enabled(self.now)[1] or "ENABLED",
            "calibration": self.cal.summary(),
            "skipped": dict(self.stats.skipped.most_common(8)),
        }
