"""Turn Jev's judgment into a yes/no and a size. Pure functions, no I/O."""

from __future__ import annotations

from dataclasses import dataclass

from .brain import NEITHER, SL, TP, Judgment
from .calibration import Calibrator
from .curve import Curve, round_trip_cost
from .events import INITIAL_V_SOL


@dataclass(frozen=True, slots=True)
class Verdict:
    buy: bool
    size_sol: float
    ev: float  # expected return per SOL, net of every cost
    p_tp: float
    p_sl: float
    reason: str


def decide(cfg, j: Judgment, cal: Calibrator, curve: Curve, balance: float, trading_enabled: bool,
           fee_rate: float | None = None, liquidity_sol: float | None = None) -> Verdict:
    p_tp = cal.calibrate(j.p_tp, "p_tp", TP)
    p_sl = cal.calibrate(j.p_sl, "p_sl", SL)
    # the three outcomes are exhaustive; renormalise if calibration overshoots
    if p_tp + p_sl > 0.98:
        s = (p_tp + p_sl) / 0.98
        p_tp, p_sl = p_tp / s, p_sl / s
    p_n = 1.0 - p_tp - p_sl

    r_tp, r_sl, r_n = cal.mean_return(TP), cal.mean_return(SL), cal.mean_return(NEITHER)
    gross = p_tp * r_tp + p_sl * r_sl + p_n * r_n

    # size first, because cost depends on size (curve impact, fixed tx fees)
    second_moment = p_tp * r_tp**2 + p_sl * r_sl**2 + p_n * r_n**2
    size = 0.0
    if gross > 0 and second_moment > 0:
        size = cfg.kelly_fraction * gross / second_moment * balance
    real_sol = max(curve.v_sol - INITIAL_V_SOL if liquidity_sol is None else liquidity_sol, 1.0)
    size = min(size, cfg.max_position_sol, cfg.max_curve_share * max(real_sol, 10.0), balance * 0.5)
    probe = max(size, cfg.min_position_sol)
    ev = gross - round_trip_cost(cfg, probe, curve, fee_rate)

    def no(reason: str) -> Verdict:
        return Verdict(False, 0.0, ev, p_tp, p_sl, reason)

    if j.rug_risk > cfg.max_rug_risk:
        return no(f"rug risk {j.rug_risk:.2f}")
    if j.confidence < cfg.min_confidence:
        return no(f"low confidence {j.confidence:.2f}")
    if ev < cfg.min_ev:
        return no(f"ev {ev:+.3f} below {cfg.min_ev:+.3f}")
    if size < cfg.min_position_sol:
        return no(f"size {size:.3f} below minimum")
    if not trading_enabled:
        return no("shadow mode")
    return Verdict(True, size, ev, p_tp, p_sl, "ok")
