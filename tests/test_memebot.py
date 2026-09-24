import asyncio
import base64
import random
import struct

import pytest

from memebot.brain import NEITHER, SL, TP, Brain, Judgment, MockJev, to_judgment
from memebot.broker import PaperBroker
from memebot.calibration import Calibrator, Label, Labeler
from memebot.cli import replay
from memebot.config import Config
from memebot.curve import Curve, round_trip_cost
from memebot.engine import Engine
from memebot.events import NewToken, Trade
from memebot.feeds import (
    CREATE_DISC,
    TRADE_DISC,
    b58encode,
    decode_pump_event,
    events_from_logs,
    parse_pumpportal,
)
from memebot.policy import decide
from memebot.synthetic import generate


def cfg(**kw):
    c = Config(data_dir="/nonexistent-unused")
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ------------------------------------------------------------------ curve


def test_round_trip_loses_fees_and_nothing_else():
    c = Curve(40.0, 30.0 * 1.073e9 / 40.0)
    tokens, after = c.buy(1.0, 0.0125)
    back, end = after.sell(tokens, 0.0125)
    assert back == pytest.approx(0.9875**2, rel=1e-9)
    assert end.v_sol == pytest.approx(c.v_sol)


def test_holding_overlay_matches_real_buy():
    c = Curve(45.0, 30.0 * 1.073e9 / 45.0)
    tokens, after = c.buy(0.5, 0.0125)
    overlaid = c.with_holding(tokens)
    assert overlaid.price == pytest.approx(after.price, rel=1e-9)


def test_round_trip_cost_is_the_hurdle():
    c = cfg()
    cost = round_trip_cost(c, 0.25, Curve(40.0, 30.0 * 1.073e9 / 40.0))
    # two pump.fun fees + two slippage haircuts + tx costs
    assert 0.045 < cost < 0.07


# ------------------------------------------------------------------ feeds


def _pk(n: int) -> bytes:
    return bytes([n]) * 32


def test_decode_trade_event():
    body = (
        TRADE_DISC + _pk(1) + struct.pack("<QQ", 2_000_000_000, 5_000_000_000_000) + b"\x01"
        + _pk(2) + struct.pack("<q", 1) + struct.pack("<QQ", 40_000_000_000, 804_750_000_000_000)
        + b"\x00" * 64  # newer trailing fields are ignored
    )
    ev = decode_pump_event(body, 123.0, "sig")
    assert isinstance(ev, Trade)
    assert ev.is_buy and ev.sol == 2.0 and ev.tokens == 5_000_000.0
    assert ev.v_sol == 40.0 and ev.mint == b58encode(_pk(1))


def test_decode_create_event_from_logs():
    def s(x: str) -> bytes:
        return struct.pack("<I", len(x)) + x.encode()

    body = CREATE_DISC + s("Doge Two") + s("DOGE2") + s("ipfs://x") + _pk(3) + _pk(4) + _pk(5) + _pk(6)
    logs = ["Program log: Instruction: Create", "Program data: " + base64.b64encode(body).decode()]
    (ev,) = events_from_logs(logs, 1.0)
    assert isinstance(ev, NewToken)
    assert ev.symbol == "DOGE2" and ev.creator == b58encode(_pk(6))


def test_parse_pumpportal():
    trade = parse_pumpportal(
        {"txType": "sell", "mint": "M", "traderPublicKey": "W", "solAmount": 0.3,
         "tokenAmount": 1e6, "vSolInBondingCurve": 35.0, "vTokensInBondingCurve": 9.2e8}, 5.0)
    assert isinstance(trade, Trade) and not trade.is_buy and trade.v_sol == 35.0
    assert parse_pumpportal({"message": "Successfully subscribed"}, 0) is None


# -------------------------------------------------------- calibration/policy


def _j(p_tp, p_sl, rug=0.1, conf=0.8):
    return Judgment(p_tp, p_sl, 1 - p_tp - p_sl, conf, rug, 0.8, 0.7, 0.3, 500, "test")


def test_calibrator_maps_overconfidence_to_reality():
    c = cfg(calibration_warmup=10)
    cal = Calibrator(c)
    rng = random.Random(1)
    # model always says 0.9 but the truth is 20%
    for i in range(300):
        cal.add(Label("m", i, 0.9, 0.05, 0.1, TP if rng.random() < 0.2 else SL, 0.0))
    assert cal.calibrate(0.9, "p_tp", TP) == pytest.approx(0.2, abs=0.07)


def test_auc():
    c = cfg()
    cal = Calibrator(c)
    for i in range(50):
        cal.add(Label("m", i, 0.8, 0.1, 0.1, TP, 0.4))
        cal.add(Label("m", i, 0.2, 0.6, 0.1, SL, -0.2))
    assert cal.auc() == 1.0


def _warm_cal(c, good=True):
    cal = Calibrator(c)
    for i in range(200):
        hi = i % 2 == 0
        outcome = (TP if hi else SL) if good else (TP if i % 5 == 0 else SL)
        cal.add(Label("m", i, 0.7 if hi else 0.1, 0.1 if hi else 0.7, 0.1, outcome, 0.45 if outcome == TP else -0.22))
    return cal


def test_policy_buys_only_with_edge_and_vetoes_rugs():
    c = cfg(calibration_warmup=50)
    cal = _warm_cal(c)
    curve = Curve(40.0, 30.0 * 1.073e9 / 40.0)
    v = decide(c, _j(0.7, 0.1), cal, curve, 10.0, True)
    assert v.buy and v.ev > c.min_ev and c.min_position_sol <= v.size_sol <= c.max_position_sol
    assert not decide(c, _j(0.1, 0.7), cal, curve, 10.0, True).buy
    assert not decide(c, _j(0.7, 0.1, rug=0.9), cal, curve, 10.0, True).buy
    assert decide(c, _j(0.7, 0.1), cal, curve, 10.0, False).reason == "shadow mode"


def test_to_judgment_normalises():
    payload = {"answers": {
        "outcome": {"type": "choice", "choice": TP, "confidence": 0.6,
                    "probabilities": {TP: 0.6, SL: 0.3, NEITHER: 0.3}},
        "rug_risk": {"type": "noul", "noul": 0.2},
        "organic": {"type": "noul", "noul": 0.7},
        "momentum": {"type": "score", "score": 1.5, "confidence": 0.5, "legend": {}, "probabilities": {}},
    }, "usage": {"input_tokens": 700}}
    j = to_judgment(payload, 0.3, "jev")
    assert j.p_tp + j.p_sl + j.p_neither == pytest.approx(1.0)
    assert j.momentum == 0.5 and j.input_tokens == 700


# ------------------------------------------------------------------ broker


def test_stop_fills_late_and_pays_the_gap():
    c = cfg(exec_latency_s=1.0)
    b = PaperBroker(c)
    curve = Curve(40.0, 30.0 * 1.073e9 / 40.0)
    b.submit_buy("M", 0.5, 0.0, {"symbol": "M"})
    curve = b.fill_due("M", 0.0, curve)
    entry = b.positions["M"].entry_price
    b.check_exits("M", 5.0, entry * 0.79)  # -21% print triggers the stop
    assert b.orders and b.orders[0].due == 6.0
    crashed = Curve(31.0, 30.0 * 1.073e9 / 31.0)  # by fill time it is far lower
    b.fill_due("M", 6.0, crashed.with_holding(b.positions["M"].tokens))
    (t,) = b.closed
    assert t.reason == "stop_loss" and t.ret < -0.21


# ------------------------------------------------------------ end to end


class NoiseJev:
    """A model with zero skill: random probabilities."""

    provider = "noise"

    def __init__(self):
        self.rng = random.Random(3)

    async def ask(self, state, questions):
        a, b = self.rng.random(), self.rng.random()
        return {"answers": {
            "outcome": {"type": "choice", "choice": TP, "confidence": 0.9,
                        "probabilities": {TP: a, SL: b, NEITHER: 0.3}},
            "rug_risk": {"type": "noul", "noul": 0.1},
            "organic": {"type": "noul", "noul": 0.9},
            "momentum": {"type": "score", "score": 2.0, "confidence": 0.9, "legend": {}, "probabilities": {}},
        }, "usage": {"input_tokens": 0}}


async def _run(c, backend, events):
    eng = Engine(c, Brain(c, backend), PaperBroker(c), Calibrator(c))
    for ev in events:
        await eng.on_event(ev)
    await eng.on_clock(events[-1].ts + 600)
    await eng.close_all(events[-1].ts + 600)
    return eng


def test_no_skill_means_no_losses_worth_mentioning():
    """The main profitability guard: a model with no edge must not be
    allowed to bleed the account. It stays in shadow/no-edge mode."""
    c = cfg()
    events = generate(250, seed=11)
    eng = asyncio.run(_run(c, NoiseJev(), events))
    pnl = eng.broker.balance - c.starting_sol
    assert len(eng.broker.closed) <= 5
    assert pnl > -0.05 * c.starting_sol
    assert not eng.cal.skill_ok()


def test_pipeline_runs_end_to_end_and_closes_everything():
    c = cfg()
    events = generate(150, seed=5)
    eng = asyncio.run(_run(c, MockJev(), events))
    assert eng.stats.evals > 50
    assert not eng.broker.positions and not eng.broker.orders
    assert eng.cal.n > 50


def test_labeler_uses_fill_time_price():
    c = cfg(exec_latency_s=1.0, max_hold_s=10)
    lab = Labeler(c)
    lab.add("M", 0.0, _j(0.5, 0.2))
    assert lab.on_price("M", 0.5, 1.0) == []  # before due: ignored
    assert lab.on_price("M", 1.4, 2.0) == []  # entry set at 2.0
    (l,) = lab.on_price("M", 2.0, 3.0)
    assert l.outcome == TP and l.ret == pytest.approx(0.5)


def test_kill_switch_pauses_after_a_losing_streak():
    from memebot.broker import ClosedTrade

    c = cfg(kill_window=10)
    eng = Engine(c, Brain(c, MockJev()), PaperBroker(c), Calibrator(c))
    eng.broker.closed = [ClosedTrade("m", "S", 0, 1, 0.2, 0.15, -0.05, -0.25 + 0.01 * (i % 3), "stop_loss", 0.1) for i in range(10)]
    asyncio.run(eng.on_clock(100.0))
    enabled, why = eng.trading_enabled(101.0)
    assert not enabled and why.startswith("paused")
