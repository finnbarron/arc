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
    c = cfg(exec_latency_s=1.0, exit_mode="barrier")
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


def test_reliably_wrong_model_counts_as_skill_and_calibrates_to_truth():
    c = cfg(calibration_warmup=50)
    cal = Calibrator(c)
    for i in range(200):
        hi = i % 2 == 0
        # the model says "take profit" exactly when it is about to dump
        cal.add(Label("m", i, 0.7 if hi else 0.1, 0.1 if hi else 0.7, 0.1,
                      SL if hi else TP, -0.22 if hi else 0.45))
    assert cal.skill_ok()
    assert cal.calibrate(0.1, "p_tp", TP) > 0.8
    assert cal.calibrate(0.7, "p_tp", TP) < 0.2


# ------------------------------------------------------------ Jev exits


def _held(c, sol=0.5):
    b = PaperBroker(c)
    curve = Curve(40.0, 30.0 * 1.073e9 / 40.0)
    b.submit_buy("M", sol, 0.0, {"symbol": "M"})
    curve = b.fill_due("M", 0.0, curve)
    return b, curve


def test_partial_then_full_sell_accounts_every_lamport():
    c = cfg(exec_latency_s=0.0)
    b, curve = _held(c)
    tokens = b.positions["M"].tokens
    b.submit_sell("M", 1.0, "jev_take_half", fraction=0.5)
    curve = b.fill_due("M", 1.0, curve)
    pos = b.positions["M"]
    assert pos.took_initials and pos.tokens == pytest.approx(tokens / 2)
    b.submit_sell("M", 2.0, "jev_take_half", fraction=0.5)  # only once
    assert not b.orders
    b.submit_sell("M", 3.0, "jev_sell")
    b.fill_due("M", 3.0, curve)
    (t,) = b.closed
    assert t.took_initials and t.reason == "jev_sell"
    assert b.balance == pytest.approx(c.starting_sol + t.pnl_sol)


def test_rails_in_jev_mode():
    c = cfg(exec_latency_s=0.0)
    b, curve = _held(c)
    e = b.positions["M"].entry_price
    b.check_exits("M", 5.0, e * 0.9)  # -10%: Jev's call, no rail fires
    assert not b.orders
    b.check_exits("M", 6.0, e * 2.05)  # doubled: take initials
    assert b.orders[-1].fraction == 0.5 and b.orders[-1].reason == "take_initials_2x"
    b.fill_due("M", 6.0, curve)
    b.check_exits("M", 7.0, e * 0.6)  # -40%: hard stop
    assert b.orders[-1].reason == "hard_stop" and b.orders[-1].fraction == 1.0


class ScriptedJev:
    """Holds until the position is up 10%, then says sell."""

    provider = "scripted"

    def __init__(self):
        self.exit_states = []

    async def ask(self, state, questions):
        self.exit_states.append(state)
        sell = 0.9 if state["position"]["return_pct"] >= 10 else 0.05
        return {"answers": {
            "action": {"type": "choice", "choice": "x", "confidence": 0.9,
                       "probabilities": {"hold": 1 - sell, "sell_half": 0.0, "sell_now": sell}},
            "dump_risk": {"type": "noul", "noul": 0.1},
            "upside": {"type": "score", "score": 2.0, "confidence": 0.5, "legend": {}, "probabilities": {}},
        }, "usage": {"input_tokens": 1000}}


def test_engine_asks_jev_every_second_and_follows_its_sell():
    c = cfg(exec_latency_s=0.0)
    jev = ScriptedJev()
    eng = Engine(c, Brain(c, jev), PaperBroker(c), Calibrator(c))
    creator = "dev"
    asyncio.run(eng.on_event(NewToken(0.0, "M", "Moon", "MOON", creator)))
    st = eng.tokens["M"]
    eng.broker.submit_buy("M", 0.3, 1.0, {"symbol": "MOON"})
    v_sol, k = 30.0, 30.0 * 1.073e9
    t = 1.0
    async def feed():
        nonlocal v_sol, t
        for i in range(60):
            t += 0.5
            v_sol *= 1.002  # steady climb, ~+10% after ~12s
            await eng.on_event(Trade(t, "M", f"w{i}", True, 0.1, 1.0, v_sol, k / v_sol))
        await eng.on_clock(t + 5)
    asyncio.run(feed())
    assert len(jev.exit_states) >= 5  # roughly one check per second held
    s = jev.exit_states[0]
    assert s["market_rules"] and s["position"]["seconds_held"] >= 0
    assert s["live_tape"]["price_path_pct_vs_entry"]
    (tr,) = eng.broker.closed
    assert tr.reason == "jev_sell" and tr.exit_checks >= 5


class AlwaysDumpOnExit(ScriptedJev):
    """Loves every entry, but the moment it is asked about holding, says sell."""

    async def ask(self, state, questions):
        if "action" in questions:
            return await super().ask({**state, "position": {**state["position"], "return_pct": 50}}, questions)
        return {"answers": {
            "outcome": {"type": "choice", "choice": TP, "confidence": 0.9,
                        "probabilities": {TP: 0.8, SL: 0.1, NEITHER: 0.1}},
            "rug_risk": {"type": "noul", "noul": 0.05},
            "organic": {"type": "noul", "noul": 0.9},
            "momentum": {"type": "score", "score": 3.0, "confidence": 0.9, "legend": {}, "probabilities": {}},
        }, "usage": {"input_tokens": 0}}


def test_never_buys_what_jev_would_sell_a_second_later():
    c = cfg(calibration_warmup=0, require_edge=False, min_ev=-1.0)
    eng = Engine(c, Brain(c, AlwaysDumpOnExit()), PaperBroker(c), Calibrator(c))
    events = generate(40, seed=4)
    asyncio.run(_run_events(eng, events))
    assert eng.stats.buys == 0
    assert eng.stats.skipped["jev_would_not_hold"] > 0


async def _run_events(eng, events):
    for ev in events:
        await eng.on_event(ev)


# ------------------------------------------------------------ scanner


def test_decode_pumpswap_buy_moves_reserves():
    from memebot.feeds import AMM_BUY_DISC

    body = (AMM_BUY_DISC + struct.pack("<q", 1)
            + struct.pack("<QQQQ", 2_000_000_000, 0, 0, 0)  # base out 2000 tokens
            + struct.pack("<QQ", 200_000_000_000_000, 80_000_000_000)  # pool 200M tokens / 80 SOL
            + struct.pack("<Q", 1_000_000_000)  # 1 SOL in
            + struct.pack("<QQQQQQ", 20, 0, 5, 0, 0, 0)
            + _pk(7) + _pk(8) + _pk(9) * 4 + _pk(10) + struct.pack("<Q", 5))
    ev = decode_pump_event(body, 1.0, "s")
    assert isinstance(ev, Trade) and ev.venue == "amm" and ev.is_buy
    assert ev.mint == b58encode(_pk(7)) and ev.creator == b58encode(_pk(10))
    assert ev.v_sol == pytest.approx(81.0) and ev.v_tokens == pytest.approx(200_000_000 - 2000)
    assert ev.fee_rate == pytest.approx(0.003)


def test_scanner_picks_up_mid_life_token_and_triggers_on_spike():
    c = cfg(max_evals_per_min=100)
    eng = Engine(c, Brain(c, MockJev()), PaperBroker(c), Calibrator(c))
    k, v = 60.0 * 5e7, 60.0
    async def run():
        nonlocal v
        t = 1000.0
        for i in range(40):  # a graduated pool nobody saw launch, quiet then spiking
            t += 1.0
            v *= 1.0 if i < 20 else 1.012
            await eng.on_event(Trade(t, "POOL", f"w{i}", True, 0.5, 1.0, v, k / v, venue="amm", fee_rate=0.0125))
    asyncio.run(run())
    st = eng.tokens["POOL"]
    assert not st.launch_seen and st.venue == "amm" and st.fee_rate == 0.0125
    assert eng.stats.evals >= 1  # the spike earned a Jev look
    assert st.features(st.last_ts)["venue"].startswith("PumpSwap")
