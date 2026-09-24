"""The judgment layer: TypeSafe's Jev reads a token's state and answers
typed questions. Code turns the answers into money decisions.

Jev never sees a position size or a balance. It answers four questions in a
single request (TypeSafe's fan-out pattern: the state is ingested once and
every question is evaluated in parallel):

* ``outcome``   Choice: which barrier the price hits first. This is the one
                that prices the trade, because a barrier race has exactly
                three outcomes and Jev returns a probability for each.
* ``rug_risk``  Noul: are insiders about to dump. A hard veto.
* ``organic``   Noul: is demand real wallets or bots/bundles.
* ``momentum``  Score: how strongly the tape leans up.

Jev is weak at arithmetic on raw numbers, so the state carries plain-language
labels next to every number (``describe``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

TP, SL, NEITHER = "take_profit_first", "stop_loss_first", "neither"
HOLD, SELL_HALF, SELL_ALL = "hold", "sell_half", "sell_now"

# What experienced pump.fun traders and the rug-pull literature agree on.
# Jev sees this with every entry and exit judgment so its read of the tape
# is grounded in how this market actually behaves.
PLAYBOOK = [
    "Most pump.fun tokens dump within their first hour; only about 1% ever graduate. Default to caution.",
    "A round trip costs about 5.5% in fees and slippage, so a small wiggle is not a reason to trade.",
    "Volume is king. When buy flow dries up or the trade rate slows after a run-up, the top is usually in.",
    "Creator or insider selling, or large sells into strength, means get out immediately.",
    "Creator holding over 5% of supply, or the top five wallets over 30%, is a rug warning.",
    "Wallets that bought in the first seconds (snipers, bundles) dump into the first pump.",
    "Healthy: a pullback that holds its low while buying volume returns. Unhealthy: lower highs with rising sells.",
    "Take initials out: once a position doubles, sell half so the rest rides risk-free.",
    "Prices move in seconds here. A winner that is fading fast should be sold before it becomes a loser.",
]


@dataclass(frozen=True, slots=True)
class Judgment:
    p_tp: float
    p_sl: float
    p_neither: float
    confidence: float
    rug_risk: float
    organic: float
    momentum: float  # 0..1
    latency_s: float
    input_tokens: int
    provider: str


def build_questions(cfg) -> dict[str, dict[str, Any]]:
    tp, sl, hold = int(cfg.take_profit * 100), int(cfg.stop_loss * 100), int(cfg.max_hold_s)
    return {
        "outcome": {
            "type": "choice",
            "instructions": (
                f"A buyer enters this pump.fun token right now. Over the next {hold} "
                f"seconds, which happens first to its price? Judge from the order "
                f"flow, holder distribution and creator behaviour in the state. "
                f"Most new memecoins fall; only pick a rise when the evidence is strong."
            ),
            "criteria": {
                TP: f"Price rises {tp}% or more above the current price before it ever falls {sl}% below it.",
                SL: f"Price falls {sl}% or more below the current price before it ever rises {tp}% above it.",
                NEITHER: f"Within {hold} seconds price neither rises {tp}% nor falls {sl}% from here.",
            },
        },
        "rug_risk": {
            "type": "noul",
            "instructions": (
                "Is the creator or a small group of insider wallets likely to dump "
                "a large share of supply on buyers in the next few minutes?"
            ),
            "criteria": {
                "true": (
                    "Creator already selling, supply concentrated in a few wallets, "
                    "bundled or bot-like early buys, or large sells into buying."
                ),
                "false": (
                    "Creator holding or small, supply spread over many wallets, no "
                    "large insider sells."
                ),
            },
        },
        "organic": {
            "type": "noul",
            "instructions": "Is the buying coming from many independent real traders?",
            "criteria": {
                "true": "Many distinct buyers with varied sizes, rising participation.",
                "false": (
                    "Few wallets, repeated buys from the same wallets, uniform dust "
                    "buys, or volume that looks like wash trading or a bot bundle."
                ),
            },
        },
        "momentum": {
            "type": "score",
            "instructions": "How strongly is this token's order flow leaning toward higher prices right now?",
            "criteria": [
                "Selling dominates. Price and flow point down.",
                "No lean. Flow is balanced or fading.",
                "Buying leads. Net inflow and price are rising.",
                "Strong, accelerating buying with broad participation and price near highs.",
            ],
        },
    }


def _lvl(x: float, cuts: list[tuple[float, str]], top: str) -> str:
    for cut, label in cuts:
        if x < cut:
            return label
    return top


def describe(features: dict, meta: dict) -> dict:
    """The state Jev sees: numbers plus words it can reason over."""
    f = features
    return {
        "market_rules": PLAYBOOK,
        "token": meta,
        "raw": f,
        "reading": {
            "age": _lvl(f["age_s"], [(30, "brand new"), (90, "young"), (180, "a few minutes old")], "older"),
            "participation": _lvl(f["unique_buyers"], [(10, "very few buyers"), (25, "some buyers"), (60, "many buyers")], "crowded"),
            "net_flow_last_15s": _lvl(f["net_flow_15s_sol"], [(-1, "heavy selling"), (-0.1, "net selling"), (0.1, "flat"), (1.5, "net buying")], "heavy buying"),
            "net_flow_last_60s": _lvl(f["net_flow_60s_sol"], [(-2, "heavy selling"), (-0.2, "net selling"), (0.2, "flat"), (3, "net buying")], "heavy buying"),
            "activity_trend": _lvl(f["trade_rate_accel"], [(0.6, "slowing down"), (1.3, "steady")], "accelerating"),
            "price_last_30s": _lvl(f["chg_30s"], [(-0.15, "dumping"), (-0.03, "falling"), (0.03, "flat"), (0.15, "rising")], "ripping"),
            "vs_all_time_high": _lvl(f["drawdown_from_ath"], [(-0.4, "far below its high"), (-0.15, "well off its high"), (-0.03, "just under its high")], "at its high"),
            "supply_concentration": _lvl(f["top5_holder_pct"], [(15, "spread out"), (30, "moderate")], "concentrated in a few wallets"),
            "creator": (
                "has dumped" if f["creator_sold_pct_of_initial"] >= 50
                else "has sold some" if f["creator_sold_pct_of_initial"] > 0
                else "has not sold"
            ),
            "snipers_still_holding": _lvl(f.get("early_sniper_holding_pct", 0), [(3, "little"), (10, "some")], "a lot, expect them to dump"),
            "bot_signs": _lvl(f["dust_buy_share"], [(0.3, "few dust buys"), (0.6, "many dust buys")], "mostly dust buys, likely bots"),
            "curve": _lvl(f["curve_progress"], [(0.1, "barely started"), (0.4, "early"), (0.8, "midway")], "near graduation"),
        },
    }


@dataclass(frozen=True, slots=True)
class ExitJudgment:
    p_hold: float
    p_half: float
    p_sell: float
    dump_risk: float
    upside: float  # 0..1
    latency_s: float
    input_tokens: int


def build_exit_questions(cfg) -> dict[str, dict[str, Any]]:
    return {
        "action": {
            "type": "choice",
            "instructions": (
                "We hold this pump.fun token (see position and live_tape). Prices move in "
                "seconds. What should we do with the position right now to make the most "
                "money? Apply market_rules. Selling costs about 3% of the position, so do "
                "not sell on noise, but do not let a winner turn into a loser."
            ),
            "criteria": {
                HOLD: "Momentum and buy flow still favour higher prices, or the dip is healthy and holding. Keep the whole position.",
                SELL_HALF: "Up meaningfully but the move is maturing or getting risky. Lock in profit on half and let the rest ride.",
                SELL_ALL: "The move is over or turning: buyers drying up, big or insider sells, lower highs, or a rug is starting. Exit everything now.",
            },
        },
        "dump_risk": {
            "type": "noul",
            "instructions": "Is a sharp drop of 20% or more likely in the next 30 seconds?",
            "criteria": {
                "true": "Big sells, creator or sniper selling, fading buy flow, or price rolling over from a spike.",
                "false": "Steady or rising buy flow and no signs of large holders exiting.",
            },
        },
        "upside": {
            "type": "score",
            "instructions": "How much further can this token run from here in the next minute?",
            "criteria": [
                "None. The run is over or it is dumping.",
                "A little. Momentum is fading.",
                "Some. Buying is steady and the chart is healthy.",
                "A lot. Accelerating buying, new wallets piling in, price breaking to new highs.",
            ],
        },
    }


def describe_exit(features: dict, meta: dict, position: dict, tape: dict) -> dict:
    base = describe(features, meta)
    base["position"] = position
    base["live_tape"] = tape
    return base


def to_exit_judgment(payload: dict, latency: float) -> ExitJudgment:
    a = payload["answers"]
    probs = a["action"]["probabilities"]
    total = sum(float(v) for v in probs.values()) or 1.0
    return ExitJudgment(
        p_hold=float(probs.get(HOLD, 0.0)) / total,
        p_half=float(probs.get(SELL_HALF, 0.0)) / total,
        p_sell=float(probs.get(SELL_ALL, 0.0)) / total,
        dump_risk=float(a["dump_risk"]["noul"]),
        upside=min(max(float(a["upside"]["score"]) / 3.0, 0.0), 1.0),
        latency_s=latency,
        input_tokens=int(payload.get("usage", {}).get("input_tokens") or 0),
    )


class JevBackend(Protocol):
    provider: str

    async def ask(self, state: dict, questions: dict) -> dict: ...


class RealJev:
    """TypeSafe's official async SDK against the live Jev API."""

    provider = "jev"

    def __init__(self, cfg) -> None:
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        # One retry only. A trading decision that shows up late is worse than none.
        self._client = AsyncTypeSafeClient(
            model=cfg.jev_model,
            timeout=cfg.jev_timeout_s,
            retry=RetryPolicy(max_retries=1, backoff_initial=0.1, backoff_max=0.3),
        )

    async def ask(self, state: dict, questions: dict) -> dict:
        resp = await self._client.system_one(state=state, questions=questions)
        return {
            "answers": {k: v.model_dump() for k, v in resp.answers.items()},
            "usage": {"input_tokens": resp.usage.input_tokens or 0},
        }

    async def close(self) -> None:
        await self._client.aclose()


class MockJev:
    """OFFLINE STAND-IN. A hand-written heuristic that returns answers in
    Jev's exact shape so the pipeline can be exercised without the API.
    Any P&L produced with this backend says nothing about Jev."""

    provider = "mock"

    async def ask(self, state: dict, questions: dict) -> dict:
        f = state["raw"]
        if "action" in questions:
            r = state["position"]["return_pct"] / 100
            fade = f["net_flow_15s_sol"] < 0 or state["live_tape"]["big_sells_last_30s"] > 0
            sell = 0.7 if (fade and r > 0.05) or r < -0.15 else 0.2
            await asyncio.sleep(0)
            return {"answers": {
                "action": {"type": "choice", "choice": SELL_ALL if sell > 0.5 else HOLD, "confidence": 0.5,
                           "probabilities": {HOLD: 1 - sell, SELL_HALF: 0.0, SELL_ALL: sell}},
                "dump_risk": {"type": "noul", "noul": 0.6 if fade else 0.2},
                "upside": {"type": "score", "score": 1.0, "confidence": 0.5, "legend": {}, "probabilities": {}},
            }, "usage": {"input_tokens": 0}}
        z = (
            1.2 * math.tanh(f["net_flow_60s_sol"] / 3)
            + 0.8 * math.tanh(f["chg_30s"] * 4)
            + 0.5 * (f["trade_rate_accel"] > 1.2)
            - 1.5 * (f["creator_sold_pct_of_initial"] >= 50)
            - 0.02 * max(f["top5_holder_pct"] - 20, 0)
            - 1.0 * (f["dust_buy_share"] > 0.6)
            - 0.6
        )
        up = 1 / (1 + math.exp(-z))
        p_tp, p_sl = 0.8 * up, 0.8 * (1 - up)
        rug = 1 / (1 + math.exp(-(1.5 * (f["creator_sold_pct_of_initial"] >= 50) + 0.05 * (f["top5_holder_pct"] - 25))))
        organic = 1 / (1 + math.exp(-(0.05 * (f["unique_buyers"] - 20) - 2 * (f["dust_buy_share"] - 0.4))))
        mom = max(0.0, min(3.0, 1.5 + 1.5 * math.tanh(z)))
        await asyncio.sleep(0)
        return {
            "answers": {
                "outcome": {
                    "type": "choice",
                    "choice": TP if p_tp > max(p_sl, 0.2) else SL if p_sl > 0.2 else NEITHER,
                    "probabilities": {TP: p_tp, SL: p_sl, NEITHER: 0.2},
                    "confidence": abs(p_tp - p_sl) + 0.2,
                },
                "rug_risk": {"type": "noul", "noul": rug},
                "organic": {"type": "noul", "noul": organic},
                "momentum": {"type": "score", "score": mom, "confidence": 0.5,
                             "legend": {}, "probabilities": {}},
            },
            "usage": {"input_tokens": 0},
        }


class JevCache:
    """Remembers every answer keyed by the exact request, so replays and
    tuning re-use paid calls instead of making them again."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.mem: dict[str, dict] = {}
        if path and path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.mem[row["key"]] = row["value"]
        self._fh = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", buffering=1)

    @staticmethod
    def key(state: dict, questions: dict, provider: str) -> str:
        blob = json.dumps([provider, state, questions], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        return self.mem.get(key)

    def put(self, key: str, value: dict) -> None:
        self.mem[key] = value
        if self._fh:
            self._fh.write(json.dumps({"key": key, "value": value}) + "\n")


class Brain:
    def __init__(self, cfg, backend: JevBackend, cache: JevCache | None = None) -> None:
        self.cfg = cfg
        self.backend = backend
        self.cache = cache or JevCache(None)
        self.questions = build_questions(cfg)
        self.exit_questions = build_exit_questions(cfg)
        self.exit_calls = 0
        self.sem = asyncio.Semaphore(cfg.jev_concurrency)
        self.calls = 0
        self.errors = 0
        self.input_tokens = 0
        self.latencies: list[float] = []

    @property
    def usd_spent(self) -> float:
        return self.input_tokens * 0.042 / 1_000_000

    async def judge(self, features: dict, meta: dict) -> Judgment | None:
        state = describe(features, meta)
        key = JevCache.key(state, self.questions, self.backend.provider)
        cached = self.cache.get(key)
        if cached is not None:
            payload, latency = cached["payload"], cached["latency_s"]
        else:
            async with self.sem:
                started = time.perf_counter()
                try:
                    payload = await self.backend.ask(state, self.questions)
                except Exception as exc:
                    self.errors += 1
                    log.warning("jev call failed: %s", exc)
                    return None
                latency = time.perf_counter() - started
            self.calls += 1
            self.input_tokens += int(payload.get("usage", {}).get("input_tokens") or 0)
            self.cache.put(key, {"payload": payload, "latency_s": latency})
        self.latencies.append(latency)
        return to_judgment(payload, latency, self.backend.provider)

    async def judge_exit(self, features: dict, meta: dict, position: dict, tape: dict) -> ExitJudgment | None:
        state = describe_exit(features, meta, position, tape)
        started = time.perf_counter()
        try:
            async with self.sem:
                payload = await self.backend.ask(state, self.exit_questions)
            j = to_exit_judgment(payload, time.perf_counter() - started)
        except Exception as exc:
            self.errors += 1
            log.warning("jev exit call failed: %s", exc)
            return None
        self.calls += 1
        self.exit_calls += 1
        self.input_tokens += j.input_tokens
        self.latencies.append(j.latency_s)
        return j


def to_judgment(payload: dict, latency: float, provider: str) -> Judgment:
    a = payload["answers"]
    probs = a["outcome"]["probabilities"]
    total = sum(float(v) for v in probs.values()) or 1.0
    mom = a["momentum"]
    return Judgment(
        p_tp=float(probs.get(TP, 0.0)) / total,
        p_sl=float(probs.get(SL, 0.0)) / total,
        p_neither=float(probs.get(NEITHER, 0.0)) / total,
        confidence=float(a["outcome"].get("confidence", 0.0)),
        rug_risk=float(a["rug_risk"]["noul"]),
        organic=float(a["organic"]["noul"]),
        momentum=min(max(float(mom["score"]) / 3.0, 0.0), 1.0),
        latency_s=latency,
        input_tokens=int(payload.get("usage", {}).get("input_tokens") or 0),
        provider=provider,
    )


def make_backend(cfg) -> JevBackend:
    want = cfg.jev_provider
    if want == "mock" or (want == "auto" and not os.environ.get("TYPESAFE_API_KEY")):
        log.warning("using MockJev: offline heuristic, NOT the Jev model")
        return MockJev()
    return RealJev(cfg)
