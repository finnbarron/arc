"""SYNTHETIC pump.fun-like market, for exercising the plumbing offline.

This is not market data and nothing learned from it transfers to the real
market. It exists so the engine, broker, labeler and kill switch can be
tested end to end without network access. Token archetypes:

* rug     creator + insiders buy early, then dump
* dud     a trickle of trades, slow bleed
* runner  broadening organic demand, a real pump, then distribution
"""

from __future__ import annotations

import random

from .curve import Curve
from .events import INITIAL_V_SOL, INITIAL_V_TOKENS, Event, NewToken, Trade


def generate(n_tokens: int = 200, seed: int = 7, fee: float = 0.0125) -> list[Event]:
    rng = random.Random(seed)
    events: list[Event] = []
    t0 = 1_700_000_000.0
    for i in range(n_tokens):
        start = t0 + i * rng.uniform(2, 8)
        kind = rng.choices(["rug", "dud", "runner"], [0.55, 0.33, 0.12])[0]
        events.extend(_token(rng, f"MINT{i:05d}", start, kind, fee))
    events.sort(key=lambda e: e.ts)
    return events


def _token(rng: random.Random, mint: str, start: float, kind: str, fee: float) -> list[Event]:
    creator = f"{mint}-dev"
    curve = Curve(INITIAL_V_SOL, INITIAL_V_TOKENS)
    dev_sol = rng.uniform(0.5, 2.5)
    dev_tokens, curve = curve.buy(dev_sol, fee)
    out: list[Event] = [NewToken(start, mint, f"{kind} coin {mint[-3:]}", mint[-4:], creator,
                                 curve.v_sol, curve.v_tokens, dev_tokens)]
    holdings: dict[str, float] = {creator: dev_tokens}
    life = {"rug": 150, "dud": 200, "runner": 420}[kind]
    dump_at = rng.uniform(25, 90) if kind == "rug" else None
    peak_at = rng.uniform(90, 200) if kind == "runner" else None
    t = start
    wallets = 0
    while t - start < life:
        age = t - start
        if kind == "rug":
            rate, p_buy = (1.2, 0.8) if age < dump_at else (0.6, 0.25)
        elif kind == "dud":
            rate, p_buy = 0.25, 0.45
        else:
            rate = 0.5 + 2.5 * min(age / peak_at, 1.0)
            p_buy = 0.72 if age < peak_at else 0.35
        t += rng.expovariate(rate)
        is_buy = rng.random() < p_buy
        if kind == "rug" and dump_at <= age < dump_at + 3 and holdings.get(creator, 0) > 0:
            trader, is_buy = creator, False
            amount = holdings[creator]
        elif is_buy:
            if kind == "rug" and rng.random() < 0.5:
                trader = f"{mint}-bot{rng.randrange(4)}"  # a few repeat wallets
                sol = rng.uniform(0.01, 0.04)
            else:
                wallets += 1
                trader = f"{mint}-w{wallets}"
                sol = rng.lognormvariate(-1.2, 0.9)
            tokens, curve = curve.buy(sol, fee)
            holdings[trader] = holdings.get(trader, 0) + tokens
            out.append(Trade(t, mint, trader, True, sol, tokens, curve.v_sol, curve.v_tokens))
            continue
        else:
            sellers = [w for w, v in holdings.items() if v > 0 and w != creator]
            if not sellers:
                continue
            trader = rng.choice(sellers)
            amount = holdings[trader] * rng.uniform(0.3, 1.0)
        sol, curve = curve.sell(amount, fee)
        holdings[trader] -= amount
        out.append(Trade(t, mint, trader, False, sol, amount, curve.v_sol, curve.v_tokens))
    return out
