"""Offline strategy research on recorded live data, without calling Jev.

Every live Jev entry judgment was stored as a label (mint, time, Jev's
probabilities). The raw events were recorded too. So for each judgment we
can rebuild the token's exact state at that moment and replay what the
price did afterwards, then simulate any entry filter + exit rule with the
same fill model as the live paper broker (curve math, venue fee, slippage,
tx cost, fill delays, our own price impact).

Overfitting is the enemy: trying many rules on one sample will always find
a winner by luck. So the data is split by token launch time into
train (search) / validate / test. The test slice is scored once, at the end.

    python -m memebot.research build        # one pass over data/events-*.jsonl
    python -m memebot.research search       # grid search, walk-forward verdict
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

from .curve import Curve
from .events import from_dict
from .token_state import TokenState

DATASET = Path("data/research.pkl")


# ------------------------------------------------------------------ build


def build(data_dir: str = "data", horizon_s: float = 900.0) -> None:
    labels = []
    for f in glob.glob(f"{data_dir}/labels-jev-*.jsonl"):
        labels += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    by_mint: dict[str, list] = {}
    for l in labels:
        by_mint.setdefault(l["mint"], []).append(l)
    for ls in by_mint.values():
        ls.sort(key=lambda l: l["ts"])
    want = set(by_mint)

    states: dict[str, TokenState] = {}
    paths: dict[str, list] = {}
    pending = {m: list(ls) for m, ls in by_mint.items()}
    rows = []
    files = sorted(glob.glob(f"{data_dir}/events-*.jsonl"))
    for f in files:
        with open(f) as fh:
            for line in fh:
                k = line.find('"mint":"')
                m = line[k + 8 : line.find('"', k + 8)] if k >= 0 else None
                if m not in want:
                    continue  # skip the json parse for tokens nobody judged
                raw = json.loads(line)
                ev = from_dict(raw)
                if ev.kind == "new":
                    if m not in states:
                        states[m] = TokenState.from_event(ev)
                        paths[m] = []
                    continue
                st = states.get(m)
                if st is None or ev.kind != "trade":
                    continue
                # emit feature rows for judgments made before this trade
                q = pending.get(m)
                while q and q[0]["ts"] <= ev.ts:
                    l = q.pop(0)
                    rows.append({**l, "features": st.features(l["ts"]), "fee": st.fee_rate})
                st.apply(ev)
                paths[m].append((ev.ts, ev.v_sol, ev.v_tokens))
    launch = {m: s.created_ts for m, s in states.items()}
    data = {"rows": [r for r in rows if r["mint"] in paths], "paths": paths, "launch": launch}
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    with DATASET.open("wb") as fh:
        pickle.dump(data, fh)
    print(f"built {len(data['rows'])} judgments over {len(paths)} tokens from {len(files)} files -> {DATASET}")


def build_triggers(data_dir: str = "data") -> None:
    """Rule-based entry points on every recorded launch, no Jev involved:
    fixed ages after launch, bonding-curve progress milestones (graduation
    plays), and pullbacks after a run (dip buys)."""
    ages = (15, 45, 90, 180)
    progress = (0.25, 0.5, 0.75, 0.9)
    dips = (0.25, 0.40)
    states: dict[str, TokenState] = {}
    paths: dict[str, list] = {}
    fired: dict[str, set] = {}
    rows = []
    files = sorted(glob.glob(f"{data_dir}/events-*.jsonl"))
    for f in files:
        with open(f) as fh:
            for line in fh:
                ev = from_dict(json.loads(line))
                m = ev.mint
                if ev.kind == "new":
                    if m not in states and ev.venue == "curve":
                        states[m] = TokenState.from_event(ev)
                        paths[m], fired[m] = [], set()
                    continue
                st = states.get(m)
                if st is None or ev.kind != "trade":
                    continue
                st.apply(ev)
                paths[m].append((ev.ts, ev.v_sol, ev.v_tokens))
                age = ev.ts - st.created_ts
                prog = min(max(st.curve.v_sol - 30.0, 0.0) / 85.0, 1.0)
                launch_price = paths[m][0][1] / paths[m][0][2]
                hits = [f"age{a}" for a in ages if age >= a] + [f"prog{int(p*100)}" for p in progress if prog >= p]
                if st.ath >= launch_price * 1.5:
                    dd = 1 - st.curve.price / st.ath
                    hits += [f"dip{int(d*100)}" for d in dips if dd >= d]
                for h in hits:
                    if h not in fired[m]:
                        fired[m].add(h)
                        rows.append({"mint": m, "ts": ev.ts, "trigger": h, "p_tp": 0.0, "p_sl": 0.0,
                                     "rug": 0.0, "features": st.features(ev.ts), "fee": st.fee_rate})
    launch = {m: s.created_ts for m, s in states.items()}
    out = Path("data/research_triggers.pkl")
    with out.open("wb") as fh:
        pickle.dump({"rows": rows, "paths": paths, "launch": launch}, fh)
    print(f"built {len(rows)} trigger entries over {len(paths)} launches -> {out}")


# --------------------------------------------------------------- simulate


@dataclass(frozen=True)
class Rule:
    # entry filter (on Jev's answers and code-side features)
    max_p_sl: float = 1.0
    min_p_tp: float = 0.0
    max_rug: float = 1.0
    min_buyers: int = 0
    max_top5: float = 100.0
    max_sniper: float = 100.0
    min_flow60: float = -1e9
    max_age: float = 1e9
    trigger: str = "any"
    max_creator_sold: float = 100.0
    # exits
    tp: float = 0.4
    sl: float = 0.2
    trail: float = 0.0  # 0 = off; else trailing stop from peak once armed
    trail_arm: float = 0.1
    half_at: float = 0.0  # 0 = off; else sell half at this gain
    hold_s: float = 180.0
    size: float = 0.3


ENTRY_DELAY = 0.8  # Jev latency + landing
EXIT_DELAY = 0.6
SLIP = 0.01
TX = 0.0006


def _state_at(path: list, t: float, start: int = 0) -> tuple[int, Curve | None]:
    """Last recorded curve at or before t, scanning forward from start."""
    i, cur = start, None
    while i < len(path) and path[i][0] <= t:
        cur = Curve(path[i][1], path[i][2])
        i += 1
    return i, cur


def _sell_curve(path: list, t: float, fallback: Curve) -> Curve:
    """Curve to sell into at t: the last one with tokens left (a completed
    curve exits at its final price, like the live broker on migration)."""
    best = fallback
    for ts, vs, vt in path:
        if ts > t:
            break
        if vt > 1e6:
            best = Curve(vs, vt)
    return best


def simulate(row: dict, path: list, r: Rule) -> tuple[float, float] | None:
    """Return (net return on capital, exit time) for one entry, or None."""
    t0 = row["ts"] + ENTRY_DELAY
    i, c = _state_at(path, t0)
    if c is None or c.price <= 0 or c.v_tokens < 1e6:
        return None  # nothing left to buy: the curve is complete
    fee = row["fee"]
    cost = r.size + TX
    tokens, after = c.buy(r.size, fee)
    tokens *= 1 - SLIP
    entry = after.price
    peak = entry
    held = tokens
    banked = 0.0
    took_half = False
    reason_t = None
    while i < len(path):
        t, vs, vt = path[i]
        i += 1
        if t - t0 > r.hold_s:
            reason_t = t0 + r.hold_s
            break
        if vt <= 0:
            reason_t = t  # curve completed: the token migrated
            break
        price = Curve(vs, vt).with_holding(held).price
        peak = max(peak, price)
        g = price / entry - 1
        if r.half_at and not took_half and g >= r.half_at:
            cx = _sell_curve(path, t + EXIT_DELAY, c).with_holding(held)
            sol, _ = cx.sell(held / 2, fee)
            banked += sol * (1 - SLIP) - TX
            held /= 2
            took_half = True
            continue
        if g >= r.tp or g <= -r.sl or (
            r.trail and peak / entry - 1 >= r.trail_arm and price <= peak * (1 - r.trail)
        ):
            reason_t = t
            break
    if reason_t is None:
        reason_t = path[-1][0] if path else t0
    cx = _sell_curve(path, reason_t + EXIT_DELAY, c).with_holding(held)
    sol, _ = cx.sell(held, fee)
    proceeds = banked + max(sol * (1 - SLIP) - TX, 0.0)
    return proceeds / cost - 1.0, reason_t + EXIT_DELAY


def admits(row: dict, r: Rule) -> bool:
    f = row["features"]
    return (
        row["p_sl"] <= r.max_p_sl and row["p_tp"] >= r.min_p_tp and row["rug"] <= r.max_rug
        and f["unique_buyers"] >= r.min_buyers and f["top5_holder_pct"] <= r.max_top5
        and f.get("early_sniper_holding_pct", 0) <= r.max_sniper
        and f["net_flow_60s_sol"] >= r.min_flow60 and f["age_s"] <= r.max_age
        and (r.trigger == "any" or row.get("trigger") == r.trigger)
        and f["creator_sold_pct_of_initial"] <= r.max_creator_sold
    )


EXIT_KEYS = ("tp", "sl", "trail", "half_at", "hold_s")
ENTRY_KEYS = ("max_p_sl", "min_p_tp", "max_rug", "min_buyers", "max_top5", "max_sniper", "min_flow60",
              "max_age", "trigger", "max_creator_sold")


def exit_key(r: Rule) -> tuple:
    return tuple(getattr(r, k) for k in EXIT_KEYS)


_G: dict = {}  # dataset shared with forked workers


def _outcomes_for(key: tuple) -> tuple:
    r = Rule(**dict(zip(EXIT_KEYS, key)))
    return key, [simulate(row, _G["paths"][row["mint"]], r) for row in _G["rows"]]


class Book:
    """Caches each exit rule's outcome per judgment, so entry filters are cheap."""

    def __init__(self, rows: list, paths: dict) -> None:
        self.rows, self.paths, self.cache = rows, paths, {}

    def precompute(self, rules: list[Rule]) -> None:
        """Simulate every distinct exit rule once, in parallel."""
        from concurrent.futures import ProcessPoolExecutor

        _G.update(rows=self.rows, paths=self.paths)
        keys = sorted({exit_key(r) for r in rules} - set(self.cache))
        with ProcessPoolExecutor() as pool:
            for key, outs in pool.map(_outcomes_for, keys, chunksize=4):
                self.cache[key] = outs

    def outcomes(self, r: Rule) -> list:
        key = exit_key(r)
        if key not in self.cache:
            self.cache[key] = [simulate(row, self.paths[row["mint"]], r) for row in self.rows]
        return self.cache[key]

    def run(self, r: Rule, rows_idx: list[int]) -> list[float]:
        """Trade every admitted judgment in rows_idx, one position per token at a time."""
        outs = self.outcomes(r)
        busy_until: dict[str, float] = {}
        rets = []
        for i in rows_idx:
            row = self.rows[i]
            if row["ts"] < busy_until.get(row["mint"], -1) or not admits(row, r) or outs[i] is None:
                continue
            ret, until = outs[i]
            busy_until[row["mint"]] = until
            rets.append(ret)
        return rets


def stats(rets: list[float]) -> dict:
    n = len(rets)
    if n < 2:
        return {"n": n, "mean": rets[0] if rets else 0.0, "t": 0.0, "win": 0.0, "sum": sum(rets)}
    m = sum(rets) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (n - 1)) or 1e-9
    return {"n": n, "mean": m, "t": m / (sd / math.sqrt(n)), "win": sum(x > 0 for x in rets) / n, "sum": sum(rets)}


# ----------------------------------------------------------------- search


def split(data: dict) -> tuple[list, list, list]:
    """Split by token launch time so no token straddles two slices."""
    rows = sorted(data["rows"], key=lambda r: r["ts"])
    order = sorted({r["mint"] for r in rows}, key=lambda m: data["launch"].get(m, 0))
    a, b = int(len(order) * 0.5), int(len(order) * 0.75)
    part = {m: 0 if i < a else 1 if i < b else 2 for i, m in enumerate(order)}
    return tuple([r for r in rows if part[r["mint"]] == k] for k in range(3))


GRID = {
    "max_p_sl": [1.0, 0.45, 0.35],
    "max_rug": [1.0, 0.4, 0.25],
    "min_buyers": [0, 15, 30],
    "max_sniper": [100.0, 5.0],
    "min_flow60": [-1e9, 0.0, 1.0],
    "tp": [0.15, 0.3, 0.6, 1.0],
    "sl": [0.1, 0.2, 0.35],
    "trail": [0.0, 0.15],
    "half_at": [0.0, 0.25],
    "hold_s": [60.0, 180.0, 600.0],
}


TRIGGER_GRID = {
    "trigger": ["age15", "age45", "age90", "age180", "prog25", "prog50", "prog75", "prog90", "dip25", "dip40"],
    "min_buyers": [0, 20, 50],
    "max_top5": [100.0, 25.0],
    "max_creator_sold": [100.0, 0.0],
    "min_flow60": [-1e9, 0.0],
    "tp": [0.2, 0.5, 1.0],
    "sl": [0.1, 0.25],
    "trail": [0.0, 0.2],
    "half_at": [0.0],
    "hold_s": [60.0, 300.0, 900.0],
}


def search(min_trades: int = 30, dataset: str = "jev") -> None:
    global GRID
    path = DATASET if dataset == "jev" else Path("data/research_triggers.pkl")
    if dataset != "jev":
        GRID = TRIGGER_GRID
    with path.open("rb") as fh:
        data = pickle.load(fh)
    train, val, test = split(data)
    rows = train + val + test
    idx = {"train": range(0, len(train)), "val": range(len(train), len(train) + len(val)),
           "test": range(len(train) + len(val), len(rows))}
    book = Book(rows, data["paths"])
    print(f"judgments: train {len(train)}  validate {len(val)}  test {len(test)} (test is scored once)")
    base = stats(book.run(Rule(tp=10, sl=10), idx["train"]))
    print(f"buy every judgment, hold 3 min: {base['n']} trades, mean {base['mean']:+.2%} per trade")

    exit_grid = {k: v for k, v in GRID.items() if k in EXIT_KEYS}
    entry_grid = {k: v for k, v in GRID.items() if k not in EXIT_KEYS}
    rules = [Rule(**dict(zip(list(entry_grid) + list(exit_grid), c)))
             for c in itertools.product(*entry_grid.values(), *exit_grid.values())]
    print(f"searching {len(rules)} rules ({len(list(itertools.product(*exit_grid.values())))} exit rules simulated once each)...", flush=True)
    book.precompute(rules)
    results = []
    for r in rules:
        s = stats(book.run(r, idx["train"]))
        if s["n"] >= min_trades:
            results.append((s["mean"], s, r))
    results.sort(key=lambda x: x[0], reverse=True)
    keys = list(GRID)

    def diff(r):
        return {k: getattr(r, k) for k in keys if getattr(r, k) != getattr(Rule(), k)}

    print(f"\n{len(results)} rules with {min_trades}+ train trades. Top 10 on TRAIN, then VALIDATE:")
    print(f"{'train mean':>10} {'n':>5} {'t':>6} | {'val mean':>9} {'n':>4} {'t':>6} | rule")
    shortlist = []
    for mean, s, r in results[:10]:
        v = stats(book.run(r, idx["val"]))
        shortlist.append((v, s, r))
        print(f"{mean:+10.2%} {s['n']:5d} {s['t']:+6.2f} | {v['mean']:+9.2%} {v['n']:4d} {v['t']:+6.2f} | {diff(r)}")
    positive = sum(1 for m, _, _ in results if m > 0)
    print(f"\n{positive} of {len(results)} rules were profitable on train (luck alone produces some)")
    shortlist.sort(key=lambda x: (x[0]["mean"] if x[0]["n"] >= 15 else -9), reverse=True)
    v, s, r = shortlist[0]
    t = stats(book.run(r, idx["test"]))
    print(f"\nchosen on validation: {diff(r)}")
    print(f"TEST (never seen): {t['n']} trades, mean {t['mean']:+.2%} per trade, win {t['win']:.0%}, t={t['t']:+.2f}, "
          f"total {t['sum'] * r.size:+.3f} SOL at {r.size} SOL/trade")
    ok = t["n"] >= 20 and t["mean"] > 0 and t["t"] > 2
    print("VERDICT:", "profitable out of sample" if ok else "NOT shown profitable out of sample")
    with open("data/research_choice.json", "w") as fh:
        json.dump({"rule": diff(r), "train": s, "val": v, "test": t, "verdict": ok}, fh, indent=1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="memebot.research")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("build-triggers")
    p = sub.add_parser("search")
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--dataset", choices=["jev", "triggers"], default="jev")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build()
    elif a.cmd == "build-triggers":
        build_triggers()
    else:
        search(a.min_trades, a.dataset)


if __name__ == "__main__":
    main(sys.argv[1:])
