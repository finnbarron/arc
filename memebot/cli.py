"""memebot: paper-trade pump.fun memecoins on live data, decisions by Jev.

    python -m memebot doctor              check the feed and the Jev key
    python -m memebot run                 live paper trading (records everything)
    python -m memebot record              record the live feed only, no Jev calls
    python -m memebot replay FILE         re-run the engine over recorded events
    python -m memebot tune FILE           walk-forward parameter search on a recording
    python -m memebot synthetic OUT       write a SYNTHETIC event file for offline tests
    python -m memebot report              summarise data/trades.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import signal
import time
from dataclasses import replace
from pathlib import Path

from .brain import Brain, JevCache, make_backend
from .broker import PaperBroker
from .calibration import Calibrator
from .config import Config, load_dotenv
from .engine import Engine
from .feeds import Recorder, make_feed, read_events

log = logging.getLogger("memebot")


def _labels_path(cfg: Config, provider: str) -> Path:
    # labels only mean something for one barrier setup and one model
    tag = f"{provider}-tp{int(cfg.take_profit*100)}-sl{int(cfg.stop_loss*100)}-h{int(cfg.max_hold_s)}"
    return Path(cfg.data_dir) / f"labels-{tag}.jsonl"


def build(cfg: Config, *, live: bool, persist: bool, feed=None) -> Engine:
    backend = make_backend(cfg)
    data = Path(cfg.data_dir)
    # Real Jev answers are cached on disk so replays and tuning re-use paid
    # calls. Mock answers are free and never persisted.
    cache = JevCache(data / "jev_cache.jsonl" if backend.provider == "jev" else None)
    brain = Brain(cfg, backend, cache)
    cal = Calibrator(cfg, _labels_path(cfg, backend.provider) if persist else None)
    broker = PaperBroker(cfg, data / "trades.jsonl" if persist else None)
    return Engine(cfg, brain, broker, cal, feed=feed, live=live)


def _print_status(engine: Engine) -> None:
    s = engine.snapshot()
    c = s["calibration"]
    print(
        f"[{time.strftime('%H:%M:%S')}] equity {s['equity_sol']:.3f} SOL "
        f"(pnl {s['pnl_sol']:+.3f}) | open {s['open']} closed {s['closed']} "
        f"win {s['win_rate'] if s['win_rate'] is not None else '-'} | "
        f"tokens {s['tokens_seen']} tracked {s['tracked']} | jev {s['jev_calls']} calls "
        f"p50 {s['jev_p50_ms']}ms ${s['jev_usd']} | labels {c['labels']} "
        f"auc {c['auc_tp']} | trading: {s['trading']}",
        flush=True,
    )


# --------------------------------------------------------------- live


async def cmd_run(cfg: Config, record_only: bool = False) -> None:
    feed = make_feed(cfg)
    data = Path(cfg.data_dir)
    rec = Recorder(data / f"events-{time.strftime('%Y%m%d-%H%M%S')}.jsonl") if cfg.record else None
    engine = None if record_only else build(cfg, live=True, persist=True, feed=feed)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # windows
            pass

    async def pump() -> None:
        n = 0
        async for ev in feed.events():
            if rec:
                rec.write(ev)
            n += 1
            if engine:
                await engine.on_event(ev)
            elif n % 500 == 0:
                print(f"recorded {n} events", flush=True)
            if stop.is_set():
                return

    async def ticker() -> None:
        last_status = 0.0
        while not stop.is_set():
            now = time.time()
            if engine:
                await engine.on_clock(now)
                if now - last_status >= 15:
                    _print_status(engine)
                    (data / "status.json").write_text(json.dumps(engine.snapshot(), indent=2))
                    last_status = now
            await asyncio.sleep(0.25)

    tasks = [asyncio.create_task(pump()), asyncio.create_task(ticker())]
    await stop.wait()
    for t in tasks:
        t.cancel()
    if engine:
        await engine.drain()
        await engine.close_all(time.time())
        _print_status(engine)
        print(json.dumps(engine.snapshot(), indent=2))
    if rec:
        rec.close()


# ------------------------------------------------------------- replay


async def replay(cfg: Config, events, *, persist: bool = False) -> Engine:
    engine = build(cfg, live=False, persist=persist)
    for ev in events:
        await engine.on_event(ev)
    if events:
        end = events[-1].ts + cfg.max_hold_s + 5
        await engine.on_clock(end)
        await engine.close_all(end)
    return engine


async def cmd_replay(cfg: Config, path: Path) -> None:
    events = read_events(path)
    engine = await replay(cfg, events)
    print(json.dumps(engine.snapshot(), indent=2))
    _print_trades(engine)


def _print_trades(engine: Engine, limit: int = 20) -> None:
    for t in engine.broker.closed[-limit:]:
        print(f"  {t.symbol:>8} {t.reason:>12} ret {t.ret:+7.1%} pnl {t.pnl_sol:+.4f} ev {t.ev:+.3f}")


def _tune_one(args) -> tuple:
    cfg, path, params, cut = args
    c = replace(cfg, **params)
    engine = asyncio.run(replay(c, read_events(path)))
    train = [t for t in engine.broker.closed if t.entry_ts < cut]
    test = [t for t in engine.broker.closed if t.entry_ts >= cut]
    return sum(t.pnl_sol for t in train), [t.ret for t in test], sum(t.pnl_sol for t in test), params


def _tstat(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return m / (sd / len(xs) ** 0.5) if sd else 0.0


def cmd_tune(cfg: Config, path: Path, split: float) -> None:
    """Pick parameters on the first part of a recording, then report how
    they do on the part they never saw. Only the test number counts.

    Calibration inside each replay is already walk-forward (every decision
    only sees labels from before it), so the remaining overfitting risk is
    the parameter choice itself, which the held-out split measures."""
    from concurrent.futures import ProcessPoolExecutor

    events = read_events(path)
    cut = events[0].ts + (events[-1].ts - events[0].ts) * split
    grid = {
        "min_ev": [0.02, 0.05, 0.10, 0.20],
        "max_rug_risk": [0.3, 0.45, 0.6],
        "take_profit": [0.3, 0.5],
        "stop_loss": [0.15, 0.25],
    }
    keys = list(grid)
    jobs = [(cfg, path, dict(zip(keys, combo)), cut) for combo in itertools.product(*grid.values())]
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(_tune_one, jobs))
    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"{'train pnl':>10} {'TEST pnl':>10} {'test n':>7} {'test t':>7}  params")
    for train, rets, test, params in rows[:10]:
        print(f"{train:+10.4f} {test:+10.4f} {len(rets):7d} {_tstat(rets):+7.2f}  {params}")
    train, rets, test, params = rows[0]
    t = _tstat(rets)
    if len(rets) >= 30 and test > 0 and t > 2.0:
        verdict = "PROFITABLE out of sample (t > 2 over 30+ trades)"
    elif test > 0:
        verdict = f"positive out of sample but NOT significant ({len(rets)} trades, t={t:+.2f}); record more data"
    else:
        verdict = "NOT profitable out of sample; do not trust these params"
    print(f"\nbest-on-train params: {test:+.4f} SOL on unseen data -> {verdict}")
    print("to use them:")
    for k, v in params.items():
        print(f"  export MEMEBOT_{k.upper()}={v}")


# ------------------------------------------------------------- misc


def cmd_report(cfg: Config) -> None:
    path = Path(cfg.data_dir) / "trades.jsonl"
    if not path.exists():
        print("no trades yet")
        return
    sells = [json.loads(l) for l in path.read_text().splitlines() if '"kind": "sell"' in l]
    if not sells:
        print("no closed trades yet")
        return
    pnl = [s["pnl_sol"] for s in sells]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    by_reason: dict[str, list[float]] = {}
    for s in sells:
        by_reason.setdefault(s["reason"], []).append(s["pnl_sol"])
    print(f"closed trades {len(pnl)}  total pnl {sum(pnl):+.4f} SOL  win rate {len(wins)/len(pnl):.1%}")
    if wins and losses:
        print(f"avg win {sum(wins)/len(wins):+.4f}  avg loss {sum(losses)/len(losses):+.4f}  "
              f"profit factor {sum(wins)/-sum(losses):.2f}")
    for reason, ps in sorted(by_reason.items()):
        print(f"  {reason:>14}: {len(ps):4d} trades  {sum(ps):+.4f} SOL")


async def cmd_doctor(cfg: Config) -> None:
    import os

    print(f"feed: {cfg.feed}")
    feed = make_feed(cfg)
    try:
        async def first():
            async for ev in feed.events():
                return ev
        ev = await asyncio.wait_for(first(), 20)
        print(f"  OK, first event: {ev.kind} {ev.mint}")
    except Exception as exc:
        print(f"  FAILED: {exc!r}")
    print("jev:", "key set" if os.environ.get("TYPESAFE_API_KEY") else "NO KEY (mock only)")
    if os.environ.get("TYPESAFE_API_KEY"):
        from typesafe_sdk import AsyncTypeSafeClient, Noul

        try:
            async with AsyncTypeSafeClient(timeout=10) as client:
                t = time.perf_counter()
                r = await client.system_one(
                    state={"note": "connectivity check"},
                    questions={"ok": Noul(instructions="Is this a connectivity check?")},
                )
                print(f"  OK, model {r.model}, p={r.nouls['ok'].noul:.2f}, {1000*(time.perf_counter()-t):.0f}ms")
        except Exception as exc:
            print(f"  FAILED: {exc!r}")


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="memebot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--provider", choices=["auto", "jev", "mock"], help="Jev backend")
    ap.add_argument("--feed", choices=["pumpportal", "rpc"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    sub.add_parser("run")
    sub.add_parser("record")
    p = sub.add_parser("replay")
    p.add_argument("file", type=Path)
    p = sub.add_parser("tune")
    p.add_argument("file", type=Path)
    p.add_argument("--split", type=float, default=0.6)
    p = sub.add_parser("synthetic")
    p.add_argument("out", type=Path)
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    sub.add_parser("report")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("typesafe_sdk", "httpx2", "httpx", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = Config.from_env(jev_provider=args.provider, feed=args.feed)

    if args.cmd == "run":
        asyncio.run(cmd_run(cfg))
    elif args.cmd == "record":
        asyncio.run(cmd_run(cfg, record_only=True))
    elif args.cmd == "replay":
        asyncio.run(cmd_replay(cfg, args.file))
    elif args.cmd == "tune":
        cmd_tune(cfg, args.file, args.split)
    elif args.cmd == "synthetic":
        from .events import to_dict
        from .synthetic import generate

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fh:
            for ev in generate(args.tokens, args.seed, cfg.fee_rate):
                fh.write(json.dumps(to_dict(ev)) + "\n")
        print(f"wrote SYNTHETIC events to {args.out} (not market data)")
    elif args.cmd == "report":
        cmd_report(cfg)
    elif args.cmd == "doctor":
        asyncio.run(cmd_doctor(cfg))
