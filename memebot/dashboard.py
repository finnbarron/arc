"""Build one compact JSON snapshot of the bot's progress for the dashboard.

Reads only files the running bot already writes, so it never needs the
bot restarted:

    python -m memebot.dashboard > data/dashboard.json
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import re
import sys
from pathlib import Path

from .calibration import Calibrator, Label
from .config import Config

STATUS_RE = re.compile(
    r"^\[(\d\d:\d\d:\d\d)\] equity ([\d.]+) SOL .*?\| open (\d+) closed (\d+) .*?"
    r"tokens (\d+) .*?jev (\d+) calls .*?labels (\d+) auc ([\d.]+) \| trading: (.*)$"
)
MODE_RE = re.compile(r"^=== (.*?) run (\d\d:\d\d:\d\d): (.*) ===$")


def _day_ts(day: dt.date, hms: str) -> float:
    t = dt.datetime.strptime(hms, "%H:%M:%S").time()
    return dt.datetime.combine(day, t, tzinfo=dt.timezone.utc).timestamp()


def equity_series(log: Path, day: dt.date, max_points: int = 400) -> tuple[list, list]:
    rows, marks = [], []
    if not log.exists():
        return rows, marks
    for line in log.read_text(errors="replace").splitlines():
        m = STATUS_RE.match(line)
        if m:
            rows.append([_day_ts(day, m[1]), float(m[2]), int(m[5]), int(m[7]), float(m[8])])
            continue
        m = MODE_RE.match(line)
        if m:
            marks.append({"ts": _day_ts(day, m[2]), "label": f"{m[1]} settings", "detail": m[3]})
    if len(rows) > max_points:
        step = len(rows) / max_points
        rows = [rows[int(i * step)] for i in range(max_points - 1)] + [rows[-1]]
    return rows, marks


def trades(path: Path, labels: list | None = None) -> tuple[list, list]:
    # the shadow label taken at the same decision says what the old fixed
    # +40% / -20% / 3 min rule did with the very same entry
    fixed = {(l.mint, round(l.ts, 2)): l for l in (labels or [])}
    buys, sells = {}, []
    if not path.exists():
        return [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["kind"] == "buy":
            buys[row["mint"]] = row
        elif row["kind"] == "sell":
            b = buys.pop(row["mint"], {})
            sells.append({
                "mint": row["mint"], "symbol": row["symbol"], "entry_ts": row["entry_ts"],
                "exit_ts": row["exit_ts"], "cost": round(row["cost_sol"], 4),
                "pnl": round(row["pnl_sol"], 4), "ret": round(row["ret"], 4),
                "reason": row["reason"], "ev": row.get("ev"), "p_tp": b.get("p_tp"),
                "p_sl": b.get("p_sl"), "rug": b.get("rug"),
                "took_half": row.get("took_initials", False),
                "checks": row.get("exit_checks", 0), "last_exit": row.get("last_exit") or None,
                "fixed_rule_move": (lambda l: round(l.ret, 4) if l else None)(
                    fixed.get((row["mint"], round(b.get("decided_ts", -1), 2)))),
            })
    open_ = [{"mint": b["mint"], "symbol": b.get("symbol", ""), "sol": b["sol"], "ts": b["ts"],
              "ev": b.get("ev"), "p_tp": b.get("p_tp"), "p_sl": b.get("p_sl")} for b in buys.values()]
    return sells[-200:], open_


def calibration(labels: list[Label]) -> dict:
    out: dict = {"n": len(labels)}
    if len(labels) < 25:
        return out

    def buckets(attr: str) -> list:
        xs = sorted(labels, key=lambda l: getattr(l, attr))
        k = len(xs) // 5
        rows = []
        for i in range(5):
            ch = xs[i * k:(i + 1) * k] if i < 4 else xs[4 * k:]
            rows.append({
                "lo": round(getattr(ch[0], attr), 3), "hi": round(getattr(ch[-1], attr), 3),
                "n": len(ch),
                "tp": round(sum(l.outcome == "take_profit_first" for l in ch) / len(ch), 3),
                "sl": round(sum(l.outcome == "stop_loss_first" for l in ch) / len(ch), 3),
                "ret": round(sum(l.ret for l in ch) / len(ch), 4),
            })
        return rows

    out["by_p_tp"] = buckets("p_tp")
    out["by_p_sl"] = buckets("p_sl")
    cal = Calibrator(Config())
    hist = []
    ordered = sorted(labels, key=lambda l: l.ts)
    for n in range(100, len(ordered) + 1, 100):
        cal.labels = ordered[:n]
        hist.append([n, round(cal.auc("p_tp", "take_profit_first"), 3), round(cal.auc("p_sl", "stop_loss_first"), 3)])
    out["auc_history"] = hist
    counts: dict = {}
    for l in labels:
        counts[l.outcome] = counts.get(l.outcome, 0) + 1
    out["outcomes"] = counts
    return out


def build(data_dir: str = "data") -> dict:
    d = Path(data_dir)
    status = json.loads((d / "status.json").read_text()) if (d / "status.json").exists() else {}
    day = dt.datetime.now(dt.timezone.utc).date()
    equity, marks = equity_series(d / "run.log", day)
    labels = []
    for f in glob.glob(str(d / "labels-jev-*.jsonl")):
        labels += [Label(**json.loads(l)) for l in Path(f).read_text().splitlines() if l.strip()]
    closed, open_ = trades(d / "trades.jsonl", labels)
    return {
        "updated": dt.datetime.now(dt.timezone.utc).timestamp(),
        "starting_sol": 10.0,
        "status": status,
        "equity": equity,  # [ts, equity, tokens_seen, labels, auc]
        "marks": marks,
        "closed": closed,
        "open": open_,
        "calibration": calibration(labels),
    }


if __name__ == "__main__":
    json.dump(build(sys.argv[1] if len(sys.argv) > 1 else "data"), sys.stdout, separators=(",", ":"))
