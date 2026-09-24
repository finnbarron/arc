"""Hold Jev to reality before any SOL is risked on it.

Every token Jev judges gets a *shadow label*: the bot keeps watching the
tape and records which barrier the price actually hit first and the return
it would have realised, whether or not it bought. That gives hundreds of
labels an hour instead of one per trade.

From those labels:

* raw ``p_tp`` / ``p_sl`` are mapped to observed frequencies (nearest-
  neighbour binning, shrunk to the base rate), so an overconfident model
  cannot talk the bot into bad trades;
* the payoff of each outcome is the *measured* mean return, which captures
  stop-loss gaps and pump.fun's habit of dumping straight through a stop;
* Jev's discrimination (AUC) is tracked, and trading stays off until it is
  clearly better than a coin flip.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .brain import NEITHER, SL, TP, Judgment


@dataclass
class Label:
    mint: str
    ts: float
    p_tp: float
    p_sl: float
    rug: float
    outcome: str
    ret: float  # price return entry->exit, before costs


@dataclass
class Pending:
    mint: str
    ts: float
    judgment: Judgment
    due: float  # when a real order would have filled
    entry: float = 0.0  # price at fill time
    hi: float = 0.0
    lo: float = 0.0


class Labeler:
    """Runs the barrier race on every judged token using live prices."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.pending: dict[str, list[Pending]] = {}

    def add(self, mint: str, ts: float, j: Judgment) -> None:
        due = ts + j.latency_s + self.cfg.exec_latency_s
        self.pending.setdefault(mint, []).append(Pending(mint, ts, j, due))

    def busy(self, mint: str) -> bool:
        return bool(self.pending.get(mint))

    def on_price(self, mint: str, ts: float, price: float) -> list[Label]:
        out: list[Label] = []
        keep: list[Pending] = []
        for p in self.pending.get(mint, []):
            if ts < p.due:
                keep.append(p)
                continue
            if p.entry == 0.0:
                p.entry = p.hi = p.lo = price
                keep.append(p)
                continue
            r = price / p.entry - 1.0
            if r >= self.cfg.take_profit:
                out.append(self._label(p, TP, r))
            elif r <= -self.cfg.stop_loss:
                out.append(self._label(p, SL, r))
            elif ts - p.due >= self.cfg.max_hold_s:
                out.append(self._label(p, NEITHER, r))
            else:
                keep.append(p)
        if keep:
            self.pending[mint] = keep
        else:
            self.pending.pop(mint, None)
        return out

    def expire(self, mint: str, ts: float, price: float) -> list[Label]:
        """Token went quiet or migrated: settle what is left at last price."""
        out = []
        for p in self.pending.pop(mint, []):
            if p.entry:
                r = price / p.entry - 1.0
                outcome = TP if r >= self.cfg.take_profit else SL if r <= -self.cfg.stop_loss else NEITHER
                out.append(self._label(p, outcome, r))
        return out

    def _label(self, p: Pending, outcome: str, r: float) -> Label:
        j = p.judgment
        return Label(p.mint, p.ts, j.p_tp, j.p_sl, j.rug_risk, outcome, r)


class Calibrator:
    def __init__(self, cfg, path: Path | None = None) -> None:
        self.cfg = cfg
        self.labels: list[Label] = []
        self._fh = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.strip():
                        self.labels.append(Label(**json.loads(line)))
            self._fh = path.open("a", buffering=1)

    def add(self, label: Label) -> None:
        self.labels.append(label)
        if self._fh:
            self._fh.write(json.dumps(asdict(label)) + "\n")

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def warmed_up(self) -> bool:
        return self.n >= self.cfg.calibration_warmup

    def base_rate(self, outcome: str) -> float:
        if not self.labels:
            return {TP: 0.1, SL: 0.6, NEITHER: 0.3}[outcome]
        return sum(l.outcome == outcome for l in self.labels) / self.n

    def mean_return(self, outcome: str) -> float:
        """Measured payoff of an outcome. Priors are the nominal barriers,
        made worse to reflect gaps, until real data replaces them."""
        prior = {
            TP: self.cfg.take_profit,
            SL: -self.cfg.stop_loss * 1.5,
            NEITHER: -0.05,
        }[outcome]
        rets = [l.ret for l in self.labels if l.outcome == outcome]
        m = 5.0  # prior weight
        return (sum(rets) + m * prior) / (len(rets) + m)

    def calibrate(self, raw: float, attr: str, outcome: str) -> float:
        """Empirical P(outcome) among the labels whose raw score is nearest."""
        if not self.labels:
            return raw
        # sort on the score alone: ties keep arrival order instead of
        # grouping all misses before all hits
        pts = sorted(((getattr(l, attr), l.outcome == outcome) for l in self.labels), key=lambda p: p[0])
        xs = [x for x, _ in pts]
        k = max(20, self.n // self.cfg.calibration_bins)
        k = min(k, self.n)
        i = (bisect.bisect_left(xs, raw) + bisect.bisect_right(xs, raw)) // 2
        lo, hi = max(0, i - k // 2), min(self.n, i - k // 2 + k)
        lo = max(0, hi - k)
        hits = sum(h for _, h in pts[lo:hi])
        base = self.base_rate(outcome)
        m = 10.0
        return (hits + m * base) / ((hi - lo) + m)

    def auc(self, attr: str = "p_tp", outcome: str = TP) -> float:
        """Probability a random positive outranks a random negative."""
        pos = [getattr(l, attr) for l in self.labels if l.outcome == outcome]
        neg = [getattr(l, attr) for l in self.labels if l.outcome != outcome]
        if not pos or not neg:
            return 0.5
        neg.sort()
        wins = 0.0
        for p in pos:
            lo = bisect.bisect_left(neg, p)
            hi = bisect.bisect_right(neg, p)
            wins += lo + 0.5 * (hi - lo)
        return wins / (len(pos) * len(neg))

    def brier(self, attr: str = "p_tp", outcome: str = TP) -> float:
        if not self.labels:
            return float("nan")
        return sum((getattr(l, attr) - (l.outcome == outcome)) ** 2 for l in self.labels) / self.n

    def skill_ok(self) -> bool:
        """Jev's ranking must carry real information before we trade on it.

        Direction does not matter: ``calibrate`` maps raw scores to observed
        frequencies, so a model that is reliably *wrong* (live data showed
        Jev chasing pumps that then dump) is as usable as one that is
        reliably right. What matters is that the separation is significant,
        on either the take-profit or the stop-loss question. The EV gate
        still has to clear costs on top of this.
        """
        if not self.warmed_up:
            return False
        for attr, outcome in (("p_tp", TP), ("p_sl", SL)):
            n_pos = sum(l.outcome == outcome for l in self.labels)
            n_neg = self.n - n_pos
            if min(n_pos, n_neg) < 5:
                continue
            a = self.auc(attr, outcome)
            # standard error of AUC under the null (conservative)
            se = math.sqrt(0.25 / min(n_pos, n_neg))
            if abs(a - 0.5) > 2 * se and abs(a - 0.5) >= self.cfg.extra.get("min_auc_gap", 0.05):
                return True
        return False

    def summary(self) -> dict:
        return {
            "labels": self.n,
            "base_tp": round(self.base_rate(TP), 3),
            "base_sl": round(self.base_rate(SL), 3),
            "ret_tp": round(self.mean_return(TP), 3),
            "ret_sl": round(self.mean_return(SL), 3),
            "ret_neither": round(self.mean_return(NEITHER), 3),
            "auc_tp": round(self.auc(), 3),
            "auc_sl": round(self.auc("p_sl", SL), 3),
            "brier_tp": round(self.brier(), 4) if self.labels else None,
            "skill_ok": self.skill_ok(),
        }
