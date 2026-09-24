"""Every tunable number in one place, overridable from the environment.

Defaults are deliberately conservative. The single biggest reason memecoin
bots lose money is paying fees and slippage on trades that had no edge, so the
defaults trade rarely and only after Jev has been shown to be calibrated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader so the key never has to be exported by hand."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Config:
    # ------------------------------------------------------------- data feed
    # "rpc": logsSubscribe on a Solana WS RPC at processed commitment,
    #        decoding pump.fun events straight from program logs. Fastest
    #        option without a Geyser stream, and free on the public endpoint.
    # "pumpportal": PumpPortal websocket. Per-token trade streams need a
    #        PumpPortal API key funded with >= 0.02 SOL (append ?api-key=...).
    feed: str = "rpc"
    pumpportal_ws: str = "wss://pumpportal.fun/api/data"
    # the public endpoint works; a private one (Helius, Triton) is faster and steadier
    rpc_ws: str = "wss://api.mainnet-beta.solana.com"
    max_tracked: int = 400  # tokens watched at once

    # ------------------------------------------------------ candidate filter
    # Cheap code-side filters that run before a Jev call is spent.
    eval_min_age_s: float = 15.0
    eval_max_age_s: float = 240.0
    eval_min_trades: int = 12
    eval_min_unique_buyers: int = 8
    eval_min_mcap_sol: float = 32.0
    eval_max_mcap_sol: float = 250.0
    reeval_every_s: float = 20.0
    max_evals_per_token: int = 3

    # -------------------------------------------------------------- barriers
    # A trade is a race: +take_profit first, -stop_loss first, or time out.
    # Jev is asked for the probability of each, which makes EV exact.
    take_profit: float = 0.40
    stop_loss: float = 0.20
    # 0 = off. Exits then match the barrier race Jev is calibrated on exactly.
    trailing_stop: float = 0.0  # from the high-water mark, once in profit
    trail_arm: float = 0.20  # profit needed before the trail arms
    max_hold_s: float = 180.0

    # ---------------------------------------------------------- execution sim
    fee_rate: float = 0.0125  # pump.fun curve fee (protocol + creator)
    tx_cost_sol: float = 0.0006  # base fee + priority fee + tip, per tx
    exec_latency_s: float = 0.6  # decision -> landed, on top of Jev latency
    extra_slippage: float = 0.01  # adverse fill beyond curve math

    # ----------------------------------------------------------------- sizing
    starting_sol: float = 10.0
    max_position_sol: float = 0.5
    min_position_sol: float = 0.05
    kelly_fraction: float = 0.25
    max_open_positions: int = 5
    max_curve_share: float = 0.03  # never be more than 3% of curve SOL

    # ------------------------------------------------------ edge requirement
    min_ev: float = 0.05  # net of all costs, per SOL risked
    max_rug_risk: float = 0.45
    # Jev is often confidently wrong on pump.fun (live data), so its
    # self-reported confidence is not a useful gate. Calibration is.
    min_confidence: float = 0.0

    # -------------------------------------------------- calibration / safety
    # No capital is risked until this many candidates have been labelled with
    # their real outcome, so Jev's probabilities can be checked against reality.
    calibration_warmup: int = 60
    calibration_bins: int = 10
    kill_window: int = 25  # recent trades checked by the kill switch
    kill_max_drawdown: float = 0.25  # of starting balance
    kill_pause_s: float = 900.0

    # --------------------------------------------------------------------- jev
    jev_model: str = "jev-latest"
    jev_timeout_s: float = 3.0
    jev_concurrency: int = 8
    jev_provider: str = "auto"  # auto | jev | mock

    # -------------------------------------------------------------------- io
    data_dir: str = "data"
    record: bool = True

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """MEMEBOT_<FIELD> environment variables override defaults."""
        cfg = cls()
        for f in fields(cls):
            if f.name == "extra":
                continue
            raw = os.environ.get(f"MEMEBOT_{f.name.upper()}")
            if raw is not None:
                setattr(cfg, f.name, _coerce(raw, type(getattr(cfg, f.name))))
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg


def _coerce(raw: str, kind: type):
    if kind is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    return kind(raw)
