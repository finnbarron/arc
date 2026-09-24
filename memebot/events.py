"""Normalised market events. Every feed emits these, the recorder writes them,
and replay reads them back, so live and backtest run the same code."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Union

# pump.fun launches every curve with the same virtual reserves.
INITIAL_V_SOL = 30.0
INITIAL_V_TOKENS = 1_073_000_000.0
TOTAL_SUPPLY = 1_000_000_000.0


@dataclass(frozen=True, slots=True)
class NewToken:
    ts: float
    mint: str
    name: str
    symbol: str
    creator: str
    v_sol: float = INITIAL_V_SOL
    v_tokens: float = INITIAL_V_TOKENS
    initial_buy_tokens: float = 0.0
    uri: str = ""
    venue: str = "curve"  # "curve" (pump.fun bonding curve) | "amm" (PumpSwap pool)
    base_mint: str = ""  # for amm pools: the token's mint (``mint`` is the pool)
    kind: str = "new"


@dataclass(frozen=True, slots=True)
class Trade:
    ts: float
    mint: str
    trader: str
    is_buy: bool
    sol: float
    tokens: float
    v_sol: float  # reserves *after* this trade (virtual on the curve, real in a pool)
    v_tokens: float
    signature: str = ""
    creator: str = ""  # token creator when the event carries it
    venue: str = "curve"
    fee_rate: float = 0.0  # total fee on this venue when the event carries it
    kind: str = "trade"


@dataclass(frozen=True, slots=True)
class Migration:
    ts: float
    mint: str
    kind: str = "migration"


Event = Union[NewToken, Trade, Migration]
_KINDS = {"new": NewToken, "trade": Trade, "migration": Migration}


def to_dict(event: Event) -> dict:
    return asdict(event)


def from_dict(raw: dict) -> Event:
    cls = _KINDS[raw["kind"]]
    return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
