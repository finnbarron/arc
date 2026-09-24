"""Live and recorded market data.

Speed ranking for pump.fun data, fastest first:

1. Your own Geyser/Yellowstone gRPC stream (Helius LaserStream, Triton,
   a self-hosted node). Paid, needs protobuf plumbing, not implemented here.
2. ``RpcLogsFeed``: ``logsSubscribe`` on a Solana websocket RPC at
   ``processed`` commitment, decoding pump.fun's Anchor events from the
   program logs ourselves. No middleman. Use a good RPC (Helius, Triton,
   QuickNode) located near you; the public endpoint rate-limits hard.
3. ``PumpPortalFeed``: PumpPortal's free websocket. It relays the same
   events with a small extra hop. Zero setup, which is why it is the default.

All feeds yield ``events.Event`` objects stamped with local receive time,
because that is the moment the bot could first have acted on them.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import struct
import time
from pathlib import Path
from typing import AsyncIterator, Iterable

from .events import (
    INITIAL_V_SOL,
    INITIAL_V_TOKENS,
    Event,
    Migration,
    NewToken,
    Trade,
    from_dict,
    to_dict,
)

log = logging.getLogger(__name__)

PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
LAMPORTS = 1_000_000_000
TOKEN_UNITS = 1_000_000  # pump.fun tokens have 6 decimals


class Feed:
    """Interface: an async stream of events plus per-mint trade subscriptions."""

    async def events(self) -> AsyncIterator[Event]:  # pragma: no cover
        raise NotImplementedError
        yield

    async def watch(self, mints: Iterable[str]) -> None:
        """Start receiving trades for these mints (no-op for firehose feeds)."""

    async def unwatch(self, mints: Iterable[str]) -> None:
        """Stop receiving trades for these mints."""


# --------------------------------------------------------------- PumpPortal


class PumpPortalFeed(Feed):
    """wss://pumpportal.fun/api/data. One connection, per-mint trade subs.

    PumpPortal asks clients to reuse a single socket, so subscriptions are
    batched onto it and replayed after a reconnect.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._ws = None
        self._watched: set[str] = set()
        self._send_lock = asyncio.Lock()

    async def events(self) -> AsyncIterator[Event]:
        import websockets

        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    self.url, ping_interval=20, max_size=2**22
                ) as ws:
                    self._ws = ws
                    backoff = 1.0
                    await self._send({"method": "subscribeNewToken"})
                    await self._send({"method": "subscribeMigration"})
                    if self._watched:
                        await self._send(
                            {"method": "subscribeTokenTrade", "keys": sorted(self._watched)}
                        )
                    log.info("pumpportal connected")
                    async for raw in ws:
                        event = parse_pumpportal(json.loads(raw), time.time())
                        if event is not None:
                            yield event
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network drops are routine
                log.warning("pumpportal disconnected: %s; retry in %.0fs", exc, backoff)
                self._ws = None
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _send(self, msg: dict) -> None:
        if self._ws is None:
            return
        async with self._send_lock:
            await self._ws.send(json.dumps(msg))

    async def watch(self, mints: Iterable[str]) -> None:
        new = [m for m in mints if m not in self._watched]
        if new:
            self._watched.update(new)
            await self._send({"method": "subscribeTokenTrade", "keys": new})

    async def unwatch(self, mints: Iterable[str]) -> None:
        gone = [m for m in mints if m in self._watched]
        if gone:
            self._watched.difference_update(gone)
            await self._send({"method": "unsubscribeTokenTrade", "keys": gone})


def parse_pumpportal(msg: dict, now: float) -> Event | None:
    tx = msg.get("txType")
    mint = msg.get("mint")
    if not tx or not mint:
        return None  # subscription acks and errors
    try:
        if tx == "create":
            v_tokens = float(msg.get("vTokensInBondingCurve", INITIAL_V_TOKENS))
            return NewToken(
                ts=now,
                mint=mint,
                name=str(msg.get("name", ""))[:64],
                symbol=str(msg.get("symbol", ""))[:32],
                creator=msg.get("traderPublicKey", ""),
                v_sol=float(msg.get("vSolInBondingCurve", INITIAL_V_SOL)),
                v_tokens=v_tokens,
                initial_buy_tokens=float(msg.get("initialBuy", 0.0) or 0.0),
                uri=str(msg.get("uri", ""))[:200],
            )
        if tx in ("buy", "sell"):
            if "vSolInBondingCurve" not in msg:
                return None  # post-migration AMM trades lack curve reserves
            return Trade(
                ts=now,
                mint=mint,
                trader=msg.get("traderPublicKey", ""),
                is_buy=tx == "buy",
                sol=float(msg.get("solAmount", 0.0)),
                tokens=float(msg.get("tokenAmount", 0.0)),
                v_sol=float(msg["vSolInBondingCurve"]),
                v_tokens=float(msg["vTokensInBondingCurve"]),
                signature=msg.get("signature", ""),
            )
        if tx == "migrate":
            return Migration(ts=now, mint=mint)
    except (TypeError, ValueError, KeyError) as exc:
        log.debug("bad pumpportal message %s: %s", msg, exc)
    return None


# ----------------------------------------------------------- raw RPC logs


def _disc(name: str) -> bytes:
    return hashlib.sha256(f"event:{name}".encode()).digest()[:8]


TRADE_DISC = _disc("TradeEvent")
CREATE_DISC = _disc("CreateEvent")
COMPLETE_DISC = _disc("CompleteEvent")

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58encode(raw: bytes) -> str:
    n = int.from_bytes(raw, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = _B58[r] + out
    pad = len(raw) - len(raw.lstrip(b"\0"))
    return "1" * pad + out


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("short event")
        chunk = self.data[self.pos : self.pos + n]
        self.pos += n
        return chunk

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.take(8))[0]

    def pubkey(self) -> str:
        return b58encode(self.take(32))

    def string(self) -> str:
        (n,) = struct.unpack("<I", self.take(4))
        return self.take(n).decode("utf-8", "replace")

    def remaining(self) -> int:
        return len(self.data) - self.pos


def decode_pump_event(data: bytes, now: float, signature: str = "") -> Event | None:
    """Decode one ``Program data:`` payload from the pump.fun program.

    Only the leading fields are read. pump.fun has appended fields to these
    events over time (fees, creator), and reading a prefix keeps working
    across those upgrades.
    """
    disc, body = data[:8], _Reader(data[8:])
    try:
        if disc == TRADE_DISC:
            mint = body.pubkey()
            sol = body.u64() / LAMPORTS
            tokens = body.u64() / TOKEN_UNITS
            is_buy = body.take(1) != b"\0"
            user = body.pubkey()
            body.i64()  # on-chain timestamp, second resolution; we use receive time
            v_sol = body.u64() / LAMPORTS
            v_tokens = body.u64() / TOKEN_UNITS
            return Trade(now, mint, user, is_buy, sol, tokens, v_sol, v_tokens, signature)
        if disc == CREATE_DISC:
            name, symbol, uri = body.string(), body.string(), body.string()
            mint = body.pubkey()
            body.pubkey()  # bonding curve account
            user = body.pubkey()
            creator = user
            if body.remaining() >= 32:
                creator = body.pubkey()
            return NewToken(now, mint, name[:64], symbol[:32], creator, uri=uri[:200])
        if disc == COMPLETE_DISC:
            body.pubkey()  # user
            return Migration(now, body.pubkey())
    except (ValueError, struct.error) as exc:
        log.debug("undecodable pump event: %s", exc)
    return None


class RpcLogsFeed(Feed):
    """Firehose of every pump.fun event via ``logsSubscribe``.

    ``watch``/``unwatch`` are no-ops: the firehose already carries every
    trade, and the engine drops what it does not track.
    """

    def __init__(self, url: str) -> None:
        if not url:
            raise ValueError("set MEMEBOT_RPC_WS to a Solana websocket RPC url")
        self.url = url

    async def events(self) -> AsyncIterator[Event]:
        import websockets

        sub = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [{"mentions": [PUMP_PROGRAM]}, {"commitment": "processed"}],
        }
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    self.url, ping_interval=20, max_size=2**24
                ) as ws:
                    await ws.send(json.dumps(sub))
                    backoff = 1.0
                    log.info("rpc logsSubscribe connected")
                    async for raw in ws:
                        now = time.time()
                        value = (
                            json.loads(raw).get("params", {}).get("result", {}).get("value")
                        )
                        if not value or value.get("err"):
                            continue  # failed txs moved nothing
                        for event in events_from_logs(value.get("logs") or [], now, value.get("signature", "")):
                            yield event
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("rpc feed disconnected: %s; retry in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)


def events_from_logs(logs: list[str], now: float, signature: str = "") -> list[Event]:
    out: list[Event] = []
    for line in logs:
        if line.startswith("Program data: "):
            try:
                data = base64.b64decode(line[14:])
            except ValueError:
                continue
            event = decode_pump_event(data, now, signature)
            if event is not None:
                out.append(event)
    return out


# ------------------------------------------------------- record and replay


class Recorder:
    """Appends every event to a JSONL file for later replay and tuning."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", buffering=1)

    def write(self, event: Event) -> None:
        self._fh.write(json.dumps(to_dict(event), separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._fh.close()


def read_events(path: Path) -> list[Event]:
    events = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(from_dict(json.loads(line)))
    events.sort(key=lambda e: e.ts)
    return events


def make_feed(cfg) -> Feed:
    if cfg.feed == "rpc":
        return RpcLogsFeed(cfg.rpc_ws)
    if cfg.feed == "pumpportal":
        return PumpPortalFeed(cfg.pumpportal_ws)
    raise ValueError(f"unknown feed {cfg.feed!r}")
