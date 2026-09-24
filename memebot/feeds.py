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
PUMP_AMM_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"  # PumpSwap
WSOL = "So11111111111111111111111111111111111111112"
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
                        msg = json.loads(raw)
                        if "message" in msg or "errors" in msg:
                            # e.g. trade streams refused without a funded API key
                            log.warning("pumpportal: %s", msg.get("message") or msg.get("errors"))
                            continue
                        event = parse_pumpportal(msg, time.time())
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
# PumpSwap (pump_amm) events, discriminators from pump-fun/pump-public-docs idl/pump_amm.json
AMM_BUY_DISC = bytes([103, 244, 82, 31, 44, 245, 119, 119])
AMM_SELL_DISC = bytes([62, 47, 55, 10, 165, 3, 220, 42])
AMM_CREATE_POOL_DISC = bytes([177, 49, 12, 210, 160, 118, 167, 116])

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
            creator, fee = "", 0.0
            if body.remaining() >= 8 + 8 + 32 + 8 + 8 + 32 + 8:
                body.u64(), body.u64()  # real reserves
                body.pubkey()  # fee recipient
                fee_bps = body.u64()
                body.u64()
                creator = body.pubkey()
                fee = (fee_bps + body.u64()) / 10_000
            return Trade(now, mint, user, is_buy, sol, tokens, v_sol, v_tokens, signature,
                         creator=creator, fee_rate=fee)
        if disc in (AMM_BUY_DISC, AMM_SELL_DISC):
            return _decode_amm_trade(disc == AMM_BUY_DISC, body, now, signature)
        if disc == AMM_CREATE_POOL_DISC:
            body.i64(), body.take(2)  # timestamp, index
            body.pubkey()  # pool creator (the migration authority)
            base_mint, quote_mint = body.pubkey(), body.pubkey()
            base_dec, quote_dec = body.take(1)[0], body.take(1)[0]
            if quote_mint != WSOL:
                return None  # only SOL-quoted pools
            body.u64(), body.u64()  # amounts in
            base = body.u64() / 10**base_dec
            quote = body.u64() / 10**quote_dec
            body.u64(), body.u64(), body.u64(), body.take(1)
            pool = body.pubkey()
            body.pubkey(), body.pubkey(), body.pubkey()
            coin_creator = body.pubkey()
            return NewToken(now, pool, "", "", coin_creator, quote, base, venue="amm", base_mint=base_mint)
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


def _decode_amm_trade(is_buy: bool, body: "_Reader", now: float, signature: str) -> Trade | None:
    """PumpSwap BuyEvent / SellEvent. The event names the pool, not the mint,
    so pools are tracked by pool address. Reserves are read before the swap
    and moved by the swap's own amounts."""
    body.i64()  # timestamp
    base_amt = body.u64()
    body.u64()  # max_quote_in / min_quote_out
    body.u64(), body.u64()  # user reserves
    pool_base, pool_quote = body.u64(), body.u64()
    quote_amt = body.u64()  # quote_amount_in / quote_amount_out
    lp_bps = body.u64(); body.u64()
    proto_bps = body.u64(); body.u64()
    body.u64(), body.u64()
    pool = body.pubkey()
    user = body.pubkey()
    body.pubkey(), body.pubkey(), body.pubkey(), body.pubkey()
    coin_creator = body.pubkey()
    creator_bps = body.u64()
    # pump.fun tokens have 6 decimals and pools are quoted in wrapped SOL (9)
    base, quote = pool_base / TOKEN_UNITS, pool_quote / LAMPORTS
    tokens, sol = base_amt / TOKEN_UNITS, quote_amt / LAMPORTS
    if base <= 0 or quote <= 0 or quote > 1e6:
        return None  # not a SOL-quoted pump pool
    if is_buy:
        base, quote = base - tokens, quote + sol
    else:
        base, quote = base + tokens, quote - sol
    if base <= 0 or quote <= 0:
        return None
    return Trade(now, pool, user, is_buy, sol, tokens, quote, base, signature,
                 creator=coin_creator, venue="amm", fee_rate=(lp_bps + proto_bps + creator_bps) / 10_000)


class RpcLogsFeed(Feed):
    """Firehose of every pump.fun event via ``logsSubscribe``.

    ``watch``/``unwatch`` are no-ops: the firehose already carries every
    trade, and the engine drops what it does not track.
    """

    def __init__(self, url: str, programs: tuple[str, ...] = (PUMP_PROGRAM, PUMP_AMM_PROGRAM)) -> None:
        if not url:
            raise ValueError("set MEMEBOT_RPC_WS to a Solana websocket RPC url")
        self.url = url
        self.programs = programs

    async def events(self) -> AsyncIterator[Event]:
        import websockets

        subs = [
            {"jsonrpc": "2.0", "id": i + 1, "method": "logsSubscribe",
             "params": [{"mentions": [prog]}, {"commitment": "processed"}]}
            for i, prog in enumerate(self.programs)
        ]
        seen: dict[str, None] = {}  # a tx touching both programs arrives twice
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    self.url, ping_interval=20, max_size=2**24
                ) as ws:
                    for sub in subs:
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
                        sig = value.get("signature", "")
                        if sig in seen:
                            continue
                        seen[sig] = None
                        if len(seen) > 20000:
                            for old in list(seen)[:10000]:
                                del seen[old]
                        for event in events_from_logs(value.get("logs") or [], now, sig):
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
        programs = (PUMP_PROGRAM, PUMP_AMM_PROGRAM) if cfg.scan_amm else (PUMP_PROGRAM,)
        return RpcLogsFeed(cfg.rpc_ws, programs)
    if cfg.feed == "pumpportal":
        return PumpPortalFeed(cfg.pumpportal_ws)
    raise ValueError(f"unknown feed {cfg.feed!r}")
