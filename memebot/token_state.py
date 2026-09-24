"""Everything the bot knows about one token, built only from the event stream."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from .curve import Curve
from .events import INITIAL_V_SOL, TOTAL_SUPPLY, NewToken, Trade

GRADUATION_SOL = 85.0  # real SOL on the curve when it migrates


@dataclass
class TokenState:
    mint: str
    name: str
    symbol: str
    creator: str
    created_ts: float
    curve: Curve
    trades: deque = field(default_factory=lambda: deque(maxlen=4000))
    prices: deque = field(default_factory=lambda: deque(maxlen=4000))  # (ts, price)
    balances: dict = field(default_factory=dict)  # wallet -> tokens held
    buyers: set = field(default_factory=set)
    sellers: set = field(default_factory=set)
    buy_count: dict = field(default_factory=dict)
    first_buy: dict = field(default_factory=dict)  # wallet -> ts of first buy
    creator_initial: float = 0.0
    creator_sold: float = 0.0
    ath: float = 0.0
    migrated: bool = False
    last_ts: float = 0.0
    evals: int = 0
    last_eval_ts: float = -1e18

    @classmethod
    def from_event(cls, ev: NewToken) -> "TokenState":
        st = cls(
            mint=ev.mint,
            name=ev.name,
            symbol=ev.symbol,
            creator=ev.creator,
            created_ts=ev.ts,
            curve=Curve(ev.v_sol, ev.v_tokens),
            creator_initial=ev.initial_buy_tokens,
            last_ts=ev.ts,
        )
        if ev.initial_buy_tokens:
            st.balances[ev.creator] = ev.initial_buy_tokens
        st.ath = st.curve.price
        st.prices.append((ev.ts, st.curve.price))
        return st

    def apply(self, tr: Trade) -> None:
        self.curve = Curve(tr.v_sol, tr.v_tokens)
        self.trades.append(tr)
        price = self.curve.price
        self.prices.append((tr.ts, price))
        self.ath = max(self.ath, price)
        self.last_ts = tr.ts
        held = self.balances.get(tr.trader, 0.0)
        if tr.is_buy:
            self.buyers.add(tr.trader)
            self.buy_count[tr.trader] = self.buy_count.get(tr.trader, 0) + 1
            self.first_buy.setdefault(tr.trader, tr.ts)
            self.balances[tr.trader] = held + tr.tokens
            if tr.trader == self.creator:
                # the RPC feed delivers the creator's launch buy as its own trade
                self.creator_initial += tr.tokens
        else:
            self.sellers.add(tr.trader)
            self.balances[tr.trader] = max(held - tr.tokens, 0.0)
            if tr.trader == self.creator:
                self.creator_sold += tr.tokens

    # ------------------------------------------------------------ features

    def price_at(self, ts: float) -> float:
        """Last known price at or before ``ts``."""
        best = self.prices[0][1] if self.prices else self.curve.price
        for t, p in self.prices:
            if t > ts:
                break
            best = p
        return best

    def features(self, now: float) -> dict:
        trades = list(self.trades)
        buys = [t for t in trades if t.is_buy]
        sells = [t for t in trades if not t.is_buy]
        price = self.curve.price

        def window(sec: float):
            w = [t for t in trades if t.ts >= now - sec]
            b = sum(t.sol for t in w if t.is_buy)
            s = sum(t.sol for t in w if not t.is_buy)
            return len(w), b, s

        n5, b5, s5 = window(5)
        n15, b15, s15 = window(15)
        n60, b60, s60 = window(60)

        def change(sec: float) -> float:
            past = self.price_at(now - sec)
            return price / past - 1.0 if past > 0 else 0.0

        holders = sorted(
            (v for k, v in self.balances.items() if v > 0), reverse=True
        )
        top5 = sum(holders[:5]) / TOTAL_SUPPLY
        top1 = holders[0] / TOTAL_SUPPLY if holders else 0.0
        buy_sizes = [t.sol for t in buys]
        dust = sum(1 for s in buy_sizes if s < 0.05) / len(buy_sizes) if buy_sizes else 0.0
        repeat = (
            sum(1 for c in self.buy_count.values() if c > 1) / len(self.buy_count)
            if self.buy_count
            else 0.0
        )
        biggest_sell = max((t.tokens for t in sells), default=0.0) / TOTAL_SUPPLY
        real_sol = max(self.curve.v_sol - INITIAL_V_SOL, 0.0)
        creator_held = self.balances.get(self.creator, 0.0) / TOTAL_SUPPLY
        age = max(now - self.created_ts, 0.0)
        # wallets that bought within 3s of launch are snipers or bundles;
        # what they still hold tends to be dumped into the first pump
        snipers = sum(
            self.balances.get(w, 0.0) for w, t in self.first_buy.items()
            if t - self.created_ts <= 3.0 and w != self.creator
        ) / TOTAL_SUPPLY
        rate_recent = n15 / 15.0
        rate_overall = len(trades) / age if age > 0 else 0.0

        return {
            "age_s": round(age, 1),
            "trades": len(trades),
            "buys": len(buys),
            "sells": len(sells),
            "unique_buyers": len(self.buyers),
            "unique_sellers": len(self.sellers),
            "buy_sol_total": round(sum(buy_sizes), 3),
            "sell_sol_total": round(sum(t.sol for t in sells), 3),
            "net_flow_5s_sol": round(b5 - s5, 3),
            "net_flow_15s_sol": round(b15 - s15, 3),
            "trades_5s": n5,
            "net_flow_60s_sol": round(b60 - s60, 3),
            "trades_15s": n15,
            "trades_60s": n60,
            "trade_rate_accel": round(rate_recent / rate_overall, 2) if rate_overall else 0.0,
            "mcap_sol": round(price * TOTAL_SUPPLY, 2),
            "curve_progress": round(min(real_sol / GRADUATION_SOL, 1.0), 3),
            "chg_10s": round(change(10), 4),
            "chg_30s": round(change(30), 4),
            "chg_60s": round(change(60), 4),
            "drawdown_from_ath": round(price / self.ath - 1.0, 4) if self.ath else 0.0,
            "top1_holder_pct": round(top1 * 100, 2),
            "top5_holder_pct": round(top5 * 100, 2),
            "creator_holding_pct": round(creator_held * 100, 2),
            "creator_sold_pct_of_initial": (
                round(min(self.creator_sold / self.creator_initial, 1.0) * 100, 1)
                if self.creator_initial
                else (100.0 if self.creator_sold else 0.0)
            ),
            "median_buy_sol": round(median(buy_sizes), 4) if buy_sizes else 0.0,
            "largest_buy_sol": round(max(buy_sizes), 3) if buy_sizes else 0.0,
            "dust_buy_share": round(dust, 3),
            "repeat_buyer_share": round(repeat, 3),
            "largest_sell_pct_supply": round(biggest_sell * 100, 2),
            "sell_buy_ratio": round(len(sells) / len(buys), 3) if buys else 0.0,
            "early_sniper_holding_pct": round(snipers * 100, 2),
        }

    def exit_view(self, now: float, entry_price: float, entry_ts: float) -> dict:
        """What Jev sees each second while we hold: the price path since
        entry and the most recent individual trades."""
        path = []
        next_t = max(entry_ts - 30, self.created_ts)
        for t, p in self.prices:
            if t >= next_t:
                path.append([round(t - entry_ts, 1), round((p / entry_price - 1) * 100, 1)])
                next_t = t + 2.0
        path = path[-60:]
        big = 0.5  # SOL
        recent = []
        for tr in list(self.trades)[-12:]:
            recent.append({
                "secs_ago": round(now - tr.ts, 1),
                "side": "buy" if tr.is_buy else "sell",
                "sol": round(tr.sol, 3),
                "who": "creator" if tr.trader == self.creator else (
                    "sniper" if self.first_buy.get(tr.trader, 1e18) - self.created_ts <= 3.0 else "trader"),
                "big": tr.sol >= big,
            })
        big_sells_30s = sum(1 for t in self.trades if not t.is_buy and t.ts >= now - 30 and t.sol >= big)
        creator_sells_60s = sum(1 for t in self.trades if not t.is_buy and t.ts >= now - 60 and t.trader == self.creator)
        return {
            "price_path_pct_vs_entry": path,
            "recent_trades": recent,
            "big_sells_last_30s": big_sells_30s,
            "creator_sells_last_60s": creator_sells_60s,
        }
