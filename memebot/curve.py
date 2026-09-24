"""pump.fun bonding-curve math.

The curve is constant product over *virtual* reserves. Quoting against the
live reserves reported with each trade is what makes the simulated fills
honest: a 0.5 SOL buy on a 35 SOL curve moves price about 3%, and the sim
pays that, same as a real wallet would.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Curve:
    v_sol: float  # virtual SOL reserves
    v_tokens: float  # virtual token reserves

    @property
    def price(self) -> float:
        """SOL per token at the margin."""
        return self.v_sol / self.v_tokens if self.v_tokens > 0 else 0.0

    def buy(self, sol_in: float, fee_rate: float) -> tuple[float, "Curve"]:
        """Spend ``sol_in`` (fee included). Returns tokens out and the new curve."""
        net = sol_in * (1.0 - fee_rate)
        k = self.v_sol * self.v_tokens
        new_sol = self.v_sol + net
        new_tokens = k / new_sol
        return self.v_tokens - new_tokens, Curve(new_sol, new_tokens)

    def with_holding(self, tokens: float) -> "Curve":
        """The curve as it would be if we really held ``tokens`` bought from it.

        Paper trades never reach the chain, so recorded reserves lack our
        impact. pump.fun keeps k constant (fees are paid outside the
        reserves), so removing our tokens from the pool is exact.
        """
        if tokens <= 0:
            return self
        k = self.v_sol * self.v_tokens
        v_tokens = max(self.v_tokens - tokens, self.v_tokens * 0.01)
        return Curve(k / v_tokens, v_tokens)

    def sell(self, tokens_in: float, fee_rate: float) -> tuple[float, "Curve"]:
        """Sell ``tokens_in``. Returns SOL out after fee and the new curve."""
        k = self.v_sol * self.v_tokens
        new_tokens = self.v_tokens + tokens_in
        new_sol = k / new_tokens
        gross = self.v_sol - new_sol
        return gross * (1.0 - fee_rate), Curve(new_sol, new_tokens)


def round_trip_cost(cfg, sol: float, curve: Curve, fee_rate: float | None = None) -> float:
    """Fraction of ``sol`` lost to buying and immediately selling.

    This is the hurdle every trade must clear before it earns anything.
    """
    fee = cfg.fee_rate if fee_rate is None else fee_rate
    tokens, after = curve.buy(sol, fee)
    back, _ = after.sell(tokens, fee)
    back *= (1.0 - cfg.extra_slippage) ** 2
    back -= 2 * cfg.tx_cost_sol
    return 1.0 - back / sol
