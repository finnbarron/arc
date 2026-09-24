# Forward test: graduation play

Registered 2026-09-24 16:39 UTC, before any forward-test data existed.

## Where the rule came from

Offline search on ~4 hours of recorded live pump.fun data (6,185 launches,
20,289 rule-based entry points), split by launch time into train / validate /
test. Over 30,000 rules were tried in total across three searches, so a few
lucky survivors are expected. This one was the best-supported:

| slice    | trades | mean per trade (after costs) | t    |
|----------|--------|------------------------------|------|
| train    | 53     | +22.3%                       | +2.0 |
| validate | 19     | -3.4%                        | -0.2 |
| test     | 20     | +19.7%                       | +1.0 |

Not significant. Jev-driven entries, Jev-filtered entries, age triggers,
spike triggers and dip buys all failed out of sample.

## The rule (fixed; no changes during the test)

- Entry: a pump.fun launch seen from creation whose bonding curve first
  crosses 50% full (42.5 real SOL of 85), with >= 20 distinct buyers and a
  creator who has sold nothing. Once per token.
- Size: 0.3 SOL paper.
- Exit: -25% stop, 5 minute time limit, take profit only at +1000%, no
  trailing stop. If the curve completes, exit at the last curve price.
- Fills: same paper broker as every other run (live curve math, venue fee,
  1% slippage each side, tx cost, 0.6 s landing delay, own price impact).
- Jev is asked at every entry in shadow; its answers are logged, never acted on.

## Success criteria

After >= 50 closed trades: mean return per trade > 0 after costs with
t > 2. Anything else, including "positive but not significant", is a fail
for the purpose of calling it profitable.
