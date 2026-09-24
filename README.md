# arc: memebot

Paper-trades pump.fun memecoins on **live Solana data**, with **TypeSafe's Jev**
making the judgment calls. No real money moves: buys and sells are simulated
against the live bonding curve, with fees, latency and price impact charged
honestly.

## Read this first: profitability

Nothing here guarantees profit, and anyone claiming a memecoin bot is
guaranteed profitable is lying. Most new pump.fun tokens go to zero, and a
round trip costs roughly 5% in fees and slippage before you're up a cent. So
the bot is built to **not trade unless it has measured an edge**, and to stop
itself when that edge disappears:

1. **Shadow labelling.** Every token Jev judges gets a real outcome from the
   live tape (did price hit +TP or -SL first, or neither), whether the bot
   bought it or not. That's hundreds of honest labels an hour.
2. **No trading until Jev proves itself.** Trading stays off until there are
   `calibration_warmup` labels *and* Jev's ranking of winners beats a coin flip
   by more than its standard error (AUC). A model with no skill never gets
   capital (`test_no_skill_means_no_losses_worth_mentioning`).
3. **Calibrated EV, not vibes.** Jev's raw probabilities are mapped to observed
   frequencies, and payoffs use the *measured* return of each outcome, stop-loss
   gaps included. A trade needs `EV >= min_ev` after the full round-trip cost at
   its actual size.
4. **Hard vetoes and sizing.** Rug-risk veto, confidence floor, quarter-Kelly
   sizing, a cap on share of the curve, and max open positions.
5. **Kill switch.** It pauses new entries after a drawdown or a statistically
   losing streak.
6. **Walk-forward tuning.** `tune` picks params on the first 60% of a recording
   and reports only the unseen 40%. It says "profitable" only when t > 2 over
   30+ trades.

The loop to actually find out if it makes money: run it live for a day,
`tune` on the recording, run again with the tuned params, check `report`.
If it isn't profitable out of sample, it isn't profitable. Don't go live
with real money on a paper result either: real fills are worse.

## How it works

```
feed (PumpPortal WS | raw RPC logs) ─► token state/features ─► filter ─► Jev (1 request, 4 questions)
                                                                              │
             paper broker ◄── policy (calibrated EV, Kelly, vetoes) ◄─────────┘
             (live curve fills)        ▲
                                       └── calibrator ◄── shadow labeler (every judged token)
```

**Data, fastest first**
- Your own Yellowstone/Geyser gRPC stream: fastest, paid, not implemented.
- `--feed rpc`: `logsSubscribe` on a Solana WS RPC at `processed` commitment,
  decoding pump.fun's Anchor events from program logs ourselves. No middleman.
  Set `MEMEBOT_RPC_WS=wss://mainnet.helius-rpc.com/?api-key=...` (or Triton,
  QuickNode). The public endpoint rate-limits hard.
- `--feed pumpportal` (default): PumpPortal's free websocket. One extra hop,
  zero setup.

**Jev** gets one `POST /v1/systemone` per evaluation through the official
`typesafe-sdk`, with four typed questions answered in one parallel pass:

| question | type | used for |
|---|---|---|
| `outcome` | Choice: `take_profit_first` / `stop_loss_first` / `neither` | the EV calculation |
| `rug_risk` | Noul | hard veto |
| `organic` | Noul | context / logging |
| `momentum` | Score (0-3 rubric) | context / logging |

The state carries raw features plus plain-language readings ("net buying",
"creator has dumped"), because Jev reasons over words better than raw numbers.
Cost is about $0.00003 per call.

**Simulated execution**
- Orders fill after `jev latency + exec_latency_s`, at whatever the curve is by then.
- Fill prices come from constant-product math on live reserves, so size moves price.
- The 1.25% pump.fun fee is charged on both sides, plus per-tx cost and extra slippage.
- Stops trigger on a print but fill later, so gaps are paid in full.
- Our simulated holding is overlaid on the recorded reserves (pump.fun's `k` is
  constant), so our own impact is neither ignored nor double-counted.
- On migration, positions exit at the last curve price. That's an approximation.

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "TYPESAFE_API_KEY=..." > .env        # gitignored

.venv/bin/python -m memebot doctor         # checks the feed and a real Jev call
.venv/bin/python -m memebot run            # live paper trading; Ctrl-C prints final report
.venv/bin/python -m memebot record         # just record the feed, no Jev spend
.venv/bin/python -m memebot replay data/events-*.jsonl
.venv/bin/python -m memebot tune   data/events-*.jsonl
.venv/bin/python -m memebot report
.venv/bin/python -m pytest -q tests
```

Every setting in `memebot/config.py` can be overridden with `MEMEBOT_<NAME>`,
e.g. `MEMEBOT_MIN_EV=0.1 MEMEBOT_FEED=rpc`. `--provider mock` swaps Jev for
an offline heuristic. Results from the mock say nothing about Jev.

Output lives in `data/`: raw events (for replay), `trades.jsonl`, `labels-*.jsonl`
(calibration persists across runs, per model and barrier setting),
`jev_cache.jsonl` (replays re-use paid answers) and `status.json`.

`memebot synthetic` writes a fake market for offline testing only. It is not
market data, and P&L on it proves nothing.
