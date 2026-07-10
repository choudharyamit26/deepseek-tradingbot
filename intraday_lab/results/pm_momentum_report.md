# PM-momentum carryover study — final report (2026-07-09)

**Hypothesis (as posed):** stocks in momentum after 14:00 remain in momentum
next trading session.

**Verdict: REJECTED for longs, REFINED to a thin short-side edge that does
not clear the lab's significance bar. Do not deploy standalone.**

## Method

- Phase 1 (`pm_momentum_study.py`): parameter-free event study, 146 symbols,
  35,450 symbol-days, 2025-07-04 → 2026-07-03. M = 14:00→15:25 return;
  next-day outcomes by decile, sign-aligned.
- Phase 1b (`pm_momentum_phase1b.py`): daily-portfolio test of the tradeable
  refinement, full Dhan cost model + 2 bps slip/side, day-clustered t-stats,
  small fully-reported grid.
- Phase 2 (`pm_momentum_holdout.py`): rule frozen, single pass on untouched
  2024-01 → 2025-07 data (22 symbols, 8,162 symbol-days).

## Findings

1. **Up-momentum does not continue intraday.** It appears only in the
   overnight gap (+0.18% top decile), which requires CNC delivery whose STT
   (~0.2% RT) exceeds the edge. Intraday next day, strong up-movers mean-revert
   (open→close −0.085%). Long version of the strategy: −0.24%/trade net,
   t = −3.1. Dead, consistent with the BUY-side findings on live fills.
2. **Down-momentum continues next morning.** Bottom-decile PM losers fall
   another ~−0.16% open→close, concentrated open→10:15, usually after a small
   gap up. This stacks with the known 09:30–11:00 bleed. Not NIFTY drift
   (index open→10:15 mean −0.02%).
3. **Tradeable rule** (short M ≤ −0.75% names at next open, K=5/day,
   Rs100k/trade):

   | period | exit 10:15 | exit close |
   |---|---|---|
   | 2025-26 exploration (909 tr) | +0.048%/tr, PF 1.09, t 0.77 | +0.114%/tr, PF 1.15, t 1.42 |
   | 2024-25 holdout (695 tr) | +0.151%/tr, PF 1.26, t 1.70 | +0.052%/tr, PF 1.06, t 0.99 |

   Both periods positive on both exits, but the better exit flips between
   periods (noise), no cell reaches t ≥ 2, and combined 10:15 daily series
   (478 days) gives t = 1.85, Sharpe ≈ 1.3, 32% negative months, worst
   portfolio day −9.45% (short-squeeze tail).

## Disposition

- Do **not** build a live runner from this. Edge/trade (Rs 50–150 on Rs 1L)
  is inside one bad fill, and the pre-registered primary confirmed weakly.
- Salvage: add `prev_day_pm_ret` (M) as a **logged feature** in the s22/OFI
  feature pipeline — it is a cheap, validated-directionally short-side
  conditioner and may stack with the OFI short gate (62% WR) rather than
  stand alone.
- The long-side rejection is durable knowledge: no "buy yesterday's PM
  winners" variants need testing again.
