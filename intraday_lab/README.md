# Intraday Strategy Lab

Standalone research folder: 20 intraday strategies × 20 high-beta NSE stocks ×
1 year of real Dhan 5-minute data, with grid optimization on in-sample data and
honest out-of-sample + walk-forward validation. Nothing here touches the live
bot; survivors are candidates only.

## Relationship to `opencode_intraday_startegy_lab/`

That lab ran first, on the same window (2025-07-04 → 2026-07-03), same
high-beta universe, realistic costs — and **every one of its 20 classic
context-free strategies failed walk-forward** (0/5 profitable folds each,
gross edge ≈ 0, costs ≈ -Rs140/trade at 10+ trades/day). This lab treats that
as the control result and changes what is tested rather than re-running it:

- **Data and universe are imported from that lab** (`run.py import-opencode`)
  so results are directly comparable; CHOLAFIN + NIFTY 5-min fetched fresh.
- **6 classics are kept as engine cross-validation controls** (s01-s06). They
  are *expected* to lose here too; if they don't, an engine is broken.
- **14 strategies are conditioned** with filters the live bot validated on real
  fills: OFI confirmation (shorts 62% vs 25% WR), no entries before 10:15
  (market-open decay), NIFTY relative strength, RSI 35-45 short zone — plus
  **two-phase exits** (bank half at +0.4%, breakeven lock, trail) instead of
  fixed brackets, and a hard 2 trades/symbol/day cap to control cost drag.

## Honesty rules

- Signals on closed bars only; fills at next bar open + slippage (tested).
- Intrabar SL before TP when both touch (conservative).
- Full Dhan intraday cost model (brokerage/STT/exchange/GST/stamp) + 2bps slip.
- IS = first 9 months (optimize here only) → single frozen-param OOS pass on
  the last 3 months → 9-fold rolling walk-forward (3m train / 1m test).
- Selection requires ≥100 IS trades and a parameter plateau (isolated grid
  spikes rejected). Survivor criteria fixed in config before any results.

## Usage

```
python run.py import-opencode   # adopt opencode lab's processed candles
python run.py fetch             # fetch gaps (NIFTY 5-min, missing symbols)
python run.py validate          # IS optimize -> OOS -> walk-forward -> report
python -m pytest tests/         # engine honesty tests
```

Outputs: `results/validation.json`, `results/report.md`.
