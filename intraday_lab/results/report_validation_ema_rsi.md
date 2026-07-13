# Intraday Lab — IS/OOS + Walk-Forward Report

Window 2025-07-04..2026-07-03 | IS ends 2026-03-31 | 20 high-beta stocks | 5-min | net of Dhan costs + slippage

| strategy | status | IS shp | IS pf | IS n | OOS shp | OOS pf | OOS n | OOS net | WF pf | WF folds+ |
|---|---|---|---|---|---|---|---|---|---|---|
| ema_rsi_long | REJECTED | -1.79 | 0.84 | 319 | 2.78 | 1.42 | 115 | 11301.0 | 0.79 | 3/9 |
| ema_rsi_short | REJECTED | -0.52 | 0.93 | 220 | -5.82 | 0.47 | 71 | -8805.0 | 0.68 | 2/9 |

**Survivors (0):** NONE

Criteria (fixed up front): OOS PF>=1.2, OOS Sharpe>=1.0, OOS trades>=30, IS->OOS Sharpe decay<50%.