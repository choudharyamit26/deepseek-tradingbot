# Intraday Lab — IS/OOS + Walk-Forward Report

Window 2025-07-04..2026-07-03 | IS ends 2026-03-31 | 20 high-beta stocks | 5-min | net of Dhan costs + slippage

| strategy | status | IS shp | IS pf | IS n | OOS shp | OOS pf | OOS n | OOS net | WF pf | WF folds+ |
|---|---|---|---|---|---|---|---|---|---|---|
| ema_rsi_filt_long | REJECTED | -1.76 | 0.84 | 306 | 2.5 | 1.36 | 111 | 9553.0 | 0.83 | 4/9 |
| ema_rsi_filt_short | REJECTED | -0.32 | 0.96 | 217 | -5.48 | 0.48 | 66 | -8245.0 | 0.56 | 1/9 |

**Survivors (0):** NONE

Criteria (fixed up front): OOS PF>=1.2, OOS Sharpe>=1.0, OOS trades>=30, IS->OOS Sharpe decay<50%.