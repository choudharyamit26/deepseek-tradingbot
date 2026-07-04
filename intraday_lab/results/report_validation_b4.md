# Intraday Lab — IS/OOS + Walk-Forward Report

Window 2025-07-04..2026-07-03 | IS ends 2026-03-31 | 20 high-beta stocks | 5-min | net of Dhan costs + slippage

| strategy | status | IS shp | IS pf | IS n | OOS shp | OOS pf | OOS n | OOS net | WF pf | WF folds+ |
|---|---|---|---|---|---|---|---|---|---|---|
| s67_smc_discount_h2c | REJECTED | -3.88 | 0.7 | 945 | 0.87 | 1.18 | 319 | 18853.0 | 0.85 | 1/9 |
| s65_smc_choch_h2c | REJECTED | -4.95 | 0.7 | 2023 | 0.1 | 1.01 | 781 | 2828.0 | 0.81 | 1/9 |
| s64_smc_bos_h2c | REJECTED | -4.33 | 0.79 | 4594 | -1.48 | 0.93 | 1640 | -44166.0 | 0.85 | 1/9 |
| s61_smc_sweep_reclaim_h2c | REJECTED | -2.62 | 0.7 | 413 | -2.11 | 0.81 | 135 | -7185.0 | 0.95 | 4/9 |
| s63_smc_order_block_h2c | REJECTED | -3.95 | 0.78 | 2087 | -2.26 | 0.86 | 683 | -32392.0 | 0.83 | 2/9 |
| s62_smc_fvg_retrace_h2c | REJECTED | -3.83 | 0.83 | 5297 | -3.8 | 0.83 | 1868 | -141586.0 | 0.86 | 2/9 |
| s66_smc_session_sweep_h2c | REJECTED | -6.88 | 0.58 | 1543 | -4.28 | 0.74 | 539 | -45275.0 | 0.63 | 0/9 |
| s68_smc_confluence_h2c | REJECTED | -3.96 | 0.76 | 2264 | -4.44 | 0.76 | 809 | -75990.0 | 0.82 | 1/9 |

**Survivors (0):** NONE

Criteria (fixed up front): OOS PF>=1.2, OOS Sharpe>=1.0, OOS trades>=30, IS->OOS Sharpe decay<50%.