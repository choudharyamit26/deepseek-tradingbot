# Intraday Lab — IS/OOS + Walk-Forward Report

Window 2025-07-04..2026-07-03 | IS ends 2026-03-31 | 20 high-beta stocks | 5-min | net of Dhan costs + slippage

| strategy | status | IS shp | IS pf | IS n | OOS shp | OOS pf | OOS n | OOS net | WF pf | WF folds+ |
|---|---|---|---|---|---|---|---|---|---|---|
| s41_expiry_day_fade_h2c | REJECTED | -6.24 | 0.6 | 284 | 4.39 | 2.21 | 136 | 64420.0 | 1.12 | 3/9 |
| s59_late_breakout | REJECTED | -3.06 | 0.68 | 754 | 0.58 | 1.08 | 261 | 4458.0 | 0.81 | 2/9 |
| s53_beta_gap_rv_h2c | REJECTED | -0.95 | 0.88 | 278 | 0.52 | 1.09 | 89 | 4244.0 | 0.79 | 3/9 |
| s45_failed_orb_rev_h2c | REJECTED | -2.71 | 0.84 | 2325 | -0.38 | 0.97 | 832 | -11853.0 | 0.88 | 1/9 |
| s54_sector_laggard_h2c | REJECTED | -4.01 | 0.57 | 120 | -0.67 | 0.92 | 30 | -1283.0 | 0.73 | 3/9 |
| s50_poc_break_h2c | REJECTED | -1.92 | 0.83 | 740 | -1.34 | 0.9 | 228 | -8432.0 | 0.8 | 2/9 |
| s46_failed_pdh_rev_h2c | REJECTED | -4.36 | 0.74 | 1679 | -2.2 | 0.85 | 591 | -34731.0 | 0.82 | 2/9 |
| s49_poc_magnet | REJECTED | -5.53 | 0.6 | 1499 | -2.24 | 0.82 | 631 | -51760.0 | 0.63 | 0/9 |
| s47_gap_trap_h2c | REJECTED | -0.58 | 0.91 | 129 | -2.9 | 0.71 | 113 | -15730.0 | 0.65 | 1/9 |
| s42_friday_trend_h2c | REJECTED | 2.32 | 1.2 | 395 | -3.03 | 0.83 | 125 | -11863.0 | 0.96 | 4/9 |
| s43_monday_gap_rev_h2c | REJECTED | -2.56 | 0.64 | 173 | -3.55 | 0.74 | 127 | -15028.0 | 0.61 | 3/9 |
| s48_spring_reversal_h2c | REJECTED | -4.47 | 0.74 | 2112 | -3.76 | 0.81 | 739 | -54145.0 | 0.8 | 1/9 |
| s58_moc_momentum | REJECTED | -4.02 | 0.68 | 1469 | -4.83 | 0.65 | 534 | -43058.0 | 0.71 | 1/9 |
| s57_range_exhaust_fade | REJECTED | -5.99 | 0.63 | 914 | -5.46 | 0.66 | 204 | -28849.0 | 0.64 | 2/9 |
| s56_trend_regime_pullback_h2c | REJECTED | -4.04 | 0.75 | 3326 | -7.46 | 0.7 | 1088 | -142500.0 | 0.78 | 2/9 |
| s60_vwap_convergence | REJECTED | -5.84 | 0.46 | 694 | -8.29 | 0.36 | 350 | -64552.0 | 0.41 | 0/9 |
| s44_month_end_long_h2c | REJECTED | -7.01 | 0.58 | 150 | -8.74 | 0.46 | 71 | -21991.0 | 0.85 | 3/9 |
| s52_pair_zscore | REJECTED | -10.3 | 0.62 | 1774 | -10.16 | 0.69 | 612 | -68400.0 | 0.61 | 0/9 |
| s51_poc_bounce | REJECTED | -12.75 | 0.6 | 2682 | -13.62 | 0.57 | 802 | -109092.0 | 0.59 | 0/9 |
| s55_vol_regime_switch | REJECTED | -17.39 | 0.57 | 4905 | -15.29 | 0.61 | 1804 | -248088.0 | 0.57 | 0/9 |

**Survivors (0):** NONE

Criteria (fixed up front): OOS PF>=1.2, OOS Sharpe>=1.0, OOS trades>=30, IS->OOS Sharpe decay<50%.