# Intraday Lab — IS/OOS + Walk-Forward Report

Window 2025-07-04..2026-07-03 | IS ends 2026-03-31 | 20 high-beta stocks | 5-min | net of Dhan costs + slippage

| strategy | status | IS shp | IS pf | IS n | OOS shp | OOS pf | OOS n | OOS net | WF pf | WF folds+ |
|---|---|---|---|---|---|---|---|---|---|---|
| s29_range_compress_break_h2c | REJECTED | -1.32 | 0.88 | 450 | 1.42 | 1.15 | 141 | 8293.0 | 0.75 | 3/9 |
| s35_open_drive_h2c | REJECTED | -2.02 | 0.8 | 504 | 1.22 | 1.15 | 139 | 11981.0 | 0.89 | 3/9 |
| s23_prev_trend_follow_h2c | REJECTED | -3.31 | 0.69 | 1048 | -0.44 | 0.95 | 424 | -8813.0 | 0.67 | 1/9 |
| s22_gap_go_h2c | REJECTED | 0.67 | 1.14 | 286 | -0.82 | 0.86 | 216 | -18813.0 | 1.06 | 5/9 |
| s32_60m_breakout_h2c | REJECTED | -0.56 | 0.95 | 2348 | -1.03 | 0.91 | 761 | -32027.0 | 0.97 | 4/9 |
| s38_failed_balance_h2c | REJECTED | 0.43 | 1.05 | 634 | -1.29 | 0.87 | 144 | -11560.0 | 0.95 | 4/9 |
| s27_xs_rs_loser_short_h2c | REJECTED | -0.51 | 0.92 | 178 | -1.5 | 0.78 | 62 | -7924.0 | 0.87 | 3/9 |
| s30_15m_donchian_h2c | REJECTED | -1.01 | 0.93 | 2958 | -1.71 | 0.88 | 994 | -59217.0 | 0.91 | 2/9 |
| s26_xs_rs_winner_h2c | REJECTED | 1.19 | 1.22 | 175 | -1.83 | 0.75 | 62 | -10468.0 | 0.85 | 4/9 |
| s33_15m_vwap_ofi_2ph | REJECTED | -5.07 | 0.62 | 579 | -2.45 | 0.77 | 185 | -10612.0 | 0.73 | 0/9 |
| s25_inside_day_break_h2c | REJECTED | -0.12 | 0.98 | 290 | -2.48 | 0.72 | 81 | -9256.0 | 1.08 | 6/9 |
| s21_first_hour_trend_h2c | REJECTED | 0.18 | 1.02 | 1366 | -3.14 | 0.73 | 508 | -83971.0 | 0.86 | 2/9 |
| s39_compressed_orb_h2c | REJECTED | -2.6 | 0.77 | 513 | -3.49 | 0.71 | 153 | -16348.0 | 0.75 | 2/9 |
| s36_lunch_reversal_h2c | REJECTED | -6.44 | 0.54 | 649 | -4.12 | 0.59 | 273 | -41351.0 | 0.54 | 0/9 |
| s24_nr7_expansion_h2c | REJECTED | 0.54 | 1.06 | 469 | -5.02 | 0.63 | 178 | -32459.0 | 0.97 | 6/9 |
| s34_high_vol_day_orb_h2c | REJECTED | -1.21 | 0.87 | 760 | -7.07 | 0.57 | 174 | -35570.0 | 0.58 | 2/9 |
| s31_15m_ema_trend | REJECTED | -6.77 | 0.63 | 2772 | -7.44 | 0.63 | 983 | -98907.0 | 0.63 | 0/9 |
| s37_second_day_momo_h2c | REJECTED | -5.68 | 0.47 | 408 | -10.07 | 0.38 | 82 | -26422.0 | 0.0 | 0/9 |
| s40_ensemble_vote_2ph | REJECTED | -11.66 | 0.63 | 5396 | -11.69 | 0.61 | 1885 | -211523.0 | 0.64 | 0/9 |
| s28_xs_gap_extend_h2c | NO-QUALIFIER | - | - | - | - | - | - | - | - | -/- |

**Survivors (0):** NONE

Criteria (fixed up front): OOS PF>=1.2, OOS Sharpe>=1.0, OOS trades>=30, IS->OOS Sharpe decay<50%.