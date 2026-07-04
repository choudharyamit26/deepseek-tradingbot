# Intraday Lab — IS/OOS + Walk-Forward Report

Window 2025-07-04..2026-07-03 | IS ends 2026-03-31 | 20 high-beta stocks | 5-min | net of Dhan costs + slippage

| strategy | status | IS shp | IS pf | IS n | OOS shp | OOS pf | OOS n | OOS net | WF pf | WF folds+ |
|---|---|---|---|---|---|---|---|---|---|---|
| s17_vol_spike_ofi | REJECTED | -7.44 | 0.6 | 829 | -4.61 | 0.71 | 306 | -21010.0 | 0.6 | 0/9 |
| s12_gap_fade_confirm | REJECTED | -3.35 | 0.59 | 196 | -4.67 | 0.52 | 199 | -45935.0 | 0.57 | 1/9 |
| s18_ofi_div_reversal | REJECTED | -8.72 | 0.46 | 644 | -4.74 | 0.64 | 211 | -23102.0 | 0.48 | 0/9 |
| s03_ema_adx_ctl | REJECTED | -5.09 | 0.71 | 1269 | -5.01 | 0.66 | 441 | -51662.0 | 0.68 | 0/9 |
| s15_range_fade_div | REJECTED | -9.47 | 0.41 | 721 | -5.09 | 0.6 | 243 | -27983.0 | 0.43 | 0/9 |
| s08_vwap_reclaim_ofi | REJECTED | -7.58 | 0.53 | 569 | -5.16 | 0.63 | 191 | -14375.0 | 0.53 | 0/9 |
| s19_afternoon_trend | REJECTED | -7.03 | 0.5 | 961 | -5.74 | 0.63 | 334 | -29515.0 | 0.53 | 0/9 |
| s11_squeeze_ofi | REJECTED | -5.69 | 0.51 | 173 | -8.46 | 0.36 | 44 | -7344.0 | 0.49 | 0/9 |
| s07_orb_ofi_2ph | REJECTED | -8.96 | 0.58 | 1575 | -8.8 | 0.55 | 494 | -59584.0 | 0.58 | 0/9 |
| s13_vwapz_ofi_div | REJECTED | -10.93 | 0.45 | 728 | -9.55 | 0.49 | 286 | -34659.0 | 0.49 | 0/9 |
| s06_boll_fade_ctl | REJECTED | -14.45 | 0.42 | 1450 | -10.28 | 0.56 | 489 | -50115.0 | 0.48 | 0/9 |
| s02_vwap_pullback_ctl | REJECTED | -6.57 | 0.74 | 4923 | -10.91 | 0.64 | 1690 | -222360.0 | 0.71 | 1/9 |
| s04_donchian_ctl | REJECTED | -9.13 | 0.69 | 5328 | -11.2 | 0.67 | 1773 | -195498.0 | 0.71 | 0/9 |
| s14_rsi_zone_short | REJECTED | -8.32 | 0.56 | 2228 | -11.2 | 0.5 | 688 | -83784.0 | 0.54 | 0/9 |
| s20_pdh_retest | REJECTED | -9.75 | 0.61 | 4333 | -11.34 | 0.55 | 1541 | -199212.0 | 0.59 | 0/9 |
| s09_trend_day_rider | REJECTED | -10.46 | 0.58 | 2496 | -11.58 | 0.56 | 871 | -95348.0 | 0.57 | 0/9 |
| s16_ofi_momentum | REJECTED | -15.18 | 0.55 | 3715 | -12.17 | 0.58 | 1297 | -129997.0 | 0.57 | 0/9 |
| s01_orb_ctl | REJECTED | -6.74 | 0.7 | 3551 | -12.25 | 0.58 | 1226 | -197338.0 | 0.69 | 0/9 |
| s10_nifty_rs | REJECTED | -12.39 | 0.56 | 2622 | -12.34 | 0.54 | 922 | -106232.0 | 0.55 | 0/9 |
| s05_rsi2_fade_ctl | REJECTED | -19.78 | 0.56 | 7290 | -21.03 | 0.63 | 2516 | -315852.0 | 0.57 | 0/9 |

**Survivors (0):** NONE

Criteria (fixed up front): OOS PF>=1.2, OOS Sharpe>=1.0, OOS trades>=30, IS->OOS Sharpe decay<50%.