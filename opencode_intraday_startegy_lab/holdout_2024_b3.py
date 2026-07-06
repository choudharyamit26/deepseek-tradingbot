"""True out-of-sample holdout for the walk-forward winners of 2026-07-06.

Candidates (named before this test): b3_three_day_reversal_h2c,
b3_late_liquidity_run_h2c, b3_vwap_compression_release — frozen at the params
their walk-forward folds recurrently chose. Window 2024-01-01..2025-07-03
(pre-study data, never optimized on by either lab), run ONCE.

Pre-registered keep rule: PF >= 1.1 AND n >= 50 AND net > 0 in BOTH 2024 and
2025H1. Data: intraday_lab's cached holdout parquets (same core-20 universe,
18 symbols with data). Engine/costs: this lab's own (5 bps + 2 bps slippage).
"""
import json
import sys
from pathlib import Path

import pandas as pd

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
from intraday_strategy_lab.backtest import BacktestSettings, backtest_portfolio, calculate_metrics
from intraday_strategy_lab.strategies import all_strategies

STORE = LAB.parent / "intraday_lab" / "data" / "store"

CANDIDATES = {
    "b3_three_day_reversal_h2c": {
        "three_day_ret_pct": 3.5, "start_bar": 9, "end_bar": 50,
        "stop_pct": 1.25, "target_pct": 99.0, "max_hold_bars": 75},
    "b3_late_liquidity_run_h2c": {
        "lookback": 18, "min_day_ret": 0.8, "start_bar": 54, "volume_mult": 1.5,
        "stop_pct": 1.25, "target_pct": 99.0, "max_hold_bars": 75},
    "b3_vwap_compression_release": {
        "lookback": 18, "max_dist_pct": 0.12, "slope_bars": 8,
        "stop_pct": 1.25, "target_pct": 99.0, "max_hold_bars": 75},
}

settings = BacktestSettings()  # matches config/backtest.yaml defaults
strategies = {s.name: s for s in all_strategies()}

data = {}
for p in sorted(STORE.glob("*_5min_holdout.parquet")):
    sym = p.name.replace("_5min_holdout.parquet", "")
    if sym == "NIFTY":
        continue
    df = pd.read_parquet(p).reset_index().rename(columns={"ts": "timestamp"})
    df["symbol"] = sym
    data[sym] = df.sort_values("timestamp").reset_index(drop=True)
print(f"holdout universe: {len(data)} symbols, "
      f"{min(d['timestamp'].min() for d in data.values()):%Y-%m-%d} .. "
      f"{max(d['timestamp'].max() for d in data.values()):%Y-%m-%d}")

out = {}
print(f"\n{'strategy':<30} {'n':>5} {'wr%':>6} {'pf':>6} {'sharpe':>7} {'net':>10}  verdict")
for name, params in CANDIDATES.items():
    trades = backtest_portfolio(data, strategies[name], params, settings)
    m = calculate_metrics(trades, settings)
    yearly = {}
    if len(trades):
        trades["year"] = pd.to_datetime(trades["exit_time"]).dt.year
        for y, g in trades.groupby("year"):
            gm = calculate_metrics(g.drop(columns="year"), settings)
            yearly[int(y)] = {"n": gm["trade_count"], "pf": round(gm["profit_factor"], 2),
                              "net": round(gm["net_pnl"])}
    keep = (m["profit_factor"] >= 1.1 and m["trade_count"] >= 50
            and all(v["net"] > 0 for v in yearly.values()) and len(yearly) == 2)
    out[name] = {"params": params, "metrics": {k: round(v, 4) if isinstance(v, float) else v
                                               for k, v in m.items()},
                 "yearly": yearly, "keep": keep}
    print(f"{name:<30} {m['trade_count']:>5} {m['win_rate_pct']:>6.1f} "
          f"{m['profit_factor']:>6.2f} {m['sharpe']:>7.2f} {m['net_pnl']:>10.0f}  "
          f"{'KEEP' if keep else 'reject'}")
    for y, v in yearly.items():
        print(f"    {y}: n={v['n']} pf={v['pf']} net={v['net']:+d}")

(LAB / "results" / "holdout_2024_b3.json").write_text(json.dumps(out, indent=2))
print("\nsaved results/holdout_2024_b3.json")
