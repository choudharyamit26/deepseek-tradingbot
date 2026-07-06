"""Reusable frozen holdout evaluator for selected strategy candidates.

Protocol: evaluate frozen parameters on intraday_lab cached 2024-01-01..2025-07-03
holdout parquets. Keep rule: PF >= 1.1, n >= 50, and net > 0 in both 2024 and
2025H1. Candidate params default to the most common walk-forward fold params.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))

from intraday_strategy_lab.backtest import BacktestSettings, backtest_portfolio, calculate_metrics
from intraday_strategy_lab.strategies import all_strategies


STORE = LAB.parent / "intraday_lab" / "data" / "store"


def load_holdout_data() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(STORE.glob("*_5min_holdout.parquet")):
        symbol = path.name.replace("_5min_holdout.parquet", "")
        if symbol == "NIFTY":
            continue
        frame = pd.read_parquet(path).reset_index().rename(columns={"ts": "timestamp"})
        frame["symbol"] = symbol
        data[symbol] = frame.sort_values("timestamp").reset_index(drop=True)
    if not data:
        raise FileNotFoundError(f"No holdout parquet files found in {STORE}")
    return data


def most_common_walk_forward_params(strategy: str) -> dict[str, Any]:
    path = LAB / "results" / "walk_forward" / "folds" / f"{strategy}_fold_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing walk-forward fold metrics for {strategy}: {path}")
    frame = pd.read_csv(path)
    if "best_params" not in frame or frame.empty:
        raise ValueError(f"No best_params available in {path}")
    params_text = Counter(frame["best_params"].dropna().astype(str)).most_common(1)[0][0]
    return json.loads(params_text)


def evaluate(strategy_names: list[str], output_name: str) -> dict[str, Any]:
    settings = BacktestSettings()
    strategies = {strategy.name: strategy for strategy in all_strategies()}
    data = load_holdout_data()
    out: dict[str, Any] = {}
    print(
        f"holdout universe: {len(data)} symbols, "
        f"{min(frame['timestamp'].min() for frame in data.values()):%Y-%m-%d} .. "
        f"{max(frame['timestamp'].max() for frame in data.values()):%Y-%m-%d}"
    )
    print(f"\n{'strategy':<42} {'n':>5} {'wr%':>6} {'pf':>6} {'sharpe':>7} {'net':>10}  verdict")
    for name in strategy_names:
        if name not in strategies:
            raise KeyError(f"Unknown strategy: {name}")
        params = most_common_walk_forward_params(name)
        trades = backtest_portfolio(data, strategies[name], params, settings)
        metrics = calculate_metrics(trades, settings)
        yearly = {}
        if not trades.empty:
            trades = trades.copy()
            trades["year"] = pd.to_datetime(trades["exit_time"]).dt.year
            for year, group in trades.groupby("year"):
                group_metrics = calculate_metrics(group.drop(columns="year"), settings)
                yearly[int(year)] = {
                    "n": group_metrics["trade_count"],
                    "pf": round(float(group_metrics["profit_factor"]), 4),
                    "net": round(float(group_metrics["net_pnl"]), 2),
                }
        keep = (
            metrics["profit_factor"] >= 1.1
            and metrics["trade_count"] >= 50
            and len(yearly) == 2
            and all(value["net"] > 0 for value in yearly.values())
        )
        out[name] = {
            "params": params,
            "metrics": {key: round(value, 4) if isinstance(value, float) else value for key, value in metrics.items()},
            "yearly": yearly,
            "keep": keep,
        }
        print(
            f"{name:<42} {metrics['trade_count']:>5} {metrics['win_rate_pct']:>6.1f} "
            f"{metrics['profit_factor']:>6.2f} {metrics['sharpe']:>7.2f} {metrics['net_pnl']:>10.0f}  "
            f"{'KEEP' if keep else 'reject'}"
        )
        for year, value in yearly.items():
            print(f"    {year}: n={value['n']} pf={value['pf']} net={value['net']:+.0f}")
    output_path = LAB / "results" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved {output_path.relative_to(LAB)}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen holdout for selected strategies")
    parser.add_argument("strategies", nargs="+", help="Strategy names to evaluate")
    parser.add_argument("--output", default="holdout_candidates.json", help="Output JSON under results/")
    args = parser.parse_args()
    evaluate(args.strategies, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
