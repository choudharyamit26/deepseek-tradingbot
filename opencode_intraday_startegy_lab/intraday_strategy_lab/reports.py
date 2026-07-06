from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .optimize import ResearchResult
from .optimize import WalkForwardResult


def _safe_value(value: Any) -> Any:
    if isinstance(value, float) and (pd.isna(value) or value in [float("inf"), float("-inf")]):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def write_research_reports(results: list[ResearchResult], output_root: Path) -> None:
    in_sample_dir = output_root / "in_sample"
    out_sample_dir = output_root / "out_of_sample"
    optimization_dir = output_root / "optimization"
    combined_dir = output_root / "combined_reports"
    for directory in [in_sample_dir, out_sample_dir, optimization_dir, combined_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    leaderboard_rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] = {}
    all_trials: list[pd.DataFrame] = []
    in_sample_trades: list[pd.DataFrame] = []
    out_sample_trades: list[pd.DataFrame] = []

    for result in results:
        result.in_sample_trades.to_csv(in_sample_dir / f"{result.strategy}_trades.csv", index=False)
        result.out_of_sample_trades.to_csv(out_sample_dir / f"{result.strategy}_trades.csv", index=False)
        result.trials.to_csv(optimization_dir / f"{result.strategy}_trials.csv", index=False)

        row: dict[str, Any] = {"strategy": result.strategy, "best_score": result.best_score, "best_params": json.dumps(result.best_params, sort_keys=True)}
        row.update({f"is_{key}": value for key, value in result.in_sample_metrics.items()})
        row.update({f"oos_{key}": value for key, value in result.out_of_sample_metrics.items()})
        leaderboard_rows.append(row)
        best_params[result.strategy] = result.best_params
        if not result.trials.empty:
            all_trials.append(result.trials)
        if not result.in_sample_trades.empty:
            in_sample_trades.append(result.in_sample_trades)
        if not result.out_of_sample_trades.empty:
            out_sample_trades.append(result.out_of_sample_trades)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("oos_total_return_pct", ascending=False)
    leaderboard.to_csv(combined_dir / "strategy_leaderboard.csv", index=False)
    _write_json(optimization_dir / "best_params.json", best_params)
    _write_json(combined_dir / "summary.json", [{key: _safe_value(value) for key, value in row.items()} for row in leaderboard_rows])
    if all_trials:
        pd.concat(all_trials, ignore_index=True).to_csv(optimization_dir / "all_trials.csv", index=False)
    if in_sample_trades:
        pd.concat(in_sample_trades, ignore_index=True).to_csv(in_sample_dir / "all_trades.csv", index=False)
    if out_sample_trades:
        pd.concat(out_sample_trades, ignore_index=True).to_csv(out_sample_dir / "all_trades.csv", index=False)


def write_walk_forward_reports(results: list[WalkForwardResult], output_root: Path) -> None:
    walk_dir = output_root / "walk_forward"
    fold_dir = walk_dir / "folds"
    trade_dir = walk_dir / "trades"
    trial_dir = walk_dir / "trials"
    for directory in [walk_dir, fold_dir, trade_dir, trial_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    leaderboard_rows: list[dict[str, Any]] = []
    all_fold_metrics: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    all_trials: list[pd.DataFrame] = []
    for result in results:
        result.fold_metrics.to_csv(fold_dir / f"{result.strategy}_fold_metrics.csv", index=False)
        result.fold_trades.to_csv(trade_dir / f"{result.strategy}_trades.csv", index=False)
        result.fold_trials.to_csv(trial_dir / f"{result.strategy}_trials.csv", index=False)
        row: dict[str, Any] = {"strategy": result.strategy}
        row.update(result.aggregate_metrics)
        leaderboard_rows.append(row)
        if not result.fold_metrics.empty:
            all_fold_metrics.append(result.fold_metrics)
        if not result.fold_trades.empty:
            all_trades.append(result.fold_trades)
        if not result.fold_trials.empty:
            all_trials.append(result.fold_trials)

    leaderboard = pd.DataFrame(leaderboard_rows).sort_values(
        ["net_pnl", "profit_factor", "sharpe"], ascending=[False, False, False]
    )
    leaderboard.to_csv(walk_dir / "walk_forward_leaderboard.csv", index=False)
    _write_json(walk_dir / "walk_forward_summary.json", leaderboard_rows)
    if all_fold_metrics:
        pd.concat(all_fold_metrics, ignore_index=True).to_csv(walk_dir / "all_fold_metrics.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(walk_dir / "all_trades.csv", index=False)
    if all_trials:
        pd.concat(all_trials, ignore_index=True).to_csv(walk_dir / "all_trials.csv", index=False)
