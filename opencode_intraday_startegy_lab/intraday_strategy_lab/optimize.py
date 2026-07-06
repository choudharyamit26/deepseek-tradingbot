from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .backtest import BacktestSettings, backtest_portfolio, calculate_metrics
from .strategies import StrategyDefinition, all_strategies


@dataclass
class OptimizationResult:
    strategy: str
    best_params: dict[str, Any]
    best_score: float
    in_sample_metrics: dict[str, float | int]
    in_sample_trades: pd.DataFrame
    trials: pd.DataFrame


@dataclass
class ResearchResult:
    strategy: str
    best_params: dict[str, Any]
    best_score: float
    in_sample_metrics: dict[str, float | int]
    out_of_sample_metrics: dict[str, float | int]
    in_sample_trades: pd.DataFrame
    out_of_sample_trades: pd.DataFrame
    trials: pd.DataFrame


@dataclass
class WalkForwardResult:
    strategy: str
    aggregate_metrics: dict[str, float | int]
    fold_metrics: pd.DataFrame
    fold_trades: pd.DataFrame
    fold_trials: pd.DataFrame


def iter_param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def sample_param_grid(grid: dict[str, list[Any]], max_evals: int, seed: int) -> list[dict[str, Any]]:
    combinations = iter_param_grid(grid)
    if max_evals <= 0 or len(combinations) <= max_evals:
        return combinations
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(combinations)), max_evals))
    return [combinations[index] for index in indices]


def score_metrics(metrics: dict[str, float | int], config: dict[str, Any]) -> float:
    objective = config.get("objective", {}) if isinstance(config, dict) else {}
    min_trades = int(config.get("min_trades_in_sample", 20)) if isinstance(config, dict) else 20
    score = (
        float(metrics.get("total_return_pct", 0)) * float(objective.get("total_return_weight", 1.0))
        + float(metrics.get("sharpe", 0)) * float(objective.get("sharpe_weight", 2.0))
        + float(metrics.get("profit_factor", 0)) * float(objective.get("profit_factor_weight", 2.0))
        - float(metrics.get("max_drawdown_pct", 0)) * float(objective.get("drawdown_penalty_weight", 1.5))
    )
    if int(metrics.get("trade_count", 0)) < min_trades:
        score -= float(objective.get("low_trade_penalty", 25.0))
    return float(score)


def split_by_date(
    data_by_symbol: dict[str, pd.DataFrame], train_ratio: float
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    all_dates: list[object] = []
    for frame in data_by_symbol.values():
        all_dates.extend(pd.to_datetime(frame["timestamp"]).dt.date.unique().tolist())
    dates = sorted(set(all_dates))
    if len(dates) < 2:
        raise ValueError("Need at least two sessions to create in-sample and out-of-sample splits.")
    cutoff_index = max(1, min(len(dates) - 1, int(len(dates) * float(train_ratio))))
    train_dates = set(dates[:cutoff_index])
    test_dates = set(dates[cutoff_index:])
    train: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    for symbol, frame in data_by_symbol.items():
        session_dates = pd.to_datetime(frame["timestamp"]).dt.date
        train[symbol] = frame[session_dates.isin(train_dates)].reset_index(drop=True)
        test[symbol] = frame[session_dates.isin(test_dates)].reset_index(drop=True)
    return train, test


def _filter_by_dates(data_by_symbol: dict[str, pd.DataFrame], dates: set[object]) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for symbol, frame in data_by_symbol.items():
        session_dates = pd.to_datetime(frame["timestamp"]).dt.date
        output[symbol] = frame[session_dates.isin(dates)].reset_index(drop=True)
    return output


def walk_forward_date_splits(
    data_by_symbol: dict[str, pd.DataFrame],
    train_sessions: int,
    test_sessions: int,
    step_sessions: int,
    max_folds: int | None = None,
) -> list[tuple[int, list[object], list[object]]]:
    all_dates: list[object] = []
    for frame in data_by_symbol.values():
        all_dates.extend(pd.to_datetime(frame["timestamp"]).dt.date.unique().tolist())
    dates = sorted(set(all_dates))
    train_sessions = int(train_sessions)
    test_sessions = int(test_sessions)
    step_sessions = max(1, int(step_sessions))
    if train_sessions <= 0 or test_sessions <= 0:
        raise ValueError("walk_forward train_sessions and test_sessions must be positive.")
    splits: list[tuple[int, list[object], list[object]]] = []
    start = 0
    fold = 1
    while start + train_sessions + test_sessions <= len(dates):
        train_dates = dates[start : start + train_sessions]
        test_dates = dates[start + train_sessions : start + train_sessions + test_sessions]
        splits.append((fold, train_dates, test_dates))
        start += step_sessions
        fold += 1
    if max_folds and max_folds > 0 and len(splits) > max_folds:
        splits = splits[-int(max_folds) :]
        splits = [(index + 1, train_dates, test_dates) for index, (_, train_dates, test_dates) in enumerate(splits)]
    if not splits:
        raise ValueError(
            f"Not enough sessions for walk-forward split. Need at least {train_sessions + test_sessions}. Found {len(dates)}."
        )
    return splits


def optimize_strategy(
    strategy: StrategyDefinition,
    data_by_symbol: dict[str, pd.DataFrame],
    settings: BacktestSettings,
    optimization_config: dict[str, Any],
) -> OptimizationResult:
    max_evals = int(optimization_config.get("max_evals_per_strategy", 40))
    seed = int(optimization_config.get("random_seed", 42))
    parameter_sets = sample_param_grid(strategy.param_grid, max_evals=max_evals, seed=seed)
    best_score = float("-inf")
    best_params: dict[str, Any] = dict(strategy.default_params)
    best_metrics: dict[str, float | int] = calculate_metrics(pd.DataFrame(), settings)
    best_trades = pd.DataFrame()
    trial_rows: list[dict[str, Any]] = []

    for params in parameter_sets:
        merged = dict(strategy.default_params)
        merged.update(params)
        trades = backtest_portfolio(data_by_symbol, strategy, merged, settings)
        metrics = calculate_metrics(trades, settings)
        score = score_metrics(metrics, optimization_config)
        row = {"strategy": strategy.name, "score": score, "params": json.dumps(merged, sort_keys=True)}
        row.update(metrics)
        trial_rows.append(row)
        if score > best_score:
            best_score = score
            best_params = merged
            best_metrics = metrics
            best_trades = trades

    trials = pd.DataFrame(trial_rows).sort_values("score", ascending=False).reset_index(drop=True)
    return OptimizationResult(strategy.name, best_params, best_score, best_metrics, best_trades, trials)


def run_research_suite(
    data_by_symbol: dict[str, pd.DataFrame],
    settings: BacktestSettings,
    optimization_config: dict[str, Any],
    strategies: list[StrategyDefinition] | None = None,
) -> list[ResearchResult]:
    train_ratio = float(optimization_config.get("train_ratio", 0.70))
    train_data, test_data = split_by_date(data_by_symbol, train_ratio)
    results: list[ResearchResult] = []
    for strategy in strategies or all_strategies():
        optimized = optimize_strategy(strategy, train_data, settings, optimization_config)
        oos_trades = backtest_portfolio(test_data, strategy, optimized.best_params, settings)
        oos_metrics = calculate_metrics(oos_trades, settings)
        results.append(
            ResearchResult(
                strategy=optimized.strategy,
                best_params=optimized.best_params,
                best_score=optimized.best_score,
                in_sample_metrics=optimized.in_sample_metrics,
                out_of_sample_metrics=oos_metrics,
                in_sample_trades=optimized.in_sample_trades,
                out_of_sample_trades=oos_trades,
                trials=optimized.trials,
            )
        )
    return results


def run_walk_forward_suite(
    data_by_symbol: dict[str, pd.DataFrame],
    settings: BacktestSettings,
    optimization_config: dict[str, Any],
    strategies: list[StrategyDefinition] | None = None,
) -> list[WalkForwardResult]:
    wf_config = dict(optimization_config.get("walk_forward", {}))
    splits = walk_forward_date_splits(
        data_by_symbol,
        train_sessions=int(wf_config.get("train_sessions", 120)),
        test_sessions=int(wf_config.get("test_sessions", 20)),
        step_sessions=int(wf_config.get("step_sessions", 20)),
        max_folds=int(wf_config.get("max_folds", 0)) or None,
    )
    fold_opt_config = dict(optimization_config)
    if "max_evals_per_strategy" in wf_config:
        fold_opt_config["max_evals_per_strategy"] = int(wf_config["max_evals_per_strategy"])

    results: list[WalkForwardResult] = []
    for strategy in strategies or all_strategies():
        metrics_rows: list[dict[str, Any]] = []
        trade_frames: list[pd.DataFrame] = []
        trial_frames: list[pd.DataFrame] = []
        for fold, train_dates, test_dates in splits:
            train_data = _filter_by_dates(data_by_symbol, set(train_dates))
            test_data = _filter_by_dates(data_by_symbol, set(test_dates))
            optimized = optimize_strategy(strategy, train_data, settings, fold_opt_config)
            oos_trades = backtest_portfolio(test_data, strategy, optimized.best_params, settings)
            oos_metrics = calculate_metrics(oos_trades, settings)
            metrics_row: dict[str, Any] = {
                "fold": fold,
                "strategy": strategy.name,
                "train_start": str(train_dates[0]),
                "train_end": str(train_dates[-1]),
                "test_start": str(test_dates[0]),
                "test_end": str(test_dates[-1]),
                "best_score": optimized.best_score,
                "best_params": json.dumps(optimized.best_params, sort_keys=True),
            }
            metrics_row.update({f"is_{key}": value for key, value in optimized.in_sample_metrics.items()})
            metrics_row.update({f"oos_{key}": value for key, value in oos_metrics.items()})
            metrics_rows.append(metrics_row)

            if not oos_trades.empty:
                trades = oos_trades.copy()
                trades.insert(0, "fold", fold)
                trade_frames.append(trades)
            if not optimized.trials.empty:
                trials = optimized.trials.copy()
                trials.insert(0, "fold", fold)
                trial_frames.append(trials)

        fold_metrics = pd.DataFrame(metrics_rows)
        fold_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        fold_trials = pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()
        aggregate_metrics = calculate_metrics(fold_trades, settings)
        aggregate_metrics["fold_count"] = int(len(splits))
        aggregate_metrics["profitable_fold_count"] = int((fold_metrics.get("oos_net_pnl", pd.Series(dtype=float)) > 0).sum())
        results.append(WalkForwardResult(strategy.name, aggregate_metrics, fold_metrics, fold_trades, fold_trials))
    return results
