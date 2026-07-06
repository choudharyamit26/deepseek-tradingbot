from __future__ import annotations

from intraday_strategy_lab.backtest import BacktestSettings, backtest_portfolio, calculate_metrics
from intraday_strategy_lab.optimize import run_research_suite, run_walk_forward_suite
from intraday_strategy_lab.pipeline import generate_synthetic_data
from intraday_strategy_lab.strategies import all_strategies


def test_registry_has_20_strategies() -> None:
    assert len(all_strategies()) == 55


def test_backtest_runs_on_synthetic_data() -> None:
    data = generate_synthetic_data(symbols=["AAA", "BBB"], sessions=12)
    strategy = all_strategies()[0]
    settings = BacktestSettings(max_trades_per_day_per_symbol=2)
    trades = backtest_portfolio(data, strategy, strategy.default_params, settings)
    metrics = calculate_metrics(trades, settings)
    assert "trade_count" in metrics
    assert "total_return_pct" in metrics


def test_research_suite_smoke_runs_all_strategies() -> None:
    data = generate_synthetic_data(symbols=["AAA", "BBB"], sessions=14)
    settings = BacktestSettings(max_trades_per_day_per_symbol=1)
    optimization_config = {
        "train_ratio": 0.70,
        "max_evals_per_strategy": 1,
        "random_seed": 1,
        "min_trades_in_sample": 1,
        "objective": {
            "total_return_weight": 1.0,
            "sharpe_weight": 1.0,
            "profit_factor_weight": 1.0,
            "drawdown_penalty_weight": 1.0,
            "low_trade_penalty": 1.0,
        },
    }
    results = run_research_suite(data, settings, optimization_config)
    assert len(results) == 55
    assert all(result.best_params for result in results)


def test_walk_forward_smoke_runs_all_strategies() -> None:
    data = generate_synthetic_data(symbols=["AAA", "BBB"], sessions=16)
    settings = BacktestSettings(max_trades_per_day_per_symbol=1)
    optimization_config = {
        "max_evals_per_strategy": 1,
        "random_seed": 1,
        "min_trades_in_sample": 1,
        "walk_forward": {"train_sessions": 8, "test_sessions": 4, "step_sessions": 4, "max_folds": 2, "max_evals_per_strategy": 1},
        "objective": {
            "total_return_weight": 1.0,
            "sharpe_weight": 1.0,
            "profit_factor_weight": 1.0,
            "drawdown_penalty_weight": 1.0,
            "low_trade_penalty": 1.0,
        },
    }
    results = run_walk_forward_suite(data, settings, optimization_config)
    assert len(results) == 55
    assert all(result.aggregate_metrics["fold_count"] == 2 for result in results)
