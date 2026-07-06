from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import BacktestSettings
from .config import PROJECT_ROOT, load_config_dir, project_path
from .pipeline import download_dhan_data, generate_synthetic_data, run_research, run_walk_forward
from .reports import write_research_reports, write_walk_forward_reports
from .optimize import run_research_suite, run_walk_forward_suite
from .strategies import all_strategies


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT.parent / ".env", override=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intraday strategy lab")
    parser.add_argument("--config-dir", type=Path, default=project_path("config"), help="Folder containing YAML configs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-strategies", help="Print the 20 registered strategy names")

    download = subparsers.add_parser("download-data", help="Download Dhan historical data")
    download.add_argument("--data-root", type=Path, default=project_path("dhan_historical_data"))

    run = subparsers.add_parser("run", help="Run optimization and OOS validation on processed data")
    run.add_argument("--data-root", type=Path, default=project_path("dhan_historical_data"))
    run.add_argument("--results-root", type=Path, default=project_path("results"))

    walk = subparsers.add_parser("walk-forward", help="Run rolling walk-forward optimization and validation")
    walk.add_argument("--data-root", type=Path, default=project_path("dhan_historical_data"))
    walk.add_argument("--results-root", type=Path, default=project_path("results"))
    walk.add_argument("--strategies", nargs="*", help="Optional strategy names, space- or comma-separated")

    smoke = subparsers.add_parser("smoke-test", help="Run all strategies on synthetic data")
    smoke.add_argument("--results-root", type=Path, default=project_path("results", "smoke_test"))
    smoke.add_argument("--max-evals", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_if_available()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-strategies":
        for index, strategy in enumerate(all_strategies(), start=1):
            print(f"{index:02d}. {strategy.name} - {strategy.description}")
        return 0

    if args.command == "download-data":
        selected = download_dhan_data(args.config_dir, args.data_root)
        print(f"Downloaded Dhan data for {len(selected)} selected high-beta symbols: {', '.join(selected)}")
        return 0

    if args.command == "run":
        run_research(args.config_dir, args.data_root, args.results_root)
        print(f"Research reports written to {args.results_root}")
        return 0

    if args.command == "walk-forward":
        run_walk_forward(args.config_dir, args.data_root, args.results_root, args.strategies)
        print(f"Walk-forward reports written to {args.results_root}")
        return 0

    if args.command == "smoke-test":
        configs = load_config_dir(args.config_dir)
        optimization_config = dict(configs["optimization"])
        optimization_config["max_evals_per_strategy"] = args.max_evals
        optimization_config["min_trades_in_sample"] = 1
        data = generate_synthetic_data()
        settings = BacktestSettings.from_dict(configs["backtest"])
        results = run_research_suite(data, settings, optimization_config)
        write_research_reports(results, args.results_root)
        walk_results = run_walk_forward_suite(data, settings, optimization_config)
        write_walk_forward_reports(walk_results, args.results_root)
        print(f"Smoke test completed for {len(results)} strategies. Reports written to {args.results_root}")
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2
