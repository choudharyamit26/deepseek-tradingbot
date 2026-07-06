from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import BacktestSettings, calculate_metrics
from .config import get_env_setting, load_config_dir, project_path
from .data.dhan import DhanClient, download_instrument_master
from .data.io import ensure_directories, load_processed_intraday, save_ohlcv_csv, write_json
from .data.universe import compute_beta_table, load_instrument_master, resolve_security_ids, select_high_beta_symbols
from .optimize import WalkForwardResult, run_research_suite
from .optimize import run_walk_forward_suite
from .reports import write_research_reports
from .reports import write_walk_forward_reports
from .strategies import all_strategies


def _chunks(start: datetime, end: datetime, days: int) -> list[tuple[datetime, datetime]]:
    output: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + timedelta(days=max(1, int(days))))
        output.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(seconds=1)
    return output


def _latest_market_datetime() -> datetime:
    today = datetime.now().date()
    while today.weekday() >= 5:
        today -= timedelta(days=1)
    return datetime.combine(today, time(15, 30))


def download_dhan_data(config_dir: Path | None = None, data_root: Path | None = None) -> list[str]:
    configs = load_config_dir(config_dir)
    dhan_config = configs["dhan"]
    universe_config = configs["universe"]
    data_root = data_root or project_path("dhan_historical_data")
    ensure_directories(data_root)

    client = DhanClient(
        client_id=get_env_setting(dhan_config, "client_id"),
        access_token=get_env_setting(dhan_config, "access_token"),
        base_url=str(dhan_config.get("base_url", "https://api.dhan.co/v2")),
        request_sleep_seconds=float(dhan_config.get("request_sleep_seconds", 0.25)),
    )
    master_path = data_root / "metadata" / "dhan_instrument_master.csv"
    if not master_path.exists():
        download_instrument_master(str(dhan_config["instrument_master_url"]), master_path)
    master = load_instrument_master(master_path)

    candidate_symbols = [str(symbol).upper() for symbol in universe_config.get("candidate_symbols", [])]
    instruments = resolve_security_ids(
        master,
        candidate_symbols,
        exchange_segment=str(dhan_config.get("exchange_segment", "NSE_EQ")),
        instrument=str(dhan_config.get("instrument", "EQUITY")),
    )
    if not instruments:
        raise RuntimeError("No candidate symbols were resolved from the Dhan instrument master.")

    end = _latest_market_datetime()
    start = end - timedelta(days=int(dhan_config.get("lookback_days", 365)))
    daily_by_symbol: dict[str, pd.DataFrame] = {}
    raw_dir = data_root / "raw"
    processed_dir = data_root / "processed"

    benchmark_config = dhan_config.get("benchmark", {})
    benchmark_frame, benchmark_raw = client.fetch_daily(
        security_id=str(benchmark_config.get("security_id", "13")),
        exchange_segment=str(benchmark_config.get("exchange_segment", "NSE_IDX")),
        instrument=str(benchmark_config.get("instrument", "INDEX")),
        from_date=start,
        to_date=end,
    )
    write_json(raw_dir / "benchmark_daily.json", benchmark_raw)
    save_ohlcv_csv(benchmark_frame, processed_dir / "NIFTY_daily.csv", symbol="NIFTY")

    for instrument in instruments:
        frame, raw = client.fetch_daily(
            security_id=instrument.security_id,
            exchange_segment=instrument.exchange_segment,
            instrument=instrument.instrument,
            from_date=start,
            to_date=end,
        )
        write_json(raw_dir / f"{instrument.symbol}_daily.json", raw)
        save_ohlcv_csv(frame, processed_dir / f"{instrument.symbol}_daily.csv", symbol=instrument.symbol)
        daily_by_symbol[instrument.symbol] = frame

    beta_table = compute_beta_table(daily_by_symbol, benchmark_frame)
    beta_table.to_csv(data_root / "metadata" / "beta_table.csv", index=False)
    selected = select_high_beta_symbols(
        beta_table,
        count=int(universe_config.get("top_beta_count", 20)),
        min_avg_daily_value=float(universe_config.get("min_avg_daily_value", 0)),
    )
    if not selected:
        selected = [instrument.symbol for instrument in instruments[: int(universe_config.get("top_beta_count", 20))]]
    write_json(data_root / "metadata" / "selected_high_beta_symbols.json", selected)

    instrument_by_symbol = {instrument.symbol: instrument for instrument in instruments}
    chunk_days = int(dhan_config.get("intraday_chunk_days", 30))
    for symbol in selected:
        instrument = instrument_by_symbol[symbol]
        frames: list[pd.DataFrame] = []
        raw_chunks: list[dict[str, Any]] = []
        for chunk_start, chunk_end in _chunks(start, end, chunk_days):
            frame, raw = client.fetch_intraday(
                security_id=instrument.security_id,
                exchange_segment=instrument.exchange_segment,
                instrument=instrument.instrument,
                interval=str(dhan_config.get("interval", "5")),
                from_datetime=chunk_start,
                to_datetime=chunk_end,
            )
            if not frame.empty:
                frames.append(frame)
            raw_chunks.append(raw)
        write_json(raw_dir / f"{symbol}_intraday_chunks.json", raw_chunks)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            save_ohlcv_csv(combined, processed_dir / f"{symbol}_intraday.csv", symbol=symbol)
    return selected


def run_research(config_dir: Path | None = None, data_root: Path | None = None, results_root: Path | None = None) -> None:
    configs = load_config_dir(config_dir)
    data_root = data_root or project_path("dhan_historical_data")
    results_root = results_root or project_path("results")
    data_by_symbol = load_processed_intraday(data_root / "processed")
    settings = BacktestSettings.from_dict(configs["backtest"])
    results = run_research_suite(data_by_symbol, settings, configs["optimization"])
    write_research_reports(results, results_root)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_existing_walk_forward_results(results_root: Path, settings: BacktestSettings) -> list[WalkForwardResult]:
    walk_dir = results_root / "walk_forward"
    fold_dir = walk_dir / "folds"
    trade_dir = walk_dir / "trades"
    trial_dir = walk_dir / "trials"
    if not fold_dir.exists():
        return []
    existing: list[WalkForwardResult] = []
    suffix = "_fold_metrics.csv"
    for fold_path in sorted(fold_dir.glob(f"*{suffix}")):
        strategy = fold_path.name[: -len(suffix)]
        fold_metrics = _read_csv_or_empty(fold_path)
        fold_trades = _read_csv_or_empty(trade_dir / f"{strategy}_trades.csv")
        fold_trials = _read_csv_or_empty(trial_dir / f"{strategy}_trials.csv")
        aggregate_metrics = calculate_metrics(fold_trades, settings)
        aggregate_metrics["fold_count"] = int(fold_metrics["fold"].nunique()) if "fold" in fold_metrics else int(len(fold_metrics))
        if "oos_net_pnl" in fold_metrics:
            aggregate_metrics["profitable_fold_count"] = int((fold_metrics["oos_net_pnl"] > 0).sum())
        else:
            aggregate_metrics["profitable_fold_count"] = 0
        existing.append(WalkForwardResult(strategy, aggregate_metrics, fold_metrics, fold_trades, fold_trials))
    return existing


def _normalise_strategy_names(strategy_names: list[str] | None) -> set[str] | None:
    if not strategy_names:
        return None
    names: set[str] = set()
    for item in strategy_names:
        for name in str(item).split(","):
            clean = name.strip()
            if clean:
                names.add(clean)
    return names or None


def run_walk_forward(
    config_dir: Path | None = None,
    data_root: Path | None = None,
    results_root: Path | None = None,
    strategy_names: list[str] | None = None,
) -> None:
    configs = load_config_dir(config_dir)
    data_root = data_root or project_path("dhan_historical_data")
    results_root = results_root or project_path("results")
    data_by_symbol = load_processed_intraday(data_root / "processed")
    settings = BacktestSettings.from_dict(configs["backtest"])
    results = _load_existing_walk_forward_results(results_root, settings)
    completed = {result.strategy for result in results}
    selected_names = _normalise_strategy_names(strategy_names)
    strategies = [strategy for strategy in all_strategies() if selected_names is None or strategy.name in selected_names]
    if selected_names:
        known = {strategy.name for strategy in all_strategies()}
        unknown = selected_names - known
        if unknown:
            raise ValueError(f"Unknown strategies: {', '.join(sorted(unknown))}")
    pending = [strategy for strategy in strategies if strategy.name not in completed]
    if not pending:
        write_walk_forward_reports(results, results_root)
        print("Walk-forward already complete for requested strategies.", flush=True)
        return
    for index, strategy in enumerate(pending, start=1):
        print(f"Walk-forward {index}/{len(pending)}: {strategy.name}", flush=True)
        results.extend(run_walk_forward_suite(data_by_symbol, settings, configs["optimization"], strategies=[strategy]))
        write_walk_forward_reports(results, results_root)


def generate_synthetic_data(symbols: list[str] | None = None, sessions: int = 45, seed: int = 7) -> dict[str, pd.DataFrame]:
    symbols = symbols or ["SYNTH1", "SYNTH2", "SYNTH3"]
    rng = np.random.default_rng(seed)
    output: dict[str, pd.DataFrame] = {}
    start_date = pd.Timestamp("2025-01-01")
    session_days = pd.bdate_range(start_date, periods=sessions)
    intraday_times = pd.date_range("09:15", "15:25", freq="5min").time
    for symbol_index, symbol in enumerate(symbols):
        price = 100 + symbol_index * 20
        rows: list[dict[str, Any]] = []
        for day in session_days:
            day_drift = rng.normal(0, 0.004)
            price *= 1 + rng.normal(0, 0.015)
            for current_time in intraday_times:
                timestamp = pd.Timestamp.combine(day.date(), current_time)
                open_price = price
                move = rng.normal(day_drift / len(intraday_times), 0.0028)
                close = max(1, open_price * (1 + move))
                high = max(open_price, close) * (1 + abs(rng.normal(0, 0.0015)))
                low = min(open_price, close) * (1 - abs(rng.normal(0, 0.0015)))
                volume = int(rng.integers(50_000, 500_000))
                rows.append(
                    {
                        "timestamp": timestamp,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "symbol": symbol,
                    }
                )
                price = close
        output[symbol] = pd.DataFrame(rows)
    return output
