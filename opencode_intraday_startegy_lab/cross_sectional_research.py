"""Cross-sectional intraday research batch.

This is intentionally separate from the per-symbol strategy registry. Each recipe
selects at most one symbol per day across the whole high-beta book, then holds the
trade intraday with a stop and square-off. The goal is to reduce turnover and test
whether relative selection clears the cost hurdle better than many independent
per-symbol entries.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))

from intraday_strategy_lab.backtest import BacktestSettings, calculate_metrics
from intraday_strategy_lab.data.io import read_ohlcv_csv
from intraday_strategy_lab.indicators import parse_time


FeatureRule = Callable[[pd.Series, dict[str, Any]], tuple[int, float]]


@dataclass(frozen=True)
class XSRecipe:
    name: str
    description: str
    default_params: dict[str, Any]
    grid: dict[str, list[Any]]
    rule: FeatureRule


def _param_grid(grid: dict[str, list[Any]], max_evals: int = 3) -> list[dict[str, Any]]:
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]
    return combos[:max_evals] if max_evals > 0 else combos


def _dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["timestamp"]).dt.date


def _session_time_mask(df: pd.DataFrame, cutoff: time) -> pd.Series:
    return pd.to_datetime(df["timestamp"]).dt.time <= cutoff


def _daily_context(df: pd.DataFrame) -> pd.DataFrame:
    dates = _dates(df)
    daily = df.groupby(dates, sort=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
    )
    daily["range"] = daily["high"] - daily["low"]
    daily["ret"] = (daily["close"] / daily["open"] - 1) * 100
    daily["close_pos"] = (daily["close"] - daily["low"]) / daily["range"].replace(0, np.nan)
    daily["volume_med20"] = daily["volume"].rolling(20).median().shift(1)
    daily["range_med20"] = daily["range"].rolling(20).median().shift(1)
    daily["prev5_ret"] = daily["ret"].rolling(5).sum().shift(1)
    daily["high20"] = daily["high"].rolling(20).max().shift(1)
    daily["low20"] = daily["low"].rolling(20).min().shift(1)
    prev = daily.shift(1).add_prefix("prev_")
    return pd.concat([daily, prev], axis=1)


def load_current_data() -> dict[str, pd.DataFrame]:
    processed = LAB / "dhan_historical_data" / "processed"
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(processed.glob("*_intraday.csv")):
        symbol = path.name.replace("_intraday.csv", "")
        data[symbol] = read_ohlcv_csv(path)
    if not data:
        raise FileNotFoundError(f"No processed intraday files found in {processed}")
    return data


def load_holdout_data() -> dict[str, pd.DataFrame]:
    store = LAB.parent / "intraday_lab" / "data" / "store"
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(store.glob("*_5min_holdout.parquet")):
        symbol = path.name.replace("_5min_holdout.parquet", "")
        if symbol == "NIFTY":
            continue
        frame = pd.read_parquet(path).reset_index().rename(columns={"ts": "timestamp"})
        frame["symbol"] = symbol
        data[symbol] = frame.sort_values("timestamp").reset_index(drop=True)
    if not data:
        raise FileNotFoundError(f"No holdout parquet files found in {store}")
    return data


def available_dates(data: dict[str, pd.DataFrame]) -> list[object]:
    dates: set[object] = set()
    for frame in data.values():
        dates.update(_dates(frame).unique().tolist())
    return sorted(dates)


def walk_forward_splits(data: dict[str, pd.DataFrame], train_sessions: int = 120, test_sessions: int = 20, step_sessions: int = 20, max_folds: int = 5) -> list[tuple[int, list[object], list[object]]]:
    dates = available_dates(data)
    splits: list[tuple[int, list[object], list[object]]] = []
    start = 0
    fold = 1
    while start + train_sessions + test_sessions <= len(dates):
        splits.append((fold, dates[start : start + train_sessions], dates[start + train_sessions : start + train_sessions + test_sessions]))
        start += step_sessions
        fold += 1
    if max_folds and len(splits) > max_folds:
        splits = splits[-max_folds:]
        splits = [(index + 1, train_dates, test_dates) for index, (_, train_dates, test_dates) in enumerate(splits)]
    if not splits:
        raise ValueError("Not enough sessions for cross-sectional walk-forward split")
    return splits


def _vwap_for_slice(frame: pd.DataFrame) -> pd.Series:
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3
    volume = frame["volume"].clip(lower=0)
    return (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)


def _feature_row(symbol: str, day: object, group: pd.DataFrame, context: pd.DataFrame, decision_bar: int, settings: BacktestSettings) -> dict[str, Any] | None:
    if day not in context.index or len(group) <= decision_bar + 1:
        return None
    squareoff = parse_time(settings.squareoff_time)
    squareoff_positions = group.index[_session_time_mask(group, squareoff)]
    if len(squareoff_positions) == 0:
        return None
    squareoff_label = squareoff_positions[-1]
    local_labels = list(group.index)
    entry_label = local_labels[decision_bar + 1]
    if local_labels.index(entry_label) > local_labels.index(squareoff_label):
        return None

    hist = group.iloc[: decision_bar + 1]
    decision = hist.iloc[-1]
    entry = group.loc[entry_label]
    exit_slice = group.loc[entry_label:squareoff_label]
    if exit_slice.empty:
        return None

    ctx = context.loc[day]
    if pd.isna(ctx.get("prev_close", np.nan)):
        return None
    vwap_series = _vwap_for_slice(hist)
    vwap = float(vwap_series.iloc[-1]) if len(vwap_series) else np.nan
    day_open = float(group.iloc[0]["open"])
    high_so_far = float(hist["high"].max())
    low_so_far = float(hist["low"].min())
    range_so_far = max(high_so_far - low_so_far, np.nan)
    first_volume = float(hist["volume"].sum())
    prev_volume_med20 = float(ctx.get("volume_med20", np.nan))
    prev_range_med20 = float(ctx.get("range_med20", np.nan))
    close = float(decision["close"])
    open_efficiency = abs(close - day_open) / range_so_far if range_so_far and not np.isnan(range_so_far) else np.nan
    close_location = (close - low_so_far) / range_so_far if range_so_far and not np.isnan(range_so_far) else np.nan

    return {
        "symbol": symbol,
        "date": day,
        "decision_time": decision["timestamp"],
        "entry_time": entry["timestamp"],
        "entry_open": float(entry["open"]),
        "exit_time": exit_slice.iloc[-1]["timestamp"],
        "exit_close": float(exit_slice.iloc[-1]["close"]),
        "high_after_entry": float(exit_slice["high"].max()),
        "low_after_entry": float(exit_slice["low"].min()),
        "day_open": day_open,
        "decision_close": close,
        "first_ret": (close / day_open - 1) * 100,
        "gap": (day_open / float(ctx["prev_close"]) - 1) * 100 if float(ctx["prev_close"]) else np.nan,
        "vwap": vwap,
        "vwap_dist": (close / vwap - 1) * 100 if vwap else np.nan,
        "vwap_slope": vwap - float(vwap_series.iloc[max(0, len(vwap_series) - 5)]) if len(vwap_series) else np.nan,
        "above_vwap_frac": float((hist["close"] > vwap_series).mean()) if len(vwap_series) else np.nan,
        "range_so_far": range_so_far,
        "range_ratio": range_so_far / prev_range_med20 if prev_range_med20 else np.nan,
        "first_volume_ratio": first_volume / prev_volume_med20 if prev_volume_med20 else np.nan,
        "open_efficiency": open_efficiency,
        "close_location": close_location,
        "prev_ret": float(ctx.get("prev_ret", np.nan)),
        "prev5_ret": float(ctx.get("prev5_ret", np.nan)),
        "prev_close_pos": float(ctx.get("prev_close_pos", np.nan)),
        "prev_range_ratio": float(ctx.get("prev_range", np.nan)) / prev_range_med20 if prev_range_med20 else np.nan,
        "prev_close": float(ctx["prev_close"]),
        "prev_high": float(ctx.get("prev_high", np.nan)),
        "prev_low": float(ctx.get("prev_low", np.nan)),
        "high20": float(ctx.get("high20", np.nan)),
        "low20": float(ctx.get("low20", np.nan)),
    }


def make_feature_table(data: dict[str, pd.DataFrame], decision_bar: int, settings: BacktestSettings, selected_dates: set[object] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, frame in data.items():
        working = frame.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"])
        working = working.sort_values("timestamp").reset_index(drop=True)
        context = _daily_context(working)
        for day, group in working.groupby(_dates(working), sort=False):
            if selected_dates is not None and day not in selected_dates:
                continue
            row = _feature_row(symbol, day, group.reset_index(drop=True), context, decision_bar, settings)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def _entry_fill(open_price: float, direction: int, settings: BacktestSettings) -> float:
    slippage = settings.slippage_bps / 10_000
    return open_price * (1 + slippage if direction == 1 else 1 - slippage)


def _exit_fill(price: float, direction: int, settings: BacktestSettings) -> float:
    slippage = settings.slippage_bps / 10_000
    return price * (1 - slippage if direction == 1 else 1 + slippage)


def _trade_from_selection(recipe: str, selected: pd.Series, direction: int, params: dict[str, Any], settings: BacktestSettings) -> dict[str, Any] | None:
    entry_price = _entry_fill(float(selected["entry_open"]), direction, settings)
    quantity = int(settings.capital_per_trade // entry_price)
    if quantity <= 0:
        return None
    stop_pct = float(params.get("stop_pct", 1.0)) / 100
    stop_price = entry_price * (1 - stop_pct if direction == 1 else 1 + stop_pct)
    hit_stop = float(selected["low_after_entry"]) <= stop_price if direction == 1 else float(selected["high_after_entry"]) >= stop_price
    raw_exit = stop_price if hit_stop else float(selected["exit_close"])
    exit_price = _exit_fill(raw_exit, direction, settings)
    gross = (exit_price - entry_price) * quantity * direction
    turnover = (entry_price + exit_price) * quantity
    costs = turnover * settings.cost_bps / 10_000
    net = gross - costs
    return {
        "strategy": recipe,
        "symbol": selected["symbol"],
        "direction": direction,
        "entry_time": selected["entry_time"],
        "exit_time": selected["exit_time"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "gross_pnl": gross,
        "costs": costs,
        "net_pnl": net,
        "return_pct": net / settings.capital_per_trade * 100,
        "exit_reason": "stop" if hit_stop else "squareoff",
        "bars_held": 0,
        "params": json.dumps(params, sort_keys=True),
    }


def run_recipe(data: dict[str, pd.DataFrame], recipe: XSRecipe, params: dict[str, Any], dates: set[object], settings: BacktestSettings) -> pd.DataFrame:
    decision_bar = int(params.get("decision_bar", params.get("open_bars", 12)))
    features = make_feature_table(data, decision_bar, settings, selected_dates=dates)
    if features.empty:
        return pd.DataFrame()
    picks: list[dict[str, Any]] = []
    for day, group in features.groupby("date", sort=False):
        candidates: list[tuple[float, int, pd.Series]] = []
        for _, row in group.iterrows():
            direction, score = recipe.rule(row, params)
            if direction and np.isfinite(score):
                candidates.append((float(score), int(direction), row))
        if not candidates:
            continue
        _, direction, selected = max(candidates, key=lambda item: item[0])
        trade = _trade_from_selection(recipe.name, selected, direction, params, settings)
        if trade is not None:
            picks.append(trade)
    return pd.DataFrame(picks)


def _score(metrics: dict[str, float | int]) -> float:
    score = float(metrics.get("total_return_pct", 0)) + 2 * float(metrics.get("profit_factor", 0)) + 2 * float(metrics.get("sharpe", 0))
    score -= 1.5 * float(metrics.get("max_drawdown_pct", 0))
    if int(metrics.get("trade_count", 0)) < 20:
        score -= 25
    return score


def optimize_recipe(data: dict[str, pd.DataFrame], recipe: XSRecipe, train_dates: set[object], settings: BacktestSettings, max_evals: int = 4) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float | int], float, pd.DataFrame]:
    best_params = dict(recipe.default_params)
    best_trades = pd.DataFrame()
    best_metrics = calculate_metrics(best_trades, settings)
    best_score = float("-inf")
    rows: list[dict[str, Any]] = []
    for params_only in _param_grid(recipe.grid, max_evals=max_evals):
        params = dict(recipe.default_params)
        params.update(params_only)
        trades = run_recipe(data, recipe, params, train_dates, settings)
        metrics = calculate_metrics(trades, settings)
        score = _score(metrics)
        rows.append({"strategy": recipe.name, "score": score, "params": json.dumps(params, sort_keys=True), **metrics})
        if score > best_score:
            best_score = score
            best_params = params
            best_trades = trades
            best_metrics = metrics
    trials = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return best_params, best_trades, best_metrics, best_score, trials


def _sign(value: float) -> int:
    return 1 if value > 0 else -1


def r_gap_extension(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if abs(row.gap) < p["gap_pct"] or abs(row.first_ret) < p["move_pct"] or np.sign(row.gap) != np.sign(row.first_ret):
        return 0, 0.0
    return _sign(row.gap), abs(row.gap) + abs(row.first_ret)


def r_gap_fade_relative(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if abs(row.gap) < p["gap_pct"] or abs(row.first_ret) < p["fade_pct"] or np.sign(row.gap) == np.sign(row.first_ret):
        return 0, 0.0
    return -_sign(row.gap), abs(row.gap) + abs(row.first_ret)


def r_open_drive_efficiency(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.open_efficiency < p["efficiency"] or abs(row.first_ret) < p["move_pct"] or row.first_volume_ratio < p["volume_ratio"]:
        return 0, 0.0
    return _sign(row.first_ret), abs(row.first_ret) * row.open_efficiency * max(row.first_volume_ratio, 0)


def r_open_climax_fade(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.range_ratio < p["range_ratio"] or row.first_volume_ratio < p["volume_ratio"]:
        return 0, 0.0
    if row.close_location > p["upper_loc"]:
        return -1, row.range_ratio * row.close_location
    if row.close_location < 1 - p["upper_loc"]:
        return 1, row.range_ratio * (1 - row.close_location)
    return 0, 0.0


def r_vwap_stretch_fade(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if abs(row.vwap_dist) < p["stretch_pct"]:
        return 0, 0.0
    return -_sign(row.vwap_dist), abs(row.vwap_dist)


def r_vwap_bandwalk_continue(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.above_vwap_frac >= p["side_frac"] and row.vwap_slope > 0 and row.vwap_dist > p["min_dist"]:
        return 1, row.above_vwap_frac + row.vwap_dist
    if row.above_vwap_frac <= 1 - p["side_frac"] and row.vwap_slope < 0 and row.vwap_dist < -p["min_dist"]:
        return -1, 1 - row.above_vwap_frac + abs(row.vwap_dist)
    return 0, 0.0


def r_prev_winner_continue(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.prev_ret > p["prev_ret"] and row.first_ret > p["confirm_pct"]:
        return 1, row.prev_ret + row.first_ret
    if row.prev_ret < -p["prev_ret"] and row.first_ret < -p["confirm_pct"]:
        return -1, abs(row.prev_ret) + abs(row.first_ret)
    return 0, 0.0


def r_prev_winner_reversal(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.prev_ret > p["prev_ret"] and row.first_ret < -p["counter_pct"]:
        return -1, row.prev_ret + abs(row.first_ret)
    if row.prev_ret < -p["prev_ret"] and row.first_ret > p["counter_pct"]:
        return 1, abs(row.prev_ret) + row.first_ret
    return 0, 0.0


def r_close_location_follow(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.prev_close_pos >= p["close_pos"] and row.first_ret > p["confirm_pct"]:
        return 1, row.prev_close_pos + row.first_ret
    if row.prev_close_pos <= 1 - p["close_pos"] and row.first_ret < -p["confirm_pct"]:
        return -1, (1 - row.prev_close_pos) + abs(row.first_ret)
    return 0, 0.0


def r_close_location_fade(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.prev_close_pos >= p["close_pos"] and row.first_ret < -p["counter_pct"]:
        return -1, row.prev_close_pos + abs(row.first_ret)
    if row.prev_close_pos <= 1 - p["close_pos"] and row.first_ret > p["counter_pct"]:
        return 1, (1 - row.prev_close_pos) + row.first_ret
    return 0, 0.0


def r_range_expansion_momo(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.range_ratio < p["range_ratio"] or abs(row.first_ret) < p["move_pct"]:
        return 0, 0.0
    return _sign(row.first_ret), row.range_ratio + abs(row.first_ret)


def r_range_expansion_fade(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.range_ratio < p["range_ratio"]:
        return 0, 0.0
    if row.close_location >= p["close_loc"]:
        return -1, row.range_ratio + row.close_location
    if row.close_location <= 1 - p["close_loc"]:
        return 1, row.range_ratio + (1 - row.close_location)
    return 0, 0.0


def r_prior5_reversal(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.prev5_ret > p["trend_pct"] and row.first_ret < -p["counter_pct"]:
        return -1, row.prev5_ret + abs(row.first_ret)
    if row.prev5_ret < -p["trend_pct"] and row.first_ret > p["counter_pct"]:
        return 1, abs(row.prev5_ret) + row.first_ret
    return 0, 0.0


def r_20day_breakout_follow(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.decision_close > row.high20 and row.first_ret > p["move_pct"]:
        return 1, row.first_ret
    if row.decision_close < row.low20 and row.first_ret < -p["move_pct"]:
        return -1, abs(row.first_ret)
    return 0, 0.0


def r_20day_breakout_fade(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.decision_close > row.high20 and row.close_location < p["reject_loc"]:
        return -1, 1 - row.close_location
    if row.decision_close < row.low20 and row.close_location > 1 - p["reject_loc"]:
        return 1, row.close_location
    return 0, 0.0


RECIPES: list[XSRecipe] = [
    XSRecipe("xs_gap_extension", "Largest gap that continues in the first hour.", {"decision_bar": 12, "gap_pct": 0.6, "move_pct": 0.25, "stop_pct": 1.25}, {"gap_pct": [0.4, 0.8], "move_pct": [0.2, 0.4]}, r_gap_extension),
    XSRecipe("xs_gap_fade_relative", "Largest gap that rejects during the first hour.", {"decision_bar": 12, "gap_pct": 0.6, "fade_pct": 0.25, "stop_pct": 1.25}, {"gap_pct": [0.4, 0.8], "fade_pct": [0.2, 0.4]}, r_gap_fade_relative),
    XSRecipe("xs_open_drive_efficiency", "Most efficient high-volume opening drive.", {"decision_bar": 12, "efficiency": 0.65, "move_pct": 0.35, "volume_ratio": 0.12, "stop_pct": 1.25}, {"efficiency": [0.55, 0.75], "move_pct": [0.25, 0.5], "volume_ratio": [0.08, 0.16]}, r_open_drive_efficiency),
    XSRecipe("xs_open_climax_fade", "Fade the most stretched high-volume opening range.", {"decision_bar": 12, "range_ratio": 0.30, "volume_ratio": 0.12, "upper_loc": 0.75, "stop_pct": 1.25}, {"range_ratio": [0.25, 0.4], "upper_loc": [0.70, 0.82]}, r_open_climax_fade),
    XSRecipe("xs_vwap_stretch_fade", "Fade the largest first-hour distance from VWAP.", {"decision_bar": 12, "stretch_pct": 0.65, "stop_pct": 1.25}, {"stretch_pct": [0.45, 0.75, 1.0]}, r_vwap_stretch_fade),
    XSRecipe("xs_vwap_bandwalk_continue", "Follow the strongest first-hour VWAP band-walk.", {"decision_bar": 18, "side_frac": 0.80, "min_dist": 0.20, "stop_pct": 1.25}, {"side_frac": [0.70, 0.85], "min_dist": [0.1, 0.3], "decision_bar": [12, 18]}, r_vwap_bandwalk_continue),
    XSRecipe("xs_prev_winner_continue", "Follow yesterday's strongest directional carry with first-hour confirmation.", {"decision_bar": 12, "prev_ret": 1.5, "confirm_pct": 0.25, "stop_pct": 1.25}, {"prev_ret": [1.0, 2.0], "confirm_pct": [0.2, 0.4]}, r_prev_winner_continue),
    XSRecipe("xs_prev_winner_reversal", "Fade yesterday's strongest mover if first hour reverses.", {"decision_bar": 12, "prev_ret": 1.5, "counter_pct": 0.25, "stop_pct": 1.25}, {"prev_ret": [1.0, 2.0], "counter_pct": [0.2, 0.4]}, r_prev_winner_reversal),
    XSRecipe("xs_close_location_follow", "Follow prior close-location strength with first-hour confirmation.", {"decision_bar": 12, "close_pos": 0.75, "confirm_pct": 0.25, "stop_pct": 1.25}, {"close_pos": [0.70, 0.82], "confirm_pct": [0.2, 0.4]}, r_close_location_follow),
    XSRecipe("xs_close_location_fade", "Fade prior close-location exhaustion with first-hour reversal.", {"decision_bar": 12, "close_pos": 0.75, "counter_pct": 0.25, "stop_pct": 1.25}, {"close_pos": [0.70, 0.82], "counter_pct": [0.2, 0.4]}, r_close_location_fade),
    XSRecipe("xs_range_expansion_momo", "Follow the largest early range expansion with directional move.", {"decision_bar": 12, "range_ratio": 0.25, "move_pct": 0.30, "stop_pct": 1.25}, {"range_ratio": [0.20, 0.35], "move_pct": [0.2, 0.45]}, r_range_expansion_momo),
    XSRecipe("xs_range_expansion_fade", "Fade the most extreme early range expansion close-location.", {"decision_bar": 12, "range_ratio": 0.30, "close_loc": 0.78, "stop_pct": 1.25}, {"range_ratio": [0.25, 0.40], "close_loc": [0.72, 0.84]}, r_range_expansion_fade),
    XSRecipe("xs_prior5_reversal", "Fade a five-day move when the first hour counters it.", {"decision_bar": 12, "trend_pct": 3.0, "counter_pct": 0.35, "stop_pct": 1.25}, {"trend_pct": [2.0, 3.5], "counter_pct": [0.25, 0.50]}, r_prior5_reversal),
    XSRecipe("xs_20day_breakout_follow", "Follow the strongest first-hour 20-day breakout.", {"decision_bar": 18, "move_pct": 0.35, "stop_pct": 1.25}, {"decision_bar": [12, 18], "move_pct": [0.25, 0.50]}, r_20day_breakout_follow),
    XSRecipe("xs_20day_breakout_fade", "Fade failed first-hour 20-day breakout attempts.", {"decision_bar": 18, "reject_loc": 0.45, "stop_pct": 1.25}, {"decision_bar": [12, 18], "reject_loc": [0.40, 0.55]}, r_20day_breakout_fade),
]


def walk_forward(data: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    settings = BacktestSettings()
    splits = walk_forward_splits(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "walk_forward_leaderboard.csv"
    fold_metrics_path = output_dir / "fold_metrics.csv"
    all_trades_path = output_dir / "all_trades.csv"
    leaderboard_rows: list[dict[str, Any]] = pd.read_csv(leaderboard_path).to_dict("records") if leaderboard_path.exists() else []
    all_fold_rows: list[dict[str, Any]] = pd.read_csv(fold_metrics_path).to_dict("records") if fold_metrics_path.exists() else []
    all_trades: list[pd.DataFrame] = [pd.read_csv(all_trades_path)] if all_trades_path.exists() else []
    completed = {str(row["strategy"]) for row in leaderboard_rows}
    for recipe in RECIPES:
        if recipe.name in completed:
            print(f"skipping completed {recipe.name}", flush=True)
            continue
        fold_trades: list[pd.DataFrame] = []
        for fold, train_dates, test_dates in splits:
            best_params, _, is_metrics, best_score, trials = optimize_recipe(data, recipe, set(train_dates), settings)
            oos_trades = run_recipe(data, recipe, best_params, set(test_dates), settings)
            oos_metrics = calculate_metrics(oos_trades, settings)
            all_fold_rows.append({
                "fold": fold,
                "strategy": recipe.name,
                "best_score": best_score,
                "best_params": json.dumps(best_params, sort_keys=True),
                **{f"is_{k}": v for k, v in is_metrics.items()},
                **{f"oos_{k}": v for k, v in oos_metrics.items()},
            })
            trials.to_csv(output_dir / f"{recipe.name}_fold{fold}_trials.csv", index=False)
            if not oos_trades.empty:
                oos_trades = oos_trades.copy()
                oos_trades.insert(0, "fold", fold)
                fold_trades.append(oos_trades)
        trades = pd.concat(fold_trades, ignore_index=True) if fold_trades else pd.DataFrame()
        metrics = calculate_metrics(trades, settings)
        profitable_folds = sum(row["strategy"] == recipe.name and row["oos_net_pnl"] > 0 for row in all_fold_rows)
        leaderboard_rows.append({"strategy": recipe.name, **metrics, "fold_count": len(splits), "profitable_fold_count": profitable_folds})
        if not trades.empty:
            all_trades.append(trades)
        pd.DataFrame(leaderboard_rows).sort_values(["net_pnl", "profit_factor"], ascending=[False, False]).to_csv(leaderboard_path, index=False)
        pd.DataFrame(all_fold_rows).to_csv(fold_metrics_path, index=False)
        if all_trades:
            pd.concat(all_trades, ignore_index=True).to_csv(all_trades_path, index=False)
        print(f"completed {recipe.name}", flush=True)
    return pd.DataFrame(leaderboard_rows).sort_values(["net_pnl", "profit_factor"], ascending=[False, False]).reset_index(drop=True)


def most_common_params(fold_metrics: pd.DataFrame, strategy: str) -> dict[str, Any]:
    values = fold_metrics.loc[fold_metrics["strategy"] == strategy, "best_params"].dropna().astype(str)
    if values.empty:
        raise ValueError(f"No fold params for {strategy}")
    return json.loads(Counter(values).most_common(1)[0][0])


def holdout(candidates: list[str], output_dir: Path) -> dict[str, Any]:
    settings = BacktestSettings()
    data = load_holdout_data()
    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    recipes = {recipe.name: recipe for recipe in RECIPES}
    out: dict[str, Any] = {}
    print("\nFrozen holdout")
    print(f"{'strategy':<30} {'n':>5} {'pf':>6} {'net':>10} verdict")
    for strategy in candidates:
        recipe = recipes[strategy]
        params = most_common_params(fold_metrics, strategy)
        trades = run_recipe(data, recipe, params, set(available_dates(data)), settings)
        metrics = calculate_metrics(trades, settings)
        yearly: dict[int, dict[str, float | int]] = {}
        if not trades.empty:
            trades = trades.copy()
            trades["year"] = pd.to_datetime(trades["exit_time"]).dt.year
            for year, group in trades.groupby("year"):
                year_metrics = calculate_metrics(group.drop(columns="year"), settings)
                yearly[int(year)] = {"n": year_metrics["trade_count"], "pf": round(float(year_metrics["profit_factor"]), 4), "net": round(float(year_metrics["net_pnl"]), 2)}
        keep = metrics["profit_factor"] >= 1.1 and metrics["trade_count"] >= 50 and len(yearly) == 2 and all(row["net"] > 0 for row in yearly.values())
        out[strategy] = {"params": params, "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}, "yearly": yearly, "keep": keep}
        print(f"{strategy:<30} {metrics['trade_count']:>5} {metrics['profit_factor']:>6.2f} {metrics['net_pnl']:>10.0f} {'KEEP' if keep else 'reject'}")
    (output_dir / "holdout_candidates.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cross-sectional intraday research")
    parser.add_argument("--output-dir", type=Path, default=LAB / "results" / "cross_sectional_b5")
    args = parser.parse_args()
    data = load_current_data()
    leaderboard = walk_forward(data, args.output_dir)
    candidates = leaderboard[(leaderboard["net_pnl"] > 0) & (leaderboard["trade_count"] >= 50) & (leaderboard["profit_factor"] >= 1.05)]["strategy"].tolist()
    if candidates:
        holdout(candidates, args.output_dir)
    else:
        (args.output_dir / "holdout_candidates.json").write_text("{}", encoding="utf-8")
        print("\nNo cross-sectional strategy passed the walk-forward pre-filter for holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
