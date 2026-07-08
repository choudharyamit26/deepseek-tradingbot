"""NIFTY/sector lead-lag intraday research.

This experiment targets a different edge source than the earlier OHLCV pattern
batches: NIFTY and sector basket movement leading high-beta stock movement. Each
recipe selects at most one stock per day, enters on the next 5-minute bar after a
decision bar, and exits intraday at stop or square-off.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))

from intraday_strategy_lab.backtest import BacktestSettings, calculate_metrics
from intraday_strategy_lab.data.io import read_ohlcv_csv
from intraday_strategy_lab.indicators import intraday_vwap, parse_time


LeadLagRule = Callable[[pd.Series, dict[str, Any]], tuple[int, float]]


@dataclass(frozen=True)
class LeadLagRecipe:
    name: str
    description: str
    default_params: dict[str, Any]
    grid: dict[str, list[Any]]
    rule: LeadLagRule


SECTOR_BY_SYMBOL = {
    "SHRIRAMFIN": "finance",
    "CHOLAFIN": "finance",
    "PAYTM": "finance",
    "BAJFINANCE": "finance",
    "BANDHANBNK": "finance",
    "INDUSINDBK": "finance",
    "MUTHOOTFIN": "finance",
    "CANBK": "finance",
    "PNB": "finance",
    "BANKBARODA": "finance",
    "ADANIENT": "infra",
    "ADANIPORTS": "infra",
    "LT": "infra",
    "BHEL": "infra",
    "INDIGO": "consumer",
    "DIXON": "consumer",
    "M&M": "consumer",
    "TRENT": "consumer",
    "BPCL": "energy",
    "IDEA": "telecom",
}


def _dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["timestamp"]).dt.date


def _param_grid(grid: dict[str, list[Any]], max_evals: int = 4) -> list[dict[str, Any]]:
    keys = list(grid)
    combos = [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]
    return combos[:max_evals] if max_evals > 0 else combos


def _normalise_parquet(path: Path, symbol: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).reset_index().rename(columns={"ts": "timestamp"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["symbol"] = symbol
    return frame[["timestamp", "open", "high", "low", "close", "volume", "symbol"]].sort_values("timestamp").reset_index(drop=True)


def load_current_stock_data() -> dict[str, pd.DataFrame]:
    processed = LAB / "dhan_historical_data" / "processed"
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(processed.glob("*_intraday.csv")):
        symbol = path.name.replace("_intraday.csv", "")
        data[symbol] = read_ohlcv_csv(path)
    if not data:
        raise FileNotFoundError(f"No current stock intraday files found in {processed}")
    return data


def load_current_nifty() -> pd.DataFrame:
    return _normalise_parquet(LAB.parent / "intraday_lab" / "data" / "store" / "NIFTY_5min.parquet", "NIFTY")


def load_holdout_stock_data() -> dict[str, pd.DataFrame]:
    store = LAB.parent / "intraday_lab" / "data" / "store"
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(store.glob("*_5min_holdout.parquet")):
        symbol = path.name.replace("_5min_holdout.parquet", "")
        if symbol == "NIFTY":
            continue
        data[symbol] = _normalise_parquet(path, symbol)
    if not data:
        raise FileNotFoundError(f"No holdout stock parquets found in {store}")
    return data


def load_holdout_nifty() -> pd.DataFrame:
    return _normalise_parquet(LAB.parent / "intraday_lab" / "data" / "store" / "NIFTY_5min_holdout.parquet", "NIFTY")


def available_dates(data: dict[str, pd.DataFrame]) -> list[object]:
    dates: set[object] = set()
    for frame in data.values():
        dates.update(_dates(frame).unique().tolist())
    return sorted(dates)


def walk_forward_splits(
    data: dict[str, pd.DataFrame], train_sessions: int = 120, test_sessions: int = 20, step_sessions: int = 20, max_folds: int = 5
) -> list[tuple[int, list[object], list[object]]]:
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
        raise ValueError("Not enough sessions for lead-lag walk-forward split")
    return splits


def _day_context(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby(_dates(df), sort=False).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
    )
    daily["ret"] = (daily["close"] / daily["open"] - 1) * 100
    daily["range"] = daily["high"] - daily["low"]
    daily["range_med20"] = daily["range"].rolling(20).median().shift(1)
    daily["prev5_ret"] = daily["ret"].rolling(5).sum().shift(1)
    return daily


def _decision_features_for_group(group: pd.DataFrame, decision_bar: int, settings: BacktestSettings) -> dict[str, Any] | None:
    if len(group) <= decision_bar + 1:
        return None
    squareoff = parse_time(settings.squareoff_time)
    squareoff_positions = group.index[pd.to_datetime(group["timestamp"]).dt.time <= squareoff]
    if len(squareoff_positions) == 0:
        return None
    entry_pos = decision_bar + 1
    squareoff_pos = int(squareoff_positions[-1])
    if entry_pos > squareoff_pos:
        return None
    hist = group.iloc[: decision_bar + 1]
    decision = hist.iloc[-1]
    entry = group.iloc[entry_pos]
    exit_slice = group.iloc[entry_pos : squareoff_pos + 1]
    vwap = intraday_vwap(hist).iloc[-1]
    recent_start = max(0, len(hist) - 4)
    recent_ret = (float(hist.iloc[-1]["close"]) / float(hist.iloc[recent_start]["close"]) - 1) * 100 if float(hist.iloc[recent_start]["close"]) else np.nan
    day_open = float(group.iloc[0]["open"])
    high_so_far = float(hist["high"].max())
    low_so_far = float(hist["low"].min())
    range_so_far = high_so_far - low_so_far
    close = float(decision["close"])
    return {
        "decision_time": decision["timestamp"],
        "entry_time": entry["timestamp"],
        "entry_open": float(entry["open"]),
        "exit_time": exit_slice.iloc[-1]["timestamp"],
        "exit_close": float(exit_slice.iloc[-1]["close"]),
        "high_after_entry": float(exit_slice["high"].max()),
        "low_after_entry": float(exit_slice["low"].min()),
        "day_open": day_open,
        "decision_close": close,
        "ret": (close / day_open - 1) * 100 if day_open else np.nan,
        "recent_ret": recent_ret,
        "vwap_dist": (close / float(vwap) - 1) * 100 if float(vwap) else np.nan,
        "above_vwap_frac": float((hist["close"] > intraday_vwap(hist)).mean()),
        "range_so_far": range_so_far,
        "close_location": (close - low_so_far) / range_so_far if range_so_far else np.nan,
    }


def make_feature_table(
    data: dict[str, pd.DataFrame], nifty: pd.DataFrame, decision_bar: int, settings: BacktestSettings, selected_dates: set[object] | None = None
) -> pd.DataFrame:
    nifty_by_day: dict[object, dict[str, Any]] = {}
    for day, group in nifty.groupby(_dates(nifty), sort=False):
        if selected_dates is not None and day not in selected_dates:
            continue
        features = _decision_features_for_group(group.reset_index(drop=True), decision_bar, settings)
        if features:
            nifty_by_day[day] = features
    rows: list[dict[str, Any]] = []
    for symbol, frame in data.items():
        working = frame.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"])
        working = working.sort_values("timestamp").reset_index(drop=True)
        context = _day_context(working)
        for day, group in working.groupby(_dates(working), sort=False):
            if selected_dates is not None and day not in selected_dates:
                continue
            if day not in nifty_by_day or day not in context.index:
                continue
            features = _decision_features_for_group(group.reset_index(drop=True), decision_bar, settings)
            if not features:
                continue
            ctx = context.loc[day]
            nifty_features = nifty_by_day[day]
            sector = SECTOR_BY_SYMBOL.get(symbol, "other")
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "sector": sector,
                    **features,
                    "stock_ret": features["ret"],
                    "stock_recent_ret": features["recent_ret"],
                    "stock_vwap_dist": features["vwap_dist"],
                    "stock_above_vwap_frac": features["above_vwap_frac"],
                    "stock_close_location": features["close_location"],
                    "stock_range_ratio": features["range_so_far"] / float(ctx["range_med20"]) if float(ctx.get("range_med20", 0) or 0) else np.nan,
                    "stock_prev5_ret": float(ctx.get("prev5_ret", np.nan)),
                    "nifty_ret": nifty_features["ret"],
                    "nifty_recent_ret": nifty_features["recent_ret"],
                    "nifty_vwap_dist": nifty_features["vwap_dist"],
                    "nifty_above_vwap_frac": nifty_features["above_vwap_frac"],
                }
            )
    features = pd.DataFrame(rows)
    if features.empty:
        return features
    features["sector_ret"] = features.groupby(["date", "sector"])["stock_ret"].transform("mean")
    features["sector_recent_ret"] = features.groupby(["date", "sector"])["stock_recent_ret"].transform("mean")
    features["market_ret"] = features.groupby("date")["stock_ret"].transform("mean")
    features["lag_to_nifty"] = features["nifty_ret"] - features["stock_ret"]
    features["lag_to_sector"] = features["sector_ret"] - features["stock_ret"]
    features["rel_to_nifty"] = features["stock_ret"] - features["nifty_ret"]
    features["rel_to_sector"] = features["stock_ret"] - features["sector_ret"]
    return features


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
    stop_pct = float(params.get("stop_pct", 1.25)) / 100
    stop_price = entry_price * (1 - stop_pct if direction == 1 else 1 + stop_pct)
    hit_stop = float(selected["low_after_entry"]) <= stop_price if direction == 1 else float(selected["high_after_entry"]) >= stop_price
    raw_exit = stop_price if hit_stop else float(selected["exit_close"])
    exit_price = _exit_fill(raw_exit, direction, settings)
    gross_pnl = (exit_price - entry_price) * quantity * direction
    turnover = (entry_price + exit_price) * quantity
    costs = turnover * settings.cost_bps / 10_000
    net_pnl = gross_pnl - costs
    return {
        "strategy": recipe,
        "symbol": selected["symbol"],
        "direction": direction,
        "entry_time": selected["entry_time"],
        "exit_time": selected["exit_time"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": quantity,
        "gross_pnl": gross_pnl,
        "costs": costs,
        "net_pnl": net_pnl,
        "return_pct": net_pnl / settings.capital_per_trade * 100,
        "exit_reason": "stop" if hit_stop else "squareoff",
        "bars_held": 0,
        "params": json.dumps(params, sort_keys=True),
    }


def run_recipe(
    data: dict[str, pd.DataFrame], nifty: pd.DataFrame, recipe: LeadLagRecipe, params: dict[str, Any], dates: set[object], settings: BacktestSettings
) -> pd.DataFrame:
    decision_bar = int(params.get("decision_bar", 12))
    features = make_feature_table(data, nifty, decision_bar, settings, selected_dates=dates)
    if features.empty:
        return pd.DataFrame()
    trades: list[dict[str, Any]] = []
    for _, day_rows in features.groupby("date", sort=False):
        candidates: list[tuple[float, int, pd.Series]] = []
        for _, row in day_rows.iterrows():
            direction, score = recipe.rule(row, params)
            if direction and np.isfinite(score):
                candidates.append((float(score), int(direction), row))
        if not candidates:
            continue
        _, direction, selected = max(candidates, key=lambda item: item[0])
        trade = _trade_from_selection(recipe.name, selected, direction, params, settings)
        if trade:
            trades.append(trade)
    return pd.DataFrame(trades)


def _score(metrics: dict[str, float | int]) -> float:
    score = float(metrics.get("total_return_pct", 0)) + 2 * float(metrics.get("profit_factor", 0)) + 2 * float(metrics.get("sharpe", 0))
    score -= 1.5 * float(metrics.get("max_drawdown_pct", 0))
    if int(metrics.get("trade_count", 0)) < 20:
        score -= 25
    return score


def optimize_recipe(
    data: dict[str, pd.DataFrame], nifty: pd.DataFrame, recipe: LeadLagRecipe, train_dates: set[object], settings: BacktestSettings, max_evals: int = 4
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, float | int], float, pd.DataFrame]:
    best_params = dict(recipe.default_params)
    best_trades = pd.DataFrame()
    best_metrics = calculate_metrics(best_trades, settings)
    best_score = float("-inf")
    rows: list[dict[str, Any]] = []
    for partial in _param_grid(recipe.grid, max_evals=max_evals):
        params = dict(recipe.default_params)
        params.update(partial)
        trades = run_recipe(data, nifty, recipe, params, train_dates, settings)
        metrics = calculate_metrics(trades, settings)
        score = _score(metrics)
        rows.append({"strategy": recipe.name, "score": score, "params": json.dumps(params, sort_keys=True), **metrics})
        if score > best_score:
            best_params = params
            best_trades = trades
            best_metrics = metrics
            best_score = score
    return best_params, best_trades, best_metrics, best_score, pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def _sgn(value: float) -> int:
    return 1 if value > 0 else -1


def r_nifty_lag_catchup(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.nifty_ret >= p["index_move"] and row.lag_to_nifty >= p["lag_min"] and row.stock_recent_ret >= p["confirm"]:
        return 1, row.nifty_ret + row.lag_to_nifty + row.stock_recent_ret
    if row.nifty_ret <= -p["index_move"] and row.lag_to_nifty <= -p["lag_min"] and row.stock_recent_ret <= -p["confirm"]:
        return -1, abs(row.nifty_ret) + abs(row.lag_to_nifty) + abs(row.stock_recent_ret)
    return 0, 0.0


def r_sector_lag_catchup(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.sector_ret >= p["sector_move"] and row.lag_to_sector >= p["lag_min"] and row.stock_recent_ret >= p["confirm"]:
        return 1, row.sector_ret + row.lag_to_sector
    if row.sector_ret <= -p["sector_move"] and row.lag_to_sector <= -p["lag_min"] and row.stock_recent_ret <= -p["confirm"]:
        return -1, abs(row.sector_ret) + abs(row.lag_to_sector)
    return 0, 0.0


def r_finance_basket_lag(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.sector != "finance":
        return 0, 0.0
    return r_sector_lag_catchup(row, p)


def r_nifty_impulse_pull(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.nifty_recent_ret >= p["impulse"] and row.stock_ret < row.nifty_ret and row.stock_vwap_dist > p["vwap_confirm"]:
        return 1, row.nifty_recent_ret + row.lag_to_nifty
    if row.nifty_recent_ret <= -p["impulse"] and row.stock_ret > row.nifty_ret and row.stock_vwap_dist < -p["vwap_confirm"]:
        return -1, abs(row.nifty_recent_ret) + abs(row.lag_to_nifty)
    return 0, 0.0


def r_nifty_reversal_contagion(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.nifty_ret <= -p["index_move"] and row.nifty_recent_ret >= p["reversal"] and row.stock_recent_ret >= p["confirm"]:
        return 1, abs(row.nifty_ret) + row.nifty_recent_ret
    if row.nifty_ret >= p["index_move"] and row.nifty_recent_ret <= -p["reversal"] and row.stock_recent_ret <= -p["confirm"]:
        return -1, row.nifty_ret + abs(row.nifty_recent_ret)
    return 0, 0.0


def r_sector_reversal_contagion(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.sector_ret <= -p["sector_move"] and row.sector_recent_ret >= p["reversal"] and row.stock_recent_ret >= p["confirm"]:
        return 1, abs(row.sector_ret) + row.sector_recent_ret
    if row.sector_ret >= p["sector_move"] and row.sector_recent_ret <= -p["reversal"] and row.stock_recent_ret <= -p["confirm"]:
        return -1, row.sector_ret + abs(row.sector_recent_ret)
    return 0, 0.0


def r_index_up_best_relative(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.nifty_ret >= p["index_move"] and row.nifty_vwap_dist > 0 and row.rel_to_nifty >= p["rel_min"]:
        return 1, row.rel_to_nifty + row.nifty_ret
    if row.nifty_ret <= -p["index_move"] and row.nifty_vwap_dist < 0 and row.rel_to_nifty <= -p["rel_min"]:
        return -1, abs(row.rel_to_nifty) + abs(row.nifty_ret)
    return 0, 0.0


def r_index_up_laggard_reversal(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.nifty_ret >= p["index_move"] and row.rel_to_nifty <= -p["lag_min"] and row.stock_recent_ret <= -p["counter"]:
        return -1, row.nifty_ret + abs(row.rel_to_nifty)
    if row.nifty_ret <= -p["index_move"] and row.rel_to_nifty >= p["lag_min"] and row.stock_recent_ret >= p["counter"]:
        return 1, abs(row.nifty_ret) + row.rel_to_nifty
    return 0, 0.0


def r_gap_against_index_fade(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    gap_dir = _sgn(row.stock_ret) if abs(row.stock_ret) >= p["stock_move"] else 0
    index_dir = _sgn(row.nifty_ret) if abs(row.nifty_ret) >= p["index_move"] else 0
    if not gap_dir or not index_dir or gap_dir == index_dir:
        return 0, 0.0
    return index_dir, abs(row.stock_ret) + abs(row.nifty_ret)


def r_gap_with_index_follow(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if abs(row.stock_ret) < p["stock_move"] or abs(row.nifty_ret) < p["index_move"] or np.sign(row.stock_ret) != np.sign(row.nifty_ret):
        return 0, 0.0
    return _sgn(row.stock_ret), abs(row.stock_ret) + abs(row.nifty_ret)


def r_sector_dispersion_reversion(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.rel_to_sector <= -p["dispersion"] and row.sector_ret > p["sector_move"] and row.stock_recent_ret > p["confirm"]:
        return 1, abs(row.rel_to_sector) + row.sector_ret
    if row.rel_to_sector >= p["dispersion"] and row.sector_ret < -p["sector_move"] and row.stock_recent_ret < -p["confirm"]:
        return -1, row.rel_to_sector + abs(row.sector_ret)
    return 0, 0.0


def r_market_breadth_lag(row: pd.Series, p: dict[str, Any]) -> tuple[int, float]:
    if row.market_ret >= p["market_move"] and row.stock_ret < row.market_ret - p["lag_min"] and row.stock_recent_ret >= p["confirm"]:
        return 1, row.market_ret - row.stock_ret
    if row.market_ret <= -p["market_move"] and row.stock_ret > row.market_ret + p["lag_min"] and row.stock_recent_ret <= -p["confirm"]:
        return -1, row.stock_ret - row.market_ret
    return 0, 0.0


RECIPES: list[LeadLagRecipe] = [
    LeadLagRecipe("ll_nifty_lag_catchup", "Stock lags a directional NIFTY move, then confirms in NIFTY direction.", {"decision_bar": 12, "index_move": 0.35, "lag_min": 0.35, "confirm": 0.08, "stop_pct": 1.25}, {"index_move": [0.25, 0.45], "lag_min": [0.25, 0.50], "confirm": [0.05, 0.12]}, r_nifty_lag_catchup),
    LeadLagRecipe("ll_sector_lag_catchup", "Stock lags its sector basket, then confirms in sector direction.", {"decision_bar": 12, "sector_move": 0.45, "lag_min": 0.35, "confirm": 0.08, "stop_pct": 1.25}, {"sector_move": [0.30, 0.55], "lag_min": [0.25, 0.50], "confirm": [0.05, 0.12]}, r_sector_lag_catchup),
    LeadLagRecipe("ll_finance_basket_lag", "Finance-only basket lead-lag catch-up.", {"decision_bar": 12, "sector_move": 0.45, "lag_min": 0.35, "confirm": 0.08, "stop_pct": 1.25}, {"sector_move": [0.30, 0.55], "lag_min": [0.25, 0.50], "confirm": [0.05, 0.12]}, r_finance_basket_lag),
    LeadLagRecipe("ll_nifty_impulse_pull", "NIFTY recent impulse pulls a lagging stock after VWAP confirmation.", {"decision_bar": 18, "impulse": 0.18, "vwap_confirm": 0.05, "stop_pct": 1.25}, {"decision_bar": [12, 18], "impulse": [0.12, 0.25], "vwap_confirm": [0.00, 0.08]}, r_nifty_impulse_pull),
    LeadLagRecipe("ll_nifty_reversal_contagion", "NIFTY reverses intraday and stock starts confirming reversal direction.", {"decision_bar": 18, "index_move": 0.35, "reversal": 0.15, "confirm": 0.05, "stop_pct": 1.25}, {"index_move": [0.25, 0.45], "reversal": [0.10, 0.22], "decision_bar": [12, 18]}, r_nifty_reversal_contagion),
    LeadLagRecipe("ll_sector_reversal_contagion", "Sector basket reverses intraday and stock starts confirming reversal direction.", {"decision_bar": 18, "sector_move": 0.45, "reversal": 0.18, "confirm": 0.06, "stop_pct": 1.25}, {"sector_move": [0.35, 0.60], "reversal": [0.12, 0.25], "decision_bar": [12, 18]}, r_sector_reversal_contagion),
    LeadLagRecipe("ll_index_best_relative", "When NIFTY trends, choose the strongest relative stock in the same direction.", {"decision_bar": 12, "index_move": 0.25, "rel_min": 0.20, "stop_pct": 1.25}, {"index_move": [0.20, 0.35], "rel_min": [0.10, 0.30], "decision_bar": [12, 18]}, r_index_up_best_relative),
    LeadLagRecipe("ll_index_laggard_reversal", "When NIFTY trends, fade a stock that diverges hard against it.", {"decision_bar": 12, "index_move": 0.30, "lag_min": 0.45, "counter": 0.10, "stop_pct": 1.25}, {"index_move": [0.25, 0.45], "lag_min": [0.35, 0.60], "counter": [0.08, 0.15]}, r_index_up_laggard_reversal),
    LeadLagRecipe("ll_gap_against_index_fade", "Trade with NIFTY when stock first-hour move diverges from NIFTY.", {"decision_bar": 12, "index_move": 0.25, "stock_move": 0.45, "stop_pct": 1.25}, {"index_move": [0.20, 0.35], "stock_move": [0.35, 0.60]}, r_gap_against_index_fade),
    LeadLagRecipe("ll_gap_with_index_follow", "Follow stocks whose first-hour move aligns with NIFTY.", {"decision_bar": 12, "index_move": 0.25, "stock_move": 0.45, "stop_pct": 1.25}, {"index_move": [0.20, 0.35], "stock_move": [0.35, 0.60]}, r_gap_with_index_follow),
    LeadLagRecipe("ll_sector_dispersion_reversion", "Fade stock-sector dispersion only when stock starts reverting toward the sector.", {"decision_bar": 18, "sector_move": 0.25, "dispersion": 0.55, "confirm": 0.06, "stop_pct": 1.25}, {"sector_move": [0.20, 0.35], "dispersion": [0.45, 0.70], "decision_bar": [12, 18]}, r_sector_dispersion_reversion),
    LeadLagRecipe("ll_market_breadth_lag", "Use equal-weight book movement as lead; trade lagging stock catch-up.", {"decision_bar": 12, "market_move": 0.35, "lag_min": 0.35, "confirm": 0.06, "stop_pct": 1.25}, {"market_move": [0.25, 0.45], "lag_min": [0.25, 0.50], "confirm": [0.04, 0.10]}, r_market_breadth_lag),
]


def walk_forward(data: dict[str, pd.DataFrame], nifty: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    settings = BacktestSettings()
    splits = walk_forward_splits(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "walk_forward_leaderboard.csv"
    fold_path = output_dir / "fold_metrics.csv"
    trades_path = output_dir / "all_trades.csv"
    leaderboard_rows = pd.read_csv(leaderboard_path).to_dict("records") if leaderboard_path.exists() else []
    fold_rows = pd.read_csv(fold_path).to_dict("records") if fold_path.exists() else []
    trade_frames = [pd.read_csv(trades_path)] if trades_path.exists() else []
    completed = {row["strategy"] for row in leaderboard_rows}
    for recipe in RECIPES:
        if recipe.name in completed:
            print(f"skipping completed {recipe.name}", flush=True)
            continue
        recipe_trades: list[pd.DataFrame] = []
        for fold, train_dates, test_dates in splits:
            best_params, _, is_metrics, score, trials = optimize_recipe(data, nifty, recipe, set(train_dates), settings)
            oos_trades = run_recipe(data, nifty, recipe, best_params, set(test_dates), settings)
            oos_metrics = calculate_metrics(oos_trades, settings)
            fold_rows.append({"fold": fold, "strategy": recipe.name, "best_score": score, "best_params": json.dumps(best_params, sort_keys=True), **{f"is_{k}": v for k, v in is_metrics.items()}, **{f"oos_{k}": v for k, v in oos_metrics.items()}})
            trials.to_csv(output_dir / f"{recipe.name}_fold{fold}_trials.csv", index=False)
            if not oos_trades.empty:
                oos_trades = oos_trades.copy()
                oos_trades.insert(0, "fold", fold)
                recipe_trades.append(oos_trades)
        trades = pd.concat(recipe_trades, ignore_index=True) if recipe_trades else pd.DataFrame()
        metrics = calculate_metrics(trades, settings)
        profitable_folds = sum(row["strategy"] == recipe.name and row["oos_net_pnl"] > 0 for row in fold_rows)
        leaderboard_rows.append({"strategy": recipe.name, **metrics, "fold_count": len(splits), "profitable_fold_count": profitable_folds})
        if not trades.empty:
            trade_frames.append(trades)
        pd.DataFrame(leaderboard_rows).sort_values(["net_pnl", "profit_factor"], ascending=[False, False]).to_csv(leaderboard_path, index=False)
        pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
        if trade_frames:
            pd.concat(trade_frames, ignore_index=True).to_csv(trades_path, index=False)
        print(f"completed {recipe.name}", flush=True)
    return pd.DataFrame(leaderboard_rows).sort_values(["net_pnl", "profit_factor"], ascending=[False, False]).reset_index(drop=True)


def _most_common_params(fold_metrics: pd.DataFrame, strategy: str) -> dict[str, Any]:
    values = fold_metrics.loc[fold_metrics["strategy"] == strategy, "best_params"].dropna().astype(str)
    if values.empty:
        raise ValueError(f"No fold params for {strategy}")
    return json.loads(Counter(values).most_common(1)[0][0])


def holdout(candidates: list[str], output_dir: Path) -> dict[str, Any]:
    settings = BacktestSettings()
    data = load_holdout_stock_data()
    nifty = load_holdout_nifty()
    recipes = {recipe.name: recipe for recipe in RECIPES}
    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    out: dict[str, Any] = {}
    print("\nFrozen holdout")
    print(f"{'strategy':<34} {'n':>5} {'pf':>6} {'net':>10} verdict")
    for name in candidates:
        params = _most_common_params(fold_metrics, name)
        trades = run_recipe(data, nifty, recipes[name], params, set(available_dates(data)), settings)
        metrics = calculate_metrics(trades, settings)
        yearly: dict[int, dict[str, float | int]] = {}
        if not trades.empty:
            trades = trades.copy()
            trades["year"] = pd.to_datetime(trades["exit_time"]).dt.year
            for year, group in trades.groupby("year"):
                year_metrics = calculate_metrics(group.drop(columns="year"), settings)
                yearly[int(year)] = {"n": year_metrics["trade_count"], "pf": round(float(year_metrics["profit_factor"]), 4), "net": round(float(year_metrics["net_pnl"]), 2)}
        keep = metrics["profit_factor"] >= 1.1 and metrics["trade_count"] >= 50 and len(yearly) == 2 and all(row["net"] > 0 for row in yearly.values())
        out[name] = {"params": params, "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}, "yearly": yearly, "keep": keep}
        print(f"{name:<34} {metrics['trade_count']:>5} {metrics['profit_factor']:>6.2f} {metrics['net_pnl']:>10.0f} {'KEEP' if keep else 'reject'}")
    (output_dir / "holdout_candidates.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NIFTY/sector lead-lag research")
    parser.add_argument("--output-dir", type=Path, default=LAB / "results" / "lead_lag_b6")
    args = parser.parse_args()
    data = load_current_stock_data()
    nifty = load_current_nifty()
    leaderboard = walk_forward(data, nifty, args.output_dir)
    candidates = leaderboard[(leaderboard["net_pnl"] > 0) & (leaderboard["trade_count"] >= 50) & (leaderboard["profit_factor"] >= 1.05)]["strategy"].tolist()
    if candidates:
        holdout(candidates, args.output_dir)
    else:
        (args.output_dir / "holdout_candidates.json").write_text("{}", encoding="utf-8")
        print("\nNo lead-lag strategy passed the walk-forward pre-filter for holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
