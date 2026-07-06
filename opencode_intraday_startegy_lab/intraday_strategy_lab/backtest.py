from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import time
from typing import Any

import numpy as np
import pandas as pd

from .indicators import ensure_datetime, parse_time
from .strategies import StrategyDefinition


@dataclass(frozen=True)
class BacktestSettings:
    initial_capital: float = 1_000_000
    capital_per_trade: float = 100_000
    cost_bps: float = 5.0
    slippage_bps: float = 2.0
    entry_start_time: str = "09:20"
    last_entry_time: str = "15:05"
    squareoff_time: str = "15:20"
    max_trades_per_day_per_symbol: int = 3
    allow_short: bool = True
    stop_first_if_ambiguous: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BacktestSettings":
        accepted = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in values.items() if key in accepted})


@dataclass
class Trade:
    strategy: str
    symbol: str
    direction: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
    exit_reason: str
    bars_held: int
    params: str


def _time_in_range(value: time, start: time, end: time) -> bool:
    return start <= value <= end


def _entry_fill(open_price: float, direction: int, settings: BacktestSettings) -> float:
    slippage = settings.slippage_bps / 10_000
    return open_price * (1 + slippage if direction == 1 else 1 - slippage)


def _exit_fill(price: float, direction: int, settings: BacktestSettings) -> float:
    slippage = settings.slippage_bps / 10_000
    return price * (1 - slippage if direction == 1 else 1 + slippage)


def _finalise_trade(open_trade: dict[str, Any], exit_time: pd.Timestamp, exit_price: float, reason: str, bars_held: int) -> Trade:
    direction = int(open_trade["direction"])
    quantity = int(open_trade["quantity"])
    entry_price = float(open_trade["entry_price"])
    gross_pnl = (exit_price - entry_price) * quantity * direction
    turnover = (entry_price + exit_price) * quantity
    costs = turnover * float(open_trade["settings"].cost_bps) / 10_000
    net_pnl = gross_pnl - costs
    return Trade(
        strategy=str(open_trade["strategy"]),
        symbol=str(open_trade["symbol"]),
        direction=direction,
        entry_time=open_trade["entry_time"],
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=float(exit_price),
        quantity=quantity,
        gross_pnl=float(gross_pnl),
        costs=float(costs),
        net_pnl=float(net_pnl),
        return_pct=float(net_pnl / float(open_trade["settings"].capital_per_trade) * 100),
        exit_reason=reason,
        bars_held=int(bars_held),
        params=str(open_trade["params"]),
    )


def backtest_strategy(
    df: pd.DataFrame,
    strategy: StrategyDefinition,
    params: dict[str, Any] | None = None,
    settings: BacktestSettings | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    settings = settings or BacktestSettings()
    working = ensure_datetime(df)
    if working.empty:
        return trades_to_frame([])
    symbol = symbol or str(working.get("symbol", pd.Series(["UNKNOWN"])).iloc[0])
    merged_params = dict(strategy.default_params)
    if params:
        merged_params.update(params)
    signals = strategy.generate(working, merged_params)

    stop_pct = float(merged_params.get("stop_pct", 0.5)) / 100
    target_pct = float(merged_params.get("target_pct", 1.0)) / 100
    max_hold_bars = int(merged_params.get("max_hold_bars", 24))
    entry_start = parse_time(settings.entry_start_time)
    last_entry = parse_time(settings.last_entry_time)
    squareoff = parse_time(settings.squareoff_time)
    params_json = json.dumps(merged_params, sort_keys=True)
    trades: list[Trade] = []

    dates = pd.to_datetime(working["timestamp"]).dt.date
    for _, group in working.groupby(dates, sort=False):
        indices = list(group.index)
        open_trade: dict[str, Any] | None = None
        trades_today = 0
        for local_pos, idx in enumerate(indices):
            row = working.loc[idx]
            current_time = pd.to_datetime(row["timestamp"]).time()

            if open_trade is not None and local_pos >= int(open_trade["entry_local_pos"]):
                direction = int(open_trade["direction"])
                stop_price = float(open_trade["stop_price"])
                target_price = float(open_trade["target_price"])
                hit_stop = row["low"] <= stop_price if direction == 1 else row["high"] >= stop_price
                hit_target = row["high"] >= target_price if direction == 1 else row["low"] <= target_price
                exit_reason = ""
                raw_exit_price = float(row["close"])

                if current_time >= squareoff or local_pos == len(indices) - 1:
                    exit_reason = "squareoff"
                    raw_exit_price = float(row["close"])
                elif local_pos - int(open_trade["entry_local_pos"]) >= max_hold_bars:
                    exit_reason = "time_exit"
                    raw_exit_price = float(row["close"])
                elif hit_stop and hit_target:
                    exit_reason = "stop" if settings.stop_first_if_ambiguous else "target"
                    raw_exit_price = stop_price if exit_reason == "stop" else target_price
                elif hit_stop:
                    exit_reason = "stop"
                    raw_exit_price = stop_price
                elif hit_target:
                    exit_reason = "target"
                    raw_exit_price = target_price

                if exit_reason:
                    exit_price = _exit_fill(raw_exit_price, direction, settings)
                    bars_held = local_pos - int(open_trade["entry_local_pos"]) + 1
                    trades.append(_finalise_trade(open_trade, row["timestamp"], exit_price, exit_reason, bars_held))
                    open_trade = None

            if open_trade is not None:
                continue
            if trades_today >= settings.max_trades_per_day_per_symbol or local_pos >= len(indices) - 1:
                continue

            direction = int(signals.loc[idx])
            if direction == 0 or (direction == -1 and not settings.allow_short):
                continue
            next_idx = indices[local_pos + 1]
            next_row = working.loc[next_idx]
            next_time = pd.to_datetime(next_row["timestamp"]).time()
            if not _time_in_range(next_time, entry_start, last_entry):
                continue

            entry_price = _entry_fill(float(next_row["open"]), direction, settings)
            quantity = int(settings.capital_per_trade // entry_price)
            if quantity <= 0:
                continue
            stop_price = entry_price * (1 - stop_pct if direction == 1 else 1 + stop_pct)
            target_price = entry_price * (1 + target_pct if direction == 1 else 1 - target_pct)
            open_trade = {
                "strategy": strategy.name,
                "symbol": symbol,
                "direction": direction,
                "entry_time": next_row["timestamp"],
                "entry_price": entry_price,
                "entry_local_pos": local_pos + 1,
                "quantity": quantity,
                "stop_price": stop_price,
                "target_price": target_price,
                "settings": settings,
                "params": params_json,
            }
            trades_today += 1

    return trades_to_frame(trades)


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    columns = list(Trade.__dataclass_fields__.keys())
    if not trades:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([asdict(trade) for trade in trades])


def calculate_metrics(trades: pd.DataFrame, settings: BacktestSettings | None = None) -> dict[str, float | int]:
    settings = settings or BacktestSettings()
    if trades.empty:
        return {
            "trade_count": 0,
            "net_pnl": 0.0,
            "total_return_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "average_trade": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "average_bars_held": 0.0,
        }

    frame = trades.copy()
    frame["exit_time"] = pd.to_datetime(frame["exit_time"])
    net_pnl = float(frame["net_pnl"].sum())
    winners = frame[frame["net_pnl"] > 0]
    losers = frame[frame["net_pnl"] < 0]
    gross_profit = float(winners["net_pnl"].sum())
    gross_loss = float(losers["net_pnl"].sum())
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else (999.0 if gross_profit > 0 else 0.0)

    ordered = frame.sort_values("exit_time")
    equity = settings.initial_capital + ordered["net_pnl"].cumsum()
    peak = equity.cummax()
    drawdown = ((peak - equity) / peak.replace(0, np.nan) * 100).fillna(0)

    daily_pnl = ordered.groupby(ordered["exit_time"].dt.date)["net_pnl"].sum()
    daily_returns = daily_pnl / settings.initial_capital
    sharpe = 0.0
    sortino = 0.0
    if len(daily_returns) > 1 and daily_returns.std(ddof=0) > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std(ddof=0) * np.sqrt(252))
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1 and downside.std(ddof=0) > 0:
        sortino = float(daily_returns.mean() / downside.std(ddof=0) * np.sqrt(252))

    trade_count = int(len(frame))
    return {
        "trade_count": trade_count,
        "net_pnl": net_pnl,
        "total_return_pct": float(net_pnl / settings.initial_capital * 100),
        "win_rate_pct": float(len(winners) / trade_count * 100),
        "profit_factor": float(profit_factor),
        "expectancy": float(frame["net_pnl"].mean()),
        "average_trade": float(frame["net_pnl"].mean()),
        "max_drawdown_pct": float(drawdown.max()),
        "sharpe": sharpe,
        "sortino": sortino,
        "average_bars_held": float(frame["bars_held"].mean()),
    }


def backtest_portfolio(
    data_by_symbol: dict[str, pd.DataFrame],
    strategy: StrategyDefinition,
    params: dict[str, Any],
    settings: BacktestSettings,
) -> pd.DataFrame:
    frames = [backtest_strategy(frame, strategy, params, settings, symbol) for symbol, frame in data_by_symbol.items()]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else trades_to_frame([])
