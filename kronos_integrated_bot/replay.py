"""Deterministic replay of stored candle data under a given strategy.

Walks each symbol's 3-minute bars for one or more dates and simulates the
bot's FILTER and EXIT pipeline: prefilter gates, direction rule, reversal
veto, R:R check, cooldowns, daily caps, SL/TP/trailing/time/reversal exits.

LIMITATION (by design): the DeepSeek and Kronos layers are NOT replayed —
entries here are filter-level decisions. That makes results comparable
across strategy parameter sets (which is what the reflection agent tunes:
filter and exit parameters), but absolute PnL will differ from live trading.

Candles come from the data store written by DhanStockTradingBot when
data_store_dir is set (kronos_integrated_bot/data/<date>/<sid>_3minute.csv).

Usage:
    python -m kronos_integrated_bot.replay --dates 2026-06-11 2026-06-12
    python -m kronos_integrated_bot.replay --dates 2026-06-11 \
        --strategy kronos_strategy.yaml --compare state/history/strategy_vkronos-v12.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import time as dtime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constant import (FNO_UNIVERSE, ETF_LIQUID, FILTERED_FNO_UNIVERSE,
                      NIFTY50_UNIVERSE)
from dhan_integration import VWAP_RECLAIM_STOCKS
from indicators import calculate_technical_indicators
from reversal_detector import detect_reversals
from kronos_integrated_bot import config as cfg
from kronos_integrated_bot.reflect import compute_metrics

logger = logging.getLogger("replay")

SECURITY_IDS = {**FNO_UNIVERSE, **ETF_LIQUID, **FILTERED_FNO_UNIVERSE,
                **VWAP_RECLAIM_STOCKS, **NIFTY50_UNIVERSE}
SID_TO_SYMBOL = {str(sid): sym for sym, sid in SECURITY_IDS.items()}

MIN_BARS = 20            # bars needed before the first decision
ENTRY_START = dtime(9, 30)
MARKET_CLOSE = dtime(15, 30)


def load_day_candles(date_str: str, data_dir=None) -> dict[str, pd.DataFrame]:
    """Load 3-minute candles for one date from the data store: symbol -> df."""
    data_dir = Path(data_dir) if data_dir else cfg.DATA_DIR
    day_dir = data_dir / date_str
    out = {}
    if not day_dir.is_dir():
        return out
    for f in day_dir.glob("*_3minute.csv"):
        sid = f.name.split("_")[0]
        symbol = SID_TO_SYMBOL.get(sid)
        if not symbol:
            continue
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            if len(df) >= MIN_BARS:
                out[symbol] = df.sort_index()
        except Exception as e:
            logger.warning("Skipping %s: %s", f, e)
    return out


def passes_prefilter(ind: dict, p: dict) -> bool:
    """Mirror of the bot's hard pre-AI gates, driven by strategy params."""
    adx = ind.get("adx", 20)
    vol = ind.get("volume_ratio", 1.0)
    atr = ind.get("atr", 0)
    close = ind.get("close", 0)
    rsi = ind.get("rsi", 50)
    atr_pct = (atr / close * 100) if close > 0 else 0

    if adx < p.get("min_adx_trending", 18):
        return False
    if vol < p.get("min_prefilter_volume_ratio", 0.15):
        return False
    if atr_pct < p.get("min_prefilter_atr_pct", 0.30):
        return False
    if rsi >= 78 or rsi <= 22:  # extreme-RSI hard gate (matches bot)
        return False
    return True


def entry_direction(ind: dict, p: dict) -> str | None:
    """Filter-level direction proxy: VWAP + SMA20 alignment with RSI gates."""
    close = ind.get("close", 0)
    vwap = ind.get("vwap", close)
    sma = ind.get("sma_20", close)
    rsi = ind.get("rsi", 50)
    vol = ind.get("volume_ratio", 1.0)

    if vol < p.get("min_volume_ratio_trending", 0.3):
        return None
    if close > vwap and close > sma and rsi < p.get("rsi_ob_limit", 70):
        return "BUY"
    if close < vwap and close < sma and rsi > p.get("rsi_os_limit", 30):
        return "SELL"
    return None


def simulate_day(symbol: str, df: pd.DataFrame, p: dict, state: dict) -> list[dict]:
    """Walk one symbol's bars; return closed simulated trades."""
    trades = []
    position = None
    last_entry_ts = None
    entries_today = 0
    close_exit_min = int(p.get("market_close_exit_minutes", 15))
    max_dur_min = int(p.get("max_trade_duration_minutes", 180))
    trail_act = float(p.get("trailing_sl_activation_pct", 3.0))
    trail_atr = float(p.get("trailing_sl_distance_atr", 2.0))
    cooldown = float(p.get("cooldown_seconds", 1800))
    eod_min = 15 * 60 + 30 - close_exit_min  # minutes-from-midnight of EOD exit
    eod_exit_t = dtime(eod_min // 60, eod_min % 60)

    for i in range(MIN_BARS, len(df)):
        bar = df.iloc[i]
        ts = df.index[i]
        window = df.iloc[: i + 1]

        # ── Manage open position ─────────────────────────────────────────────
        if position is not None:
            is_buy = position["direction"] == "BUY"
            held_min = (ts - position["entry_ts"]).total_seconds() / 60
            hit_sl = bar["low"] <= position["sl"] if is_buy else bar["high"] >= position["sl"]
            hit_tp = bar["high"] >= position["tp"] if is_buy else bar["low"] <= position["tp"]
            exit_reason = None
            exit_price = None

            if hit_sl:
                exit_reason, exit_price = "SL", position["sl"]
            elif hit_tp:
                exit_reason, exit_price = "TP", position["tp"]
            elif held_min >= max_dur_min:
                exit_reason, exit_price = "TIME", float(bar["close"])
            elif ts.time() >= eod_exit_t:
                exit_reason, exit_price = "EOD", float(bar["close"])
            else:
                ind = calculate_technical_indicators(window, MIN_BARS)
                if ind:
                    rev = detect_reversals(window.copy(), is_buy=is_buy, indicators=ind)
                    if rev.score >= 75:
                        exit_reason, exit_price = "REVERSAL", float(bar["close"])
                    # Trailing SL
                    pnl_pct = ((bar["close"] - position["entry"]) / position["entry"] * 100) * (1 if is_buy else -1)
                    if exit_reason is None and pnl_pct >= trail_act:
                        atr = ind.get("atr", 1)
                        new_sl = bar["close"] - trail_atr * atr if is_buy else bar["close"] + trail_atr * atr
                        if (is_buy and new_sl > position["sl"]) or (not is_buy and new_sl < position["sl"]):
                            position["sl"] = new_sl

            if exit_reason:
                direction_mult = 1 if is_buy else -1
                pnl = (exit_price - position["entry"]) * direction_mult
                trades.append({
                    "timestamp": str(position["entry_ts"]),
                    "symbol": symbol,
                    "direction": position["direction"],
                    "entry": position["entry"],
                    "exit": round(float(exit_price), 2),
                    "pnl": round(float(pnl), 2),
                    "exit_reason": exit_reason,
                    "confidence": 0,
                    "market_regime": "",
                })
                position = None
            continue

        # ── Consider a new entry ─────────────────────────────────────────────
        if ts.time() < ENTRY_START:
            continue
        if close_exit_min and ts.time() >= dtime(15, 0):
            continue  # no fresh entries in the last stretch
        if entries_today >= int(p.get("max_signals_per_stock_per_day", 1)):
            continue
        if state["total_entries"] >= int(p.get("max_daily_signals", 10)):
            continue
        if last_entry_ts is not None and (ts - last_entry_ts).total_seconds() < cooldown:
            continue

        ind = calculate_technical_indicators(window, MIN_BARS)
        if not ind or not passes_prefilter(ind, p):
            continue
        direction = entry_direction(ind, p)
        if direction is None:
            continue

        # Reversal veto on entry (same threshold as live bot)
        rev = detect_reversals(window.copy(), is_buy=(direction == "BUY"), indicators=ind)
        if rev.score >= 40:
            continue

        atr = ind.get("atr", 1)
        close = float(bar["close"])
        atr_pct = atr / close * 100 if close > 0 else 0
        sl_pct = round(atr_pct * 1.5, 2)
        tp_pct = round(atr_pct * 3.0, 2)
        if sl_pct <= 0 or tp_pct / sl_pct < float(p.get("min_rr_ratio", 1.8)):
            continue

        mult = 1 if direction == "BUY" else -1
        position = {
            "direction": direction,
            "entry": close,
            "entry_ts": ts,
            "sl": close * (1 - mult * sl_pct / 100),
            "tp": close * (1 + mult * tp_pct / 100),
        }
        last_entry_ts = ts
        entries_today += 1
        state["total_entries"] += 1

    # Close any position still open at end of data
    if position is not None:
        last_close = float(df["close"].iloc[-1])
        mult = 1 if position["direction"] == "BUY" else -1
        trades.append({
            "timestamp": str(position["entry_ts"]),
            "symbol": symbol,
            "direction": position["direction"],
            "entry": position["entry"],
            "exit": round(last_close, 2),
            "pnl": round((last_close - position["entry"]) * mult, 2),
            "exit_reason": "EOD-FORCED",
            "confidence": 0,
            "market_regime": "",
        })
    return trades


def run_replay(dates: list[str], params: dict, data_dir=None) -> tuple[dict, list[dict]]:
    """Replay the given dates under params. Returns (metrics, trades)."""
    all_trades = []
    for date_str in dates:
        candles = load_day_candles(date_str, data_dir)
        if not candles:
            logger.warning("No stored candles for %s — run the bot with the "
                           "data store enabled to collect them.", date_str)
            continue
        state = {"total_entries": 0}
        for symbol, df in sorted(candles.items()):
            all_trades.extend(simulate_day(symbol, df, params, state))
    return compute_metrics(all_trades), all_trades


def replay_compare(dates: list[str], params_a: dict, params_b: dict,
                   data_dir=None) -> tuple[dict, dict]:
    """Replay both parameter sets over the same dates."""
    metrics_a, _ = run_replay(dates, params_a, data_dir)
    metrics_b, _ = run_replay(dates, params_b, data_dir)
    return metrics_a, metrics_b


def _load_params(path: str) -> dict:
    with open(path) as f:
        strategy = yaml.safe_load(f) or {}
    return strategy.get("params", strategy)


def main():
    parser = argparse.ArgumentParser(description="Replay stored candles under a strategy.")
    parser.add_argument("--dates", nargs="+", required=True, help="Dates (YYYY-MM-DD)")
    parser.add_argument("--strategy", default=str(cfg.STRATEGY_FILE),
                        help="Strategy YAML (default: current kronos_strategy.yaml)")
    parser.add_argument("--compare", default=None,
                        help="Second strategy YAML to compare against")
    parser.add_argument("--data-dir", default=None, help="Override candle store dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    params = _load_params(args.strategy)
    metrics, trades = run_replay(args.dates, params, args.data_dir)
    print(f"\n=== {os.path.basename(args.strategy)} over {args.dates} "
          f"(filter/exit-level replay, no LLM) ===")
    print(json.dumps(metrics, indent=2))
    by_reason: dict[str, int] = {}
    for t in trades:
        by_reason[t["exit_reason"]] = by_reason.get(t["exit_reason"], 0) + 1
    print("Exits by reason:", by_reason)

    if args.compare:
        params_b = _load_params(args.compare)
        metrics_b, _ = run_replay(args.dates, params_b, args.data_dir)
        print(f"\n=== {os.path.basename(args.compare)} ===")
        print(json.dumps(metrics_b, indent=2))
        delta_pnl = metrics["total_pnl"] - metrics_b["total_pnl"]
        print(f"\nDelta ({os.path.basename(args.strategy)} - {os.path.basename(args.compare)}): "
              f"pnl={delta_pnl:+.2f}, win_rate={metrics['win_rate'] - metrics_b['win_rate']:+.2f}pp")


if __name__ == "__main__":
    main()
