"""Momentum bot dry-run replay for 2026-06-22.

Fetches today's historical OHLCV data from Dhan, runs the full opening-range
scan, then replays every 3-min candle from 09:30 to 14:00 to show exactly
which signals would have fired, when time exits would have triggered, and
what the P&L would have been.

Run: python momentum_dryrun_2026_06_22.py
"""

import sys
import time
import logging
from datetime import datetime, timedelta, date as date_t

import pandas as pd
import talib

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,   # suppress Dhan chatter
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dryrun")

# ── Config (copy from momentum_bot.config so we are standalone) ───────────────
BREAKOUT_BUFFER_PCT       = 0.05
ENTRY_VOLUME_MULTIPLIER   = 1.5
RR_RATIO                  = 1.5
MAX_STOP_PCT              = 1.5
MIN_STOP_PCT              = 0.15
TOP_N_SECTORS             = 2
MIN_SECTOR_MOVE_PCT       = 0.20
MIN_SECTOR_STOCKS         = 2
TOP_STOCKS_PER_SECTOR     = 2
MIN_OR_WIDTH_PCT          = 0.15
TIME_EXIT_MINUTES         = 60
ENTRY_END                 = "14:00"

TODAY = date_t(2026, 6, 22)

# New gates added to scanner.py
MIN_SECTOR_DIRECTION_PCT = 0.20   # neutral zone — skip ambiguous direction
REQUIRE_TREND_ALIGNMENT  = True   # OR direction must match 5-day trend


# ── Dhan initialisation ───────────────────────────────────────────────────────
print("Connecting to Dhan …")
from dhan_integration import DhanStockTradingBot
dhan = DhanStockTradingBot()
print(f"Watchlist: {len(dhan.security_ids)} stocks\n")

from momentum_bot.sector_map import SECTORS, SYMBOL_TO_SECTOR


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: Opening Range Scan (09:15 – 09:30 candle)
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("PHASE 1 — Opening Range Scan (first 15-min candle)")
print("=" * 65)

all_or   = []   # list of dicts

for symbol, sid in dhan.security_ids.items():
    sector = SYMBOL_TO_SECTOR.get(symbol)
    if not sector:
        continue

    try:
        df15 = dhan.get_historical_data(str(sid), interval="15minute", min_bars=1)
        time.sleep(0.15)
        if df15 is None or df15.empty:
            continue

        # Filter to today
        if hasattr(df15.index, "date"):
            df15 = df15[df15.index.date == TODAY]
        if df15.empty:
            continue

        row = df15.iloc[0]
        o, h, l, c, v = (
            float(row["open"]), float(row["high"]),
            float(row["low"]),  float(row["close"]), float(row["volume"]),
        )
        if o <= 0 or c <= 0:
            continue

        # Daily bars: vol baseline + 5-day trend (single API call)
        df_day = dhan.get_historical_data(str(sid), interval="1day", min_bars=6)
        time.sleep(0.1)
        avg_daily_vol = 0.0
        five_day_ret  = 0.0
        if df_day is not None and len(df_day) >= 6:
            avg_daily_vol = float(df_day["volume"].tail(10).mean())
            five_day_ret  = (float(df_day["close"].iloc[-1]) - float(df_day["close"].iloc[-6])) \
                            / float(df_day["close"].iloc[-6]) * 100.0
        elif df_day is not None and len(df_day) >= 2:
            avg_daily_vol = float(df_day["volume"].tail(10).mean())

        avg_vol_15 = (avg_daily_vol / 25.0) if avg_daily_vol > 0 else 1.0
        vol_ratio  = v / avg_vol_15

        pct_move = (c - o) / o * 100.0
        score    = abs(pct_move) * vol_ratio

        or_width_pct = (h - l) / c * 100.0
        if or_width_pct < MIN_OR_WIDTH_PCT:
            continue

        all_or.append({
            "symbol": symbol, "sid": str(sid), "sector": sector,
            "or_open": o, "or_high": h, "or_low": l, "or_close": c,
            "or_volume": v, "avg_vol_15": avg_vol_15,
            "pct_move": pct_move, "vol_ratio": vol_ratio, "score": score,
            "or_width_pct": or_width_pct, "five_day_ret": five_day_ret,
        })
    except Exception as exc:
        logger.debug("%s  OR error: %s", symbol, exc)

print(f"\n{len(all_or)} stocks with valid opening ranges\n")

# ── Sector scoring ─────────────────────────────────────────────────────────────
by_sector: dict[str, list] = {}
for r in all_or:
    by_sector.setdefault(r["sector"], []).append(r)

sector_results = []
for sector, records in by_sector.items():
    if len(records) < MIN_SECTOR_STOCKS:
        continue
    avg_move = sum(r["pct_move"] for r in records) / len(records)
    total_vr = sum(r["vol_ratio"] for r in records) or 1.0
    wscore   = sum(abs(r["pct_move"]) * r["vol_ratio"] for r in records) / total_vr
    sector_5d = sum(r.get("five_day_ret", 0) for r in records) / len(records)

    if wscore < MIN_SECTOR_MOVE_PCT:
        continue

    # Gate: neutral zone — direction must be decisive
    if abs(avg_move) < MIN_SECTOR_DIRECTION_PCT:
        print(f"  SKIP {sector:<14}  neutral zone (avg_move={avg_move:+.2f}%)")
        continue

    direction   = "BULL" if avg_move > 0 else "BEAR"
    five_d_dir  = "BULL" if sector_5d > 0 else "BEAR"

    # Gate: 5-day trend alignment
    if REQUIRE_TREND_ALIGNMENT and direction != five_d_dir:
        print(f"  SKIP {sector:<14}  trend mismatch OR={direction}({avg_move:+.2f}%) vs 5d={five_d_dir}({sector_5d:+.2f}%)")
        continue

    sector_results.append({
        "sector": sector, "direction": direction,
        "score": wscore, "avg_move": avg_move, "sector_5d": sector_5d,
        "stocks": sorted(records, key=lambda x: x["score"], reverse=True),
    })

sector_results.sort(key=lambda s: s["score"], reverse=True)

print(f"{'SECTOR':<14}  {'DIR':<4}  {'SCORE':>6}  {'OR_MOVE':>8}  {'5D_TREND':>9}  {'STOCKS':>6}")
print("-" * 60)
for s in sector_results:
    print(f"{s['sector']:<14}  {s['direction']:<4}  {s['score']:>6.3f}  "
          f"{s['avg_move']:>+7.2f}%  {s['sector_5d']:>+8.2f}%  {len(s['stocks']):>6}")

# ── Watchlist selection ────────────────────────────────────────────────────────
watchlist  = []   # list of dicts (augmented with entry direction)
for sector_res in sector_results[: TOP_N_SECTORS]:
    count = 0
    for r in sector_res["stocks"]:
        if count >= TOP_STOCKS_PER_SECTOR:
            break
        # Stock-direction alignment: individual stock OR must match sector
        stock_bull = r["pct_move"] >= 0
        if sector_res["direction"] == "BULL" and not stock_bull:
            print(f"  SKIP {r['symbol']:<12}  OR {r['pct_move']:+.2f}% opposes BULL sector")
            continue
        if sector_res["direction"] == "BEAR" and stock_bull:
            print(f"  SKIP {r['symbol']:<12}  OR {r['pct_move']:+.2f}% opposes BEAR sector")
            continue
        r["entry_dir"] = "BUY" if sector_res["direction"] == "BULL" else "SELL"
        watchlist.append(r)
        count += 1

print(f"\nWATCHLIST ({len(watchlist)} stocks):")
print(f"{'SYMBOL':<12}  {'SECTOR':<14}  {'DIR':<4}  {'OR_LOW':>8}  {'OR_HIGH':>8}  {'MOVE':>7}  {'VOL_R':>6}")
print("-" * 72)
for w in watchlist:
    print(f"{w['symbol']:<12}  {w['sector']:<14}  {w['entry_dir']:<4}  "
          f"{w['or_low']:>8.2f}  {w['or_high']:>8.2f}  "
          f"{w['pct_move']:>+6.2f}%  {w['vol_ratio']:>5.1f}x")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: Replay 3-min candles (09:30 → 14:00)
# ═════════════════════════════════════════════════════════════════════════════

print("\n")
print("=" * 65)
print("PHASE 2 — Signal replay (3-min candles, 09:30 – 14:00)")
print("=" * 65)

ENTRY_END_TIME = datetime.combine(TODAY, datetime.strptime(ENTRY_END, "%H:%M").time())

sim_positions  = {}   # symbol → dict
all_signals    = []
all_exits      = []

for w in watchlist:
    symbol = w["symbol"]
    sid    = w["sid"]

    try:
        df3 = dhan.get_historical_data(sid, interval="3minute", min_bars=100)
        time.sleep(0.2)
    except Exception as exc:
        print(f"\n{symbol}  3-min fetch failed: {exc}")
        continue

    if df3 is None or df3.empty:
        print(f"\n{symbol}  no 3-min data")
        continue

    if hasattr(df3.index, "date"):
        df3 = df3[df3.index.date == TODAY]

    if df3.empty:
        print(f"\n{symbol}  no today bars in 3-min frame")
        continue

    entry_start = datetime.combine(TODAY, datetime.strptime("09:30", "%H:%M").time())
    orb         = w
    direction   = w["entry_dir"]
    buy_trigger = orb["or_high"] * (1 + BREAKOUT_BUFFER_PCT / 100.0)
    sell_trigger= orb["or_low"]  * (1 - BREAKOUT_BUFFER_PCT / 100.0)

    print(f"\n{symbol}  dir={direction}  OR={orb['or_low']:.2f}-{orb['or_high']:.2f}"
          f"  trigger={'>' if direction=='BUY' else '<'}"
          f"{'%.2f' % (buy_trigger if direction=='BUY' else sell_trigger)}")

    position    = None
    vol_window  = []

    for ts, bar in df3.iterrows():
        # Strip timezone so comparisons with naive entry_start work
        bar_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if getattr(bar_dt, "tzinfo", None) is not None:
            bar_dt = bar_dt.replace(tzinfo=None)
        if bar_dt < entry_start:
            continue

        close  = float(bar["close"])
        high   = float(bar["high"])
        low    = float(bar["low"])
        volume = float(bar["volume"])
        vol_window.append(volume)
        vol_mean = sum(vol_window[-20:]) / len(vol_window[-20:]) if vol_window else 1.0
        vol_ratio_bar = volume / vol_mean if vol_mean > 0 else 1.0

        # ── Active position: check SL, target, or time exit ──────────────────
        if position is not None:
            held_min = (bar_dt - position["entry_time"]).total_seconds() / 60.0

            sl_hit     = (position["dir"] == "BUY"  and low  <= position["stop"])  or \
                         (position["dir"] == "SELL" and high >= position["stop"])
            target_hit = (position["dir"] == "BUY"  and high >= position["target"]) or \
                         (position["dir"] == "SELL" and low  <= position["target"])
            time_hit   = held_min >= TIME_EXIT_MINUTES

            if target_hit or sl_hit or time_hit:
                exit_price = position["target"] if target_hit else (
                    position["stop"]  if sl_hit else close
                )
                exit_type  = "TARGET" if target_hit else ("SL" if sl_hit else "TIME-EXIT")
                pnl_per    = (exit_price - position["entry"]) if position["dir"] == "BUY" \
                             else (position["entry"] - exit_price)
                pnl_total  = pnl_per * position["qty"]
                pnl_pct    = pnl_per / position["entry"] * 100

                exit_rec = {
                    "symbol": symbol, "dir": position["dir"],
                    "entry": position["entry"], "stop": position["stop"],
                    "target": position["target"], "exit_price": exit_price,
                    "exit_type": exit_type, "exit_time": bar_dt.strftime("%H:%M"),
                    "pnl_pct": pnl_pct, "held_min": int(held_min),
                }
                all_exits.append(exit_rec)

                tag = ("TARGET" if target_hit else ("SL    " if sl_hit else "TIME  "))
                sign = "+" if pnl_pct >= 0 else ""
                print(f"  {bar_dt.strftime('%H:%M')}  {tag}  exit={exit_price:.2f}"
                      f"  pnl={sign}{pnl_pct:.2f}%  held={int(held_min)}min")
                position = None
            continue

        # ── No position: check entry ──────────────────────────────────────────
        if bar_dt > ENTRY_END_TIME:
            continue

        triggered = (direction == "BUY"  and close >= buy_trigger) or \
                    (direction == "SELL" and close <= sell_trigger)
        if not triggered:
            continue

        if vol_ratio_bar < ENTRY_VOLUME_MULTIPLIER:
            print(f"  {bar_dt.strftime('%H:%M')}  breakout but vol only {vol_ratio_bar:.1f}x — skip")
            continue

        # Compute RSI from available bars up to this point
        rsi = 50.0
        try:
            idx = list(df3.index).index(ts)
            closes_so_far = df3["close"].iloc[max(0, idx-29): idx+1].values.astype(float)
            if len(closes_so_far) >= 15:
                rsi_arr = talib.RSI(closes_so_far, timeperiod=14)
                if not pd.isna(rsi_arr[-1]):
                    rsi = float(rsi_arr[-1])
        except Exception:
            pass

        if direction == "BUY" and rsi > 78:
            print(f"  {bar_dt.strftime('%H:%M')}  breakout but RSI={rsi:.0f} overbought — skip")
            continue
        if direction == "SELL" and rsi < 22:
            print(f"  {bar_dt.strftime('%H:%M')}  breakout but RSI={rsi:.0f} oversold — skip")
            continue

        entry = close
        if direction == "BUY":
            raw_stop_dist = entry - orb["or_low"]
            stop_pct      = max(MIN_STOP_PCT, min(MAX_STOP_PCT, raw_stop_dist / entry * 100))
            stop          = entry * (1 - stop_pct / 100)
            target        = entry + RR_RATIO * (entry - stop)
        else:
            raw_stop_dist = orb["or_high"] - entry
            stop_pct      = max(MIN_STOP_PCT, min(MAX_STOP_PCT, raw_stop_dist / entry * 100))
            stop          = entry * (1 + stop_pct / 100)
            target        = entry - RR_RATIO * (stop - entry)

        tgt_pct = abs(target - entry) / entry * 100

        position = {
            "symbol": symbol, "dir": direction, "entry": entry,
            "stop": stop, "target": target, "qty": 10,  # dummy qty
            "entry_time": bar_dt,
        }

        sig_rec = {
            "symbol": symbol, "dir": direction,
            "entry_time": bar_dt.strftime("%H:%M"), "entry": entry,
            "stop": stop, "target": target,
            "stop_pct": stop_pct, "target_pct": tgt_pct,
            "vol_ratio": vol_ratio_bar, "rsi": rsi,
        }
        all_signals.append(sig_rec)

        print(f"  {bar_dt.strftime('%H:%M')}  SIGNAL {direction}"
              f"  entry={entry:.2f}  stop={stop:.2f}(-{stop_pct:.2f}%)"
              f"  target={target:.2f}(+{tgt_pct:.2f}%)"
              f"  vol={vol_ratio_bar:.1f}x  RSI={rsi:.0f}")

    # Position still open at replay end (would have been time-exited)
    if position is not None:
        last_close = float(df3.iloc[-1]["close"])
        last_dt = df3.index[-1].to_pydatetime()
        if getattr(last_dt, "tzinfo", None) is not None:
            last_dt = last_dt.replace(tzinfo=None)
        held_min   = (last_dt - position["entry_time"]).total_seconds() / 60
        pnl_pct    = (
            (last_close - position["entry"]) / position["entry"] * 100
            if direction == "BUY"
            else (position["entry"] - last_close) / position["entry"] * 100
        )
        sign = "+" if pnl_pct >= 0 else ""
        print(f"  14:00  TIME-EXIT (eod)  exit~{last_close:.2f}"
              f"  pnl~{sign}{pnl_pct:.2f}%  held={int(held_min)}min")
        all_exits.append({
            "symbol": symbol, "dir": direction,
            "entry": position["entry"], "exit_price": last_close,
            "exit_type": "TIME-EXIT(EOD)", "pnl_pct": pnl_pct, "held_min": int(held_min),
        })


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print("\n")
print("=" * 65)
print(f"SUMMARY  ({TODAY})")
print("=" * 65)
print(f"Signals fired   : {len(all_signals)}")
print(f"Exits recorded  : {len(all_exits)}")

if all_exits:
    print(f"\n{'SYMBOL':<12}  {'DIR':<4}  {'TYPE':<14}  {'ENTRY':>7}  {'EXIT':>7}  {'PNL':>7}  {'HELD':>5}")
    print("-" * 65)
    total_pnl = 0.0
    for e in all_exits:
        sign = "+" if e["pnl_pct"] >= 0 else ""
        print(f"{e['symbol']:<12}  {e['dir']:<4}  {e['exit_type']:<14}  "
              f"{e['entry']:>7.2f}  {e['exit_price']:>7.2f}  "
              f"{sign}{e['pnl_pct']:>6.2f}%  {e.get('held_min', '?'):>5}")
        total_pnl += e["pnl_pct"]
    if all_exits:
        avg_pnl = total_pnl / len(all_exits)
        wins    = sum(1 for e in all_exits if e["pnl_pct"] > 0)
        print(f"\nAvg P&L : {'+' if avg_pnl>=0 else ''}{avg_pnl:.2f}%  |  "
              f"Win rate: {wins}/{len(all_exits)} = {wins/len(all_exits)*100:.0f}%")

if not all_signals:
    print("\nNo ORB signals fired today — possible reasons:")
    print("  • Breakout volume < 1.5x required")
    print("  • Price did not close above/below OR level + buffer")
    print("  • All sectors moved < MIN_SECTOR_MOVE_PCT (%.2f%%)" % MIN_SECTOR_MOVE_PCT)
