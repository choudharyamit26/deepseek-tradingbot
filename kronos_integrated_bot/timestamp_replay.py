#!/usr/bin/env python3
"""Replay today's 5 actual trades through the enhanced bot at their exact timestamps."""
import sys, os, asyncio, logging, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["DRY_RUN"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.disable(logging.CRITICAL)

from dotenv import load_dotenv; load_dotenv()
from dhan_integration import DhanStockTradingBot
from deepseek_analyzer import DeepSeekStockAnalyzer
from risk_manager import RiskManager
from kronos_integrated_bot import config as cfg
from kronos_integrated_bot.kronos_integration import KronosIntegration
from kronos_integrated_bot.enhanced_bot import EnhancedIntradayBot

IST = ZoneInfo("Asia/Kolkata")

# Actual trades from today's signal log
TRADES = [
    {"time": "11:31", "symbol": "BAJFINANCE", "action": "SHORT", "price": 894.5,  "exit": 895.65, "pnl": -1.15},
    {"time": "12:21", "symbol": "ICICIBANK",  "action": "SHORT", "price": 1246.2, "exit": 1241.5, "pnl": +4.70},
    {"time": "12:41", "symbol": "ICICIBANK",  "action": "SHORT", "price": 1245.2, "exit": None,   "pnl": None},
    {"time": "12:59", "symbol": "AXISBANK",   "action": "SHORT", "price": 1270.2, "exit": 1277.3, "pnl": -6.70},
    {"time": "13:08", "symbol": "NTPC",       "action": "SHORT", "price": 379.1,  "exit": 379.05,"pnl": +0.10},
]

async def run_analysis(bot, symbol, ts_str):
    """Fetch data up to timestamp, run indicators + AI + Kronos, return results."""
    sid = bot.dhan.security_ids.get(symbol)
    if not sid:
        return None, "NO SECURITY ID"

    target_t = datetime.strptime(f"2026-06-01 {ts_str}", "%Y-%m-%d %H:%M").replace(tzinfo=IST)

    df = bot.dhan.get_historical_data(sid, "3minute", min_bars=20)
    if df is None or len(df) < 20:
        return None, f"NO DATA ({len(df) if df is not None else 0})"

    # Slice up to target timestamp (inclusive of bars at or before target)
    df_before = df[df.index <= target_t].copy()
    if len(df_before) < 20:
        return None, f"ONLY {len(df_before)} BARS BEFORE {ts_str}"

    ind = bot.calculate_technical_indicators(df_before)
    if not ind:
        return None, "INDICATORS FAILED"

    ltp = float(df_before["close"].iloc[-1])
    rsi = ind.get("rsi", 0)
    adx = ind.get("adx", 0)
    vwap = ind.get("vwap_distance_pct", 0)
    vol = ind.get("volume_ratio", 0)
    atr = ind.get("atr", 1) or 1

    reg = bot.regime.get_regime(symbol)
    ctx = bot.regime.format_regime_context(symbol, reg)
    recent = df_before.tail(10)

    # ── Kronos prediction + inject into AI context ─────────────────────
    kronos_section = None
    if bot.kronos.ready:
        kronos_input_15m = df_before.resample("15min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        enough_15m = len(kronos_input_15m) >= 10
        enough_3m = len(df_before) >= 20
        if enough_15m:
            kronos_input = kronos_input_15m
        elif enough_3m:
            kronos_input = df_before
        else:
            kronos_input = None

        if kronos_input is not None:
            pred = bot.kronos.predict(kronos_input, symbol=symbol)
            if pred is not None and not pred.empty:
                kronos_section = bot.kronos.build_prompt_section(pred, ltp)
                ctx += (
                    "\n\n" + kronos_section +
                    "\n\nIMPORTANT: The Kronos forecast above is a supplementary signal."
                    "\nFactor it into your decision but do NOT follow it blindly."
                    "\nYour primary technical rules (VWAP, RSI, ADX, volume, MTF) still apply."
                )

    # ── AI signal ───────────────────────────────────────────────────────
    raw = bot.ai.get_trading_signal(symbol, {"ltp": ltp}, ind, ctx, recent_bars=recent)
    ai_sig = raw.get("signal", "HOLD")
    ai_conf = raw.get("confidence", 0)
    bot._last_signals[symbol] = {
        "signal": ai_sig, "confidence": ai_conf,
        "setup_type": raw.get("setup_type"), "reasoning": raw.get("reasoning"),
    }

    pf_pass, pf_reason = bot._passes_prefilter(ind, reg)
    kronos_note = "N/A"
    kronos_ratio = 1.0
    if bot.kronos.ready and ai_sig in ("BUY", "SELL"):
        if pred is not None and not pred.empty:
            kc = bot.kronos.compute_confirmation(ai_sig, pred, ltp, historical_df=kronos_input)
            kronos_ratio = kc["adjustment"]
            mag = kc.get("magnitude", 0)
            rng = kc.get("pred_range_pct", 0)
            src = "15m" if enough_15m else "3m"
            if kc["conflict"]:
                kronos_note = f"CONFLICT ratio={kronos_ratio:.2f} mag={mag:.2f} rng={rng:.2f}% [{src}]"
            else:
                kronos_note = f"ALIGN    ratio={kronos_ratio:.2f} mag={mag:.2f} rng={rng:.2f}% [{src}]"
        else:
            kronos_note = "NO PRED"
    elif bot.kronos.ready:
        kronos_note = "HOLD(no conf needed)"

    return {
        "symbol": symbol, "ts": ts_str, "ltp": ltp, "bars": len(df_before),
        "rsi": rsi, "adx": adx, "vwap": vwap, "vol": vol,
        "pf": pf_pass, "pf_reason": pf_reason,
        "ai_sig": ai_sig, "ai_conf": ai_conf,
        "kronos": kronos_note, "kronos_ratio": kronos_ratio,
    }, None

async def main():
    cfg.KRONOS_DEVICE = "cpu"
    cfg.MIN_CONFIDENCE = 75

    dhan = DhanStockTradingBot()
    bal = dhan.get_available_balance()
    ai = DeepSeekStockAnalyzer(api_key=os.getenv("DEEPSEEK_API_KEY",""))
    risk = RiskManager(dhan, max_daily_trades=10, max_daily_loss_percent=5,
                       risk_per_trade_percent=2, min_confidence=50)

    cfg_map = {k: getattr(cfg, k) for k in [
        "KRONOS_MODEL","KRONOS_TOKENIZER","KRONOS_MAX_CONTEXT",
        "KRONOS_PRED_LEN","KRONOS_LOOKBACK","KRONOS_TEMPERATURE",
        "KRONOS_SAMPLE_COUNT","KRONOS_TOP_P","KRONOS_PENALTY_CONFLICT",
        "KRONOS_BONUS_ALIGN","KRONOS_EXIT_THRESHOLD"]}
    cfg_map["device"] = "cpu"
    ki = KronosIntegration({
        "model_name": cfg_map["KRONOS_MODEL"],
        "tokenizer_name": cfg_map["KRONOS_TOKENIZER"],
        "max_context": cfg_map["KRONOS_MAX_CONTEXT"],
        "device": "cpu",
        "pred_len": cfg_map["KRONOS_PRED_LEN"],
        "lookback": cfg_map["KRONOS_LOOKBACK"],
        "temperature": cfg_map["KRONOS_TEMPERATURE"],
        "sample_count": cfg_map["KRONOS_SAMPLE_COUNT"],
        "top_p": cfg_map["KRONOS_TOP_P"],
        "penalty_conflict": cfg_map["KRONOS_PENALTY_CONFLICT"],
        "bonus_align": cfg_map["KRONOS_BONUS_ALIGN"],
        "exit_threshold": cfg_map["KRONOS_EXIT_THRESHOLD"],
    })
    ki.load()

    bot = EnhancedIntradayBot(dhan, ai, risk, ki,
        watchlist=list({t["symbol"] for t in TRADES}),
        enable_telegram=False, dry_run=True)
    bot._ENTRY_START_T = dtime(0, 0)
    bot._ENTRY_END_T = dtime(23, 59)
    bot.cooldown_seconds = 0

    print("=" * 120)
    print("  TIMESTAMP REPLAY  |  Enhanced Bot vs Actual Trades on 2026-06-01")
    print(f"  Balance: Rs{bal:,.2f}  |  Kronos: {cfg.KRONOS_MODEL}")
    print("=" * 120)

    hdr = (f"  {'Time':>5s}  {'Stock':12s}  {'Bars':>4s}  {'LTP':>8s}  {'RSI':>5s}  {'ADX':>5s}  "
           f"{'VWAP':>7s}  {'Vol':>5s}  {'PF':3s}  {'AI':12s}  {'Enh Bot':10s}  "
           f"Kronos(mag/rng)")
    sep = "  " + "-" * (120 - 4)
    print(hdr)
    print(sep)

    enhanced_verdicts = []

    for t in TRADES:
        res, err = await run_analysis(bot, t["symbol"], t["time"])
        if err or res is None:
            print(f"  {t['time']:>5s}  {t['symbol']:12s}  ERROR: {err}")
            continue

        pf_str = "Y" if res["pf"] else f"N({res['pf_reason'][:20]})"
        ai_str = f"{res['ai_sig']:5s}({res['ai_conf']:3d})"

        # Enhanced bot decision
        sig = res["ai_sig"]
        kratio = res.get("kronos_ratio", 1.0)
        if not res["pf"]:
            enh_verdict = "BLOCKED(PF)"
        elif sig in ("BUY", "SELL") and res["ai_conf"] >= 50:
            enh_verdict = f"ENTRY-{sig}"
            if kratio != 1.0:
                enh_verdict += f"(x{kratio:.2f})"
        else:
            enh_verdict = f"HOLD"

        actual_side = "SHORT" if t["action"] == "SHORT" else "LONG"
        pnl_str = f"PnL={t['pnl']:+.2f}" if t["pnl"] is not None else "OPEN"
        is_short = sig == "SELL"
        match_str = ""
        if enh_verdict.startswith("ENTRY-"):
            if (is_short and actual_side == "SHORT") or (not is_short and actual_side == "LONG"):
                match_str = "MATCH(dir)"
            else:
                match_str = "MISMATCH(dir)"
        elif enh_verdict == "HOLD":
            match_str = f"SKIPPED({t['action']})"
        else:
            match_str = "N/A"

        print(f"  {t['time']:>5s}  {t['symbol']:12s}  {res['bars']:>4d}  "
              f"{res['ltp']:>8.2f}  {res['rsi']:>5.1f}  {res['adx']:>5.1f}  "
              f"{res['vwap']:+7.3f}%  {res['vol']:>5.2f}x  {pf_str:3s}  {ai_str:12s}  "
              f"{enh_verdict:10s}  {res['kronos']}")

        enhanced_verdicts.append((t, enh_verdict, match_str))

    print(sep)
    print(f"\n  {'Stock':12s}  {'Time':>5s}  {'Actual':>12s}  {'Price':>8s}  {'Exit/PnL':>14s}  "
          f"{'Enhanced':>16s}  {'Match':>14s}")
    print("  " + "-" * 85)
    for (t, enh_v, match) in enhanced_verdicts:
        entry_label = f"ENTRY-{t['action']}"
        pnl_s = f"exit={t['exit']} PnL={t['pnl']:+.2f}" if t["pnl"] is not None else "OPEN (no exit)"
        print(f"  {t['symbol']:12s}  {t['time']:>5s}  {entry_label:>12s}  "
              f"{t['price']:>8.1f}  {pnl_s:>14s}  {enh_v:>16s}  {match:>14s}")

    # Summary
    skipped = sum(1 for _, v, m in enhanced_verdicts if v == "HOLD" and "SHORT" in m)
    entered = sum(1 for _, v, _ in enhanced_verdicts if v.startswith("ENTRY-"))
    averted_losses = sum(1 for (t, v, m) in enhanced_verdicts if v == "HOLD" and t["pnl"] is not None and t["pnl"] < 0)
    missed_gains = sum(1 for (t, v, m) in enhanced_verdicts if v == "HOLD" and t["pnl"] is not None and t["pnl"] > 0)
    print(f"\n  Summary: {len(enhanced_verdicts)} trades replayed."
          f"  {entered} entered (directional match)."
          f"  {skipped} skipped.  Would have averted {averted_losses} losses, missed {missed_gains} gains.")
    print("=" * 120)

if __name__ == "__main__":
    asyncio.run(main())
