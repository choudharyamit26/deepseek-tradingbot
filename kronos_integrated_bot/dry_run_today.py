#!/usr/bin/env python3
"""Dry-run the enhanced bot on today's data — compares AI vs Kronos."""
import sys, os, asyncio, logging
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

STOCKS = ["BAJFINANCE", "ICICIBANK", "AXISBANK", "NTPC", "HDFCBANK"]
IST = ZoneInfo("Asia/Kolkata")

ACTUAL = """2026-06-01 11:31  BAJFINANCE  ENTRY-SHORT@894.5  exit=895.65  PnL=-1.15
2026-06-01 12:21  ICICIBANK   ENTRY-SHORT@1246.2  exit=1241.5  PnL=+4.70
2026-06-01 12:41  ICICIBANK   ENTRY-SHORT@1245.2  (open)
2026-06-01 12:59  AXISBANK    TRAILING-SL@1270.2  exit=1277.3  PnL=-6.70
2026-06-01 13:08  NTPC        TRAILING-SL@379.1   exit=379.05 PnL=+0.10"""

async def analyze_one(bot, symbol):
    sid = bot.dhan.security_ids.get(symbol)
    if not sid:
        return f"  {symbol:12s}  NO SECURITY ID"

    df = bot.dhan.get_historical_data(sid, "3minute", min_bars=50)
    if df is None or len(df) < 20:
        return f"  {symbol:12s}  NO DATA ({len(df) if df is not None else 0} bars)"

    ind = bot.calculate_technical_indicators(df)
    if not ind:
        return f"  {symbol:12s}  INDICATORS FAILED"

    ltp = float(df["close"].iloc[-1])
    rsi = ind.get("rsi", 0)
    adx = ind.get("adx", 0)
    vwap = ind.get("vwap_distance_pct", 0)
    vol  = ind.get("volume_ratio", 0)
    atr  = ind.get("atr", 1) or 1

    reg = bot.regime.get_regime(symbol)
    ctx = bot.regime.format_regime_context(symbol, reg)
    recent = df.tail(10)
    raw = bot.ai.get_trading_signal(symbol, {"ltp": ltp}, ind, ctx, recent_bars=recent)
    ai_sig = raw.get("signal", "HOLD")
    ai_conf = raw.get("confidence", 0)

    # Kronos
    kronos_line = ""
    if bot.kronos.ready:
        pred = bot.kronos.predict(df, symbol=symbol)
        if pred is not None and not pred.empty:
            po, pc = pred["open"].iloc[0], pred["close"].iloc[-1]
            pret = (pc - po) / po * 100
            prng = f"Rs{pred['low'].min():.2f}-Rs{pred['high'].max():.2f}"

            kc = bot.kronos.compute_confirmation(ai_sig, pred, ltp)
            newc = min(100, max(0, int(ai_conf * kc["adjustment"])))
            adj_s = f"adj={kc['adjustment']:.2f} conf={ai_conf}->{newc}"
            if kc["conflict"]:
                kronos_line = f"CONFLICT {adj_s} pred={po:.1f}->{pc:.1f}({pret:+.2f}%) [{prng}]"
            elif ai_sig in ("BUY","SELL"):
                kronos_line = f"AGREES   {adj_s} pred={po:.1f}->{pc:.1f}({pret:+.2f}%) [{prng}]"
            else:
                kronos_line = f"HOLD     pred={po:.1f}->{pc:.1f}({pret:+.2f}%) [{prng}]"
        else:
            kronos_line = "NO PREDICTION"

    pf = "Y" if bot._passes_prefilter(ind, {})[0] else "N"
    return (f"  {symbol:12s}  LTP={ltp:>8.2f}  RSI={rsi:>5.1f}  ADX={adx:>5.1f}  "
            f"VWAP={vwap:+6.2f}%  Vol={vol:.2f}x  PF={pf}  "
            f"AI={ai_sig:5s}({ai_conf:3d})  Kronos:{kronos_line}")

async def main():
    cfg.KRONOS_DEVICE = "cpu"
    cfg.MIN_CONFIDENCE = 50  # lower for max signal visibility

    dhan = DhanStockTradingBot()
    bal = dhan.get_available_balance()
    ai  = DeepSeekStockAnalyzer(api_key=os.getenv("DEEPSEEK_API_KEY",""))
    risk = RiskManager(dhan, max_daily_trades=10, max_daily_loss_percent=5,
                       risk_per_trade_percent=2, min_confidence=50)

    cfg_map = {
        "model_name": cfg.KRONOS_MODEL,
        "tokenizer_name": cfg.KRONOS_TOKENIZER,
        "max_context": cfg.KRONOS_MAX_CONTEXT,
        "device": "cpu",
        "pred_len": cfg.KRONOS_PRED_LEN,
        "lookback": cfg.KRONOS_LOOKBACK,
        "temperature": cfg.KRONOS_TEMPERATURE,
        "sample_count": cfg.KRONOS_SAMPLE_COUNT,
        "top_p": cfg.KRONOS_TOP_P,
        "penalty_conflict": cfg.KRONOS_PENALTY_CONFLICT,
        "bonus_align": cfg.KRONOS_BONUS_ALIGN,
        "exit_threshold": cfg.KRONOS_EXIT_THRESHOLD,
    }
    ki = KronosIntegration(cfg_map)
    ki.load()

    bot = EnhancedIntradayBot(dhan, ai, risk, ki,
        watchlist=STOCKS, enable_telegram=False, dry_run=True)
    bot._ENTRY_START_T = dtime(0,0)
    bot._ENTRY_END_T   = dtime(23,59)
    bot.cooldown_seconds = 0

    print("=" * 110)
    print("  ENHANCED BOT DRY-RUN  |  Today's Data (2026-06-01)")
    print(f"  Time: {datetime.now(IST).strftime('%H:%M:%S %Z')}  |  Balance: Rs{bal:,.2f}")
    print("=" * 110)
    print("\n  Actual trades today:")
    for L in ACTUAL.split("\n"):
        print(f"   {L}")

    print(f"\n  {'Stock':12s}  {'LTP':>8s}  {'RSI':>5s}  {'ADX':>5s}  {'VWAP':>7s}  {'Vol':>5s}  PF  "
          f"{'AI':12s}  Kronos")
    print(f"  {'-'*12}  {'-'*8}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*5}  ---  "
          f"{'-'*12}  {'-'*55}")

    for sym in STOCKS:
        result = await analyze_one(bot, sym)
        print(result)

    print()
    print("  PF=Prefilter(Y/N)  AI=signal(conf)  Kronos=AGREES|CONFLICT adj=conf_mult")
    print("  HOLD=Kronos ran but AI was HOLD  NO PREDICTION=model didn't produce output")
    print("=" * 110)

if __name__ == "__main__":
    asyncio.run(main())
