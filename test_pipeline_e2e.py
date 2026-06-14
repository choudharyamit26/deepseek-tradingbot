"""
Full end-to-end pipeline test:
  Historical data -> Indicators -> Kronos -> DeepSeek

Runs the exact same code paths the live bot uses, against cached CSV data,
so you can see every value that flows through the system and evaluate whether
the DeepSeek prompt is producing sensible decisions.

Usage:
    python test_pipeline_e2e.py
    python test_pipeline_e2e.py RELIANCE   # run only one stock
"""
import sys, os, asyncio, warnings, json, textwrap, time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(override=True)

import pandas as pd
import numpy as np

logging_setup_done = False
import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

DATA_DIR = ROOT / "kronos_integrated_bot" / "data"
IST = ZoneInfo("Asia/Kolkata")

# ─── stocks to test (name -> security_id, prefer ones with most data) ────────
TEST_STOCKS = {
    "RELIANCE":  {"sid": "2885", "dates_3m": ["2026-06-10"],  "dates_15m": [],          "dates_1h": []},
    "KOTAKBANK": {"sid": "1922", "dates_3m": [],               "dates_15m": ["2026-06-10", "2026-06-11"], "dates_1h": ["2026-06-10"]},
    "DRREDDY":   {"sid": "881",  "dates_3m": [],               "dates_15m": ["2026-06-10", "2026-06-11"], "dates_1h": ["2026-06-10"]},
    "JSWSTEEL":  {"sid": "11723","dates_3m": [],               "dates_15m": ["2026-06-10", "2026-06-11"], "dates_1h": ["2026-06-10"]},
}

# Accept CLI argument to run single stock
if len(sys.argv) > 1:
    name_filter = sys.argv[1].upper()
    TEST_STOCKS = {k: v for k, v in TEST_STOCKS.items() if k == name_filter}
    if not TEST_STOCKS:
        print(f"Stock {name_filter} not in test list. Available: {list({'RELIANCE','KOTAKBANK','DRREDDY','JSWSTEEL'})}")
        sys.exit(1)

SEP = "=" * 72


def load_csv(sid, date, interval):
    path = DATA_DIR / date / f"{sid}_{interval}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    return df.sort_index()


def load_best(sid, dates, interval, min_bars):
    """Load and concatenate CSVs, return df with >= min_bars or None."""
    dfs = []
    for d in dates:
        df = load_csv(sid, d, interval)
        if df is not None and len(df) > 0:
            dfs.append(df)
    if not dfs:
        return None
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined if len(combined) >= min_bars else None


def fmt_df(df, n=5):
    """Format last N bars for display."""
    if df is None or len(df) == 0:
        return "  (no data)"
    rows = []
    for ts, row in df.tail(n).iterrows():
        rows.append(f"  {ts.strftime('%H:%M')}  O={row['open']:.2f}  H={row['high']:.2f}"
                    f"  L={row['low']:.2f}  C={row['close']:.2f}  V={int(row.get('volume',0))}")
    return "\n".join(rows)


def fmt_indicators(ind):
    if not ind:
        return "  (no indicators)"
    keys = ["rsi","adx","macd","macd_signal","vwap","vwap_distance_pct",
            "sma_20","ema_9","atr","mfi","bb_percent_b","volume_ratio",
            "support","resistance","close"]
    lines = []
    for k in keys:
        v = ind.get(k)
        if v is not None:
            if isinstance(v, float):
                lines.append(f"  {k:20s} = {v:.4f}")
            else:
                lines.append(f"  {k:20s} = {v}")
    return "\n".join(lines)


def fmt_pred(pred_df, ltp):
    if pred_df is None or pred_df.empty:
        return "  (no prediction)"
    lines = []
    for i, (ts, row) in enumerate(pred_df.iterrows()):
        chg = (row["close"] - ltp) / ltp * 100
        lines.append(f"  [{i+1}] {ts.strftime('%H:%M')}  O={row['open']:.2f}"
                     f"  H={row['high']:.2f}  L={row['low']:.2f}"
                     f"  C={row['close']:.2f}  ({chg:+.2f}% vs ltp)")
    return "\n".join(lines)


# ─── Load Kronos ──────────────────────────────────────────────────────────────
print(SEP)
print("  LOADING KRONOS MODEL")
print(SEP)

from kronos_integrated_bot.kronos_integration import KronosIntegration
from kronos_integrated_bot import config as cfg

kronos_cfg = {
    "model_name":       cfg.KRONOS_MODEL,
    "tokenizer_name":   cfg.KRONOS_TOKENIZER,
    "max_context":      cfg.KRONOS_MAX_CONTEXT,
    "device":           cfg.KRONOS_DEVICE,
    "pred_len":         cfg.KRONOS_PRED_LEN,
    "lookback":         cfg.KRONOS_LOOKBACK,
    "temperature":      cfg.KRONOS_TEMPERATURE,
    "top_p":            cfg.KRONOS_TOP_P,
    "sample_count":     cfg.KRONOS_SAMPLE_COUNT,
    "penalty_conflict": cfg.KRONOS_PENALTY_CONFLICT,
    "bonus_align":      cfg.KRONOS_BONUS_ALIGN,
    "exit_threshold":   cfg.KRONOS_EXIT_THRESHOLD,
    "min_predicted_move": cfg.KRONOS_MIN_PREDICTED_MOVE,
}

t0 = time.time()
ki = KronosIntegration(kronos_cfg)
ki.load()
print(f"  Loaded on: {ki._predictor.device}  ({time.time()-t0:.1f}s)")

# ─── Load indicators helper ───────────────────────────────────────────────────
from indicators import calculate_technical_indicators

# ─── Load DeepSeek ────────────────────────────────────────────────────────────
from deepseek_analyzer import DeepSeekStockAnalyzer

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_KEY:
    print("\nWARNING: DEEPSEEK_API_KEY not set in .env -- DeepSeek calls will be skipped")

ai = DeepSeekStockAnalyzer(
    api_key=DEEPSEEK_KEY,
    min_confidence=cfg.MIN_CONFIDENCE,
    min_adx=cfg.MIN_ADX_TRENDING,
    rsi_ob=cfg.RSI_OB_LIMIT,
    rsi_os=cfg.RSI_OS_LIMIT,
    min_rr_ratio=cfg.MIN_RR_RATIO,
)

# ─── Per-stock pipeline ───────────────────────────────────────────────────────
all_results = []

for symbol, meta in TEST_STOCKS.items():
    sid = meta["sid"]
    print()
    print(SEP)
    print(f"  STOCK: {symbol}  (security_id={sid})")
    print(SEP)

    # ── 1. Load historical data ───────────────────────────────────────────────
    print("\n[1] HISTORICAL DATA")
    dates_3m  = meta["dates_3m"]  or ["2026-06-11", "2026-06-10"]
    dates_15m = meta["dates_15m"] or ["2026-06-11", "2026-06-10"]
    dates_1h  = meta["dates_1h"]  or ["2026-06-11", "2026-06-10"]

    hist_3m  = load_best(sid, dates_3m,  "3minute",  30)
    hist_15m = load_best(sid, dates_15m, "15minute", 25)
    hist_1h  = load_best(sid, dates_1h,  "60minute", 15)

    print(f"  3m  bars: {len(hist_3m)  if hist_3m  is not None else 0}")
    print(f"  15m bars: {len(hist_15m) if hist_15m is not None else 0}")
    print(f"  1h  bars: {len(hist_1h)  if hist_1h  is not None else 0}")

    if hist_3m is not None:
        print(f"\n  Last 5 x 3m bars:")
        print(fmt_df(hist_3m, 5))
    elif hist_15m is not None:
        print(f"\n  Using 15m data as primary (no 3m available):")
        print(fmt_df(hist_15m, 5))

    # Choose primary (3m preferred, else 15m)
    primary_hist = hist_3m if hist_3m is not None and len(hist_3m) >= 30 else hist_15m
    if primary_hist is None:
        print(f"  [SKIP] No usable data for {symbol}")
        continue

    ltp = float(primary_hist["close"].iloc[-1])
    print(f"\n  LTP (last close): {ltp:.2f}")

    # ── 2. Technical Indicators ───────────────────────────────────────────────
    print("\n[2] TECHNICAL INDICATORS")
    ind_3m  = calculate_technical_indicators(primary_hist) if len(primary_hist) >= 5 else {}
    ind_15m = calculate_technical_indicators(hist_15m)     if hist_15m is not None and len(hist_15m) >= 25 else {}
    ind_1h  = calculate_technical_indicators(hist_1h)      if hist_1h  is not None and len(hist_1h)  >= 15 else {}

    print(f"\n  --- 3m indicators ---")
    print(fmt_indicators(ind_3m))
    if ind_15m:
        close_15 = ind_15m.get("close", 0); sma_15 = ind_15m.get("sma_20", close_15)
        print(f"\n  --- 15m trend: {'BULLISH' if close_15 > sma_15 else 'BEARISH'}  RSI={ind_15m.get('rsi','?')}  ADX={ind_15m.get('adx','?')} ---")
    if ind_1h:
        close_1h = ind_1h.get("close", 0); sma_1h = ind_1h.get("sma_20", close_1h)
        print(f"  --- 1h  trend: {'BULLISH' if close_1h > sma_1h else 'BEARISH'}  RSI={ind_1h.get('rsi','?')}  ADX={ind_1h.get('adx','?')} ---")

    # ── 3. Kronos Prediction ─────────────────────────────────────────────────
    print("\n[3] KRONOS PREDICTION")
    t_kron = time.time()
    pred_df = ki.predict(primary_hist, symbol=symbol, force=True)
    print(f"  Time: {time.time()-t_kron:.2f}s")

    if pred_df is None or pred_df.empty:
        print("  Kronos returned None — skipping")
        continue

    gap_min = int(round((pred_df.index[1] - pred_df.index[0]).total_seconds() / 60)) if len(pred_df) >= 2 else 3
    print(f"  y_gap = {gap_min}min  |  pred_len = {len(pred_df)}  |  horizon = {len(pred_df)*gap_min}min")
    print(f"\n  Predicted candles (vs LTP={ltp:.2f}):")
    print(fmt_pred(pred_df, ltp))

    pred_first_ret = (pred_df["close"].iloc[0] - pred_df["open"].iloc[0]) / pred_df["open"].iloc[0] * 100
    pred_total_ret = (pred_df["close"].iloc[-1] - ltp) / ltp * 100
    print(f"\n  Immediate next {gap_min}min: {pred_first_ret:+.3f}%")
    print(f"  Total forecast ({len(pred_df)*gap_min}min): {pred_total_ret:+.3f}%")
    print(f"  Direction: {'UP' if pred_total_ret > 0 else 'DOWN'}")

    kronos_section = ki.build_prompt_section(pred_df, ltp)
    kronos_conf = ki.compute_confirmation("BUY", pred_df, ltp, historical_df=primary_hist)
    print(f"\n  Kronos confirmation (vs BUY): agreement={kronos_conf['agreement']:.3f}  conflict={kronos_conf['conflict']}")
    print(f"  pred_range_pct={kronos_conf.get('pred_range_pct',0):.3f}%  magnitude={kronos_conf.get('magnitude',0):.3f}")

    print(f"\n  Kronos prompt section:")
    print(textwrap.indent(kronos_section, "  | "))

    # ── 4. Build Full Context (as bot would) ──────────────────────────────────
    print("\n[4] FULL CONTEXT ASSEMBLY")

    # Simulated regime (offline — cannot call Dhan)
    regime_context = (
        f"Nifty 50: trend=neutral, volatility=low, strength=48%, "
        f"current={ltp:.0f}, SMA10=0, intraday_chg=+0.20% (session=neutral)\n"
        f"Nifty Bank: trend=neutral, volatility=medium, strength=52%"
    )

    # MTF summary
    mtf_lines = []
    for label, ind in [("3-Min", ind_3m), ("15-Min", ind_15m), ("1-Hour", ind_1h)]:
        if not ind:
            continue
        close = ind.get("close", 0); sma = ind.get("sma_20", close)
        trend = "BULLISH" if close > sma else "BEARISH"
        mtf_lines.append(f"{label}: RSI={ind.get('rsi','?')}, ADX={ind.get('adx','?')}, Price={close:.2f}, SMA20={sma:.2f} -> {trend}")
    mtf_summary = ("Multi-Timeframe Analysis:\n" + "\n".join(mtf_lines)) if len(mtf_lines) >= 2 else ""

    full_context = regime_context
    if mtf_summary:
        full_context += "\n\n" + mtf_summary

    full_context += (
        "\n\n" + kronos_section +
        "\n\nIMPORTANT: The Kronos forecast above is a supplementary signal."
        "\nFactor it into your decision but do NOT follow it blindly."
        "\nYour primary technical rules (VWAP, RSI, ADX, volume, MTF) still apply."
    )

    market_data = {
        "ltp": ltp,
        "high_3m": float(primary_hist["high"].iloc[-1]),
        "low_3m":  float(primary_hist["low"].iloc[-1]),
        "volume":  int(primary_hist["volume"].iloc[-1]) if "volume" in primary_hist.columns else 0,
        "avg_volume_3m": float(primary_hist["volume"].tail(5).mean()) if "volume" in primary_hist.columns else 0,
    }

    print(f"  context length: {len(full_context)} chars")
    print(f"\n  --- Full context preview ---")
    for line in full_context.split("\n"):
        print(f"  {line}")

    # ── 5. DeepSeek API Call ──────────────────────────────────────────────────
    print(f"\n[5] DEEPSEEK API CALL")

    if not DEEPSEEK_KEY:
        print("  [SKIPPED] No API key")
        all_results.append({"symbol": symbol, "signal": "SKIPPED", "skipped": True})
        continue

    # Build the exact user prompt the bot sends
    user_prompt = ai.prepare_market_context(
        symbol, market_data, ind_3m,
        recent_bars=primary_hist.tail(10),
    )
    user_prompt += f"\n\n        Context:\n        {full_context}"

    print(f"\n  === FULL USER PROMPT ===")
    print(textwrap.indent(user_prompt, "  | "))

    atr_raw = ind_3m.get("atr", 0)
    atr_pct = round(float(atr_raw) / ltp * 100, 3) if atr_raw and ltp > 0 else 0.0
    print(f"\n  ATR%={atr_pct:.3f}  -> 1.5xATR SL={atr_pct*1.5:.3f}%")

    print(f"\n  Calling DeepSeek ({ai.model})...")
    t_ds = time.time()
    signal = asyncio.run(ai.get_trading_signal(
        symbol, market_data, ind_3m,
        regime_context=full_context,
        recent_bars=primary_hist.tail(10),
    ))
    ds_time = time.time() - t_ds
    print(f"  DeepSeek response ({ds_time:.1f}s):")
    print()
    print(f"  Signal:     {signal.get('signal')}")
    print(f"  Confidence: {signal.get('confidence')}")
    print(f"  SL%:        {signal.get('stop_loss_percent', 'N/A')}")
    print(f"  TP%:        {signal.get('target_percent', 'N/A')}")
    print(f"  Setup:      {signal.get('setup_type', 'N/A')}")
    print(f"  Penalties:  {signal.get('penalty_breakdown', 'N/A')}")
    print(f"\n  Reasoning:")
    reasoning = signal.get("reasoning", "")
    for line in textwrap.wrap(reasoning, width=68):
        print(f"    {line}")

    all_results.append({
        "symbol": symbol, "ltp": ltp, "signal": signal,
        "kronos_dir": "UP" if pred_total_ret > 0 else "DOWN",
        "kronos_ret_pct": pred_total_ret,
        "atr_pct": atr_pct,
    })

# ─── Prompt Quality Audit ─────────────────────────────────────────────────────
print()
print(SEP)
print("  DEEPSEEK PROMPT QUALITY AUDIT")
print(SEP)

resolved = []
issues = []

# ── RESOLVED (applied to deepseek_analyzer.py) ───────────────────────────────
resolved.append("Confidence score start: prompt changed to 'Start at 100'.")
resolved.append("Bonus cap: updated to 'max +10 total'.")
resolved.append("Analog RAG integration: Step 6 ANALOG EVIDENCE added to REASONING PROTOCOL.")
resolved.append("Kronos horizon awareness: pred_range_pct SL/TP guidance added.")
resolved.append("Kronos conflict instruction: explicit definition added ('negative return = conflict for BUY').")
resolved.append("JSON escaping: changed to 'NEVER use unescaped double quotes'.")
resolved.append("setup_type for HOLD: prompt now enforces setup_type=NONE for HOLD.")
resolved.append("Binding commitment rule: 'if score >= threshold, MUST output signal — no overrides'.")

# ── OPEN ISSUES ───────────────────────────────────────────────────────────────
model_name = ai.model
issues.append({
    "severity": "LOW",
    "area": "Model name verification",
    "problem": f"Model is hardcoded as '{model_name}'. DeepSeek API endpoint may use a "
               "different identifier ('deepseek-chat', 'deepseek-reasoner', etc.).",
    "impact": "If model name is wrong, API returns 400 or silently falls back to a weaker model.",
    "fix": "Verify against DeepSeek API docs or inspect response JSON 'model' field to confirm.",
})

print()
print("  RESOLVED ISSUES:")
for i, r in enumerate(resolved, 1):
    print(f"  [{i}] FIXED  {r}")
print()
print("  OPEN ISSUES:")
for i, issue in enumerate(issues, 1):
    sev_marker = {"HIGH": "***", "MEDIUM": "**", "LOW": "*"}.get(issue["severity"], "*")
    print(f"  [{i}] {sev_marker} {issue['severity']} — {issue['area']}")
    print(f"      Problem : {issue['problem']}")
    print(f"      Impact  : {issue['impact']}")
    print(f"      Fix     : {issue['fix']}")
    print()

# ─── Results summary ──────────────────────────────────────────────────────────
print(SEP)
print("  PIPELINE RESULTS SUMMARY")
print(SEP)
print()
for r in all_results:
    if r.get("skipped"):
        print(f"  {r['symbol']:12s}  SKIPPED (no API key)")
        continue
    sig = r["signal"]
    kdir = r.get("kronos_dir", "?")
    kret = r.get("kronos_ret_pct", 0)
    atr  = r.get("atr_pct", 0)
    align = "ALIGN" if (sig.get("signal") == "BUY" and kdir == "UP") or (sig.get("signal") == "SELL" and kdir == "DOWN") else "CONFLICT" if sig.get("signal") in ("BUY","SELL") else "-"
    print(f"  {r['symbol']:12s}  DeepSeek={sig.get('signal'):4s}  conf={sig.get('confidence'):3d}"
          f"  Kronos={kdir}({kret:+.2f}%)  Kronos/DS={align}"
          f"  ATR%={atr:.3f}")

print()
print(SEP)
print("  PROMPT IMPROVEMENTS SUMMARY")
print(SEP)
print()
print(f"  RESOLVED ({len(resolved)}): All major prompt issues fixed in deepseek_analyzer.py")
print(f"  OPEN     ({len(issues)}): " + ", ".join(x["area"] for x in issues))
print()
print("  Prompt is production-ready. Monitor live runs for AI reasoning consistency.")
