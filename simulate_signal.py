"""
simulate_signal.py — Replay the full signal pipeline on saved June 12 candle data.

Uses NTPC (security_id=11630) with Thursday's saved 3m/15m/60m CSVs.
Slices bars up to 11:30 AM to replay what the bot would have seen at that scan.

Stages printed:
  [1] Candle data loaded
  [2] Technical indicators (3m / 15m / 60m)
  [3] Pre-filter gate
  [4] Analog RAG query
  [5] DeepSeek AI signal
  [6] Post-signal gates (MTF, RSI veto, reversal)
  [7] Risk: position sizing
  [8] Final decision (ENTER / SKIP)

Run:  python simulate_signal.py
"""
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("simulate")

# ── Parameters ────────────────────────────────────────────────────────────────
SYMBOL      = "NTPC"
SECURITY_ID = "11630"
DATE        = "2026-06-12"
SIGNAL_TIME = "11:30"          # slice bars up to this time
CAPITAL     = 100_000.0

DATA_DIR   = ROOT / "kronos_integrated_bot" / "data" / DATE
DB_PATH    = ROOT / "analog_history.db"
DIVIDER    = "-" * 72

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_candles(interval: str) -> pd.DataFrame:
    path = DATA_DIR / f"{SECURITY_ID}_{interval}.csv"
    if not path.exists():
        print(f"  [WARN] candle file missing: {path.name}")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    cutoff = pd.Timestamp(f"{DATE} {SIGNAL_TIME}:59", tz="Asia/Kolkata")
    df = df[df.index <= cutoff]
    return df


def sep(title: str = ""):
    if title:
        print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")
    else:
        print(DIVIDER)


def fmt_ind(ind: dict) -> str:
    if not ind:
        return "  (no indicators)"
    keys = ["rsi", "adx", "volume_ratio", "mfi", "atr", "vwap",
            "close", "sma_20", "vwap_distance_pct"]
    lines = []
    for k in keys:
        v = ind.get(k)
        if v is not None:
            lines.append(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    return "\n".join(lines)

# ── Main simulation ───────────────────────────────────────────────────────────

class MockRisk:
    min_confidence = 80
    current_capital = CAPITAL
    cash_buffer_pct = 20.0
    def check_daily_trade_limit(self): return True
    def check_daily_loss_limit(self): return True
    def check_cash_buffer(self, d): return True
    def calculate_position_size(self, cap, sl_pct, price):
        if sl_pct <= 0 or price <= 0: return 0
        risk_amt = cap * 0.02
        rps = price * sl_pct / 100
        qty = int(risk_amt / rps) if rps > 0 else 0
        return min(qty, int(cap / price))


async def simulate():
    # ─────────────────────────────────────────────────────────────────────────
    sep(f"[1]  CANDLE DATA — {SYMBOL} ({DATE}  up to {SIGNAL_TIME})")
    # ─────────────────────────────────────────────────────────────────────────
    df_3m  = load_candles("3minute")
    df_15m = load_candles("15minute")
    df_60m = load_candles("60minute")

    for label, df in [("3min", df_3m), ("15min", df_15m), ("60min", df_60m)]:
        print(f"  {label:6s}: {len(df)} bars  "
              f"last={df.index[-1].strftime('%H:%M') if len(df) else 'N/A'}  "
              f"close={df['close'].iloc[-1]:.2f}" if len(df) else f"  {label}: 0 bars")

    if len(df_3m) < 20:
        print("  ERROR: insufficient 3m bars (<20). Aborting.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    sep("[2]  TECHNICAL INDICATORS")
    # ─────────────────────────────────────────────────────────────────────────
    from indicators import calculate_technical_indicators

    ind_3m  = calculate_technical_indicators(df_3m)
    ind_15m = calculate_technical_indicators(df_15m) if len(df_15m) >= 10 else {}
    ind_60m = calculate_technical_indicators(df_60m) if len(df_60m) >= 5 else {}

    print("\n  -- 3-minute --")
    print(fmt_ind(ind_3m))
    print("\n  -- 15-minute --")
    print(fmt_ind(ind_15m))
    print("\n  -- 60-minute --")
    print(fmt_ind(ind_60m))

    # ─────────────────────────────────────────────────────────────────────────
    sep("[3]  PRE-FILTER GATE")
    # ─────────────────────────────────────────────────────────────────────────
    from stock_trading_bot import IntradayStockBot

    # Thursday context: BEARISH market, NIFTY ENERGY=NEUTRAL
    regime_data = {
        "nifty": {"trend": "BEARISH", "rsi": 45, "adx": 30},
        "sector_name": "NIFTY ENERGY",
        "sector": {"trend": "NEUTRAL"},
    }

    # Call as unbound method (self not used in body — it reads module-level constants)
    passed, reason = IntradayStockBot._passes_prefilter(None, ind_3m, regime_data)
    status = "PASS [OK]" if passed else "FAIL [X]"
    print(f"\n  Pre-filter: {status}")
    print(f"  Reason    : {reason}")

    if not passed:
        print("\n  Pipeline stops here — pre-filter rejected this bar.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    sep("[4]  ANALOG RAG QUERY")
    # ─────────────────────────────────────────────────────────────────────────
    from analog_rag import AnalogRAG

    rag = AnalogRAG(db_path=DB_PATH)
    atr_pct_rag = (ind_3m.get("atr", 1) / ind_3m.get("close", 350) * 100
                   if ind_3m.get("close") else 0.3)
    analog_text = rag.query_similar(
        indicators=ind_3m,
        nifty_trend="BEARISH",
        market_regime="bearish",
        signal_type="SELL",
    )
    if analog_text:
        print(f"\n  Analog setups found:\n{analog_text}")
    else:
        print("\n  No analog setups returned (< 3 entries or poor similarity).")

    # ─────────────────────────────────────────────────────────────────────────
    sep("[5]  DEEPSEEK AI SIGNAL")
    # ─────────────────────────────────────────────────────────────────────────
    from deepseek_analyzer import DeepSeekStockAnalyzer
    import os

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("  ERROR: DEEPSEEK_API_KEY not set in .env. Skipping AI call.")
        return

    ai = DeepSeekStockAnalyzer(api_key=api_key, min_confidence=80)

    ltp   = df_3m["close"].iloc[-1]
    high  = df_3m["high"].iloc[-1]
    low   = df_3m["low"].iloc[-1]
    vol   = df_3m["volume"].iloc[-1]
    avg_v = df_3m["volume"].tail(5).mean()

    market_data = {
        "ltp": ltp,
        "high_3m": high,
        "low_3m": low,
        "volume": vol,
        "avg_volume_3m": avg_v,
    }

    # Build the same MTF summary string that the bot builds
    def _tf_trend(ind):
        c = ind.get("close", 0); s = ind.get("sma_20", c)
        return "BULLISH" if c > s else "BEARISH" if c < s else "NEUTRAL"

    regime_ctx = (
        "MARKET REGIME: BEARISH (Nifty below SMA20, RSI=45, ADX=30)\n"
        "SECTOR: NIFTY ENERGY = NEUTRAL\n"
    )
    if analog_text:
        regime_ctx += "\n" + analog_text

    mtf_parts = []
    for label, ind in [("3m", ind_3m), ("15m", ind_15m), ("1h", ind_60m)]:
        if ind:
            trend = _tf_trend(ind)
            mtf_parts.append(
                f"{label}: {trend}  RSI={ind.get('rsi',0):.1f}  "
                f"ADX={ind.get('adx',0):.1f}  "
                f"close={ind.get('close',0):.2f}  "
                f"SMA20={ind.get('sma_20',0):.2f}"
            )
    if mtf_parts:
        regime_ctx += "\nMTF CONTEXT:\n" + "\n".join(mtf_parts)

    recent_bars = df_3m.tail(10)

    print(f"\n  Calling DeepSeek... ({SYMBOL}  LTP={ltp:.2f})")
    signal = await ai.get_trading_signal(SYMBOL, market_data, ind_3m,
                                          regime_ctx, recent_bars=recent_bars)

    sig_type   = signal.get("signal", "HOLD")
    confidence = signal.get("confidence", 0)
    reasoning  = signal.get("reasoning", "")
    sl_pct     = signal.get("stop_loss_percent", 1.2)
    tp_pct     = signal.get("target_percent", 3.0)
    setup_type = signal.get("setup_type", "")

    print(f"\n  Signal     : {sig_type}")
    print(f"  Confidence : {confidence}")
    print(f"  Setup type : {setup_type}")
    print(f"  SL %       : {sl_pct}")
    print(f"  Target %   : {tp_pct}")
    print(f"\n  Reasoning  :\n{reasoning}")

    # ─────────────────────────────────────────────────────────────────────────
    sep("[6]  POST-SIGNAL GATES")
    # ─────────────────────────────────────────────────────────────────────────
    if sig_type not in ("BUY", "SELL"):
        print(f"\n  Signal is {sig_type} — no further checks needed. SKIP.")
        sep("[8]  FINAL DECISION")
        print(f"\n  RESULT: SKIP (AI said {sig_type}, conf={confidence})")
        return

    if confidence < 80:
        print(f"\n  Confidence {confidence} < 80 — SKIP.")
        sep("[8]  FINAL DECISION")
        print(f"\n  RESULT: SKIP (confidence below threshold)")
        return

    # MTF alignment check
    mtf_ok, mtf_reason = IntradayStockBot._validate_mtf_alignment(sig_type, ind_3m, ind_15m, ind_60m)
    print(f"\n  MTF alignment : {'PASS [OK]' if mtf_ok else 'FAIL [X]'}  ({mtf_reason})")

    # RSI veto
    rsi_3m = ind_3m.get("rsi", 50)
    RSI_OB, RSI_OS = 70, 30
    rsi_ok = True
    if sig_type == "BUY" and rsi_3m >= RSI_OB:
        rsi_ok = False
        print(f"  RSI OB veto   : FAIL [X]  RSI={rsi_3m:.1f} >= {RSI_OB}")
    elif sig_type == "SELL" and rsi_3m <= RSI_OS:
        rsi_ok = False
        print(f"  RSI OS veto   : FAIL [X]  RSI={rsi_3m:.1f} <= {RSI_OS}")
    else:
        print(f"  RSI veto      : PASS [OK]  RSI={rsi_3m:.1f} (BUY OB>{RSI_OB} / SELL OS<{RSI_OS})")

    # Sector regime penalty
    sector_trend = regime_data.get("sector", {}).get("trend", "").upper()
    conflict = (sig_type == "SELL" and sector_trend == "BULLISH") or \
               (sig_type == "BUY" and sector_trend == "BEARISH")
    if conflict:
        penalty_conf = confidence - 5
        print(f"  Sector penalty: conf {confidence}→{penalty_conf} ({sig_type} vs ENERGY={sector_trend})")
        confidence = penalty_conf
    else:
        print(f"  Sector check  : PASS [OK]  no conflict ({sig_type} vs ENERGY={sector_trend})")

    if not mtf_ok or not rsi_ok or confidence < 80:
        sep("[8]  FINAL DECISION")
        reasons = []
        if not mtf_ok: reasons.append("MTF misaligned")
        if not rsi_ok: reasons.append("RSI veto")
        if confidence < 80: reasons.append(f"confidence {confidence} < 80")
        print(f"\n  RESULT: SKIP  ({'; '.join(reasons)})")
        return

    # ─────────────────────────────────────────────────────────────────────────
    sep("[7]  RISK — POSITION SIZING")
    # ─────────────────────────────────────────────────────────────────────────
    risk = MockRisk()

    # ATR-based SL/TP
    atr_val = ind_3m.get("atr", 1)
    atr_pct_val = atr_val / ltp * 100
    sl_price  = ltp * (1 - sl_pct / 100) if sig_type == "BUY" else ltp * (1 + sl_pct / 100)
    tp_price  = ltp * (1 + tp_pct / 100) if sig_type == "BUY" else ltp * (1 - tp_pct / 100)
    trail_sl  = ltp - (2.0 * atr_val)     if sig_type == "BUY" else ltp + (2.0 * atr_val)
    rr_ratio  = tp_pct / sl_pct if sl_pct > 0 else 0

    qty = risk.calculate_position_size(CAPITAL, sl_pct, ltp)
    position_val = qty * ltp
    risk_amt     = qty * ltp * sl_pct / 100

    print(f"\n  Capital       : ₹{CAPITAL:,.0f}")
    print(f"  Risk/trade    : 2%  →  ₹{CAPITAL*0.02:,.0f}")
    print(f"  LTP           : ₹{ltp:.2f}")
    print(f"  ATR (3m)      : {atr_val:.2f}  ({atr_pct_val:.3f}%)")
    print(f"  SL %          : {sl_pct:.2f}%  →  ₹{sl_price:.2f}")
    print(f"  Target %      : {tp_pct:.2f}%  →  ₹{tp_price:.2f}")
    print(f"  Trailing SL   : ₹{trail_sl:.2f}")
    print(f"  R:R ratio     : {rr_ratio:.2f}x  {'✓' if rr_ratio >= 1.8 else '✗ (need 1.8x)'}")
    print(f"  Quantity      : {qty} shares")
    print(f"  Position value: ₹{position_val:,.2f}  ({position_val/CAPITAL*100:.1f}% of capital)")
    print(f"  Max risk      : ₹{risk_amt:.2f}  ({risk_amt/CAPITAL*100:.2f}% of capital)")
    print(f"  Cash buffer   : 20% kept undeployed  →  max deployable ₹{CAPITAL*0.8:,.0f}")

    # ─────────────────────────────────────────────────────────────────────────
    sep("[8]  FINAL DECISION")
    # ─────────────────────────────────────────────────────────────────────────
    if qty < 1:
        print(f"\n  RESULT: SKIP  (quantity=0, price too high for capital)")
        return

    if rr_ratio < 1.8:
        print(f"\n  RESULT: SKIP  (R:R={rr_ratio:.2f}x < 1.8x minimum)")
        return

    mode = "WOULD PLACE ORDER (DRY-RUN)"
    print(f"\n  RESULT: ENTER {sig_type}  [{mode}]")
    print(f"\n  Order summary:")
    print(f"    Symbol    : {SYMBOL}")
    print(f"    Direction : {sig_type}")
    print(f"    Quantity  : {qty}")
    print(f"    Entry     : ₹{ltp:.2f}")
    print(f"    Stop loss : ₹{sl_price:.2f}  ({sl_pct:.2f}%)")
    print(f"    Target    : ₹{tp_price:.2f}  ({tp_pct:.2f}%)")
    print(f"    Trail SL  : ₹{trail_sl:.2f}")
    print(f"    Confidence: {confidence}")
    print(f"    Setup     : {setup_type}")
    sep()


if __name__ == "__main__":
    asyncio.run(simulate())
