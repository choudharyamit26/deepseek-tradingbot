"""
Kronos prediction simulation test.

Loads persisted 15m OHLCV CSVs and runs three tests per stock:

  A  Thin context (5 bars)   -- old bad path below Fix-3 threshold
  B  Normal context (15 bars) -- within a single day
  C  Multi-day context (25 bars, full day1) -- predict first 10 bars of day2

Validates:
  Fix-2: y_timestamps must be 15-min spaced when input is 15m data
  Fix-3: prediction quality degrades sharply below 30 bars context
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from kronos_integrated_bot.kronos_integration import KronosIntegration
from kronos_integrated_bot import config as cfg

DATA_DIR = ROOT / "kronos_integrated_bot" / "data"
DAY1 = "2026-06-10"
DAY2 = "2026-06-11"

STOCKS = [
    ("KOTAKBANK",  "1922"),
    ("DRREDDY",    "881"),
    ("TATAPOWER",  "3426"),
    ("RELIANCE",   "2885"),
    ("JSWSTEEL",   "11723"),
    ("DABUR",      "772"),
    ("COLPAL",     "15141"),
    ("POLYCAB",    "9590"),
]


def load_csv(security_id, date):
    path = DATA_DIR / date / f"{security_id}_15minute.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    return df.sort_index()


def bar_dir(ref, cur):
    return "UP" if cur > ref else "DOWN"


def evaluate(pred_df, truth_df, last_ctx_close):
    n = min(len(pred_df), len(truth_df))
    dir_ok = 0
    mapes, maes = [], []
    ref = last_ctx_close
    for i in range(n):
        p = pred_df["close"].iloc[i]
        a = truth_df["close"].iloc[i]
        mapes.append(abs(p - a) / a * 100)
        maes.append(abs(p - a))
        if bar_dir(ref, p) == bar_dir(ref, a):
            dir_ok += 1
        ref = a
    return {
        "n": n,
        "dir_acc": dir_ok / n if n else 0,
        "dir_ok": dir_ok,
        "mape": float(np.mean(mapes)) if mapes else 0.0,
        "mae":  float(np.mean(maes))  if maes  else 0.0,
    }


def y_gap_minutes(pred_df):
    if len(pred_df) < 2:
        return float("nan")
    return (pred_df.index[1] - pred_df.index[0]).total_seconds() / 60


def print_comparison(pred_df, truth_df, last_close):
    n = min(len(pred_df), len(truth_df))
    ref = last_close
    print(f"    {'#':>2}  {'Time':>5}  {'Pred_C':>8}  {'Actual_C':>8}  {'Err%':>7}  Dir_P  Dir_A  Match")
    print(f"    {'--':>2}  {'-----':>5}  {'--------':>8}  {'--------':>8}  {'-------':>7}  -----  -----  -----")
    for i in range(n):
        p = pred_df["close"].iloc[i]
        a = truth_df["close"].iloc[i]
        t = truth_df.index[i].strftime("%H:%M")
        err = (p - a) / a * 100
        dp = bar_dir(ref, p)
        da = bar_dir(ref, a)
        match = " OK " if dp == da else "MISS"
        print(f"    {i+1:>2}  {t:>5}  {p:>8.2f}  {a:>8.2f}  {err:>+6.2f}%  {dp:>5}  {da:>5}  {match}")
        ref = a


# ---- Load model -------------------------------------------------------
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

print("=" * 70)
print("  KRONOS SIMULATION TEST")
print("=" * 70)
print(f"  Model    : {cfg.KRONOS_MODEL}")
print(f"  Pred len : {cfg.KRONOS_PRED_LEN} candles")
print(f"  Lookback : {cfg.KRONOS_LOOKBACK} bars (max context {cfg.KRONOS_MAX_CONTEXT})")
print(f"  Samples  : {cfg.KRONOS_SAMPLE_COUNT}")
print()
print("Loading Kronos (may download from HuggingFace on first run)...")
ki = KronosIntegration(kronos_cfg)
try:
    ki.load()
except Exception as e:
    print(f"\n[ERROR] Failed to load Kronos: {e}")
    sys.exit(1)

print(f"Kronos loaded on: {ki._predictor.device}\n")

# ---- Run tests --------------------------------------------------------
rows = []

for stock_name, sid in STOCKS:
    day1 = load_csv(sid, DAY1)
    day2 = load_csv(sid, DAY2)

    if day1 is None or len(day1) < 20:
        bars = len(day1) if day1 is not None else 0
        print(f"[SKIP] {stock_name}: only {bars} bars on day1, need 20+")
        continue

    print("=" * 70)
    print(f"  {stock_name}  (id={sid})")
    print(f"  Day1 ({DAY1}): {len(day1)} bars  |  "
          f"Day2 ({DAY2}): {len(day2) if day2 is not None else 'N/A'} bars")

    # -- TEST A: thin context (5 bars) ----------------------------------
    ctx_A   = day1.iloc[:5]
    truth_A = day1.iloc[5:15]

    print(f"\n  [A] THIN CONTEXT  5 bars  (below 30-bar Fix-3 threshold)")
    print(f"      context : {ctx_A.index[0].strftime('%H:%M')} to {ctx_A.index[-1].strftime('%H:%M')} "
          f" last_close={ctx_A['close'].iloc[-1]:.2f}")
    print(f"      truth   : {truth_A.index[0].strftime('%H:%M')} to {truth_A.index[-1].strftime('%H:%M')}")

    pred_A = ki.predict(ctx_A, symbol=f"{stock_name}_A", force=True)
    if pred_A is not None and not pred_A.empty:
        gap = y_gap_minutes(pred_A)
        gap_ok = abs(gap - 15) < 1
        print(f"      Fix-2 y_gap : {gap:.0f}min  "
              f"{'[OK] 15min spacing' if gap_ok else '[WRONG] expected 15min'}")
        m = evaluate(pred_A, truth_A, ctx_A["close"].iloc[-1])
        print_comparison(pred_A, truth_A, ctx_A["close"].iloc[-1])
        print(f"      Dir acc : {m['dir_ok']}/{m['n']} = {m['dir_acc']*100:.0f}%  "
              f"MAPE : {m['mape']:.3f}%  MAE : Rs.{m['mae']:.2f}")
        rows.append({**m, "stock": stock_name, "test": "A-thin(5)", "gap_ok": gap_ok})
    else:
        print("      Kronos returned None")

    # -- TEST B: normal context (15 bars) within same day ---------------
    ctx_B   = day1.iloc[:15]
    truth_B = day1.iloc[15:25]

    if len(truth_B) >= 5:
        print(f"\n  [B] NORMAL CONTEXT  15 bars  (within-day)")
        print(f"      context : {ctx_B.index[0].strftime('%H:%M')} to {ctx_B.index[-1].strftime('%H:%M')} "
              f" last_close={ctx_B['close'].iloc[-1]:.2f}")
        print(f"      truth   : {truth_B.index[0].strftime('%H:%M')} to {truth_B.index[-1].strftime('%H:%M')}")

        pred_B = ki.predict(ctx_B, symbol=f"{stock_name}_B", force=True)
        if pred_B is not None and not pred_B.empty:
            gap = y_gap_minutes(pred_B)
            gap_ok = abs(gap - 15) < 1
            print(f"      Fix-2 y_gap : {gap:.0f}min  "
                  f"{'[OK] 15min spacing' if gap_ok else '[WRONG] expected 15min'}")
            m = evaluate(pred_B, truth_B, ctx_B["close"].iloc[-1])
            print_comparison(pred_B, truth_B, ctx_B["close"].iloc[-1])
            print(f"      Dir acc : {m['dir_ok']}/{m['n']} = {m['dir_acc']*100:.0f}%  "
                  f"MAPE : {m['mape']:.3f}%  MAE : Rs.{m['mae']:.2f}")
            rows.append({**m, "stock": stock_name, "test": "B-normal(15)", "gap_ok": gap_ok})
        else:
            print("      Kronos returned None")
    else:
        print(f"\n  [B] NORMAL CONTEXT  -- skipped (only {len(truth_B)} truth bars)")

    # -- TEST C: multi-day context, cross-day prediction ----------------
    if day2 is not None and len(day2) >= 10:
        ctx_C   = day1           # full day1  (~25 bars)
        truth_C = day2.iloc[:10] # first 10 bars of day2

        print(f"\n  [C] MULTI-DAY CONTEXT  {len(ctx_C)} bars  (cross-day)")
        print(f"      context : all of {DAY1}  last_close={ctx_C['close'].iloc[-1]:.2f}")
        print(f"      truth   : {DAY2}  {truth_C.index[0].strftime('%H:%M')} to "
              f"{truth_C.index[-1].strftime('%H:%M')}")

        pred_C = ki.predict(ctx_C, symbol=f"{stock_name}_C", force=True)
        if pred_C is not None and not pred_C.empty:
            gap = y_gap_minutes(pred_C)
            gap_ok = abs(gap - 15) < 1
            print(f"      Fix-2 y_gap : {gap:.0f}min  "
                  f"{'[OK] 15min spacing' if gap_ok else '[WRONG] expected 15min'}")
            m = evaluate(pred_C, truth_C, ctx_C["close"].iloc[-1])
            print_comparison(pred_C, truth_C, ctx_C["close"].iloc[-1])
            print(f"      Dir acc : {m['dir_ok']}/{m['n']} = {m['dir_acc']*100:.0f}%  "
                  f"MAPE : {m['mape']:.3f}%  MAE : Rs.{m['mae']:.2f}")
            rows.append({**m, "stock": stock_name, "test": "C-multiday(25)", "gap_ok": gap_ok})
        else:
            print("      Kronos returned None")
    else:
        print(f"\n  [C] MULTI-DAY  -- skipped (no day2 data)")

    print()

# ---- Summary ----------------------------------------------------------
if not rows:
    print("No results collected.")
    sys.exit(0)

print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  {'Stock':12s}  {'Test':18s}  {'y_gap':6s}  {'DirAcc':>7}  {'MAPE':>7}  {'MAE':>8}")
print(f"  {'-'*12}  {'-'*18}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}")
for r in rows:
    gs = "[OK]" if r.get("gap_ok") else "[BAD]"
    print(f"  {r['stock']:12s}  {r['test']:18s}  {gs:6s}  "
          f"{r['dir_acc']*100:>6.0f}%  {r['mape']:>6.3f}%  Rs.{r['mae']:>6.2f}")

print()
for ttype in ["A-thin(5)", "B-normal(15)", "C-multiday(25)"]:
    grp = [r for r in rows if r["test"] == ttype]
    if not grp:
        continue
    avg_dir  = np.mean([r["dir_acc"] for r in grp]) * 100
    avg_mape = np.mean([r["mape"]    for r in grp])
    avg_mae  = np.mean([r["mae"]     for r in grp])
    all_ok   = all(r.get("gap_ok") for r in grp)
    print(f"  {ttype:18s}  Fix-2={'ALL OK' if all_ok else 'SOME BAD'}  "
          f"AvgDir={avg_dir:.0f}%  AvgMAPE={avg_mape:.3f}%  AvgMAE=Rs.{avg_mae:.2f}")

thin  = [r for r in rows if "A-thin"    in r["test"]]
multi = [r for r in rows if "C-multiday" in r["test"]]
if thin and multi:
    td, md = np.mean([r["dir_acc"] for r in thin])*100, np.mean([r["dir_acc"] for r in multi])*100
    tm, mm = np.mean([r["mape"]    for r in thin]),     np.mean([r["mape"]    for r in multi])
    print()
    print("  Context quality improvement (thin 5 bars -> multi-day 25 bars):")
    print(f"    Dir acc : {td:.0f}% -> {md:.0f}%   {'BETTER' if md > td else 'WORSE'}")
    print(f"    MAPE    : {tm:.3f}% -> {mm:.3f}%   {'BETTER' if mm < tm else 'WORSE'}")
