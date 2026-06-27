"""Quick test: Kronos with 3-minute bar input (the actual bot timeframe)."""
import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from kronos_integrated_bot.kronos_integration import KronosIntegration
from kronos_integrated_bot import config as cfg

DATA_DIR = ROOT / "kronos_integrated_bot" / "data"

def load_csv(sid, date, interval="15minute"):
    path = DATA_DIR / date / f"{sid}_{interval}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    return df.sort_index()

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

print("Loading Kronos...")
ki = KronosIntegration(kronos_cfg)
ki.load()
print(f"Ready on: {ki._predictor.device}\n")

# KOTAKBANK has 3m data from 2026-06-10
day1_3m  = load_csv("1922", "2026-06-10", "3minute")
day1_15m = load_csv("1922", "2026-06-10", "15minute")
day2_15m = load_csv("1922", "2026-06-11", "15minute")

print(f"3m bars (2026-06-10) : {len(day1_3m) if day1_3m is not None else 0}")
print(f"15m bars (2026-06-10): {len(day1_15m) if day1_15m is not None else 0}")
print(f"15m bars (2026-06-11): {len(day2_15m) if day2_15m is not None else 0}")

# ─── Simulate bot situation mid-day: ~60 bars of 3m data available ────────────
# (9:15 to ~12:15 = 3h = 60 bars)
# RELIANCE (2885) has 3m data on both days
# Combine day1 + day2 into one continuous series so we can test
# various context window sizes within-day and across days
day1_3m = load_csv("2885", "2026-06-10", "3minute")
day2_3m = load_csv("2885", "2026-06-11", "3minute")
combined = pd.concat([day1_3m, day2_3m]).sort_index()
combined = combined[~combined.index.duplicated(keep="last")]

print(f"RELIANCE (2885) combined 3m bars: {len(combined)}  "
      f"({combined.index[0].strftime('%Y-%m-%d %H:%M')} to "
      f"{combined.index[-1].strftime('%Y-%m-%d %H:%M')})")
print()

STOCKS = [("RELIANCE-3m", combined)]

for name, data in STOCKS:
    if data is None or len(data) < 40:
        print(f"[SKIP] {name}: only {len(data) if data is not None else 0} bars")
        continue

    print("=" * 65)
    print(f"  {name}  total 3m bars: {len(data)}")

    for ctx_size, label in [(30,  "30 bars = 90min  (early morning threshold)"),
                             (60,  "60 bars = 3h     (mid-morning typical)"),
                             (125, "125 bars = full day1 (cross-day test)")]:
        if ctx_size + 10 > len(data):
            print(f"\n  [{ctx_size} bars] -- skipped (not enough data)")
            continue

        ctx   = data.iloc[:ctx_size]
        truth = data.iloc[ctx_size:ctx_size + 10]
        ltp   = ctx["close"].iloc[-1]

        print(f"\n  [{ctx_size} bars] {label}")
        print(f"    context: {ctx.index[0].strftime('%H:%M')} to "
              f"{ctx.index[-1].strftime('%H:%M')}  last_close={ltp:.2f}")
        print(f"    truth  : {truth.index[0].strftime('%H:%M')} to "
              f"{truth.index[-1].strftime('%H:%M')}")

        pred = ki.predict(ctx, symbol=f"{name}_{ctx_size}", force=True)
        if pred is None or pred.empty:
            print("    Kronos returned None")
            continue

        # Verify y_timestamp gap
        gap = (pred.index[1] - pred.index[0]).total_seconds() / 60
        gap_ok = abs(gap - 3) < 0.5
        print(f"    y_gap   : {gap:.0f}min  {'[OK] 3min' if gap_ok else f'[WRONG] expected 3min, got {gap:.0f}min'}")

        # Show the exact prompt section
        section = ki.build_prompt_section(pred, ltp)
        print(f"    Kronos prompt section:")
        for line in section.split("\n"):
            print(f"      {line}")

        # Evaluate predictions
        n = min(len(pred), len(truth))
        ref = ltp
        dir_ok = 0
        mapes = []
        print(f"    {'#':>2}  {'Time':>5}  {'Pred_C':>8}  {'Act_C':>8}  {'Err%':>7}  Dir_P  Dir_A  Match")
        for i in range(n):
            p = pred["close"].iloc[i]
            a = truth["close"].iloc[i]
            t = truth.index[i].strftime("%H:%M")
            err = (p - a) / a * 100
            dp = "UP" if p > ref else "DOWN"
            da = "UP" if a > ref else "DOWN"
            ok = " OK " if dp == da else "MISS"
            mapes.append(abs(err))
            if dp == da:
                dir_ok += 1
            ref = a
            print(f"    {i+1:>2}  {t:>5}  {p:>8.2f}  {a:>8.2f}  {err:>+6.2f}%  {dp:>5}  {da:>5}  {ok}")
        print(f"    Dir acc: {dir_ok}/{n} = {dir_ok/n*100:.0f}%  MAPE: {np.mean(mapes):.3f}%  "
              f"horizon={n * int(gap):.0f}min")

print()
print("=" * 65)
print("KEY CHECKS:")
print("  1. y_gap should be 3min (not 5min or 15min) for 3m input")
print("  2. Kronos prompt says '10 x 3-min candles = 30 min ahead'")
print("  3. Direction accuracy should improve vs thin context")
