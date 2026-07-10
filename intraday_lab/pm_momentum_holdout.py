"""Phase 2: single pre-registered holdout pass for the PM-momentum short.

Frozen BEFORE this data was touched (see pm_momentum_phase1b.py results):
  PRIMARY : signal M = 14:00->15:25 ret <= -0.75%; short next open, K=5/day
            (most-negative M first), cover at 15:25 close. Rs100k/trade,
            full Dhan costs + 2bps slip/side.
  SECONDARY: same, cover at 10:15.
Exploration-year reference numbers: primary net +0.114%/trade, PF 1.15,
t_daily 1.42 (NOT significant). Holdout decides: same-sign weak-positive =
"real but thin"; flat/negative = kill.

Data: *_holdout.parquet + *_new20_holdout.parquet (2024-01-01 -> 2025-07-03),
~24 symbols — never used in phase 1.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pm_momentum_study import STORE, day_rows, build_panel  # noqa: E402
from pm_momentum_phase1b import run_rule  # noqa: E402


def load_holdout():
    frames = {}
    for p in sorted(STORE.glob("*holdout*.parquet")):
        sym = p.stem.split("_5min")[0]
        if sym == "NIFTY" or sym in frames:
            continue
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        frames[sym] = df
    return frames


def main():
    frames = load_holdout()
    print(f"Holdout symbols ({len(frames)}): {sorted(frames)}", file=sys.stderr)
    panel = build_panel(frames)
    print(f"Panel: {len(panel)} rows, {panel['date'].min().date()} -> "
          f"{panel['date'].max().date()}", file=sys.stderr)

    rows = []
    for exit_col, tag in [("n_close", "PRIMARY exit=close"),
                          ("n_px_1015", "SECONDARY exit=10:15")]:
        r, cand = run_rule(panel, -0.0075, False, exit_col, "short")
        if r is None:
            print(f"{tag}: <30 trades, no result")
            continue
        r["rule"] = tag
        rows.append(r)
        cand = cand.copy()
        cand["month"] = cand["next_date"].dt.to_period("M")
        m = cand.groupby("month")["net"].agg(["mean", "sum", "count"])
        print(f"\n=== {tag} monthly net % ===")
        print(m.round(3).to_string())

    res = pd.DataFrame(rows)
    print("\n=== HOLDOUT RESULT (net of full costs) ===")
    print(res.round(3).to_string(index=False))
    res.round(4).to_csv(ROOT / "results/pm_momentum_holdout.csv", index=False)


if __name__ == "__main__":
    main()
