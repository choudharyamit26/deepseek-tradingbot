"""Phase 1b: portfolio-level test of the refined PM-momentum hypothesis.

Phase 1 showed: post-2PM DOWN momentum continues next morning (open->10:15);
up momentum only gaps (untradeable after CNC costs). This script tests the
tradeable version as a daily portfolio with full Dhan costs + slippage and
day-clustered statistics.

Rule family (small, fully-reported grid — no cherry-picking):
  signal day t:  M = 14:00->15:25 return <= threshold  (and optional clv<=0.2)
  entry  day t+1: short at open (fill = open * (1 - slip))
  exit           : cover at 10:15 or at 15:25 close (both reported)
  portfolio      : max K positions/day (most negative M first), Rs100k each

Also reports: long-side mirror (expected dead), NIFTY open->10:15 drift
(market-drift attribution), and monthly stability.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from engine import costs  # noqa: E402
from pm_momentum_study import load_recent_universe, build_panel  # noqa: E402

CAPITAL = 100_000.0
MAX_POS = 5


def trade_net_pct(entry_px, exit_px, short=True):
    """Net % return on CAPITAL for one trade incl. slippage + charges."""
    e = costs.slip(entry_px, direction_hurts_up=not short)  # short: sell lower
    x = costs.slip(exit_px, direction_hurts_up=short)       # cover higher
    qty = int(CAPITAL // e)
    if qty <= 0:
        return np.nan
    gross = (e - x) * qty if short else (x - e) * qty
    ch = costs.round_trip(e, x, qty)
    return (gross - ch) / CAPITAL * 100


def run_rule(panel, m_thr, use_clv, exit_col, side="short"):
    if side == "short":
        sel = panel["M"] <= m_thr
        if use_clv:
            sel &= panel["clv"] <= 0.2
    else:
        sel = panel["M"] >= -m_thr
        if use_clv:
            sel &= panel["clv"] >= 0.8
    cand = panel[sel].copy()
    # top-K per day by momentum strength
    cand["rk"] = cand.groupby("next_date")["M"].rank(ascending=(side == "short"))
    cand = cand[cand["rk"] <= MAX_POS]
    cand = cand.dropna(subset=["n_open", exit_col])
    cand["net"] = [trade_net_pct(o, x, short=(side == "short"))
                   for o, x in zip(cand["n_open"], cand[exit_col])]
    cand = cand.dropna(subset=["net"])
    if len(cand) < 30:
        return None, cand
    daily = cand.groupby("next_date")["net"].mean()
    t = daily.mean() / (daily.std() / np.sqrt(len(daily)))
    wins, losses = cand.loc[cand["net"] > 0, "net"], cand.loc[cand["net"] <= 0, "net"]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf
    return {
        "m_thr": m_thr, "clv": use_clv, "exit": exit_col, "side": side,
        "trades": len(cand), "days": len(daily),
        "net_mean_%": cand["net"].mean(), "daily_mean_%": daily.mean(),
        "t_daily": t, "pf": pf, "win_rate": (cand["net"] > 0).mean(),
        "total_R": cand["net"].sum(),
    }, cand


def main():
    frames = load_recent_universe()
    panel = build_panel(frames)
    print(f"Panel: {len(panel)} rows", file=sys.stderr)

    rows, keep = [], {}
    for m_thr in [-0.005, -0.0075, -0.01]:
        for use_clv in [False, True]:
            for exit_col in ["n_px_1015", "n_close"]:
                r, cand = run_rule(panel, m_thr, use_clv, exit_col, "short")
                if r:
                    rows.append(r)
                    keep[(m_thr, use_clv, exit_col)] = cand
    # long mirror, headline cell only (expected dead)
    for exit_col in ["n_px_1015", "n_close"]:
        r, _ = run_rule(panel, -0.0075, True, exit_col, "long")
        if r:
            rows.append(r)

    res = pd.DataFrame(rows)
    print("\n=== Grid (net of full Dhan costs + 2bps slip/side, K=5) ===")
    print(res.round(3).to_string(index=False))

    # NIFTY drift attribution over the same window
    nifty = pd.read_parquet(ROOT / "data/store/NIFTY_5min.parquet")
    if not isinstance(nifty.index, pd.DatetimeIndex):
        nifty.index = pd.to_datetime(nifty.index)
    drift = []
    for day, g in nifty.groupby(nifty.index.date):
        o = g["open"].iloc[0]
        b = g.between_time("10:10", "10:10")
        c = g["close"].iloc[-1]
        if len(b):
            drift.append({"o1015": b["close"].iloc[0] / o - 1,
                          "oc": c / o - 1})
    dr = pd.DataFrame(drift)
    print(f"\nNIFTY mean open->10:15: {dr['o1015'].mean()*100:.3f}%  "
          f"open->close: {dr['oc'].mean()*100:.3f}%  (n={len(dr)} days)")

    # monthly stability of the headline cell
    hl = keep.get((-0.0075, True, "n_px_1015"))
    if hl is not None:
        hl = hl.copy()
        hl["month"] = hl["next_date"].dt.to_period("M")
        m = hl.groupby("month")["net"].agg(["mean", "sum", "count"])
        print("\n=== Monthly (thr=-0.75%, clv, exit 10:15) net % ===")
        print(m.round(3).to_string())

    res.round(4).to_csv(ROOT / "results/pm_momentum_phase1b.csv", index=False)


if __name__ == "__main__":
    main()
