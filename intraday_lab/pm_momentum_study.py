"""Event study: does post-2PM momentum carry into the next session?

Hypothesis (user, 2026-07-09): stocks in momentum after 14:00 remain in
momentum next trading session.

Phase 1 (this script): NO strategy parameters, no costs — just conditional
next-day return tables so the hypothesis can be accepted/refined/rejected on
evidence before any backtest is built. Runs only on the recent-year data
(2025-07-04 → 2026-07-03); the 2024-01 → 2025-07 *_holdout* files stay
untouched for phase 2 confirmation.

Definitions
- PM momentum  M   = close(15:25) / close(13:55 bar, i.e. 14:00 price) - 1
- Vol-adjusted Mz  = M / (day ATR proxy: (high-low)/open of day t)
- Next-day outcomes (t+1): gap (close_t -> open), open->close, open->10:15,
  10:15->close, 14:00->close (same window next day)
- All outcomes reported SIGN-ALIGNED with M (positive = continuation).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "data" / "store"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

MAX_SESSION_GAP_DAYS = 5  # skip pairs across suspensions/long gaps


def load_recent_universe():
    """All recent-year files (plain + _ext), excluding NIFTY and holdouts."""
    frames = {}
    for p in sorted(STORE.glob("*.parquet")):
        name = p.stem
        if "holdout" in name or name.startswith("NIFTY"):
            continue
        sym = name.replace("_5min_ext", "").replace("_5min", "")
        if sym in frames:  # prefer plain over ext if both exist (same range)
            continue
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        frames[sym] = df
    return frames


def day_rows(sym, df):
    """One row per (symbol, day) with PM momentum + context features."""
    rows = []
    for day, g in df.groupby(df.index.date):
        g = g.between_time("09:15", "15:25")
        if len(g) < 60:  # partial session, skip
            continue
        try:
            px_1400 = g.between_time("13:55", "13:55")["close"].iloc[0]
        except IndexError:
            continue
        o, c = g["open"].iloc[0], g["close"].iloc[-1]
        hi, lo = g["high"].max(), g["low"].min()
        pm = g.between_time("14:00", "15:25")
        vol_day = g["volume"].sum()
        rows.append({
            "symbol": sym,
            "date": pd.Timestamp(day),
            "open": o, "close": c,
            "day_ret": c / o - 1,
            "day_range": (hi - lo) / o if o else np.nan,
            "M": c / px_1400 - 1,
            "clv": (c - lo) / (hi - lo) if hi > lo else 0.5,  # close location
            "pm_vol_share": pm["volume"].sum() / vol_day if vol_day else np.nan,
            "px_1015": _px_at(g, "10:10"),
        })
    return rows


def _px_at(g, bar_start):
    b = g.between_time(bar_start, bar_start)
    return b["close"].iloc[0] if len(b) else np.nan


def build_panel(frames):
    all_rows = []
    for sym, df in frames.items():
        all_rows.extend(day_rows(sym, df))
    panel = pd.DataFrame(all_rows).sort_values(["symbol", "date"])

    # next-session outcomes
    panel["next_date"] = panel.groupby("symbol")["date"].shift(-1)
    for col in ["open", "close", "px_1015", "day_ret"]:
        panel[f"n_{col}"] = panel.groupby("symbol")[col].shift(-1)
    ok = (panel["next_date"] - panel["date"]).dt.days <= MAX_SESSION_GAP_DAYS
    panel = panel[ok & panel["next_date"].notna()].copy()

    panel["Mz"] = panel["M"] / panel["day_range"].replace(0, np.nan)
    panel["gap"] = panel["n_open"] / panel["close"] - 1
    panel["oc"] = panel["n_close"] / panel["n_open"] - 1
    panel["o_1015"] = panel["n_px_1015"] / panel["n_open"] - 1
    panel["r1015_c"] = panel["n_close"] / panel["n_px_1015"] - 1
    panel["c_c"] = panel["n_close"] / panel["close"] - 1  # close-to-close

    s = np.sign(panel["M"])
    for c in ["gap", "oc", "o_1015", "r1015_c", "c_c"]:
        panel[f"al_{c}"] = panel[c] * s  # aligned: + = continuation
    return panel


def decile_table(panel, key, outcomes):
    q = pd.qcut(panel[key], 10, labels=False, duplicates="drop")
    tab = panel.groupby(q)[outcomes].mean() * 100  # in %
    tab["n"] = panel.groupby(q).size()
    tab[key + "_mean"] = panel.groupby(q)[key].mean() * 100
    return tab


def main():
    frames = load_recent_universe()
    print(f"Loaded {len(frames)} symbols", file=sys.stderr)
    panel = build_panel(frames)
    print(f"Panel: {len(panel)} symbol-days, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}",
          file=sys.stderr)

    out = {}
    raw = ["gap", "oc", "o_1015", "r1015_c", "c_c"]
    aligned = ["al_" + c for c in raw]

    # 1. Deciles of signed M -> raw next-day outcomes (direction visible)
    out["deciles_M_raw"] = decile_table(panel, "M", raw)
    # 2. Deciles of |M| -> aligned outcomes (continuation strength)
    panel["absM"] = panel["M"].abs()
    out["deciles_absM_aligned"] = decile_table(panel, "absM", aligned)
    # 3. Vol-adjusted momentum
    p2 = panel.dropna(subset=["Mz"]).copy()
    out["deciles_Mz_raw"] = decile_table(p2, "Mz", raw)

    # 4. Cross-sectional: each day rank |M|, top decile with sign split
    panel["cs_rank"] = panel.groupby("date")["M"].rank(pct=True)
    top = panel[panel["cs_rank"] >= 0.9]     # strongest up-moves
    bot = panel[panel["cs_rank"] <= 0.1]     # strongest down-moves
    out["cs_top_decile_up"] = top[raw].mean().to_frame("mean_%").T * 100
    out["cs_bot_decile_down"] = bot[raw].mean().to_frame("mean_%").T * 100
    out["cs_top_n"] = pd.DataFrame({"n": [len(top), len(bot)]},
                                   index=["top", "bot"])

    # 5. Interaction: strong |M| + close near extreme (clv confirms)
    strong = panel[panel["absM"] >= panel["absM"].quantile(0.8)]
    conf = strong[((strong["M"] > 0) & (strong["clv"] >= 0.8)) |
                  ((strong["M"] < 0) & (strong["clv"] <= 0.2))]
    out["strong_M"] = strong[aligned].mean().to_frame("mean_%").T * 100
    out["strong_M_clv_confirmed"] = conf[aligned].mean().to_frame("mean_%").T * 100
    out["strong_conf_n"] = pd.DataFrame({"n": [len(strong), len(conf)]},
                                        index=["strong", "confirmed"])

    # 6. t-stats on the headline aligned outcomes for strong-|M| bucket
    stats = {}
    for c in aligned:
        x = conf[c].dropna()
        stats[c] = {"mean_%": float(x.mean() * 100),
                    "t": float(x.mean() / (x.std() / np.sqrt(len(x)))),
                    "n": int(len(x)),
                    "win_rate": float((x > 0).mean())}
    out_json = {"strong_clv_confirmed_stats": stats,
                "panel_days": int(panel["date"].nunique()),
                "panel_rows": int(len(panel)),
                "symbols": int(panel["symbol"].nunique())}

    with open(OUT / "pm_momentum_phase1.json", "w") as f:
        json.dump(out_json, f, indent=2)

    lines = ["# PM-momentum phase-1 event study (recent year, no costs)\n"]
    for name, tab in out.items():
        lines.append(f"## {name}\n")
        lines.append("```\n" + tab.round(3).to_string() + "\n```")
        lines.append("")
    lines.append("## strong_clv_confirmed_stats\n")
    lines.append("```\n" + pd.DataFrame(out_json["strong_clv_confirmed_stats"]).T
                 .round(3).to_string() + "\n```")
    (OUT / "pm_momentum_phase1.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
