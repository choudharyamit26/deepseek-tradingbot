#!/usr/bin/env python3
"""Feature-separation study -- the validation gate for any new signal.

The 2026-07 one-month post-mortem found that none of the logged entry features
(RSI/ADX/confidence/kronos/regime) separate winners from losers -- outcomes were
a coin flip w.r.t. everything recorded at entry. The lesson: prove separation in
analog_history.db BEFORE a feature earns a live gate.

This script measures how well a feature discriminates winners from losers using
three lenses, none of which can be fooled by a single lucky bucket:

  1. Medians + Mann-Whitney U (distribution-level separation, non-parametric)
  2. Rank AUC  (P(feature ranks a random winner above a random loser); 0.5 = none)
  3. Quantile monotonicity (win-rate & mean-return per feature quartile) -- a real
     edge trends across quartiles, noise zig-zags.

Usage:
    python -m kronos_integrated_bot.feature_study                 # all numeric features
    python -m kronos_integrated_bot.feature_study ofi ofi_trend   # specific ones
    python -m kronos_integrated_bot.feature_study --min-n 40      # require N samples

Exit code is 0 always; this is a report, not a test. Read the verdict column:
GATE-READY features are candidates to trade; everything else is noise so far.
"""
import sqlite3
import sys
import statistics
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "analog_history.db"

# Features worth studying (numeric, entry-time). ofi/ofi_trend are the new ones.
DEFAULT_FEATURES = [
    "ofi", "ofi_trend", "rsi", "adx", "volume_ratio", "mfi", "atr_pct",
    "confidence", "kronos_pred_return", "matrix_score", "analog_wr",
]


def _load(min_n_nonnull=1):
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM setups WHERE pnl_pct IS NOT NULL"
    )]
    conn.close()
    return rows


def _rank_auc(win_vals, loss_vals):
    """P(random winner's feature > random loser's), ties=0.5. 0.5 == no signal.
    Mann-Whitney U / (n1*n2). Reported as max(auc, 1-auc) with a direction flag
    so a feature that separates *negatively* still shows its strength."""
    n1, n2 = len(win_vals), len(loss_vals)
    if n1 == 0 or n2 == 0:
        return None, None
    greater = 0.0
    for w in win_vals:
        for l in loss_vals:
            if w > l:
                greater += 1
            elif w == l:
                greater += 0.5
    auc = greater / (n1 * n2)
    direction = "higher=win" if auc >= 0.5 else "lower=win"
    return max(auc, 1 - auc), direction


def _quartiles(rows, feat):
    vals = [(r[feat], r["pnl_pct"]) for r in rows if r.get(feat) is not None
            and r["pnl_pct"] is not None]
    vals.sort(key=lambda x: x[0])
    n = len(vals)
    if n < 8:
        return None
    q = n // 4
    out = []
    for i in range(4):
        seg = vals[i * q:(i + 1) * q] if i < 3 else vals[i * q:]
        wr = 100 * sum(1 for _, p in seg if p > 0) / len(seg)
        avg = statistics.mean(p for _, p in seg)
        lo, hi = seg[0][0], seg[-1][0]
        out.append((lo, hi, len(seg), wr, avg))
    return out


def _monotonic(quartiles):
    """A crude monotonicity score: fraction of adjacent quartile steps whose
    mean-return moves in a consistent direction. 1.0 = perfectly monotonic."""
    avgs = [q[4] for q in quartiles]
    ups = sum(1 for a, b in zip(avgs, avgs[1:]) if b > a)
    downs = sum(1 for a, b in zip(avgs, avgs[1:]) if b < a)
    return max(ups, downs) / 3.0


def study(feat, rows):
    win = [r[feat] for r in rows if r.get(feat) is not None and r["pnl_pct"] > 0]
    loss = [r[feat] for r in rows if r.get(feat) is not None and r["pnl_pct"] < 0]
    n_nonnull = len([r for r in rows if r.get(feat) is not None])
    print(f"\n{'='*68}\nFEATURE: {feat}   (non-null n={n_nonnull}, win={len(win)}, loss={len(loss)})")
    if len(win) < 3 or len(loss) < 3:
        print("  INSUFFICIENT DATA -- need >=3 winners and >=3 losers with this feature.")
        print("  VERDICT: NO-DATA (populate via dry-run, then re-run)")
        return
    wmed, lmed = statistics.median(win), statistics.median(loss)
    auc, direction = _rank_auc(win, loss)
    print(f"  win median = {wmed:+.4f}   loss median = {lmed:+.4f}   delta = {wmed - lmed:+.4f}")
    print(f"  rank-AUC   = {auc:.3f}  ({direction})   [0.50=no signal, >0.60 promising, >0.65 strong]")
    qs = _quartiles(rows, feat)
    if qs:
        mono = _monotonic(qs)
        print(f"  quartiles (monotonicity={mono:.2f}):")
        for i, (lo, hi, cnt, wr, avg) in enumerate(qs):
            print(f"    Q{i+1} [{lo:+.3f}..{hi:+.3f}] n={cnt:3d}  wr={wr:4.0f}%  mean%={avg:+.3f}")
    else:
        mono = 0.0
        print("  quartiles: insufficient data")
    # Verdict: needs both distribution separation (AUC) AND a monotonic gradient.
    strong = auc is not None and auc >= 0.62 and mono >= 0.66
    weak = auc is not None and auc >= 0.58
    verdict = "GATE-READY" if strong else ("PROMISING" if weak else "NOISE")
    print(f"  VERDICT: {verdict}")


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    feats = args if args else DEFAULT_FEATURES
    rows = _load()
    print(f"Loaded {len(rows)} trades with realized PnL from {DB.name}")
    have_ofi = len([r for r in rows if r.get("ofi") is not None])
    print(f"Rows with OFI populated: {have_ofi}  "
          f"({'accumulating -- dry-run to grow this' if have_ofi < 30 else 'ready to judge'})")
    for f in feats:
        if f not in rows[0].keys():
            print(f"\n(skip {f}: no such column)")
            continue
        study(f, rows)


if __name__ == "__main__":
    main(sys.argv[1:])
