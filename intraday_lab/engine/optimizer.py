"""Grid optimization over a strategy's declared param space — IS data only.

Selection = best Sharpe subject to a minimum trade count AND a parameter
plateau: the best combo's one-step neighbors must average a positive score,
otherwise the peak is treated as noise and the best *plateau* combo wins.
"""
import itertools
import logging

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from engine import backtester, metrics

logger = logging.getLogger(__name__)


def grid(space: dict):
    keys = list(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, combo))


def _neighbors(combo, space):
    out = []
    for k, vals in space.items():
        i = vals.index(combo[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                nb = dict(combo)
                nb[k] = vals[j]
                out.append(tuple(sorted(nb.items())))
    return out


def optimize(strategy, data, ctx, start, end, min_trades=cfg.MIN_IS_TRADES):
    """Returns (best_params, best_row, results list). Score = net Sharpe."""
    results = {}
    rows = []
    for params in grid(strategy.space):
        trades = backtester.run_strategy(strategy, data, params, ctx, start, end)
        m = metrics.compute(trades)
        score = m["sharpe"] if m["trades"] >= min_trades else -99.0
        key = tuple(sorted(params.items()))
        results[key] = score
        rows.append({"params": params, "score": score, **m})

    if not rows:
        return None, None, rows
    rows.sort(key=lambda r: -r["score"])

    # plateau: prefer the highest-scored combo whose neighbors also work
    for r in rows:
        if r["score"] <= -99.0:
            break
        nbs = _neighbors(r["params"], strategy.space)
        nb_scores = [results[k] for k in nbs if k in results]
        if not nb_scores or np.mean(nb_scores) > 0:
            r["plateau"] = round(float(np.mean(nb_scores)), 2) if nb_scores else None
            return r["params"], r, rows
    best = rows[0]
    best["plateau"] = None
    return (best["params"], best, rows) if best["score"] > -99.0 else (None, None, rows)
