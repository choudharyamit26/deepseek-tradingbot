"""Universe selection: rank the repo's mapped F&O/liquid names by 1-year daily
beta vs NIFTY, apply liquidity + data-quality screens, lock the top 20."""
import ast
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data import fetcher

logger = logging.getLogger(__name__)
UNIVERSE_JSON = cfg.ROOT / "data" / "universe.json"


def candidate_map():
    """symbol -> security_id from the repo's constant.py + the
    VWAP_RECLAIM_STOCKS dict in dhan_integration.py (parsed via ast so we don't
    import the heavy live-bot module)."""
    sys.path.insert(0, str(cfg.REPO))
    from constant import (FNO_UNIVERSE, ETF_LIQUID, FILTERED_FNO_UNIVERSE,
                          NIFTY50_UNIVERSE)
    m = {**FNO_UNIVERSE, **FILTERED_FNO_UNIVERSE, **NIFTY50_UNIVERSE}
    src = (cfg.REPO / "dhan_integration.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "VWAP_RECLAIM_STOCKS":
            m.update(ast.literal_eval(node.value))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", "") == "VWAP_RECLAIM_STOCKS":
                    m.update(ast.literal_eval(node.value))
    # ETFs are not stocks — exclude
    for etf in ETF_LIQUID:
        m.pop(etf, None)
    return {s: str(sid) for s, sid in m.items()}


def _beta(stock_ret, nifty_ret):
    a, b = stock_ret.align(nifty_ret, join="inner")
    if len(a) < cfg.MIN_SESSIONS - 20:
        return np.nan
    v = np.var(b.values)
    return float(np.cov(a.values, b.values)[0, 1] / v) if v > 0 else np.nan


def build_universe(force=False):
    if UNIVERSE_JSON.exists() and not force:
        return json.loads(UNIVERSE_JSON.read_text())

    cands = candidate_map()
    logger.info("scanning %d candidates for beta/liquidity", len(cands))
    nifty = fetcher.fetch_daily(cfg.NIFTY_SID, cfg.START, cfg.END,
                                segment="IDX_I", instrument="INDEX")
    if nifty.empty:
        raise RuntimeError("NIFTY daily fetch failed")
    nifty_ret = nifty["close"].pct_change().dropna()

    rows = []
    for sym, sid in sorted(cands.items()):
        d = fetcher.fetch_daily(sid, cfg.START, cfg.END)
        if d.empty or len(d) < cfg.MIN_SESSIONS:
            continue
        ret = d["close"].pct_change().dropna()
        jump = float(abs(d["open"].values[1:] / d["close"].values[:-1] - 1).max())
        adv = float((d["close"] * d["volume"]).tail(60).mean())
        beta = _beta(ret, nifty_ret)
        if np.isnan(beta) or adv < cfg.MIN_ADV_RS or jump > cfg.MAX_OVERNIGHT_JUMP:
            continue
        rows.append({"symbol": sym, "sid": sid, "beta": round(beta, 3),
                     "adv_cr": round(adv / 1e7, 1), "sessions": len(d)})
        logger.info("  %-12s beta=%.2f adv=%.0fcr n=%d", sym, beta, adv / 1e7, len(d))

    rows.sort(key=lambda r: -r["beta"])
    top = rows[: cfg.N_STOCKS]
    UNIVERSE_JSON.write_text(json.dumps(top, indent=2))
    logger.info("universe locked: %s", [r["symbol"] for r in top])
    return top


def load_universe():
    return json.loads(UNIVERSE_JSON.read_text())
