"""True out-of-sample holdout: 2024-01-01 .. 2025-07-03 (pre-study data).

Pre-registered candidates (named BEFORE this test existed, from WF gates):
  s22_gap_go_h2c, s25_inside_day_break_h2c — run ONCE with the exact frozen
  params their 2025-26 IS optimization chose. No re-optimization, no selection.
Controls s01-s06 run alongside for baseline context (expected to lose — they
tell us the holdout period has the same cost-frontier character, not which
strategy to pick).
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data import fetcher
from data.universe import candidate_map
from engine import backtester, metrics
from strategies import REGISTRY
from run import build_xs, UNIVERSE_JSON

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("holdout")

H_START, H_END = "2024-01-01", "2025-07-03"
CANDIDATES = ["s22_gap_go_h2c", "s25_inside_day_break_h2c"]
CONTROLS = [s.name for s in REGISTRY if s.name < "s07"]


def fetch_holdout():
    cands = candidate_map()
    uni = [u["symbol"] for u in json.loads(UNIVERSE_JSON.read_text())]
    data = {}
    for sym in uni:
        p = cfg.STORE / f"{sym}_5min_holdout.parquet"
        if p.exists():
            data[sym] = pd.read_parquet(p)
            continue
        sid = cands.get(sym)
        if not sid:
            log.warning("%s: no sid", sym)
            continue
        df = fetcher.fetch_intraday(sid, H_START, H_END)
        if len(df):
            df.to_parquet(p)
            data[sym] = df
            log.info("fetched %-12s %6d bars %s -> %s", sym, len(df),
                     df.index[0].date(), df.index[-1].date())
        else:
            log.warning("%s: NO holdout data", sym)
    np_ = cfg.STORE / "NIFTY_5min_holdout.parquet"
    if np_.exists():
        nifty = pd.read_parquet(np_)
    else:
        nifty = fetcher.fetch_intraday(cfg.NIFTY_SID, H_START, H_END,
                                       segment="IDX_I", instrument="INDEX")
        if len(nifty):
            nifty.to_parquet(np_)
    return data, nifty


def main():
    raw, nifty = fetch_holdout()
    data = {s: backtester.prepare(d) for s, d in raw.items() if len(d) > 1000}
    ctx = {"xs": build_xs(data),
           "prices": pd.DataFrame({s: d["close"] for s, d in data.items()}).sort_index()}
    if len(nifty):
        ctx["nifty"] = backtester.prepare(nifty)
    log.info("holdout universe: %d symbols", len(data))

    frozen = {}
    for f in ("validation.json", "validation_b2.json"):
        for r in json.loads((cfg.RESULTS / f).read_text()):
            if r.get("best_params"):
                frozen[r["strategy"]] = r["best_params"]

    print(f"\n{'strategy':30s} {'role':10s} {'n':>5s} {'wr%':>5s} {'pf':>6s} "
          f"{'sharpe':>7s} {'net':>10s} {'exp%':>8s}")
    out = {}
    for name in CANDIDATES + CONTROLS:
        strat = next(s for s in REGISTRY if s.name == name)
        params = frozen.get(name)
        if not params:
            continue
        t = backtester.run_strategy(strat, data, params, ctx, H_START, H_END)
        m = metrics.compute(t)
        role = "CANDIDATE" if name in CANDIDATES else "control"
        out[name] = {"role": role, "params": params, **m}
        print(f"{name:30s} {role:10s} {m['trades']:>5d} {m['wr']:>5.1f} {m['pf']:>6.2f} "
              f"{m['sharpe']:>7.2f} {m['net']:>10.0f} {m['exp_pct']:>8.4f}")
        # yearly consistency for candidates
        if name in CANDIDATES and len(t):
            t["y"] = pd.to_datetime(t["exit_ts"]).dt.year
            for y, g in t.groupby("y"):
                gm = metrics.compute(g.drop(columns="y"))
                print(f"    {y}: n={gm['trades']} pf={gm['pf']} net={gm['net']:.0f}")
    (cfg.RESULTS / "holdout_2024.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nsaved results/holdout_2024.json")


if __name__ == "__main__":
    main()
