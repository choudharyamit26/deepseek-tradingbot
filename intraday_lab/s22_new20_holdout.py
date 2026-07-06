"""2024 holdout for the 5 profitable new-20 symbols (VEDL, TATAMOTORS, RECLTD,
RVNL, PFC — selected on 2025-26 study data in s22_new20.py).

True out-of-sample for this selection: 2024-01-01..2025-07-03, frozen s22
params, run ONCE. Pre-registered keep rule: holdout PF >= 1.0 AND n >= 20.
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data import fetcher
from engine import backtester, metrics
from strategies import REGISTRY

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("new20_holdout")

H_START, H_END = "2024-01-01", "2025-07-03"
PARAMS = {"g": 0.8, "sl_atr": 3.0, "tp_atr": 99.0, "max_hold": 75}
CANDIDATES = {"VEDL": "3063", "TATAMOTORS": "759782", "RECLTD": "15355",
              "RVNL": "9552", "PFC": "14299"}

strat = next(s for s in REGISTRY if s.name == "s22_gap_go_h2c")
rows, all_t = [], []
print(f"{'symbol':<12} {'n':>4} {'wr%':>6} {'pf':>6} {'net':>9} {'exp%':>8}  verdict")
for sym, sid in CANDIDATES.items():
    p = cfg.STORE / f"{sym}_5min_new20_holdout.parquet"
    try:
        if p.exists():
            df = pd.read_parquet(p)
        else:
            df = fetcher.fetch_intraday(sid, H_START, H_END)
            if len(df):
                df.to_parquet(p)
        if len(df) < 5000:
            print(f"{sym:<12} insufficient data ({len(df)} bars)")
            continue
        d = backtester.prepare(df)
        t = backtester.run_strategy(strat, {sym: d}, PARAMS, {}, H_START, H_END)
        m = metrics.compute(t)
        keep = m["pf"] >= 1.0 and m["trades"] >= 20
        rows.append({"symbol": sym, **m, "keep": keep})
        all_t.append(t)
        print(f"{sym:<12} {m['trades']:>4} {m['wr']:>6.1f} {m['pf']:>6.2f} "
              f"{m['net']:>9.0f} {m['exp_pct']:>8.4f}  {'KEEP' if keep else 'drop'}")
    except Exception as e:
        log.warning("%s: %s", sym, e)

if all_t:
    agg = metrics.compute(pd.concat(all_t, ignore_index=True))
    print(f"\naggregate: n={agg['trades']} pf={agg['pf']} net={agg['net']:+.0f} "
          f"sharpe={agg['sharpe']}")
(cfg.RESULTS / "s22_new20_holdout.json").write_text(
    json.dumps({"window": [H_START, H_END], "params": PARAMS,
                "keep_rule": "pf>=1.0 and n>=20", "per_symbol": rows},
               indent=2, default=str))
print("saved results/s22_new20_holdout.json")
