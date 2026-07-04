"""s22 breadth test: frozen params on ALL other mapped universe stocks
(study window 2025-07-04..2026-07-03), no re-optimization."""
import json, logging, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from data import fetcher
from data.universe import candidate_map
from engine import backtester, metrics
from strategies import REGISTRY
from run import UNIVERSE_JSON

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("breadth")
PARAMS = {"g": 0.8, "sl_atr": 3.0, "tp_atr": 99.0, "max_hold": 75}

core = {u["symbol"] for u in json.loads(UNIVERSE_JSON.read_text())}
cands = {s: sid for s, sid in candidate_map().items() if s not in core}
strat = next(s for s in REGISTRY if s.name == "s22_gap_go_h2c")

rows, all_t = [], []
for sym, sid in sorted(cands.items()):
    p = cfg.STORE / f"{sym}_5min_ext.parquet"
    try:
        if p.exists():
            df = pd.read_parquet(p)
        else:
            df = fetcher.fetch_intraday(sid, cfg.START, cfg.END)
            if len(df):
                df.to_parquet(p)
        if len(df) < 5000:
            continue
        d = backtester.prepare(df)
        t = backtester.run_strategy(strat, {sym: d}, PARAMS, {})
        m = metrics.compute(t)
        rows.append({"symbol": sym, **m})
        all_t.append(t)
        log.info("%-12s n=%3d pf=%5.2f net=%+9.0f", sym, m["trades"], m["pf"], m["net"])
    except Exception as e:
        log.warning("%s: %s", sym, e)

agg = metrics.compute(pd.concat(all_t, ignore_index=True)) if all_t else {}
out = {"aggregate": agg, "per_symbol": rows}
(cfg.RESULTS / "s22_breadth.json").write_text(json.dumps(out, indent=2, default=str))
pos = sum(1 for r in rows if r["net"] > 0)
print(f"\nS22 BREADTH: {len(rows)} new symbols | aggregate: n={agg.get('trades')} "
      f"pf={agg.get('pf')} net={agg.get('net')} sharpe={agg.get('sharpe')} | "
      f"profitable symbols: {pos}/{len(rows)}")
