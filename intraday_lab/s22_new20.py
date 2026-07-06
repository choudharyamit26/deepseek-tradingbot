"""s22 transfer test for the 20 symbols added to the live runner 2026-07-06.

Frozen s22 params (no re-optimization), study window 2025-07-04..2026-07-03.
14/20 already have per-symbol results in results/s22_breadth.json; this fetches
and runs the 6 that were outside the breadth candidate map (JINDALSTEL, SUZLON,
IRFC, DLF, BSE, ANGELONE), then reports the combined new-20 verdict.
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
log = logging.getLogger("new20")

PARAMS = {"g": 0.8, "sl_atr": 3.0, "tp_atr": 99.0, "max_hold": 75}
NEW20 = ["TATAMOTORS", "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL",
         "SAIL", "ADANIPOWER", "ADANIGREEN", "TATAPOWER", "SUZLON", "RVNL",
         "IRFC", "HAL", "BEL", "RECLTD", "PFC", "DLF", "BSE", "ANGELONE"]
# sids for names missing from the repo maps (Dhan scrip master, 2026-07-06)
EXTRA_SIDS = {"JINDALSTEL": "6733", "SUZLON": "12018", "IRFC": "2029",
              "DLF": "14732", "BSE": "19585", "ANGELONE": "324"}

breadth = {r["symbol"]: r
           for r in json.loads((cfg.RESULTS / "s22_breadth.json").read_text())["per_symbol"]}
strat = next(s for s in REGISTRY if s.name == "s22_gap_go_h2c")

rows, new_trades = [], []
for sym in NEW20:
    if sym in breadth:
        rows.append({**breadth[sym], "source": "breadth"})
        continue
    sid = EXTRA_SIDS[sym]
    p = cfg.STORE / f"{sym}_5min_ext.parquet"
    try:
        if p.exists():
            df = pd.read_parquet(p)
        else:
            df = fetcher.fetch_intraday(sid, cfg.START, cfg.END)
            if len(df):
                df.to_parquet(p)
        if len(df) < 5000:
            log.warning("%s: only %d bars, skipped", sym, len(df))
            continue
        d = backtester.prepare(df)
        t = backtester.run_strategy(strat, {sym: d}, PARAMS, {})
        m = metrics.compute(t)
        rows.append({"symbol": sym, **m, "source": "fresh"})
        new_trades.append(t)
        log.info("%-12s n=%3d pf=%5.2f net=%+9.0f", sym, m["trades"], m["pf"], m["net"])
    except Exception as e:
        log.warning("%s: %s", sym, e)

print(f"\n{'symbol':<12} {'src':<8} {'n':>4} {'wr%':>6} {'pf':>6} {'net':>9} {'exp%':>8}")
for r in sorted(rows, key=lambda r: -r["net"]):
    print(f"{r['symbol']:<12} {r['source']:<8} {r['trades']:>4} {r['wr']:>6.1f} "
          f"{r['pf']:>6.2f} {r['net']:>9.0f} {r['exp_pct']:>8.4f}")

n = sum(r["trades"] for r in rows)
net = sum(r["net"] for r in rows)
gross_win = sum(r["avg_win"] * r["trades"] * r["wr"] / 100 for r in rows)
gross_loss = sum(-r["avg_loss"] * r["trades"] * (1 - r["wr"] / 100) for r in rows)
pf = gross_win / gross_loss if gross_loss else float("nan")
pos = sum(1 for r in rows if r["net"] > 0)
print(f"\nNEW-20 TRANSFER: {len(rows)} symbols | n={n} net={net:+.0f} "
      f"pf~{pf:.2f} | profitable symbols: {pos}/{len(rows)}")
(cfg.RESULTS / "s22_new20.json").write_text(
    json.dumps({"params": PARAMS, "per_symbol": rows,
                "aggregate": {"trades": n, "net": net, "pf_approx": round(pf, 2),
                              "profitable_symbols": f"{pos}/{len(rows)}"}},
               indent=2, default=str))
print("saved results/s22_new20.json")
