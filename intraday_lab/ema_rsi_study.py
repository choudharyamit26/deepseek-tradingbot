"""EMA13/EMA34 + RSI(60-80) study, round 2: VWAP/ADX/Supertrend filters + 3-min.

  python ema_rsi_study.py baseline [5|3]   # exact user spec, no optimization
  python ema_rsi_study.py ablate   [5|3]   # filter on/off table, IS window only
  python ema_rsi_study.py validate [5|3]   # round-1 grid -> OOS -> walk-forward
  python ema_rsi_study.py validate-filt [5|3]  # filtered grid -> OOS -> WF

Baseline = user's literal spec (EMA13/34, RSI 60-80, ATR 1.5 SL / 3.0 TP
brackets, entries from open). Ablation toggles each filter at those frozen
base params on IN-SAMPLE data only (OOS stays untouched until a validate
run). Validation searches the declared grid on IS and judges frozen params
on OOS + 9-fold walk-forward, same survivor criteria as every lab strategy.

3-min bars are resampled Dhan 1-min (data/fetch_3min.py); max_hold is scaled
to keep the same 3-hour clock as the 5-min runs.
"""
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from engine import backtester, metrics
from strategies.ema_rsi import (EmaRsiFilteredLong, EmaRsiFilteredShort,
                                EmaRsiLong, EmaRsiShort)
from validation import report, splits

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("ema_rsi")

BASE = {"fast": 13, "slow": 34, "rsi_lo": 60, "rsi_hi": 80,
        "sl_atr": 1.5, "tp_atr": 3.0, "tp_mode": "atr"}


def load_data(interval="5"):
    uni = json.loads((cfg.ROOT / "data" / "universe.json").read_text())
    data = {}
    for u in uni:
        p = cfg.STORE / f"{u['symbol']}_{interval}min.parquet"
        if p.exists():
            data[u["symbol"]] = backtester.prepare(pd.read_parquet(p))
    logger.info("interval=%smin: loaded %d symbols", interval, len(data))
    return data, {}


def hold_bars(interval):
    """Keep the same 3-hour max-hold clock across timeframes."""
    return int(cfg.MAX_HOLD_BARS * 5 / int(interval))


def baseline(data, ctx, interval):
    (is_s, is_e), (oos_s, oos_e) = splits.is_oos()
    rows = []
    for strat in (EmaRsiLong(), EmaRsiShort()):
        for trig in ("ema_cross", "rsi_entry"):
            p = {**BASE, "trigger": trig, "max_hold": hold_bars(interval)}
            t0 = time.time()
            full = backtester.run_strategy(strat, data, p, ctx)
            dt = time.time() - t0
            m_full = metrics.compute(full)
            m_is = metrics.compute(
                backtester.run_strategy(strat, data, p, ctx, is_s, is_e))
            m_oos = metrics.compute(
                backtester.run_strategy(strat, data, p, ctx, oos_s, oos_e))
            rows.append({"strategy": strat.name, "trigger": trig,
                         "full": m_full, "is": m_is, "oos": m_oos})
            logger.info("%-14s %-9s | full n=%-4d net=%8.0f pf=%.2f shp=%5.2f "
                        "| IS shp=%5.2f | OOS shp=%5.2f  (%.1fs)",
                        strat.name, trig, m_full["trades"], m_full["net"],
                        m_full["pf"], m_full["sharpe"], m_is["sharpe"],
                        m_oos["sharpe"], dt)
    out = cfg.RESULTS / f"ema_rsi_baseline_{interval}min.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    logger.info("baseline written to %s", out)
    return rows


def ablate(data, ctx, interval):
    """Toggle VWAP/ADX/Supertrend at frozen base params — IS window only."""
    (is_s, is_e), _ = splits.is_oos()
    rows = []
    for strat in (EmaRsiLong(), EmaRsiShort()):
        for vw, ath, st in itertools.product((0, 1), (0, 20, 25), (0, 1)):
            p = {**BASE, "trigger": "ema_cross", "use_vwap": vw,
                 "adx_th": ath, "use_st": st, "max_hold": hold_bars(interval)}
            m = metrics.compute(
                backtester.run_strategy(strat, data, p, ctx, is_s, is_e))
            rows.append({"strategy": strat.name, "vwap": vw, "adx": ath,
                         "st": st, **m})
            logger.info("%-14s vwap=%d adx=%-2d st=%d | n=%-4d net=%8.0f "
                        "pf=%4.2f shp=%6.2f wr=%4.1f", strat.name, vw, ath,
                        st, m["trades"], m["net"], m["pf"], m["sharpe"],
                        m["wr"])
    df = pd.DataFrame(rows)
    out = cfg.RESULTS / f"ema_rsi_ablation_{interval}min.csv"
    df.to_csv(out, index=False)
    logger.info("ablation written to %s", out)
    return df


def validate(data, ctx, interval, filtered=False):
    strats = ([EmaRsiFilteredLong(), EmaRsiFilteredShort()] if filtered
              else [EmaRsiLong(), EmaRsiShort()])
    for s in strats:   # pin timeframe-scaled max_hold as a 1-point grid dim
        s.space = {**s.space, "max_hold": [hold_bars(interval)]}
    suffix = "_filt" if filtered else ""
    report.run_all(data, ctx, strats,
                   prefix=f"validation_ema_rsi{suffix}_{interval}min")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    interval = sys.argv[2] if len(sys.argv) > 2 else "5"
    data, ctx = load_data(interval)
    if cmd in ("baseline", "all"):
        baseline(data, ctx, interval)
    if cmd in ("ablate", "all"):
        ablate(data, ctx, interval)
    if cmd in ("validate", "all"):
        validate(data, ctx, interval, filtered=False)
    if cmd in ("validate-filt", "all"):
        validate(data, ctx, interval, filtered=True)
