"""Pipeline CLI.

  python run.py import-opencode   # reuse the opencode lab's processed candles
  python run.py fetch             # fetch anything missing (incl. NIFTY 5-min)
  python run.py validate          # optimize IS -> OOS -> walk-forward -> report
  python run.py all
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from engine import backtester

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("lab")

OPENCODE = cfg.REPO / "opencode_intraday_startegy_lab"
UNIVERSE_JSON = cfg.ROOT / "data" / "universe.json"


def import_opencode():
    """Adopt the opencode lab's universe + processed 5-min candles (same real
    Dhan data, same window) so results are directly comparable."""
    sel = json.loads((OPENCODE / "dhan_historical_data" / "metadata" /
                      "selected_high_beta_symbols.json").read_text())
    have = []
    for sym in sel:
        f = OPENCODE / "dhan_historical_data" / "processed" / f"{sym}_intraday.csv"
        if not f.exists():
            logger.warning("%s: no intraday csv in opencode lab", sym)
            continue
        df = pd.read_csv(f, parse_dates=["timestamp"], index_col="timestamp")
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        df = df[~df.index.duplicated()]
        df.to_parquet(cfg.STORE / f"{sym}_5min.parquet")
        have.append(sym)
        logger.info("imported %-12s %6d bars  %s -> %s", sym, len(df),
                    df.index[0].date(), df.index[-1].date())
    UNIVERSE_JSON.write_text(json.dumps(
        [{"symbol": s, "source": "opencode"} for s in have], indent=2))
    logger.info("universe: %d symbols (opencode high-beta selection)", len(have))
    return have


def fetch_missing():
    """NIFTY 5-min (for RS strategies) + any universe symbol without bars."""
    from data import fetcher
    sel = json.loads((OPENCODE / "dhan_historical_data" / "metadata" /
                      "selected_high_beta_symbols.json").read_text())
    uni = json.loads(UNIVERSE_JSON.read_text()) if UNIVERSE_JSON.exists() else []
    have = {u["symbol"] for u in uni}
    missing = [s for s in sel if s not in have]
    if missing:
        from data.universe import candidate_map
        cands = candidate_map()
        for sym in missing:
            sid = cands.get(sym)
            if not sid:
                logger.warning("%s: no security id known, skipping", sym)
                continue
            df = fetcher.fetch_and_store(sym, sid)
            if len(df):
                uni.append({"symbol": sym, "source": "dhan"})
                logger.info("fetched %s: %d bars", sym, len(df))
        UNIVERSE_JSON.write_text(json.dumps(uni, indent=2))
    nifty_p = cfg.STORE / "NIFTY_5min.parquet"
    if not nifty_p.exists():
        df = fetcher.fetch_intraday(cfg.NIFTY_SID, cfg.START, cfg.END,
                                    segment="IDX_I", instrument="INDEX")
        if len(df):
            df.to_parquet(nifty_p)
            logger.info("fetched NIFTY 5-min: %d bars", len(df))
        else:
            logger.error("NIFTY 5-min fetch failed — s10_nifty_rs will no-op")


def build_xs(data):
    """Cross-sectional tables (day x symbol) known at the 10:15 bar close:
    return-since-open and opening gap. Used by the s26-s28 strategies."""
    rows = []
    for sym, df in data.items():
        at = df[df["tod"] == 615]
        if not len(at):
            continue
        rows.append(pd.DataFrame({
            "sym": sym, "day": at["day"].values,
            "ret": ((at["close"] / at["day_open"] - 1) * 100).values,
            "gap": ((at["day_open"] / at["prev_close"] - 1) * 100).values}))
    x = pd.concat(rows, ignore_index=True)
    return {"ret1015": x.pivot_table(index="day", columns="sym", values="ret"),
            "gap": x.pivot_table(index="day", columns="sym", values="gap")}


def load_all():
    uni = json.loads(UNIVERSE_JSON.read_text())
    data = {}
    for u in uni:
        p = cfg.STORE / f"{u['symbol']}_5min.parquet"
        if p.exists():
            data[u["symbol"]] = backtester.prepare(pd.read_parquet(p))
    ctx = {}
    nifty_p = cfg.STORE / "NIFTY_5min.parquet"
    if nifty_p.exists():
        ctx["nifty"] = backtester.prepare(pd.read_parquet(nifty_p))
    ctx["xs"] = build_xs(data)
    ctx["prices"] = pd.DataFrame({s: d["close"] for s, d in data.items()}).sort_index()
    logger.info("loaded %d symbols, nifty=%s", len(data), "yes" if "nifty" in ctx else "no")
    return data, ctx


def validate(batch=None):
    from strategies import REGISTRY
    from validation import report
    data, ctx = load_all()
    reg = REGISTRY
    prefix = "validation"
    if batch == "2":
        reg = [s for s in REGISTRY if "s21" <= s.name < "s41"]
        prefix = "validation_b2"
    elif batch == "3":
        reg = [s for s in REGISTRY if "s41" <= s.name < "s61"]
        prefix = "validation_b3"
    elif batch == "4":
        reg = [s for s in REGISTRY if s.name >= "s61"]
        prefix = "validation_b4"
    elif batch == "1":
        reg = [s for s in REGISTRY if s.name < "s21"]
    report.run_all(data, ctx, reg, prefix=prefix)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd in ("import-opencode", "all"):
        import_opencode()
    if cmd in ("fetch", "all"):
        fetch_missing()
    if cmd in ("validate", "all"):
        validate(batch=arg)
