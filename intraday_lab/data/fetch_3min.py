"""Fetch 1-min bars for the universe and resample to 3-min parquet.

Dhan's intraday API has no native 3-min interval, so: fetch interval="1"
(full study window, chunked), store {SYM}_1min.parquet, resample to
{SYM}_3min.parquet. 09:15 is 555 min from midnight (divisible by 3) so
default resample bins align exactly with session opens.

  python data/fetch_3min.py
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg
from data import fetcher
from data.universe import candidate_map

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
logger = logging.getLogger("fetch3m")

AGG = {"open": "first", "high": "max", "low": "min",
       "close": "last", "volume": "sum"}

# absent from candidate_map; looked up in scrip-master_NSE_EQ (1).csv
SID_OVERRIDES = {"IDEA": "14366", "MUTHOOTFIN": "23650"}


def resample_3min(df1):
    out = df1.resample("3min", label="left", closed="left").agg(AGG).dropna()
    # drop any partial bin spilling past session close
    return out[(out.index.hour * 60 + out.index.minute) <= 15 * 60 + 27]


def main():
    uni = json.loads((cfg.ROOT / "data" / "universe.json").read_text())
    cands = candidate_map()
    for u in uni:
        sym = u["symbol"]
        p3 = cfg.STORE / f"{sym}_3min.parquet"
        if p3.exists():
            logger.info("%s: 3min exists, skip", sym)
            continue
        sid = cands.get(sym) or SID_OVERRIDES.get(sym)
        if not sid:
            logger.warning("%s: no security id, skip", sym)
            continue
        p1 = cfg.STORE / f"{sym}_1min.parquet"
        if p1.exists():
            df1 = pd.read_parquet(p1)
        else:
            df1 = fetcher.fetch_intraday(sid, cfg.START, cfg.END, interval="1")
            if df1.empty:
                logger.error("%s: 1-min fetch EMPTY", sym)
                continue
            df1.to_parquet(p1)
        df3 = resample_3min(df1)
        df3.to_parquet(p3)
        logger.info("%s: %d 1-min -> %d 3-min bars  %s..%s", sym, len(df1),
                    len(df3), df3.index[0].date(), df3.index[-1].date())


if __name__ == "__main__":
    main()
