"""IS/OOS boundary + walk-forward folds. OOS is touched once per strategy."""
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def is_oos():
    return (cfg.START, cfg.IS_END), (cfg.OOS_START, cfg.END)


def walk_forward_folds():
    """Rolling (train 3m -> test 1m) across the study window."""
    start = pd.Timestamp(cfg.START)
    end = pd.Timestamp(cfg.END)
    folds = []
    t0 = start
    while True:
        train_end = t0 + pd.DateOffset(months=cfg.WF_TRAIN_MONTHS) - pd.Timedelta(days=1)
        test_end = train_end + pd.DateOffset(months=cfg.WF_TEST_MONTHS)
        if test_end > end:
            break
        folds.append((t0.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d"),
                      (train_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                      test_end.strftime("%Y-%m-%d")))
        t0 = t0 + pd.DateOffset(months=cfg.WF_TEST_MONTHS)
    return folds
