"""Dhan intraday equity cost model — every backtest number is NET of these."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg

BROKERAGE_CAP = 20.0        # Rs per executed order
BROKERAGE_PCT = 0.0003      # 0.03%, whichever lower
STT_SELL = 0.00025          # 0.025% sell side (intraday equity)
EXCH_TXN = 0.0000297        # NSE 0.00297%
SEBI = 0.000001             # 0.0001%
STAMP_BUY = 0.00003         # 0.003% buy side
GST = 0.18                  # on brokerage + exchange + sebi


def side_cost(turnover, is_buy):
    brok = min(BROKERAGE_CAP, turnover * BROKERAGE_PCT)
    exch = turnover * EXCH_TXN
    sebi = turnover * SEBI
    stt = 0.0 if is_buy else turnover * STT_SELL
    stamp = turnover * STAMP_BUY if is_buy else 0.0
    return brok + exch + sebi + stt + stamp + GST * (brok + exch + sebi)


def round_trip(entry_px, exit_px, qty):
    """Total charges for one intraday round trip (direction-agnostic: one buy
    leg + one sell leg either order)."""
    return side_cost(entry_px * qty, True) + side_cost(exit_px * qty, False)


def slip(px, direction_hurts_up):
    """Apply slippage against us. direction_hurts_up=True -> we pay higher."""
    s = cfg.SLIPPAGE_PCT / 100.0
    return px * (1 + s) if direction_hurts_up else px * (1 - s)
