"""Order execution wrapper for the momentum bot.

Sits on top of DhanStockTradingBot (imported fresh — no shared state with
the main bot). Responsibilities:
  - Size positions (fraction of available cash)
  - Place super orders (market entry + SL + target in one shot)
  - Detect time exits vs SL/target hits and alert via Telegram
  - Write a per-session trade log CSV (one row on entry, one row on exit)
"""

from __future__ import annotations
import csv
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from momentum_bot import config as cfg
from momentum_bot import telegram as tg
from momentum_bot.signals import Signal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol:        str
    sid:           str
    direction:     str        # "BUY" | "SELL"
    quantity:      int
    entry_price:   float
    stop_price:    float
    target_price:  float
    entry_time:    datetime = field(default_factory=datetime.now)
    order_id:      str = ""
    open:          bool = True


class MomentumExecutor:
    def __init__(self, dhan, dry_run: bool = False):
        self._dhan       = dhan
        self._dry_run    = dry_run
        self._positions: dict[str, Position] = {}   # symbol → Position
        self._log_path   = _init_log(cfg.LOG_DIR)

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def open_count(self) -> int:
        return sum(1 for p in self._positions.values() if p.open)

    @property
    def traded_symbols(self) -> set[str]:
        return set(self._positions.keys())

    def execute(self, signal: Signal, sid: str) -> bool:
        """Place a super order for the signal. Returns True on success."""
        if self.open_count >= cfg.MAX_OPEN_POSITIONS:
            logger.warning(
                "MAX_OPEN_POSITIONS=%d reached — skipping %s %s",
                cfg.MAX_OPEN_POSITIONS, signal.symbol, signal.direction,
            )
            return False

        quantity = self._calc_quantity(signal.entry_price)
        if quantity < 1:
            logger.warning("%s  calculated quantity < 1 — skipping", signal.symbol)
            return False

        tx_type = self._dhan.dhan.BUY if signal.direction == "BUY" else self._dhan.dhan.SELL

        logger.info(
            "ORDER  %-12s  %s  qty=%d  entry~%.2f  stop=%.2f  target=%.2f  [%s]",
            signal.symbol, signal.direction, quantity,
            signal.entry_price, signal.stop_price, signal.target_price,
            "DRY-RUN" if self._dry_run else "LIVE",
        )

        order_id = ""
        if not self._dry_run:
            try:
                resp = self._dhan.place_super_order(
                    security_id=sid,
                    transaction_type=tx_type,
                    quantity=quantity,
                    entry_price=signal.entry_price,
                    sl_percent=signal.stop_pct,
                    target_percent=signal.target_pct,
                    symbol=signal.symbol,
                )
                if isinstance(resp, dict) and resp.get("status") == "success":
                    order_id = str(resp.get("data", {}).get("orderId", ""))
                    logger.info("%s  order placed  id=%s", signal.symbol, order_id)
                else:
                    logger.error("%s  order FAILED: %s", signal.symbol, resp)
                    self._log_entry(signal, quantity, "FAILED", order_id)
                    return False
            except Exception as exc:
                logger.error("%s  order exception: %s", signal.symbol, exc)
                self._log_entry(signal, quantity, "ERROR", "")
                return False

        pos = Position(
            symbol=signal.symbol, sid=sid, direction=signal.direction,
            quantity=quantity, entry_price=signal.entry_price,
            stop_price=signal.stop_price, target_price=signal.target_price,
            order_id=order_id, open=True,
        )
        self._positions[signal.symbol] = pos
        self._log_entry(signal, quantity, "DRY-RUN" if self._dry_run else "OPEN", order_id)
        return True

    def timed_out_positions(self) -> list[Position]:
        """Return open positions that have been held longer than TIME_EXIT_MINUTES."""
        cutoff = timedelta(minutes=cfg.TIME_EXIT_MINUTES)
        now    = datetime.now()
        return [
            p for p in self._positions.values()
            if p.open and (now - p.entry_time) >= cutoff
        ]

    def time_exit_one(self, pos: Position, live_positions: dict[str, dict] | None = None) -> None:
        """Time-exit a single position. Fetches its live price if live_positions not supplied."""
        if live_positions is None and not self._dry_run:
            try:
                sid_int = self._dhan.security_ids.get(pos.symbol)
                chunk   = self._dhan.fetch_live_data_multi([sid_int]) if sid_int else {}
                live_positions = {
                    pos.symbol: {"last_price": chunk.get(str(sid_int), {}).get("last_price", 0)}
                } if chunk else {}
            except Exception as exc:
                logger.error("live price fetch for %s failed: %s", pos.symbol, exc)
                live_positions = {}
        self._do_time_exit(pos, live_positions or {})

    def check_and_time_exit(self) -> None:
        """Called at EXIT_ALL_TIME (hard day-end backstop).

        For each still-open position:
          - Dhan netQty == 0 → SL or target fired server-side. Log as SL/TARGET-HIT.
          - Still open in Dhan → neither triggered in session. Time-exit at market,
            send Telegram, log as TIME-EXIT.
        """
        if not self._positions:
            return

        live_positions: dict[str, dict] = {}
        if not self._dry_run:
            try:
                live_positions = self._dhan.fetch_positions()
            except Exception as exc:
                logger.error("fetch_positions failed at time-exit: %s — assuming all open", exc)

        exit_time = datetime.now().strftime("%H:%M:%S")

        for pos in list(self._positions.values()):
            if not pos.open:
                continue
            if not (pos.symbol in live_positions) and not self._dry_run:
                logger.info("%-12s  already closed by SL/target (netQty=0)", pos.symbol)
                pos.open = False
                self._log_exit(pos, exit_price=0.0, pnl=0.0, status="SL/TARGET-HIT", exit_time=exit_time)
            else:
                self._do_time_exit(pos, live_positions)

    # ── Private ───────────────────────────────────────────────────────────────

    def _do_time_exit(self, pos: Position, live_positions: dict[str, dict]) -> None:
        """Core time-exit logic: market close + Telegram + CSV row."""
        exit_time  = datetime.now().strftime("%H:%M:%S")
        live_data  = live_positions.get(pos.symbol, {})
        exit_price = float(
            live_data.get("last_price")
            or live_data.get("entry_price")
            or pos.entry_price
        )

        pnl = (
            (exit_price - pos.entry_price) * pos.quantity
            if pos.direction == "BUY"
            else (pos.entry_price - exit_price) * pos.quantity
        )

        held_min = int((datetime.now() - pos.entry_time).total_seconds() / 60)
        logger.warning(
            "TIME-EXIT  %-12s  %s  qty=%d  entry=%.2f  exit~%.2f  pnl=%.2f  held=%dmin  [%s]",
            pos.symbol, pos.direction, pos.quantity,
            pos.entry_price, exit_price, pnl, held_min,
            "DRY-RUN" if self._dry_run else "LIVE",
        )

        if not self._dry_run:
            exit_tx = self._dhan.dhan.SELL if pos.direction == "BUY" else self._dhan.dhan.BUY
            try:
                self._dhan.place_equity_order(
                    security_id=pos.sid,
                    transaction_type=exit_tx,
                    quantity=pos.quantity,
                )
            except Exception as exc:
                logger.error("TIME-EXIT order failed for %s: %s", pos.symbol, exc)

        tg.send(tg.time_exit_msg(
            symbol=pos.symbol, direction=pos.direction,
            quantity=pos.quantity, entry_price=pos.entry_price,
            exit_price=exit_price, pnl=pnl, exit_time=exit_time,
        ))

        pos.open = False
        self._log_exit(pos, exit_price=exit_price, pnl=pnl, status="TIME-EXIT", exit_time=exit_time)

    def _calc_quantity(self, entry_price: float) -> int:
        """Return share quantity using a fixed fraction of available cash."""
        try:
            cash = self._dhan.get_available_balance()
        except Exception:
            cash = 0.0

        if cash <= 0:
            logger.warning("Available balance is 0 — using fallback Rs 50,000")
            cash = 50_000.0

        alloc = cash * cfg.POSITION_SIZE_PCT
        qty   = int(alloc // entry_price)
        return max(qty, 0)

    def _log_entry(self, signal: Signal, quantity: int, status: str, order_id: str) -> None:
        row = {
            "row_type":     "ENTRY",
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol":       signal.symbol,
            "direction":    signal.direction,
            "quantity":     quantity,
            "entry_price":  signal.entry_price,
            "stop_price":   signal.stop_price,
            "target_price": signal.target_price,
            "stop_pct":     round(signal.stop_pct, 3),
            "target_pct":   round(signal.target_pct, 3),
            "or_high":      signal.or_high,
            "or_low":       signal.or_low,
            "volume_ratio": round(signal.volume_ratio, 2),
            "rsi":          round(signal.rsi, 1),
            "exit_price":   "",
            "exit_time":    "",
            "pnl":          "",
            "status":       status,
            "order_id":     order_id,
            "reasoning":    signal.reasoning,
        }
        _append_csv(self._log_path, row)

    def _log_exit(self, pos: Position, exit_price: float, pnl: float, status: str, exit_time: str) -> None:
        row = {
            "row_type":     "EXIT",
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol":       pos.symbol,
            "direction":    pos.direction,
            "quantity":     pos.quantity,
            "entry_price":  pos.entry_price,
            "stop_price":   pos.stop_price,
            "target_price": pos.target_price,
            "stop_pct":     "",
            "target_pct":   "",
            "or_high":      "",
            "or_low":       "",
            "volume_ratio": "",
            "rsi":          "",
            "exit_price":   round(exit_price, 2) if exit_price else "",
            "exit_time":    exit_time,
            "pnl":          round(pnl, 2) if pnl else "",
            "status":       status,
            "order_id":     pos.order_id,
            "reasoning":    "",
        }
        _append_csv(self._log_path, row)
        logger.info("LOG-EXIT  %s  %s  exit_time=%s  pnl=%s  status=%s",
                    pos.symbol, pos.direction, exit_time,
                    f"{pnl:.2f}" if pnl else "n/a", status)


# ── Module helpers ────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "row_type", "timestamp", "symbol", "direction", "quantity",
    "entry_price", "stop_price", "target_price", "stop_pct", "target_pct",
    "or_high", "or_low", "volume_ratio", "rsi",
    "exit_price", "exit_time", "pnl",
    "status", "order_id", "reasoning",
]


def _init_log(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path     = os.path.join(log_dir, f"momentum_{date_str}.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()
    return path


def _append_csv(path: str, row: dict) -> None:
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)
    except Exception as exc:
        logger.error("CSV write failed: %s", exc)
