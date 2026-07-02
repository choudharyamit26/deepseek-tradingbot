import os
import csv
import logging
import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_logs")

_CSV_FIELDS = [
    "timestamp", "symbol", "signal_type", "direction",
    "entry_price", "exit_price", "quantity", "stop_loss",
    "trailing_stop", "target", "confidence", "reasoning", "pnl", "mode",
    "market_regime", "sector_regime",
    "mtf_3m", "mtf_15m", "mtf_1h",
    "kronos_direction", "kronos_pred_return", "kronos_aligned",
    # Leading microstructure (Order-Flow Imbalance), captured at entry so the
    # feature study can validate it. Appended last to keep existing column order.
    "ofi", "ofi_trend",
]


class SignalLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._daily_counts: dict[str, int] = {}
        self._current_date: str | None = None
        self._lock = asyncio.Lock()

    def _csv_path(self, today: str) -> str:
        return os.path.join(LOG_DIR, f"signals_{today}.csv")

    def _ensure_date(self) -> str:
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if self._current_date != today:
            self._daily_counts.clear()
            self._current_date = today
        return today

    def get_daily_count(self, symbol: str) -> int:
        self._ensure_date()
        return self._daily_counts.get(symbol, 0)

    def can_trade(self, symbol: str, max_per_day: int = 2) -> bool:
        return self.get_daily_count(symbol) < max_per_day

    def get_total_daily_count(self) -> int:
        """Return total entry signals across all stocks today."""
        self._ensure_date()
        return sum(self._daily_counts.values())

    def reserve_daily_slot(self, symbol: str, max_total: int, max_per_stock: int) -> bool:
        """Atomically check caps AND reserve a slot. Returns True if reserved.

        This prevents the TOCTOU race where multiple concurrent coroutines
        pass get_total_daily_count() before any of them increment the counter.
        """
        self._ensure_date()
        if sum(self._daily_counts.values()) >= max_total:
            return False
        if self._daily_counts.get(symbol, 0) >= max_per_stock:
            return False
        # Reserve immediately — log_signal() will NOT double-count because
        # we set a '_reserved' flag that log_signal checks.
        self._daily_counts[symbol] = self._daily_counts.get(symbol, 0) + 1
        return True

    def release_daily_slot(self, symbol: str) -> None:
        """Release a previously reserved slot (e.g. if the signal was vetoed after reservation)."""
        self._ensure_date()
        count = self._daily_counts.get(symbol, 0)
        if count > 0:
            self._daily_counts[symbol] = count - 1

    async def log_signal(self, symbol: str, signal_type: str, direction: str,
                   entry_price: float, quantity: int, stop_loss: float,
                   trailing_stop: float, target: float, confidence: int, reasoning: str,
                   mode: str = "DRY-RUN",
                   market_regime: str = "", sector_regime: str = "",
                   mtf_3m: str = "", mtf_15m: str = "", mtf_1h: str = "",
                   kronos_direction: str = "", kronos_pred_return: str = "",
                   kronos_aligned: str = "",
                   ofi: str = "", ofi_trend: str = "",
                   slot_reserved: bool = False) -> None:
        today = self._ensure_date()
        path = self._csv_path(today)
        now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        is_entry = signal_type.upper() in ("BUY", "SELL", "ENTRY-LONG", "ENTRY-SHORT")

        row = {
            "timestamp": now_str,
            "symbol": symbol,
            "signal_type": signal_type,
            "direction": direction,
            "entry_price": f"{entry_price:.2f}",
            "exit_price": "",
            "quantity": quantity,
            "stop_loss": f"{stop_loss:.2f}" if stop_loss else "",
            "trailing_stop": f"{trailing_stop:.2f}" if trailing_stop else "",
            "target": f"{target:.2f}" if target else "",
            "confidence": confidence,
            "reasoning": reasoning.replace('"', "'") if reasoning else "",
            "pnl": "",
            "mode": mode,
            "market_regime": market_regime,
            "sector_regime": sector_regime,
            "mtf_3m": mtf_3m,
            "mtf_15m": mtf_15m,
            "mtf_1h": mtf_1h,
            "kronos_direction": kronos_direction,
            "kronos_pred_return": kronos_pred_return,
            "kronos_aligned": kronos_aligned,
            "ofi": ofi,
            "ofi_trend": ofi_trend,
        }

        def _write_csv():
            file_exists = os.path.isfile(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        async with self._lock:
            await asyncio.to_thread(_write_csv)

        # Increment counter only if the slot was NOT pre-reserved by reserve_daily_slot()
        if is_entry and not slot_reserved:
            self._daily_counts[symbol] = self._daily_counts.get(symbol, 0) + 1

    async def log_exit(self, symbol: str, exit_price: float, pnl: float,
                 exit_type: str = "EXIT") -> None:
        today = self._ensure_date()
        path = self._csv_path(today)

        def _modify_and_write():
            rows = []
            file_exists = os.path.isfile(path)
            if file_exists:
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                
                matched = False
                for row in reversed(rows):
                    if not matched and row["symbol"] == symbol and not row["exit_price"]:
                        row["exit_price"] = f"{exit_price:.2f}"
                        row["pnl"] = f"{pnl:.2f}"
                        row["signal_type"] = exit_type
                        matched = True

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

        async with self._lock:
            await asyncio.to_thread(_modify_and_write)

    def reset_daily(self) -> None:
        self._daily_counts.clear()
        self._current_date = None
