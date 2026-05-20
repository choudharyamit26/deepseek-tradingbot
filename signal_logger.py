import os
import csv
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_logs")

_CSV_FIELDS = [
    "timestamp", "symbol", "signal_type", "direction",
    "entry_price", "exit_price", "quantity", "stop_loss",
    "target", "confidence", "reasoning", "pnl", "mode",
    "market_regime", "sector_regime",
    "mtf_3m", "mtf_15m", "mtf_1h",
]


class SignalLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._daily_counts: dict[str, int] = {}
        self._current_date: str | None = None

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

    def log_signal(self, symbol: str, signal_type: str, direction: str,
                   entry_price: float, quantity: int, stop_loss: float,
                   target: float, confidence: int, reasoning: str,
                   mode: str = "DRY-RUN",
                   market_regime: str = "", sector_regime: str = "",
                   mtf_3m: str = "", mtf_15m: str = "", mtf_1h: str = "") -> None:
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
        }

        file_exists = os.path.isfile(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        if is_entry:
            self._daily_counts[symbol] = self._daily_counts.get(symbol, 0) + 1

    def log_exit(self, symbol: str, exit_price: float, pnl: float,
                 exit_type: str = "EXIT") -> None:
        today = self._ensure_date()
        path = self._csv_path(today)
        now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        rows = []
        file_exists = os.path.isfile(path)
        if file_exists:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["symbol"] == symbol and not row["exit_price"]:
                        row["exit_price"] = f"{exit_price:.2f}"
                        row["pnl"] = f"{pnl:.2f}"
                        row["signal_type"] = exit_type
                    rows.append(row)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def reset_daily(self) -> None:
        self._daily_counts.clear()
        self._current_date = None
