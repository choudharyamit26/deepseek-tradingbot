from __future__ import annotations

import time as time_module
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .io import normalize_ohlcv


class DhanAPIError(RuntimeError):
    """Raised when the Dhan API returns an error response."""


@dataclass(frozen=True)
class DhanClient:
    client_id: str
    access_token: str
    base_url: str = "https://api.dhan.co/v2"
    request_sleep_seconds: float = 0.25

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "client-id": self.client_id,
            "access-token": self.access_token,
        }

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.client_id or not self.access_token:
            raise DhanAPIError("Dhan credentials are missing. Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN.")
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        response = requests.post(url, json=payload, headers=self.headers, timeout=60)
        time_module.sleep(max(0.0, self.request_sleep_seconds))
        if response.status_code >= 400:
            raise DhanAPIError(f"Dhan API {response.status_code} for {path}: {response.text[:500]}")
        data = response.json()
        if isinstance(data, dict) and data.get("status") == "failure":
            raise DhanAPIError(f"Dhan API failure for {path}: {data}")
        return data

    def fetch_intraday(
        self,
        security_id: str,
        exchange_segment: str,
        instrument: str,
        interval: str,
        from_datetime: datetime,
        to_datetime: datetime,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": str(interval),
            "oi": False,
            "fromDate": from_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": to_datetime.strftime("%Y-%m-%d %H:%M:%S"),
        }
        raw = self._post("/charts/intraday", payload)
        return chart_response_to_frame(raw), raw

    def fetch_daily(
        self,
        security_id: str,
        exchange_segment: str,
        instrument: str,
        from_date: datetime,
        to_date: datetime,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        payload = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "expiryCode": 0,
            "oi": False,
            "fromDate": from_date.strftime("%Y-%m-%d"),
            "toDate": to_date.strftime("%Y-%m-%d"),
        }
        raw = self._post("/charts/historical", payload)
        return chart_response_to_frame(raw), raw


def download_instrument_master(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    if response.status_code >= 400:
        raise DhanAPIError(f"Instrument master download failed {response.status_code}: {response.text[:500]}")
    output_path.write_bytes(response.content)
    return output_path


def chart_response_to_frame(raw: dict[str, Any]) -> pd.DataFrame:
    payload: dict[str, Any] = raw.get("data", raw) if isinstance(raw, dict) else {}

    def pick(*names: str) -> list[Any]:
        for name in names:
            value = payload.get(name)
            if value is not None:
                return list(value)
        return []

    timestamps = pick("timestamp", "timeStamp", "time", "t")
    if not timestamps:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    numeric_timestamps = pd.to_numeric(pd.Series(timestamps), errors="coerce")
    if numeric_timestamps.notna().all():
        unit = "ms" if numeric_timestamps.max() > 10_000_000_000 else "s"
        parsed_timestamp = pd.to_datetime(numeric_timestamps, unit=unit, utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        parsed_timestamp = pd.to_datetime(timestamps)

    frame = pd.DataFrame(
        {
            "timestamp": parsed_timestamp,
            "open": pick("open", "o"),
            "high": pick("high", "h"),
            "low": pick("low", "l"),
            "close": pick("close", "c"),
            "volume": pick("volume", "v"),
        }
    )
    return normalize_ohlcv(frame)
