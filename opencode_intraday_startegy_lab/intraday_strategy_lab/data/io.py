from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def ensure_directories(base_dir: Path) -> None:
    for relative in ["raw", "processed", "metadata"]:
        base_dir.joinpath(relative).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_ohlcv(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*REQUIRED_COLUMNS, "symbol"])

    rename = {column: str(column).strip().lower() for column in df.columns}
    out = df.rename(columns=rename).copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in out.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    out = out[[*REQUIRED_COLUMNS, *(["symbol"] if "symbol" in out.columns else [])]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=REQUIRED_COLUMNS)
    out = out[(out["high"] >= out["low"]) & (out["open"] > 0) & (out["close"] > 0)]
    if symbol is not None:
        out["symbol"] = symbol
    elif "symbol" not in out.columns:
        out["symbol"] = "UNKNOWN"

    return out.sort_values("timestamp").drop_duplicates(["timestamp", "symbol"]).reset_index(drop=True)


def save_ohlcv_csv(df: pd.DataFrame, path: Path, symbol: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalize_ohlcv(df, symbol=symbol).to_csv(path, index=False)
    return path


def read_ohlcv_csv(path: Path) -> pd.DataFrame:
    return normalize_ohlcv(pd.read_csv(path))


def load_processed_intraday(processed_dir: Path, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    selected = {symbol.upper() for symbol in symbols} if symbols else None
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(processed_dir.glob("*_intraday.csv")):
        symbol = path.name[: -len("_intraday.csv")].upper()
        if selected and symbol not in selected:
            continue
        data[symbol] = read_ohlcv_csv(path)
    if not data:
        raise FileNotFoundError(f"No processed intraday CSV files found in {processed_dir}")
    return data
