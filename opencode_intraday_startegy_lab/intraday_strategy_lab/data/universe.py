from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Instrument:
    symbol: str
    security_id: str
    exchange_segment: str
    instrument: str


def _normalise_name(value: object) -> str:
    return str(value).strip().upper().replace("-EQ", "")


def _first_existing(columns: list[str], aliases: list[str]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def load_instrument_master(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    columns = list(raw.columns)
    mapping = {
        "symbol": _first_existing(columns, ["trading_symbol", "tradingsymbol", "symbol", "sem_trading_symbol"]),
        "security_id": _first_existing(columns, ["security_id", "securityid", "security id", "sem_smst_security_id"]),
        "exchange_segment": _first_existing(columns, ["exchange_segment", "exchangesegment", "segment", "sem_exm_exch_id", "sem_segment"]),
        "instrument": _first_existing(columns, ["instrument", "instrument_type", "sem_instrument_name"]),
    }
    missing = [target for target, source in mapping.items() if source is None]
    if missing:
        raise ValueError(f"Instrument master is missing columns for {missing}. Found: {columns[:20]}")
    out = raw.rename(columns={source: target for target, source in mapping.items() if source is not None})
    out = out[["symbol", "security_id", "exchange_segment", "instrument"]].dropna(subset=["symbol", "security_id"])
    out["symbol"] = out["symbol"].map(_normalise_name)
    out["security_id"] = out["security_id"].astype(str).str.strip()
    out["exchange_segment"] = out["exchange_segment"].astype(str).str.upper().str.strip()
    out["instrument"] = out["instrument"].astype(str).str.upper().str.strip()
    return out.drop_duplicates(["symbol", "security_id"]).reset_index(drop=True)


def resolve_security_ids(
    master: pd.DataFrame,
    symbols: list[str],
    exchange_segment: str = "NSE_EQ",
    instrument: str = "EQUITY",
) -> list[Instrument]:
    target_symbols = {_normalise_name(symbol) for symbol in symbols}
    exchange_hint = exchange_segment.upper().split("_")[0]
    instrument_hint = instrument.upper()
    candidates = master[master["symbol"].isin(target_symbols)].copy()
    if exchange_hint:
        exchange_filtered = candidates[candidates["exchange_segment"].str.contains(exchange_hint, na=False)]
        if not exchange_filtered.empty:
            candidates = exchange_filtered
    if instrument_hint:
        instrument_filtered = candidates[candidates["instrument"].str.contains(instrument_hint, na=False)]
        if not instrument_filtered.empty:
            candidates = instrument_filtered

    resolved: list[Instrument] = []
    for symbol in symbols:
        normalised = _normalise_name(symbol)
        match = candidates[candidates["symbol"] == normalised].head(1)
        if match.empty:
            continue
        row = match.iloc[0]
        resolved.append(
            Instrument(
                symbol=normalised,
                security_id=str(row["security_id"]),
                exchange_segment=exchange_segment,
                instrument=instrument,
            )
        )
    return resolved


def compute_beta_table(
    daily_by_symbol: dict[str, pd.DataFrame], benchmark_daily: pd.DataFrame, min_observations: int = 80
) -> pd.DataFrame:
    benchmark = benchmark_daily.copy()
    benchmark["date"] = pd.to_datetime(benchmark["timestamp"]).dt.date
    benchmark_returns = benchmark.sort_values("date").set_index("date")["close"].pct_change().dropna()
    rows: list[dict[str, float | str | int]] = []
    benchmark_variance = float(benchmark_returns.var())
    if benchmark_variance == 0 or np.isnan(benchmark_variance):
        return pd.DataFrame(columns=["symbol", "beta", "observations", "avg_daily_value"])

    for symbol, frame in daily_by_symbol.items():
        data = frame.copy()
        data["date"] = pd.to_datetime(data["timestamp"]).dt.date
        close = data.sort_values("date").set_index("date")["close"]
        returns = close.pct_change().dropna()
        aligned = pd.concat([returns.rename("stock"), benchmark_returns.rename("benchmark")], axis=1).dropna()
        if len(aligned) < min_observations:
            continue
        beta = float(aligned["stock"].cov(aligned["benchmark"]) / benchmark_variance)
        avg_daily_value = float((data["close"] * data["volume"]).mean())
        rows.append(
            {
                "symbol": symbol,
                "beta": beta,
                "observations": int(len(aligned)),
                "avg_daily_value": avg_daily_value,
            }
        )
    return pd.DataFrame(rows).sort_values(["beta", "avg_daily_value"], ascending=[False, False]).reset_index(drop=True)


def select_high_beta_symbols(beta_table: pd.DataFrame, count: int, min_avg_daily_value: float = 0) -> list[str]:
    if beta_table.empty:
        return []
    filtered = beta_table[beta_table["avg_daily_value"] >= float(min_avg_daily_value)]
    if filtered.empty:
        filtered = beta_table
    return filtered.sort_values(["beta", "avg_daily_value"], ascending=[False, False]).head(int(count))["symbol"].tolist()
