from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["amount", "symbol"]


def normalize_ohlcv(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Normalize OHLCV data to the internal schema."""
    if df.empty:
        raise ValueError("OHLCV data is empty.")

    renamed = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "代码": "symbol",
        "股票代码": "symbol",
    }
    data = df.rename(columns={k: v for k, v in renamed.items() if k in df.columns}).copy()
    data.columns = [str(c).strip().lower() for c in data.columns]

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")

    data = data[[c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in data.columns]].copy()
    data["date"] = pd.to_datetime(data["date"])
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if "symbol" not in data.columns:
        data["symbol"] = symbol or ""
    if "amount" not in data.columns:
        data["amount"] = pd.NA

    data = data.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    data = data.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    data = data.reset_index(drop=True)

    if data.empty:
        raise ValueError("OHLCV data is empty after normalization.")
    return data
