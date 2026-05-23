import pandas as pd

from stockbuyornot.data.schema import normalize_ohlcv


def test_normalize_chinese_akshare_columns():
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-01"],
            "开盘": [10],
            "最高": [11],
            "最低": [9],
            "收盘": [10.5],
            "成交量": [1000],
        }
    )

    result = normalize_ohlcv(raw, symbol="000001")

    assert list(result.columns) == ["date", "open", "high", "low", "close", "volume", "symbol", "amount"]
    assert result.loc[0, "symbol"] == "000001"
    assert result.loc[0, "close"] == 10.5
