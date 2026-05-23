import numpy as np
import pandas as pd

from stockbuyornot.analysis import analyze_ohlcv
from stockbuyornot.radar import diagnose_radar, market_state_from_benchmark


def _benchmark(rows: int = 180, weak: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = np.linspace(100, 84 if weak else 128, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(rows, 1_000_000),
            "amount": close * 1_000_000,
        }
    )


def _strong_setup(rows: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = np.linspace(10, 16, rows) + np.sin(np.arange(rows) / 8) * 0.08
    close[-65:-8] = np.linspace(14.2, 15.1, 57) + np.sin(np.arange(57) / 3) * 0.08
    close[-8:] = [15.20, 15.35, 15.48, 15.62, 15.72, 15.68, 15.74, 15.82]
    open_ = close * 0.998
    high = np.maximum(open_, close) * 1.008
    low = np.minimum(open_, close) * 0.992
    volume = np.linspace(2_000_000, 2_300_000, rows)
    volume[-20:] = np.linspace(1_600_000, 900_000, 20)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": volume * close,
            "symbol": "STRONG",
        }
    )


def _pullback_without_volume_shrink(rows: int = 180) -> pd.DataFrame:
    df = _strong_setup(rows)
    close = df["close"].to_numpy(copy=True)
    close[-8:] = [16.8, 16.5, 16.22, 16.05, 15.95, 15.90, 15.86, 15.92]
    df["close"] = close
    df["open"] = close * 1.002
    df["high"] = np.maximum(df["open"], df["close"]) * 1.008
    df["low"] = np.minimum(df["open"], df["close"]) * 0.992
    df["volume"] = 2_400_000
    df["amount"] = df["volume"] * df["close"]
    return df


def _high_risk_top(rows: int = 180) -> pd.DataFrame:
    df = _strong_setup(rows)
    close = df["close"].to_numpy(copy=True)
    close[-6:] = [18.0, 18.4, 18.6, 18.55, 18.40, 18.10]
    df["close"] = close
    df["open"] = close * 1.01
    df["high"] = close * 1.08
    df["low"] = close * 0.99
    df["volume"] = np.linspace(2_000_000, 5_800_000, rows)
    df["amount"] = df["volume"] * df["close"]
    return df


def test_strong_setup_gets_watch_or_better():
    stock = _strong_setup()
    benchmark = _benchmark()
    sector = _benchmark()
    sector["close"] = np.linspace(100, 138, len(sector))

    analysis = analyze_ohlcv(stock, symbol="STRONG", benchmark=benchmark, sector=sector, sector_name="AI")

    assert analysis.radar is not None
    assert analysis.radar.setup_score >= 50
    assert analysis.radar.expected_action in {"可关注", "等待买点", "可交易"}


def test_pullback_without_volume_shrink_has_lower_entry_quality():
    stock = _pullback_without_volume_shrink()
    benchmark = _benchmark()
    analysis = analyze_ohlcv(stock, symbol="PB", benchmark=benchmark)

    assert analysis.radar is not None
    assert analysis.radar.entry_quality_score < 75


def test_downtrend_rebound_is_rejected_by_radar():
    dates = pd.date_range("2024-01-01", periods=180, freq="B")
    close = np.linspace(24, 12, 180)
    close[-8:] = [12.0, 12.15, 12.25, 12.32, 12.38, 12.42, 12.46, 12.50]
    stock = pd.DataFrame(
        {
            "date": dates,
            "open": close * 1.002,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(2_000_000, 1_200_000, 180),
            "amount": close * 2_000_000,
        }
    )
    analysis = analyze_ohlcv(stock, symbol="DOWN", benchmark=_benchmark())

    assert analysis.radar is not None
    assert analysis.radar.expected_action == "回避"
    assert "weak_or_late_structure" in analysis.radar.reject_reason


def test_high_volume_top_raises_exit_risk():
    stock = _high_risk_top()
    analysis = analyze_ohlcv(stock, symbol="TOP", benchmark=_benchmark())

    assert analysis.radar is not None
    assert analysis.radar.exit_risk_score >= 45


def test_weak_market_blocks_active_buy_action():
    stock = _strong_setup()
    benchmark = _benchmark(weak=True)
    analysis = analyze_ohlcv(stock, symbol="STRONG", benchmark=benchmark)

    assert analysis.radar is not None
    assert market_state_from_benchmark(benchmark) == "weak"
    assert analysis.radar.expected_action != "可交易"
