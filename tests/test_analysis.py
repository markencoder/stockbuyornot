import numpy as np
import pandas as pd

from stockbuyornot.analysis import analyze_ohlcv
from stockbuyornot.features import add_features
from stockbuyornot.levels import identify_levels
from stockbuyornot.signals import detect_signals, dynamic_invalidation_buffer
from stockbuyornot.structure import classify_structure
from stockbuyornot.models import SignalSide, Stage


def _trend_fixture(rows: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    base = np.linspace(10, 18, rows)
    noise = np.sin(np.arange(rows) / 4) * 0.15
    close = base + noise
    open_ = close * (1 + np.random.default_rng(1).normal(0, 0.004, rows))
    high = np.maximum(open_, close) * 1.015
    low = np.minimum(open_, close) * 0.985
    volume = np.linspace(100000, 140000, rows)

    # Latest pullback: prices near support, narrower bars, shrinking volume.
    close[-6:] = [18.0, 17.75, 17.55, 17.42, 17.36, 17.45]
    open_[-6:] = [18.1, 17.85, 17.62, 17.46, 17.38, 17.40]
    high[-6:] = np.maximum(open_[-6:], close[-6:]) * 1.006
    low[-6:] = np.minimum(open_[-6:], close[-6:]) * 0.994
    volume[-6:] = [130000, 115000, 100000, 88000, 76000, 70000]

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": volume * close,
            "symbol": "TEST",
        }
    )


def test_analyze_returns_structured_result():
    result = analyze_ohlcv(_trend_fixture(), symbol="TEST")

    assert result.symbol == "TEST"
    assert result.score.total >= 0
    assert result.structure.stage.value
    assert result.support_levels
    assert isinstance(result.signals, list)


def _downtrend_fixture(rows: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = np.linspace(25, 14, rows) + np.sin(np.arange(rows) / 5) * 0.12
    open_ = close * 1.004
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = np.linspace(150000, 90000, rows)

    close[-6:] = [14.0, 14.25, 14.36, 14.44, 14.50, 14.54]
    open_[-6:] = [14.05, 14.22, 14.34, 14.43, 14.49, 14.53]
    high[-6:] = np.maximum(open_[-6:], close[-6:]) * 1.006
    low[-6:] = np.minimum(open_[-6:], close[-6:]) * 0.994
    volume[-6:] = [85000, 76000, 69000, 62000, 56000, 52000]

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": volume * close,
            "symbol": "DOWN",
        }
    )


def test_downtrend_low_volume_rebound_is_not_buy_signal():
    featured = add_features(_downtrend_fixture()).dropna(subset=["ma_slow"]).reset_index(drop=True)
    structure = classify_structure(featured)
    supports, resistances = identify_levels(featured)
    signals = detect_signals(featured, structure, supports, resistances)

    assert structure.stage == Stage.MARKDOWN
    assert not any(signal.side == SignalSide.BUY for signal in signals)
    assert any(signal.side == SignalSide.AVOID for signal in signals)


def test_dynamic_invalidation_buffer_respects_min_volatility_and_cap():
    low_vol = dynamic_invalidation_buffer(pd.Series({"avg_amplitude_20": 0.01}))
    mid_vol = dynamic_invalidation_buffer(pd.Series({"avg_amplitude_20": 0.06}))
    high_vol = dynamic_invalidation_buffer(pd.Series({"avg_amplitude_20": 0.20}))

    assert low_vol == 0.015
    assert mid_vol == 0.03
    assert high_vol == 0.04


def test_buy_like_signals_have_structured_invalidation_price():
    result = analyze_ohlcv(_trend_fixture(), symbol="TEST")
    buy_like = [signal for signal in result.signals if signal.side in {SignalSide.BUY, SignalSide.WATCH} and signal.stop_loss]

    assert buy_like
    for signal in buy_like:
        assert signal.trigger_level is not None
        assert signal.invalidation_price is not None
        assert signal.risk_buffer_pct is not None
        assert signal.invalidation_basis
        assert signal.invalidation_price < signal.trigger_level
        assert signal.risk_buffer_pct >= 0.015
        assert signal.risk_buffer_pct <= 0.04
