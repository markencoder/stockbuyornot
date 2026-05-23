import numpy as np
import pandas as pd
import pytest

import stockbuyornot.forecasting as forecasting
from stockbuyornot.forecasting import ForecastConfig, forecast_ohlcv, forecast_points_frame


def _price_fixture(rows: int = 360, symbol: str = "TEST") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-02", periods=rows, freq="B")
    trend = np.linspace(10, 16, rows)
    cycle = np.sin(np.arange(rows) / 9) * 0.18
    close = trend + cycle + rng.normal(0, 0.03, rows)
    open_ = close * (1 + rng.normal(0, 0.003, rows))
    high = np.maximum(open_, close) * 1.012
    low = np.minimum(open_, close) * 0.988
    volume = 120000 + np.sin(np.arange(rows) / 11) * 12000 + np.linspace(0, 40000, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": volume * close,
            "symbol": symbol,
        }
    )


def test_forecast_returns_ten_points_with_enough_history():
    result = forecast_ohlcv(
        _price_fixture(),
        benchmark=_price_fixture(symbol="BENCH"),
        symbol="TEST",
        config=ForecastConfig(model="ridge"),
    )

    assert result.symbol == "TEST"
    assert len(result.points) == 10
    assert result.points[-1].step == 10
    assert -0.35 <= result.expected_return_10d <= 0.35
    assert 0 <= result.up_probability <= 1
    assert not forecast_points_frame(result).empty


def test_forecast_rejects_short_history():
    with pytest.raises(ValueError, match="样本不足"):
        forecast_ohlcv(_price_fixture(rows=120), config=ForecastConfig(model="ridge"))


def test_forecast_as_of_prevents_future_rows():
    data = _price_fixture(rows=360)
    cutoff = data["date"].iloc[330]

    result = forecast_ohlcv(data, as_of=cutoff, config=ForecastConfig(model="ridge"))

    assert result.as_of <= cutoff
    assert result.points[0].date > result.as_of


def test_var_failure_falls_back_to_direct_ridge(monkeypatch):
    def boom(frame, config):
        raise RuntimeError("forced var failure")

    monkeypatch.setattr(forecasting, "_var_forecast", boom)

    result = forecast_ohlcv(_price_fixture(), config=ForecastConfig(model="light_var"))

    assert result.diagnostics["model_used"] == "Direct Ridge"
    assert "forced var failure" in result.diagnostics["fallback_error"]


def test_forecast_does_not_override_avoid_action():
    result = forecast_ohlcv(
        _price_fixture(),
        volume_price_action="回避",
        config=ForecastConfig(model="ridge"),
    )

    assert result.conclusion != "预测支持当前买点"
