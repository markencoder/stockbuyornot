from __future__ import annotations

import numpy as np
import pandas as pd

from stockbuyornot.config import FeatureConfig


def add_features(df: pd.DataFrame, config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    data = df.copy()
    price_range = (data["high"] - data["low"]).replace(0, np.nan)
    body = (data["close"] - data["open"]).abs()

    data["pct_change"] = data["close"].pct_change()
    data["amplitude"] = price_range / data["close"].shift(1)
    data["intraday_range_pct"] = data["amplitude"]
    data["body_pct"] = body / data["close"].shift(1)
    data["body_ratio"] = (body / price_range).fillna(0)
    data["upper_shadow_ratio"] = ((data["high"] - data[["open", "close"]].max(axis=1)) / price_range).fillna(0)
    data["lower_shadow_ratio"] = ((data[["open", "close"]].min(axis=1) - data["low"]) / price_range).fillna(0)
    data["close_position"] = ((data["close"] - data["low"]) / price_range).fillna(0.5)
    data["is_up"] = data["close"] > data["open"]
    data["is_down"] = data["close"] < data["open"]
    data["price_up"] = data["close"] > data["close"].shift(1)
    data["price_down"] = data["close"] < data["close"].shift(1)

    data["vol_ma5"] = data["volume"].rolling(config.volume_short_window, min_periods=3).mean()
    data["vol_ma10"] = data["volume"].rolling(config.volume_mid_window, min_periods=5).mean()
    data["vol_ma20"] = data["volume"].rolling(config.volume_window, min_periods=5).mean()
    data["vol_ma60"] = data["volume"].rolling(config.volume_long_window, min_periods=20).mean()
    data["vol_ma"] = data["vol_ma20"]
    data["vol_ratio"] = data["volume"] / data["vol_ma"]
    data["vol_slope"] = data["volume"].rolling(5, min_periods=3).mean().pct_change(4)
    data["vol_percentile_120"] = data["volume"].rolling(
        config.volume_percentile_window,
        min_periods=30,
    ).apply(_last_percentile_rank, raw=True)
    data["volume_trend_down"] = (data["vol_ma5"] < data["vol_ma10"]) & (data["vol_ma10"] < data["vol_ma20"])
    data["volume_trend_up"] = (data["vol_ma5"] > data["vol_ma10"]) & (data["vol_ma10"] > data["vol_ma20"])
    data["is_volume_expand"] = data["vol_ratio"] >= 1.5
    data["is_volume_strong_expand"] = data["vol_ratio"] >= 2.0
    data["is_high_volume"] = data["vol_percentile_120"] >= 0.90
    data["is_extreme_high_volume"] = data["vol_percentile_120"] >= 0.97
    data["is_volume_shrink"] = data["vol_ratio"] <= 0.70
    data["is_extreme_low_volume"] = data["vol_percentile_120"] <= 0.10

    data["ma_fast"] = data["close"].rolling(config.trend_fast_window, min_periods=10).mean()
    data["ma_slow"] = data["close"].rolling(config.trend_slow_window, min_periods=30).mean()
    data["ma_long"] = data["close"].rolling(config.long_window, min_periods=60).mean()
    data["ma_fast_slope"] = data["ma_fast"].pct_change(5)
    data["ma_slow_slope"] = data["ma_slow"].pct_change(10)

    data["rolling_high_20"] = data["high"].rolling(20, min_periods=10).max()
    data["rolling_low_20"] = data["low"].rolling(20, min_periods=10).min()
    data["rolling_high_60"] = data["high"].rolling(60, min_periods=30).max()
    data["rolling_low_60"] = data["low"].rolling(60, min_periods=30).min()
    data["rolling_high_120"] = data["high"].rolling(120, min_periods=60).max()
    data["rolling_low_120"] = data["low"].rolling(120, min_periods=60).min()
    data["volatility"] = data["pct_change"].rolling(config.volatility_window, min_periods=10).std()
    data["avg_amplitude_20"] = data["amplitude"].rolling(20, min_periods=10).mean()
    data["narrow_range"] = data["amplitude"] <= data["avg_amplitude_20"] * 0.60
    data["range_pct_60"] = (data["rolling_high_60"] - data["rolling_low_60"]) / data["rolling_low_60"]
    data["gain_from_120_low"] = data["close"] / data["rolling_low_120"] - 1
    data["drawdown_from_120_high"] = data["close"] / data["rolling_high_120"] - 1
    data["bull_count_20"] = data["is_up"].rolling(20, min_periods=10).sum()
    data["bear_count_20"] = data["is_down"].rolling(20, min_periods=10).sum()
    data["bull_body_sum_20"] = (data["body_pct"].where(data["is_up"], 0)).rolling(20, min_periods=10).sum()
    data["bear_body_sum_20"] = (data["body_pct"].where(data["is_down"], 0)).rolling(20, min_periods=10).sum()
    data["red_fat_green_thin"] = (data["bull_count_20"] > data["bear_count_20"]) & (
        data["bull_body_sum_20"] > data["bear_body_sum_20"] * 1.05
    )
    data["green_fat_red_thin"] = (data["bear_count_20"] > data["bull_count_20"]) & (
        data["bear_body_sum_20"] > data["bull_body_sum_20"] * 1.05
    )

    return data


def _last_percentile_rank(values: np.ndarray) -> float:
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return np.nan
    last = clean[-1]
    return float((clean <= last).sum() / clean.size)
