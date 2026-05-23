from __future__ import annotations

import pandas as pd

from stockbuyornot.models import MarketStructure, Stage


def classify_structure(df: pd.DataFrame) -> MarketStructure:
    if len(df) < 60:
        return MarketStructure(Stage.UNKNOWN, "unknown", "数据不足，无法稳定判断结构。", 0.2)

    latest = df.iloc[-1]
    recent = df.tail(60)
    prev = df.iloc[-20] if len(df) >= 20 else latest

    close = float(latest["close"])
    ma_fast = float(latest.get("ma_fast", close) or close)
    ma_slow = float(latest.get("ma_slow", close) or close)
    ma_long = float(latest.get("ma_long", ma_slow) or ma_slow)
    ma_fast_slope = float(latest.get("ma_fast_slope", 0) or 0)
    ma_slow_slope = float(latest.get("ma_slow_slope", 0) or 0)
    range_pct = float(latest.get("range_pct_60", 0) or 0)
    gain_from_low = float(latest.get("gain_from_120_low", 0) or 0)
    drawdown = float(latest.get("drawdown_from_120_high", 0) or 0)
    volatility = float(latest.get("volatility", 0) or 0)
    avg_vol_ratio = float(recent["vol_ratio"].tail(20).mean() or 0)
    avg_range = float(recent["amplitude"].tail(20).mean() or 0)
    red_fat_green_thin = bool(latest.get("red_fat_green_thin", False))
    green_fat_red_thin = bool(latest.get("green_fat_red_thin", False))
    low_volume_balance = avg_vol_ratio < 0.95 or bool(recent["is_extreme_low_volume"].tail(20).sum() >= 2)

    recent_highs_up = latest["rolling_high_20"] >= prev.get("rolling_high_20", latest["rolling_high_20"])
    recent_lows_up = latest["rolling_low_20"] >= prev.get("rolling_low_20", latest["rolling_low_20"])
    recent_highs_down = latest["rolling_high_20"] <= prev.get("rolling_high_20", latest["rolling_high_20"])
    recent_lows_down = latest["rolling_low_20"] <= prev.get("rolling_low_20", latest["rolling_low_20"])

    above_trend = close > ma_fast > ma_slow and ma_fast_slope > 0 and ma_slow_slope > -0.01
    below_trend = close < ma_fast < ma_slow and ma_fast_slope < 0 and ma_slow_slope < 0.01
    tight_range = range_pct < 0.30 and avg_range < 0.035
    high_area = gain_from_low > 0.45
    had_prior_rise = gain_from_low > 0.35 and close >= ma_long * 0.95
    wide_high_chop = high_area and range_pct > 0.28 and avg_vol_ratio > 1.05 and not above_trend
    low_base = tight_range and abs(ma_slow_slope) < 0.025 and low_volume_balance and drawdown < -0.12

    metrics = {
        "ma_fast": ma_fast,
        "ma_slow": ma_slow,
        "ma_long": ma_long,
        "ma_fast_slope": ma_fast_slope,
        "ma_slow_slope": ma_slow_slope,
        "gain_from_120_low": gain_from_low,
        "drawdown_from_120_high": drawdown,
        "range_pct_60": range_pct,
        "avg_vol_ratio_20": avg_vol_ratio,
        "avg_amplitude_20": avg_range,
        "red_fat_green_thin": red_fat_green_thin,
        "green_fat_red_thin": green_fat_red_thin,
    }

    if above_trend and (recent_highs_up and recent_lows_up or red_fat_green_thin):
        return MarketStructure(
            Stage.MARKUP,
            "up",
            "价格位于上升均线结构中，近端高低点抬高或呈现红肥绿瘦，需求持续占优。",
            0.86,
            metrics,
        )

    if below_trend and (recent_highs_down and recent_lows_down or green_fat_red_thin):
        return MarketStructure(
            Stage.MARKDOWN,
            "down",
            "价格位于下降均线结构中，近端高低点下移或呈现绿肥红瘦，供应持续占优。",
            0.86,
            metrics,
        )

    if wide_high_chop:
        distribution_metrics = dict(metrics)
        distribution_metrics["volatility"] = volatility
        return MarketStructure(
            Stage.DISTRIBUTION,
            "range_high",
            "阶段涨幅较大后进入高位宽幅震荡，成交活跃且波动放大，供需两强。",
            0.70,
            distribution_metrics,
        )

    if low_base:
        return MarketStructure(
            Stage.ACCUMULATION,
            "range",
            "长期回落后低位窄幅横盘，成交量偏低且慢均线走平，属于供需两弱的筑底平衡。",
            0.74,
            metrics,
        )

    if tight_range and abs(ma_slow_slope) < 0.03:
        stage = Stage.RANGE if had_prior_rise else Stage.ACCUMULATION
        desc = "近60日区间较窄且慢均线走平，供需处于平衡状态。"
        return MarketStructure(
            stage,
            "range",
            desc,
            0.68,
            metrics,
        )

    return MarketStructure(
        Stage.UNKNOWN,
        "mixed",
        "结构信号混杂，暂不适合强行归类。",
        0.45,
        metrics,
    )
