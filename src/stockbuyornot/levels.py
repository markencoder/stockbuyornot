from __future__ import annotations

import pandas as pd

from stockbuyornot.config import LevelConfig
from stockbuyornot.models import KeyLevel


def identify_levels(df: pd.DataFrame, config: LevelConfig = LevelConfig()) -> tuple[list[KeyLevel], list[KeyLevel]]:
    recent = df.tail(config.range_window)
    latest = df.iloc[-1]
    supports: list[KeyLevel] = []
    resistances: list[KeyLevel] = []

    range_low_idx = recent["low"].idxmin()
    range_high_idx = recent["high"].idxmax()
    supports.append(KeyLevel("区间底部", float(df.loc[range_low_idx, "low"]), "range_low", df.loc[range_low_idx, "date"], 1.0))
    resistances.append(KeyLevel("区间顶部", float(df.loc[range_high_idx, "high"]), "range_high", df.loc[range_high_idx, "date"], 1.0))

    pivots_high, pivots_low = _pivot_levels(df.tail(120), config.pivot_window)
    supports.extend(pivots_low[-3:])
    resistances.extend(pivots_high[-3:])

    high_volume = df[(df["vol_ratio"] >= config.high_volume_ratio) | (df.get("is_high_volume", False))].tail(5)
    for _, row in high_volume.iterrows():
        supports.append(KeyLevel("放量K线低点", float(row["low"]), "high_volume_low", row["date"], 0.8))
        resistances.append(KeyLevel("放量K线高点", float(row["high"]), "high_volume_high", row["date"], 0.8))

    low_volume = df[(df["vol_ratio"] <= config.low_volume_ratio) | (df.get("is_extreme_low_volume", False))].tail(5)
    for _, row in low_volume.iterrows():
        supports.append(KeyLevel("地量K线低点", float(row["low"]), "low_volume_low", row["date"], 0.5))
        resistances.append(KeyLevel("地量K线高点", float(row["high"]), "low_volume_high", row["date"], 0.5))

    supports = _dedupe_levels([level for level in supports if level.price <= latest["close"] * 1.08])
    resistances = _dedupe_levels([level for level in resistances if level.price >= latest["close"] * 0.92])
    return supports, resistances


def nearest_level(price: float, levels: list[KeyLevel], max_distance_pct: float) -> tuple[KeyLevel | None, float | None]:
    if not levels:
        return None, None
    pairs = [(level, abs(price - level.price) / level.price) for level in levels if level.price > 0]
    if not pairs:
        return None, None
    level, distance = min(pairs, key=lambda item: item[1])
    if distance <= max_distance_pct:
        return level, distance
    return None, distance


def _pivot_levels(df: pd.DataFrame, window: int) -> tuple[list[KeyLevel], list[KeyLevel]]:
    highs: list[KeyLevel] = []
    lows: list[KeyLevel] = []
    if len(df) < window * 2 + 1:
        return highs, lows

    for pos in range(window, len(df) - window):
        segment = df.iloc[pos - window : pos + window + 1]
        row = df.iloc[pos]
        if row["high"] == segment["high"].max():
            highs.append(KeyLevel("局部前高", float(row["high"]), "pivot_high", row["date"], 0.7))
        if row["low"] == segment["low"].min():
            lows.append(KeyLevel("局部前低", float(row["low"]), "pivot_low", row["date"], 0.7))
    return highs, lows


def _dedupe_levels(levels: list[KeyLevel], tolerance: float = 0.008) -> list[KeyLevel]:
    ordered = sorted(levels, key=lambda level: (level.price, -level.weight))
    deduped: list[KeyLevel] = []
    for level in ordered:
        if not deduped or abs(level.price - deduped[-1].price) / deduped[-1].price > tolerance:
            deduped.append(level)
        elif level.weight > deduped[-1].weight:
            deduped[-1] = level
    return deduped
