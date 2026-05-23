from __future__ import annotations

import pandas as pd

from stockbuyornot.config import SignalConfig
from stockbuyornot.levels import nearest_level
from stockbuyornot.models import KeyLevel, MarketStructure, SignalSide, Stage, TradeSignal


def detect_signals(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig = SignalConfig(),
) -> list[TradeSignal]:
    detectors = [
        detect_bearish_breakdown_sell,
        detect_down_reversal_sell,
        detect_volume_stall_sell,
        detect_demand_climax_sell,
        detect_markdown_continuation_avoid,
        detect_breakout_buy,
        detect_markup_pullback_buy,
        detect_range_bottom_buy,
        detect_bottom_reversal_buy,
        detect_bottom_pullback_confirm_buy,
        detect_range_top_sell,
    ]
    signals = [signal for detector in detectors if (signal := detector(df, structure, supports, resistances, config))]

    severe_risk = [signal for signal in signals if signal.side in {SignalSide.SELL, SignalSide.AVOID}]
    if severe_risk:
        buy_names_allowed_with_risk = {"底部反转买点"}
        signals = [
            signal
            for signal in signals
            if signal.side in {SignalSide.SELL, SignalSide.AVOID} or signal.name in buy_names_allowed_with_risk
        ]
    return sorted(signals, key=lambda signal: signal.strength, reverse=True)


def detect_breakout_buy(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig,
) -> TradeSignal | None:
    if structure.stage in {Stage.MARKDOWN, Stage.DISTRIBUTION}:
        return None

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    resistance = _best_breakout_resistance(float(latest["close"]), resistances)
    if not resistance:
        return None

    prior = df.iloc[-16:-1]
    breakout = latest["close"] > resistance.price * (1 + config.breakout_pct)
    was_below = previous["close"] <= resistance.price * (1 + config.breakout_pct)
    volume_confirm = _volume_expand(latest, config) or bool(latest.get("is_high_volume", False))
    strong_bar = _strong_bull(latest, config)
    prior_contraction = (
        prior["amplitude"].tail(10).mean() <= df["amplitude"].tail(40).mean() * 0.85
        or prior["vol_ratio"].tail(10).mean() <= 1.0
        or bool(prior["volume_trend_down"].tail(5).any())
    )
    from_base = structure.stage in {Stage.ACCUMULATION, Stage.RANGE, Stage.UNKNOWN}

    if breakout and was_below and volume_confirm and strong_bar and prior_contraction and from_base:
        risk = _invalidation_risk(resistance.price, latest, config, "突破位下方动态缓冲")
        stop = min(float(latest["low"]), resistance.price) * (1 - config.stop_buffer_pct)
        return TradeSignal(
            "向上突破买点",
            SignalSide.BUY,
            86,
            "低位或区间右侧出现放量强阳突破，需求打破原有供需平衡。",
            [
                f"收盘价有效突破{resistance.name} {resistance.price:.2f}",
                f"量比 {latest['vol_ratio']:.2f}，成交量确认突破",
                "突破前存在缩量或波动收窄，突破K线收盘靠近高位",
            ],
            stop_loss=round(stop, 3),
            entry_zone=(round(resistance.price, 3), round(float(latest["close"]) * 1.02, 3)),
            invalidation=_invalidation_text(risk),
            level=resistance,
            trigger_level=round(risk["trigger_level"], 3),
            invalidation_price=round(risk["invalidation_price"], 3),
            risk_buffer_pct=round(risk["risk_buffer_pct"], 4),
            invalidation_basis=risk["invalidation_basis"],
        )
    return None


def detect_markup_pullback_buy(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig,
) -> TradeSignal | None:
    if structure.stage != Stage.MARKUP:
        return None

    latest = df.iloc[-1]
    support, distance = nearest_level(float(latest["close"]), supports, 0.035)
    if not support:
        return None

    pullback = df.tail(7)
    had_pullback = pullback["close"].iloc[-1] <= pullback["close"].max() * 0.975 or pullback["low"].iloc[-1] <= support.price * 1.025
    volume_shrink = (
        pullback["vol_ratio"].tail(3).mean() <= config.dry_volume_ratio
        or bool(pullback["volume_trend_down"].tail(3).any())
        or pullback["volume"].tail(4).is_monotonic_decreasing
    )
    range_contract = pullback["amplitude"].tail(3).mean() <= df["amplitude"].tail(20).mean() * 0.75
    narrow_bars = pullback["body_ratio"].tail(3).mean() <= config.small_body_ratio or bool(pullback["narrow_range"].tail(3).any())
    holds_support = latest["close"] >= support.price * (1 - config.support_tolerance_pct)
    no_bearish_pressure = not (_strong_bear(latest, config) and _volume_expand(latest, config))
    stop_bar = latest["lower_shadow_ratio"] >= 0.25 or latest["close"] >= latest["open"] or latest["close_position"] >= 0.45

    if had_pullback and volume_shrink and (range_contract or narrow_bars) and holds_support and stop_bar and no_bearish_pressure:
        risk = _invalidation_risk(support.price, latest, config, "支撑价下方动态缓冲")
        stop = min(support.price, float(pullback["low"].tail(5).min())) * (1 - config.stop_buffer_pct)
        return TradeSignal(
            "上涨中继买点",
            SignalSide.BUY,
            90,
            "主升结构中的缩量回踩，供应不足且关键支撑未破，符合顺大势、逆小势。",
            [
                f"当前处于{structure.stage.value}",
                f"距离{support.name}约{(distance or 0) * 100:.1f}%",
                "回踩段成交量萎缩，K线波动收窄",
                "未有效跌破关键支撑",
            ],
            stop_loss=round(stop, 3),
            entry_zone=(round(support.price * 1.005, 3), round(support.price * 1.03, 3)),
            invalidation=_invalidation_text(risk),
            level=support,
            trigger_level=round(risk["trigger_level"], 3),
            invalidation_price=round(risk["invalidation_price"], 3),
            risk_buffer_pct=round(risk["risk_buffer_pct"], 4),
            invalidation_basis=risk["invalidation_basis"],
        )
    return None


def detect_range_bottom_buy(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig,
) -> TradeSignal | None:
    if structure.stage not in {Stage.ACCUMULATION, Stage.RANGE}:
        return None
    latest = df.iloc[-1]
    support = next((level for level in supports if level.kind == "range_low"), None)
    if not support:
        return None

    close_near_bottom = abs(float(latest["close"]) - support.price) / support.price <= 0.04
    false_break = latest["low"] < support.price * (1 - config.breakdown_pct) and latest["close"] > support.price and latest["close_position"] >= 0.55
    volume_stall_down = _volume_price_stall_down(latest, config)
    dry_stop = _volume_shrink(latest, config) and latest["body_ratio"] <= config.small_body_ratio and latest["close_position"] >= 0.45

    if close_near_bottom and (false_break or volume_stall_down or dry_stop):
        risk = _invalidation_risk(support.price, latest, config, "区间底部下方动态缓冲")
        stop = support.price * (1 - config.stop_buffer_pct)
        return TradeSignal(
            "区间底部低吸",
            SignalSide.WATCH,
            72,
            "区间底部供应尝试下压失败，或放量滞跌显示需求吸收供应。",
            [
                f"接近区间底部 {support.price:.2f}",
                "出现跌破失败、放量滞跌或缩量止跌",
            ],
            stop_loss=round(stop, 3),
            entry_zone=(round(support.price * 1.005, 3), round(support.price * 1.03, 3)),
            invalidation=_invalidation_text(risk),
            level=support,
            trigger_level=round(risk["trigger_level"], 3),
            invalidation_price=round(risk["invalidation_price"], 3),
            risk_buffer_pct=round(risk["risk_buffer_pct"], 4),
            invalidation_basis=risk["invalidation_basis"],
        )
    return None


def detect_bottom_reversal_buy(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig,
) -> TradeSignal | None:
    if len(df) < 30:
        return None
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    recent = df.tail(25)
    had_downtrend = structure.stage == Stage.MARKDOWN or recent["close"].iloc[0] > recent["close"].iloc[-2] * 1.12
    panic_before = (
        (previous["is_down"] and previous["vol_ratio"] >= config.volume_expand_ratio and previous["body_ratio"] >= 0.45)
        or bool(recent["is_extreme_high_volume"].tail(5).any())
        or recent["low"].iloc[-2] <= recent["low"].iloc[:-2].min() * 1.01
    )
    bullish_engulf = _bullish_engulfing(previous, latest)
    strong_demand = latest["is_up"] and (_volume_expand(latest, config) or latest["vol_ratio"] >= 1.2)
    reversal_shape = bullish_engulf or latest["lower_shadow_ratio"] >= config.long_shadow_ratio or _volume_price_stall_down(latest, config)
    no_new_low = latest["close"] > recent["low"].iloc[:-1].min() or latest["close"] > previous["low"]

    if had_downtrend and panic_before and strong_demand and reversal_shape and no_new_low:
        trigger_level = min(float(latest["low"]), float(previous["low"]))
        risk = _invalidation_risk(trigger_level, latest, config, "反转低点下方动态缓冲")
        stop = min(float(latest["low"]), float(previous["low"])) * (1 - config.stop_buffer_pct)
        return TradeSignal(
            "底部反转买点",
            SignalSide.WATCH,
            76,
            "下降趋势末端恐慌或新低后出现放量需求，供应可能开始出清，但该类买点需要轻仓和确认。",
            [
                "前面存在下降趋势、阶段新低或恐慌放量",
                f"当日量比 {latest['vol_ratio']:.2f}，出现强需求或放量滞跌",
                "长下影/阳吞阴/收盘收回低点显示供应未能延续",
            ],
            stop_loss=round(stop, 3),
            entry_zone=(round(float(latest["low"]), 3), round(float(latest["close"]) * 1.02, 3)),
            invalidation=_invalidation_text(risk),
            trigger_level=round(risk["trigger_level"], 3),
            invalidation_price=round(risk["invalidation_price"], 3),
            risk_buffer_pct=round(risk["risk_buffer_pct"], 4),
            invalidation_basis=risk["invalidation_basis"],
        )
    return None


def detect_bottom_pullback_confirm_buy(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig,
) -> TradeSignal | None:
    if structure.stage not in {Stage.ACCUMULATION, Stage.RANGE, Stage.UNKNOWN} or len(df) < 45:
        return None

    latest = df.iloc[-1]
    recent = df.tail(25)
    prior_reversal = recent.iloc[:-3]
    reversal_seen = (
        (prior_reversal["is_up"] & (prior_reversal["vol_ratio"] >= config.volume_expand_ratio) & (prior_reversal["close_position"] >= 0.65)).any()
        or (prior_reversal["lower_shadow_ratio"] >= config.long_shadow_ratio).any()
    )
    bottom_low = float(recent["low"].min())
    near_bottom_support = latest["close"] >= bottom_low * (1 - config.support_tolerance_pct) and latest["close"] <= bottom_low * 1.08
    shrink_pullback = recent["vol_ratio"].tail(3).mean() <= config.dry_volume_ratio or bool(recent["volume_trend_down"].tail(3).any())
    no_new_low = latest["low"] >= bottom_low * (1 - config.breakdown_pct)

    if reversal_seen and near_bottom_support and shrink_pullback and no_new_low:
        risk = _invalidation_risk(bottom_low, latest, config, "底部低点下方动态缓冲")
        stop = bottom_low * (1 - config.stop_buffer_pct)
        return TradeSignal(
            "底部反转后缩量回踩",
            SignalSide.WATCH,
            73,
            "前期需求已经出现，随后回踩缩量且不创新低，说明供应没有重新占优。",
            [
                "前期出现放量阳线、长下影或放量滞跌",
                "回踩成交量萎缩",
                "未有效跌破底部低点",
            ],
            stop_loss=round(stop, 3),
            entry_zone=(round(bottom_low * 1.01, 3), round(bottom_low * 1.06, 3)),
            invalidation=_invalidation_text(risk),
            trigger_level=round(risk["trigger_level"], 3),
            invalidation_price=round(risk["invalidation_price"], 3),
            risk_buffer_pct=round(risk["risk_buffer_pct"], 4),
            invalidation_basis=risk["invalidation_basis"],
        )
    return None


def detect_demand_climax_sell(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig,
) -> TradeSignal | None:
    if structure.stage not in {Stage.MARKUP, Stage.DISTRIBUTION}:
        return None
    latest = df.iloc[-1]
    recent = df.tail(10)
    high_gain = latest.get("gain_from_120_low", 0) >= config.high_gain_pct
    short_surge = recent["pct_change"].tail(3).sum() >= 0.08
    prior_slope = max(abs(recent["pct_change"].iloc[:5].sum()), 0.01)
    acceleration = short_surge and recent["pct_change"].tail(3).sum() >= prior_slope * 1.8
    big_up_days = (recent["is_up"] & (recent["vol_ratio"] >= config.volume_expand_ratio)).tail(5).sum() >= 2
    climax_volume = latest["vol_ratio"] >= config.climax_volume_ratio or bool(latest.get("is_extreme_high_volume", False))
    wide_bar = latest["amplitude"] >= df["amplitude"].tail(60).quantile(0.8)

    if high_gain and acceleration and (big_up_days or climax_volume) and wide_bar:
        return TradeSignal(
            "需求衰竭/加速卖点",
            SignalSide.SELL,
            82,
            "高位连续加速后需求可能快速透支，加速不是买点，已有仓位应降低风险。",
            [
                f"阶段涨幅 {latest.get('gain_from_120_low', 0) * 100:.1f}%",
                "短线斜率明显变陡",
                "放量上涨并伴随波动加剧",
            ],
            invalidation="缩量横盘消化高位放量K线并继续健康上行",
        )
    return None


def detect_volume_stall_sell(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig,
) -> TradeSignal | None:
    latest = df.iloc[-1]
    resistance, _ = nearest_level(float(latest["close"]), resistances, 0.04)
    high_area = latest.get("gain_from_120_low", 0) >= config.high_gain_pct or structure.stage == Stage.DISTRIBUTION
    near_resistance = resistance is not None

    if (high_area or near_resistance or structure.stage in {Stage.MARKUP, Stage.DISTRIBUTION}) and _volume_price_stall_up(latest, config):
        return TradeSignal(
            "放量滞涨卖点",
            SignalSide.SELL,
            86,
            "成交量放大但价格涨不动，放大的需求被供应抵消，属于高位供应出现的风险信号。",
            [
                f"量比 {latest['vol_ratio']:.2f}",
                "涨幅有限、收盘不强或上影线明显",
                f"接近压力位 {resistance.price:.2f}" if resistance else "位于阶段高位或上升趋势后段",
            ],
            invalidation="放量滞涨K线高点被有效突破并站稳",
            level=resistance,
        )
    return None


def detect_down_reversal_sell(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig,
) -> TradeSignal | None:
    if structure.stage not in {Stage.MARKUP, Stage.DISTRIBUTION, Stage.RANGE}:
        return None
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    bearish_engulf = _bearish_engulfing(previous, latest)
    failed_new_high = latest["high"] >= df["high"].tail(10).iloc[:-1].max() and latest["close_position"] <= config.close_near_low
    volume_confirm = _volume_expand(latest, config) or bool(latest.get("is_high_volume", False))
    weak_shape = _strong_bear(latest, config) or latest["upper_shadow_ratio"] >= config.long_shadow_ratio
    breaks_short_low = latest["close"] < df["low"].tail(6).iloc[:-1].min()

    if volume_confirm and (bearish_engulf or failed_new_high or weak_shape) and (breaks_short_low or latest["close_position"] <= 0.35):
        return TradeSignal(
            "向下反转卖点",
            SignalSide.SELL,
            88,
            "高位或上升结构中放量转弱，供应反包或压制需求，原需求占优格局被破坏。",
            [
                "前面处于上升、区间顶部或高位结构",
                "冲高失败、阴包阳、长上影或放量长阴",
                f"量比 {latest['vol_ratio']:.2f} 提供供应确认",
            ],
            invalidation="重新收复反转K线高点",
        )
    return None


def detect_bearish_breakdown_sell(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig,
) -> TradeSignal | None:
    latest = df.iloc[-1]
    support = next((level for level in supports if level.kind == "range_low"), None)
    if not support:
        return None
    breakdown = latest["close"] < support.price * (1 - config.breakdown_pct)
    volume_confirm = _volume_expand(latest, config) or _strong_bear(latest, config)
    risk_structure = structure.stage in {Stage.RANGE, Stage.DISTRIBUTION, Stage.MARKUP, Stage.UNKNOWN}

    if breakdown and volume_confirm and risk_structure:
        return TradeSignal(
            "向下突破卖点",
            SignalSide.SELL,
            90,
            "价格跌破区间底部或关键支撑，供应打破平衡，可能进入下跌阶段。",
            [
                f"收盘跌破关键支撑 {support.price:.2f}",
                "放量或长阴确认供应占优",
            ],
            invalidation=f"快速收回 {support.price:.2f} 上方",
            level=support,
        )
    return None


def detect_range_top_sell(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig,
) -> TradeSignal | None:
    if structure.stage not in {Stage.RANGE, Stage.DISTRIBUTION, Stage.ACCUMULATION}:
        return None
    latest = df.iloc[-1]
    resistance = next((level for level in resistances if level.kind == "range_high"), None)
    if not resistance:
        return None
    close_near_top = abs(float(latest["close"]) - resistance.price) / resistance.price <= 0.04
    failed_break = latest["high"] > resistance.price and latest["close"] <= resistance.price
    weak_demand = _volume_price_stall_up(latest, config) or latest["upper_shadow_ratio"] >= 0.35 or (
        _volume_shrink(latest, config) and latest["close_position"] < 0.65
    )

    if close_near_top and (failed_break or weak_demand):
        return TradeSignal(
            "区间顶部卖点",
            SignalSide.SELL,
            74,
            "区间顶部需求无法打破平衡，出现滞涨、长上影或突破失败，价格倾向折回区间。",
            [
                f"接近区间顶部 {resistance.price:.2f}",
                "突破失败、放量滞涨或缩量滞涨",
            ],
            invalidation=f"有效站稳区间顶部 {resistance.price:.2f}",
            level=resistance,
        )
    return None


def detect_markdown_continuation_avoid(
    df: pd.DataFrame,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    config: SignalConfig,
) -> TradeSignal | None:
    if structure.stage != Stage.MARKDOWN:
        return None
    recent = df.tail(8)
    latest = df.iloc[-1]
    resistance, _ = nearest_level(float(latest["close"]), resistances, 0.06)
    rebound = latest["close"] > recent["low"].min() * 1.04
    low_volume_rebound = recent["vol_ratio"].tail(4).mean() <= config.dry_volume_ratio or bool(recent["volume_trend_down"].tail(4).any())
    cannot_break_resistance = latest["close"] < latest["ma_fast"] or (resistance is not None and latest["close"] <= resistance.price * 1.01)
    weak_up = recent["body_ratio"].tail(4).mean() <= 0.45
    no_reversal = not (_strong_bull(latest, config) and _volume_expand(latest, config))

    if rebound and low_volume_rebound and cannot_break_resistance and weak_up and no_reversal:
        return TradeSignal(
            "下跌中继/回避信号",
            SignalSide.AVOID,
            82,
            "下降趋势中的反弹缺乏成交量配合，需求不足，不能把缩量反弹当作底部。",
            [
                "当前处于主跌结构",
                "反弹期间成交量低于均量",
                "反弹未能突破短期压力或均线",
            ],
            invalidation="放量突破短期压力并形成更高低点",
            level=resistance,
        )
    return None


def _best_breakout_resistance(close: float, resistances: list[KeyLevel]) -> KeyLevel | None:
    candidates = [
        level for level in resistances if level.price < close and level.kind in {"range_high", "pivot_high", "high_volume_high"}
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda level: level.price)


def dynamic_invalidation_buffer(row: pd.Series, config: SignalConfig = SignalConfig()) -> float:
    avg_amplitude = float(row.get("avg_amplitude_20", 0) or 0)
    if pd.isna(avg_amplitude):
        avg_amplitude = 0.0
    volatility_buffer = avg_amplitude * 0.5
    return min(config.invalidation_max_buffer_pct, max(config.invalidation_base_buffer_pct, volatility_buffer))


def _invalidation_risk(trigger_level: float, row: pd.Series, config: SignalConfig, basis: str) -> dict[str, float | str]:
    buffer_pct = dynamic_invalidation_buffer(row, config)
    invalidation_price = trigger_level * (1 - buffer_pct)
    return {
        "trigger_level": float(trigger_level),
        "invalidation_price": float(invalidation_price),
        "risk_buffer_pct": float(buffer_pct),
        "invalidation_basis": basis,
    }


def _invalidation_text(risk: dict[str, float | str]) -> str:
    return (
        f"关键位：{float(risk['trigger_level']):.2f}；"
        f"实时失效价：{float(risk['invalidation_price']):.2f}；"
        "实时价跌破失效价则信号失效，未跌破前仍可按计划观察。"
    )


def _volume_expand(row: pd.Series, config: SignalConfig) -> bool:
    return bool(row.get("vol_ratio", 0) >= config.volume_expand_ratio or row.get("is_volume_expand", False))


def _volume_shrink(row: pd.Series, config: SignalConfig) -> bool:
    return bool(row.get("vol_ratio", 1) <= config.dry_volume_ratio or row.get("is_volume_shrink", False))


def _strong_bull(row: pd.Series, config: SignalConfig) -> bool:
    return bool(row["is_up"] and row["body_ratio"] >= config.strong_body_ratio and row["close_position"] >= config.close_near_high)


def _strong_bear(row: pd.Series, config: SignalConfig) -> bool:
    return bool(row["is_down"] and row["body_ratio"] >= config.strong_body_ratio and row["close_position"] <= config.close_near_low)


def _volume_price_stall_up(row: pd.Series, config: SignalConfig) -> bool:
    price_not_up_much = abs(float(row.get("pct_change", 0) or 0)) <= config.small_change_pct or row["close_position"] <= 0.55
    has_upper_shadow = row["upper_shadow_ratio"] >= 0.35
    return _volume_expand(row, config) and (price_not_up_much or has_upper_shadow)


def _volume_price_stall_down(row: pd.Series, config: SignalConfig) -> bool:
    price_not_down_much = abs(float(row.get("pct_change", 0) or 0)) <= config.small_change_pct or row["close_position"] >= 0.50
    has_lower_shadow = row["lower_shadow_ratio"] >= 0.35
    return _volume_expand(row, config) and (price_not_down_much or has_lower_shadow)


def _bullish_engulfing(previous: pd.Series, latest: pd.Series) -> bool:
    return bool(previous["is_down"] and latest["is_up"] and latest["close"] > previous["open"] and latest["open"] <= previous["close"])


def _bearish_engulfing(previous: pd.Series, latest: pd.Series) -> bool:
    return bool(previous["is_up"] and latest["is_down"] and latest["close"] < previous["open"] and latest["open"] >= previous["close"])
