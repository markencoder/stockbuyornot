from __future__ import annotations

from stockbuyornot.config import ScoringConfig
from stockbuyornot.levels import nearest_level
from stockbuyornot.models import KeyLevel, MarketStructure, ScoreBreakdown, SignalSide, Stage, TradeSignal


def score_analysis(
    close: float,
    structure: MarketStructure,
    supports: list[KeyLevel],
    resistances: list[KeyLevel],
    signals: list[TradeSignal],
    relative_strength: float | None = None,
    config: ScoringConfig = ScoringConfig(),
) -> ScoreBreakdown:
    explanations: list[str] = []
    structure_score = _structure_score(structure)
    explanations.append(f"结构分 {structure_score}: {structure.description}")

    support, support_distance = nearest_level(close, supports, 0.08)
    resistance, resistance_distance = nearest_level(close, resistances, 0.08)
    position_score = _position_score(support_distance, resistance_distance, signals)
    if support and support_distance is not None:
        explanations.append(f"位置分参考: 距离{support.name}{support_distance * 100:.1f}%")
    elif resistance and resistance_distance is not None:
        explanations.append(f"位置分参考: 距离{resistance.name}{resistance_distance * 100:.1f}%")
    else:
        explanations.append("位置分参考: 未贴近关键支撑/压力")

    signal_score = _signal_score(signals)
    if signals:
        explanations.append("量价信号: " + "、".join(signal.name for signal in signals[:3]))
    else:
        explanations.append("量价信号: 暂无明确关键信号")

    risk_score = _risk_score(close, signals, config)
    explanations.append(f"风险分 {risk_score}: 止损距离越近、失效条件越明确，得分越高")

    rs_score = _relative_strength_score(relative_strength, config)
    if relative_strength is not None:
        explanations.append(f"相对强度分 {rs_score}: 相对强度 {relative_strength:.2%}")
    else:
        explanations.append("相对强度分: 未提供指数/行业基准，暂按中性处理")

    total = int(max(0, min(100, structure_score + position_score + signal_score + risk_score + rs_score)))
    return ScoreBreakdown(
        total=total,
        structure=structure_score,
        position=position_score,
        signal=signal_score,
        risk=risk_score,
        relative_strength=rs_score,
        explanation=explanations,
    )


def _structure_score(structure: MarketStructure) -> int:
    if structure.stage == Stage.MARKUP:
        return 30
    if structure.stage == Stage.ACCUMULATION:
        return 15
    if structure.stage == Stage.RANGE:
        return 12
    if structure.stage == Stage.DISTRIBUTION:
        return -5
    if structure.stage == Stage.MARKDOWN:
        return -30
    return 0


def _position_score(
    support_distance: float | None,
    resistance_distance: float | None,
    signals: list[TradeSignal],
) -> int:
    buy_like = any(signal.side in {SignalSide.BUY, SignalSide.WATCH} for signal in signals)
    sell_like = any(signal.side in {SignalSide.SELL, SignalSide.AVOID} for signal in signals)
    distance = support_distance if buy_like else resistance_distance if sell_like else support_distance
    if distance is None:
        return -5
    if distance <= 0.015:
        return 20
    if distance <= 0.03:
        return 16
    if distance <= 0.05:
        return 10
    return -5


def _signal_score(signals: list[TradeSignal]) -> int:
    if not signals:
        return 0
    score = 0
    for signal in signals:
        if signal.name == "上涨中继买点":
            score += 25
        elif signal.name == "向上突破买点":
            score += 25
        elif signal.name == "底部反转买点":
            score += 15
        elif signal.name in {"区间底部低吸", "底部反转后缩量回踩"}:
            score += 12
        elif signal.name in {"需求衰竭/加速卖点", "放量滞涨卖点", "区间顶部卖点"}:
            score -= 15
        elif signal.name in {"向下反转卖点", "向下突破卖点"}:
            score -= 30
        elif signal.name == "下跌中继/回避信号":
            score -= 25
    return max(-30, min(30, score))


def _risk_score(close: float, signals: list[TradeSignal], config: ScoringConfig) -> int:
    stops = [signal.stop_loss for signal in signals if signal.stop_loss and signal.stop_loss < close]
    if not stops:
        if any(signal.side in {SignalSide.SELL, SignalSide.AVOID} for signal in signals):
            return -5
        return 0
    stop = max(stops)
    distance = (close - stop) / close
    if distance <= 0.03:
        return 10
    if distance <= 0.06:
        return 8
    if distance <= config.max_stop_distance_pct:
        return 5
    return -5


def _relative_strength_score(relative_strength: float | None, config: ScoringConfig) -> int:
    if relative_strength is None:
        return 5
    if relative_strength >= 0.10:
        return 10
    if relative_strength >= 0.03:
        return 8
    if relative_strength >= -0.03:
        return 5
    return 2
