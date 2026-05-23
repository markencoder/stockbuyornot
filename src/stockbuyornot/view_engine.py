from __future__ import annotations

from typing import Any

from stockbuyornot.config import MultiFactorConfig
from stockbuyornot.models import (
    AnalysisResult,
    FactorSnapshot,
    LongTermView,
    MultiFactorScores,
    ShortTermView,
    SignalSide,
    Stage,
    TradeSignal,
)


STAGE_LABELS = {
    Stage.ACCUMULATION: "第一阶段-筑底",
    Stage.MARKUP: "第二阶段-主升",
    Stage.DISTRIBUTION: "第三阶段-筑顶/高位震荡",
    Stage.MARKDOWN: "第四阶段-主跌",
    Stage.RANGE: "区间震荡",
    Stage.UNKNOWN: "结构不明",
}


def build_investment_views(
    analysis: AnalysisResult,
    factors: FactorSnapshot,
    scores: MultiFactorScores,
    config: MultiFactorConfig = MultiFactorConfig(),
) -> tuple[LongTermView, ShortTermView]:
    return (
        build_long_term_view(factors, scores, config),
        build_short_term_view(analysis, scores, config),
    )


def build_long_term_view(
    factors: FactorSnapshot,
    scores: MultiFactorScores,
    config: MultiFactorConfig = MultiFactorConfig(),
) -> LongTermView:
    score = float(scores.long_term_score)
    long_factors = factors.long_term
    components = scores.components.get("long_term", {}) if scores.components else {}
    risks = _long_term_risks(long_factors)

    if any(item.startswith("财务风险") for item in risks):
        advice = "长期回避" if score < config.long_good_long_min else "长期谨慎关注"
        rating = "长期风险偏高"
    elif score >= max(config.long_good_long_min + 7, 75):
        advice = "长期可关注"
        rating = "长期质量较好"
    elif score >= 60:
        advice = "长期谨慎关注"
        rating = "长期条件尚可"
    elif score >= 45:
        advice = "长期暂不关注"
        rating = "长期优势不足"
    else:
        advice = "长期回避"
        rating = "长期偏弱"

    key_factors = _long_term_key_factors(long_factors, components)
    explanation = (
        "长期模块只评价中长期关注价值，主要看长期均线、长期收益、基本面质量、成长、估值和财务风险；"
        "它不会单独触发短期买入。"
    )
    return LongTermView(
        score=round(score, 2),
        rating=rating,
        advice=advice,
        key_factors=key_factors,
        risk_warnings=risks,
        explanation=explanation,
    )


def build_short_term_view(
    analysis: AnalysisResult,
    scores: MultiFactorScores,
    config: MultiFactorConfig = MultiFactorConfig(),
) -> ShortTermView:
    risk_signal = _strongest_signal(analysis.signals, {SignalSide.SELL, SignalSide.AVOID})
    buy_signal = _strongest_signal(analysis.signals, {SignalSide.BUY})
    watch_signal = _strongest_signal(analysis.signals, {SignalSide.WATCH})
    main_signal = risk_signal or buy_signal or watch_signal

    signal_type = _signal_label(main_signal, analysis.structure.stage)
    signal_strength = int(main_signal.strength) if main_signal else 0
    entry_zone = buy_signal.entry_zone if buy_signal else None
    stop_loss = buy_signal.stop_loss if buy_signal else _best_stop(analysis.signals)
    risk_pct = _risk_pct(analysis.close, stop_loss)
    risk_warnings = _short_term_risks(analysis, risk_signal, risk_pct, config)
    execution_score, execution_flags = _execution_window(scores)

    if risk_signal is not None:
        signal_direction = "偏空"
        if risk_signal.side == SignalSide.AVOID or analysis.structure.stage == Stage.MARKDOWN:
            advice = "短期回避"
            action_code = "avoid"
        elif risk_signal.strength >= 88:
            advice = "短期卖出"
            action_code = "sell"
        else:
            advice = "短期减仓"
            action_code = "reduce"
    elif buy_signal is not None:
        signal_direction = "偏多"
        if risk_pct is not None and risk_pct > config.max_buy_risk_pct:
            advice = "短期等待"
            action_code = "wait"
            risk_warnings.append("止损距离偏大，买点需要等待更好的价格或确认")
        elif execution_score < config.execution_good_min:
            advice = "短期等待"
            action_code = "wait"
            risk_warnings.append(_execution_wait_reason(execution_score, execution_flags))
        elif analysis.structure.stage == Stage.MARKUP and buy_signal.strength >= 90:
            advice = "短期加仓"
            action_code = "add"
        else:
            advice = "短期买入"
            action_code = "buy"
    elif watch_signal is not None:
        signal_direction = "中性"
        advice = "短期等待"
        action_code = "wait"
    elif analysis.structure.stage == Stage.MARKDOWN:
        signal_direction = "偏空"
        advice = "短期回避"
        action_code = "avoid"
        risk_warnings.append("当前处于第四阶段主跌，缩量反弹不能当作买点")
    elif analysis.structure.stage == Stage.MARKUP and scores.liangjia_score >= config.short_good_liangjia_min:
        signal_direction = "中性"
        if execution_score < 55:
            advice = "短期等待"
            action_code = "wait"
            risk_warnings.append(_execution_wait_reason(execution_score, execution_flags))
        else:
            advice = "短期持有"
            action_code = "hold"
    else:
        signal_direction = "中性"
        advice = "短期等待"
        action_code = "wait"

    key_factors = _short_term_key_factors(analysis, main_signal, scores)
    explanation = _short_term_explanation(signal_direction, advice)
    return ShortTermView(
        liangjia_score=round(float(scores.liangjia_score), 2),
        short_term_score=round(float(scores.short_term_score), 2),
        signal_type=signal_type,
        signal_strength=signal_strength,
        signal_direction=signal_direction,
        advice=advice,
        action_code=action_code,
        entry_zone=entry_zone,
        stop_loss=stop_loss,
        risk_pct=risk_pct,
        key_factors=key_factors,
        risk_warnings=list(dict.fromkeys(risk_warnings)),
        explanation=explanation,
    )


def stage_label(stage: Stage) -> str:
    return STAGE_LABELS.get(stage, str(stage.value))


def _strongest_signal(signals: list[TradeSignal], sides: set[SignalSide]) -> TradeSignal | None:
    candidates = [signal for signal in signals if signal.side in sides]
    return max(candidates, key=lambda signal: signal.strength) if candidates else None


def _signal_label(signal: TradeSignal | None, stage: Stage) -> str:
    if signal is None:
        return "暂无明确量价信号"
    if signal.side == SignalSide.BUY:
        return "上涨中继买点" if stage == Stage.MARKUP else "量价买点"
    if signal.side == SignalSide.WATCH:
        return "平衡/观察信号"
    if signal.side == SignalSide.AVOID:
        return "下跌中继/回避信号"
    if signal.side == SignalSide.SELL:
        if signal.strength >= 88:
            return "向下反转/向下突破卖点"
        return "放量滞涨/需求衰竭风险"
    return signal.name


def _long_term_key_factors(factors: dict[str, Any], components: dict[str, Any]) -> list[str]:
    items: list[str] = []
    ma60, ma120, ma250 = _num(factors.get("ma60")), _num(factors.get("ma120")), _num(factors.get("ma250"))
    if ma60 is not None and ma120 is not None:
        if ma250 is not None and ma60 > ma120 > ma250:
            items.append("长期均线呈多头排列，长期趋势较健康")
        elif ma60 > ma120:
            items.append("MA60 位于 MA120 上方，长期趋势有改善")
        else:
            items.append("长期均线尚未形成多头排列")
    for key, label in [("return_60d", "60日收益"), ("return_120d", "120日收益"), ("return_250d", "250日收益")]:
        value = _num(factors.get(key))
        if value is not None:
            items.append(f"{label} {value:.1%}")
    for key, label in [
        ("quality_score", "质量分"),
        ("growth_score", "成长分"),
        ("valuation_score", "估值分"),
        ("risk_score", "长期风险分"),
    ]:
        value = _num(components.get(key))
        if value is not None:
            items.append(f"{label} {value:.0f}")
    return items[:8]


def _long_term_risks(factors: dict[str, Any]) -> list[str]:
    risks: list[str] = []
    debt = _num(factors.get("debt_to_assets"))
    interest_debt = _num(factors.get("interest_debt_ratio"))
    pe_pct = _num(factors.get("pe_percentile"))
    pb_pct = _num(factors.get("pb_percentile"))
    drawdown = _num(factors.get("long_max_drawdown"))
    volatility = _num(factors.get("long_volatility"))
    if debt is not None and debt >= 0.80:
        risks.append("财务风险：资产负债率过高")
    elif debt is not None and debt >= 0.70:
        risks.append("资产负债率偏高")
    if interest_debt is not None and interest_debt >= 0.55:
        risks.append("财务风险：有息负债率过高")
    if pe_pct is not None and pe_pct >= 0.85:
        risks.append("PE历史分位偏高，估值安全边际不足")
    if pb_pct is not None and pb_pct >= 0.85:
        risks.append("PB历史分位偏高")
    if drawdown is not None and drawdown <= -0.35:
        risks.append("长期最大回撤偏大")
    if volatility is not None and volatility >= 0.45:
        risks.append("长期波动率偏高")
    return risks


def _short_term_key_factors(analysis: AnalysisResult, signal: TradeSignal | None, scores: MultiFactorScores) -> list[str]:
    execution_score, execution_flags = _execution_window(scores)
    items = [
        f"结构：{stage_label(analysis.structure.stage)}",
        f"量价分：{scores.liangjia_score:.0f}",
        f"买点执行窗口分：{execution_score:.0f}",
    ]
    if signal is not None:
        items.append(f"当前信号强度：{signal.strength}")
        if signal.logic:
            items.append(signal.logic)
    if analysis.support_levels:
        items.append(f"最近支撑：{analysis.support_levels[0].name} {analysis.support_levels[0].price:.2f}")
    if analysis.resistance_levels:
        items.append(f"最近压力：{analysis.resistance_levels[0].name} {analysis.resistance_levels[0].price:.2f}")
    items.extend(execution_flags[:3])
    return items[:8]


def _short_term_risks(
    analysis: AnalysisResult,
    risk_signal: TradeSignal | None,
    risk_pct: float | None,
    config: MultiFactorConfig,
) -> list[str]:
    risks: list[str] = []
    if analysis.structure.stage == Stage.MARKDOWN:
        risks.append("第四阶段主跌，短期优先回避")
    if risk_signal is not None:
        risks.append(_signal_label(risk_signal, analysis.structure.stage))
    radar = getattr(analysis, "radar", None)
    exit_risk = float(getattr(radar, "exit_risk_score", 0.0) or 0.0)
    if exit_risk >= config.major_risk_exit_score:
        risks.append("卖出风险评分较高")
    if risk_pct is not None and risk_pct > config.max_buy_risk_pct:
        risks.append("止损距离超过短期交易风险上限")
    return risks


def _short_term_explanation(direction: str, advice: str) -> str:
    return (
        f"短期模块只根据量价供需主线判断，当前方向为{direction}，建议为{advice}。"
        "短期买入/加仓必须来自原量价买点；长期价值较好不会自动转成短期买入。"
    )


def _execution_window(scores: MultiFactorScores) -> tuple[float, list[str]]:
    components = scores.components or {}
    item = components.get("execution_window", {}) or {}
    if isinstance(item, dict):
        score = _num(item.get("score")) or _num(components.get("execution_score")) or 70.0
        flags = item.get("flags", []) or []
        return float(score), [str(flag) for flag in flags]
    score = _num(item) or _num(components.get("execution_score")) or 70.0
    return float(score), []


def _execution_wait_reason(score: float, flags: list[str]) -> str:
    joined = "；".join(flags[:3])
    if any("追高" in flag or "远离MA20" in flag for flag in flags):
        return f"执行窗口分 {score:.0f}，短期已偏热或远离MA20，当前不追高"
    if any("尚未止跌" in flag or "承接不足" in flag for flag in flags):
        return f"执行窗口分 {score:.0f}，量价结构尚可但短期还没确认止跌"
    return f"执行窗口分 {score:.0f}，量价、短期确认和风险收益比尚未形成最佳买点窗口" + (f"：{joined}" if joined else "")


def _best_stop(signals: list[TradeSignal]) -> float | None:
    stops = [signal.stop_loss for signal in signals if signal.stop_loss]
    return max(stops) if stops else None


def _risk_pct(close: float, stop_loss: float | None) -> float | None:
    if stop_loss is None or stop_loss <= 0 or close <= 0:
        return None
    return round(max(0.0, float(close - stop_loss) / close), 4)


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
