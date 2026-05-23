from __future__ import annotations

from stockbuyornot.config import MultiFactorConfig
from stockbuyornot.models import AnalysisResult, FactorSnapshot, MultiFactorScores, SignalSide, Stage, StockClassification, TradeSignal
from stockbuyornot.view_engine import stage_label


def classify_stock(
    analysis: AnalysisResult,
    factors: FactorSnapshot,
    scores: MultiFactorScores,
    config: MultiFactorConfig = MultiFactorConfig(),
) -> StockClassification:
    main_signal = _main_signal(analysis.signals)
    buy_signal = _best_signal(analysis.signals, SignalSide.BUY)
    key_signals = [signal.name for signal in analysis.signals]
    risk_flags = _risk_flags(analysis, factors, config)
    entry_zone = buy_signal.entry_zone if buy_signal else None
    stop_loss = buy_signal.stop_loss if buy_signal else _best_stop(analysis.signals)
    risk_pct = _risk_pct(analysis.close, stop_loss)
    buy_point_type = buy_signal.name if buy_signal else ""

    if risk_flags:
        action = _risk_action(analysis)
        return StockClassification(
            category="剔除池/回避池",
            action=action,
            priority=0,
            main_signal=main_signal,
            buy_point_type=buy_point_type,
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            risk_pct=risk_pct,
            key_signals=key_signals,
            risk_flags=risk_flags,
            explanation=[
                "风险信号优先于机会信号。",
                "即使长期基本面较好，只要量价出现转弱、主跌或高位风险，也优先降级处理。",
            ],
        )

    has_liangjia_buy = buy_signal is not None
    has_strong_signal = has_liangjia_buy or any(signal.side == SignalSide.WATCH and signal.strength >= 70 for signal in analysis.signals)
    execution_score = _execution_score(scores)
    short_good = (
        has_strong_signal
        and scores.liangjia_score >= config.short_good_liangjia_min
        and scores.short_term_score >= config.short_good_short_min
        and execution_score >= config.execution_good_min
        and (risk_pct is None or risk_pct <= config.max_buy_risk_pct)
    )
    long_good = scores.long_term_score >= config.long_good_long_min and _long_risk_ok(factors)

    if short_good and long_good:
        action = "ADD" if has_liangjia_buy else "WAIT"
        category = "短期与长期兼具"
        priority = 5
        explanation = ["量价买点、短期走势和长期分层同时较好，属于最高优先级候选。"]
    elif short_good:
        action = "BUY" if has_liangjia_buy else "WAIT"
        category = "短期好股"
        priority = 4
        explanation = ["原量价算法已有买点或强信号，短期走势确认较好。"]
    elif long_good:
        action = "WAIT" if scores.liangjia_score >= config.watch_min_score else "WATCH"
        category = "长期好股"
        priority = 3
        explanation = ["长期趋势、质量或估值层面较好；没有原量价买点时只能等待或观察，不能直接买入。"]
    elif max(scores.liangjia_score, scores.short_term_score, scores.long_term_score) >= config.watch_min_score:
        action = "WATCH"
        category = "观察池"
        priority = 2
        explanation = ["部分条件较好，但买点、趋势或长期质量尚未同时确认。"]
    else:
        action = "AVOID"
        category = "剔除池/回避池"
        priority = 1
        explanation = ["缺少明确量价优势，综合评分不足。"]

    if action in {"BUY", "ADD"} and not has_liangjia_buy:
        action = "WAIT"
        explanation.append("BUY/ADD 必须来自原量价买点，当前没有买点，因此降级为 WAIT。")
    if has_liangjia_buy and execution_score < config.execution_good_min:
        action = "WAIT"
        explanation.append("执行窗口分不足，说明量价、短期确认、乖离或回踩质量尚未平衡，降级为 WAIT。")

    return StockClassification(
        category=category,
        action=action,
        priority=priority,
        main_signal=main_signal,
        buy_point_type=buy_point_type,
        entry_zone=entry_zone,
        stop_loss=stop_loss,
        risk_pct=risk_pct,
        key_signals=key_signals,
        risk_flags=risk_flags,
        explanation=explanation,
    )


def _risk_flags(analysis: AnalysisResult, factors: FactorSnapshot, config: MultiFactorConfig) -> list[str]:
    flags: list[str] = []
    if analysis.structure.stage == Stage.MARKDOWN:
        flags.append(stage_label(Stage.MARKDOWN))
    for signal in analysis.signals:
        if signal.side == SignalSide.AVOID:
            flags.append(signal.name or "回避信号")
        elif signal.side == SignalSide.SELL:
            flags.append(signal.name or "卖出信号")
    radar = getattr(analysis, "radar", None)
    exit_risk = float(getattr(radar, "exit_risk_score", 0.0) or 0.0)
    if exit_risk >= config.major_risk_exit_score:
        flags.append("卖出风险评分过高")
    debt = factors.long_term.get("debt_to_assets")
    if debt is not None and float(debt) >= 0.80:
        flags.append("资产负债率过高")
    interest_debt = factors.long_term.get("interest_debt_ratio")
    if interest_debt is not None and float(interest_debt) >= 0.55:
        flags.append("有息负债率过高")
    return list(dict.fromkeys(flags))


def _risk_action(analysis: AnalysisResult) -> str:
    sides = {signal.side for signal in analysis.signals}
    if SignalSide.SELL in sides:
        return "SELL"
    if SignalSide.AVOID in sides or analysis.structure.stage == Stage.MARKDOWN:
        return "AVOID"
    return "REDUCE"


def _long_risk_ok(factors: FactorSnapshot) -> bool:
    debt = factors.long_term.get("debt_to_assets")
    interest_debt = factors.long_term.get("interest_debt_ratio")
    return not ((debt is not None and float(debt) >= 0.80) or (interest_debt is not None and float(interest_debt) >= 0.55))


def _main_signal(signals: list[TradeSignal]) -> str:
    if not signals:
        return "暂无明确量价信号"
    return max(signals, key=lambda signal: abs(signal.strength)).name


def _best_signal(signals: list[TradeSignal], side: SignalSide) -> TradeSignal | None:
    candidates = [signal for signal in signals if signal.side == side]
    return max(candidates, key=lambda signal: signal.strength) if candidates else None


def _best_stop(signals: list[TradeSignal]) -> float | None:
    stops = [signal.stop_loss for signal in signals if signal.stop_loss]
    return max(stops) if stops else None


def _risk_pct(close: float, stop_loss: float | None) -> float | None:
    if stop_loss is None or stop_loss <= 0 or close <= 0:
        return None
    return max(0.0, float(close - stop_loss) / close)


def _execution_score(scores: MultiFactorScores) -> float:
    components = scores.components or {}
    raw = components.get("execution_score")
    try:
        if raw is not None:
            return float(raw)
        item = components.get("execution_window", {}) or {}
        if isinstance(item, dict):
            return float(item.get("score", 70.0))
        return float(item)
    except (TypeError, ValueError, AttributeError):
        return 70.0
