from __future__ import annotations

from typing import Any

from stockbuyornot.config import FactorConfig, MultiFactorConfig
from stockbuyornot.models import AnalysisResult, FactorSnapshot, MultiFactorScores, SignalSide, Stage


def compute_multi_factor_scores(
    analysis: AnalysisResult,
    factors: FactorSnapshot,
    factor_config: FactorConfig = FactorConfig(),
    multi_config: MultiFactorConfig = MultiFactorConfig(),
) -> MultiFactorScores:
    liangjia_score, liangjia_components = _liangjia_score(analysis, factors.short_term, factor_config)
    short_score, short_components = _short_term_score(factors.short_term, factor_config)
    long_score, long_components = _long_term_score(factors.long_term, factor_config)
    execution_score, execution_components = _execution_window_score(analysis, factors.short_term, liangjia_score, short_score, factor_config)
    risk_penalty = _risk_penalty(analysis, factors, multi_config)
    overall = (
        liangjia_score * multi_config.liangjia_weight
        + short_score * multi_config.short_term_weight
        + long_score * multi_config.long_term_weight
        - risk_penalty * multi_config.risk_penalty_weight
    )
    return MultiFactorScores(
        liangjia_score=round(_clamp(liangjia_score), 2),
        short_term_score=round(_clamp(short_score), 2),
        long_term_score=round(_clamp(long_score), 2),
        overall_score=round(_clamp(overall), 2),
        components={
            "liangjia": liangjia_components,
            "short_term": short_components,
            "long_term": long_components,
            "execution_window": execution_components,
            "execution_score": execution_score,
            "risk_penalty": risk_penalty,
        },
    )


def _execution_window_score(
    analysis: AnalysisResult,
    factors: dict[str, Any],
    liangjia_score: float,
    short_score: float,
    config: FactorConfig,
) -> tuple[float, dict[str, Any]]:
    flags: list[str] = []
    has_buy_signal = any(signal.side == SignalSide.BUY for signal in analysis.signals)

    structure_score = 45.0
    if analysis.structure.stage == Stage.MARKUP:
        structure_score = 82.0
        flags.append("第二阶段主升")
    elif analysis.structure.stage in {Stage.ACCUMULATION, Stage.RANGE}:
        structure_score = 62.0
        flags.append("结构处于观察/转折区")
    elif analysis.structure.stage == Stage.MARKDOWN:
        structure_score = 18.0
        flags.append("第四阶段主跌")

    signal_score = 45.0
    if has_buy_signal:
        signal_score = 86.0
        flags.append("日K量价买点")
    elif liangjia_score >= 70:
        signal_score = 62.0
        flags.append("量价结构较好但缺少买点")
    elif liangjia_score < 55:
        signal_score = 35.0
        flags.append("量价结构优势不足")

    short_temperature_score = 50.0
    if config.ideal_short_score_low <= short_score <= config.ideal_short_score_high:
        short_temperature_score = 86.0
        flags.append("短期分处于刚转强区间")
    elif short_score < 50:
        short_temperature_score = 30.0
        flags.append("短期尚未止跌确认")
    elif short_score < config.ideal_short_score_low:
        short_temperature_score = 58.0
        flags.append("短期分偏低，仍需确认")
    elif short_score <= 82:
        short_temperature_score = 66.0
        flags.append("短期分偏强，注意追高")
    elif short_score > 82:
        short_temperature_score = 38.0
        flags.append("短期分过高，存在追高风险")

    ma20_gap = _num(factors.get("ma20_gap_pct"))
    ma_gap_score = 55.0
    if ma20_gap is not None:
        if -0.02 <= ma20_gap <= 0.05:
            ma_gap_score = 88.0
            flags.append("价格贴近MA20，未明显追高")
        elif 0.05 < ma20_gap <= config.chase_ma20_gap_pct:
            ma_gap_score = 72.0
            flags.append("价格略高于MA20，仍可观察")
        elif ma20_gap > config.severe_chase_ma20_gap_pct:
            ma_gap_score = 18.0
            flags.append("价格远离MA20，追高风险较大")
        elif ma20_gap > config.chase_ma20_gap_pct:
            ma_gap_score = 38.0
            flags.append("价格偏离MA20，需等回踩")
        elif ma20_gap < -0.05:
            ma_gap_score = 35.0
            flags.append("价格跌破MA20较多，承接不足")
        else:
            ma_gap_score = 62.0

    pullback = _num(factors.get("pullback_from_20d_high"))
    pullback_score = 55.0
    if pullback is not None:
        if config.ideal_pullback_min_pct <= pullback <= config.ideal_pullback_max_pct:
            pullback_score = 88.0
            flags.append("相对20日高点回撤处于3%-10%的较优买点窗口")
        elif pullback > -0.015 and short_score > config.ideal_short_score_high:
            pullback_score = 32.0
            flags.append("接近20日高点且短期较热，不适合追")
        elif -0.03 < pullback <= 0:
            pullback_score = 58.0
            flags.append("回撤较浅，等待更好的风险收益比")
        elif pullback < -0.14:
            pullback_score = 35.0
            flags.append("回撤过深，可能不是健康中继")
        else:
            pullback_score = 62.0

    volume_ratio = _num(factors.get("volume_ratio"))
    volume_score = 55.0
    if volume_ratio is not None:
        if volume_ratio <= 0.85 and pullback is not None and pullback < 0:
            volume_score = 78.0
            flags.append("回踩阶段量能收缩")
        elif 0.85 < volume_ratio < config.volume_expand_good:
            volume_score = 62.0
            flags.append("量能正常")
        elif volume_ratio >= config.volume_expand_good and pullback is not None and pullback > -0.02:
            volume_score = 32.0
            flags.append("靠近高位放量，谨防追涨")
        elif volume_ratio >= config.volume_expand_good:
            volume_score = 58.0
            flags.append("量能放大，需要确认方向")

    penalty = 0.0
    if any(signal.side in {SignalSide.SELL, SignalSide.AVOID} for signal in analysis.signals):
        penalty += 35.0
        flags.append("存在卖出/回避信号")
    if not has_buy_signal:
        penalty += 8.0
    if short_score > 88:
        penalty += 8.0
    if ma20_gap is not None and ma20_gap > config.severe_chase_ma20_gap_pct:
        penalty += 10.0

    raw_score = (
        structure_score * 0.18
        + signal_score * 0.24
        + short_temperature_score * 0.18
        + ma_gap_score * 0.18
        + pullback_score * 0.14
        + volume_score * 0.08
        - penalty
    )
    score = _clamp(raw_score)
    return score, {
        "score": round(score, 2),
        "structure_score": round(structure_score, 2),
        "signal_score": round(signal_score, 2),
        "short_temperature_score": round(short_temperature_score, 2),
        "ma_gap_score": round(ma_gap_score, 2),
        "pullback_score": round(pullback_score, 2),
        "volume_score": round(volume_score, 2),
        "penalty": round(penalty, 2),
        "ma20_gap_pct": ma20_gap,
        "pullback_from_20d_high": pullback,
        "has_buy_signal": has_buy_signal,
        "flags": flags[:8],
    }


def _liangjia_score(analysis: AnalysisResult, factors: dict[str, Any] | None = None, config: FactorConfig = FactorConfig()) -> tuple[float, dict[str, Any]]:
    base = float(analysis.score.total)
    radar = getattr(analysis, "radar", None)
    market_state = str(getattr(radar, "market_state", "unknown") or "unknown")
    market_modifier = {"strong": 5.0, "neutral": 0.0, "weak": -12.0, "unknown": 0.0}.get(market_state, 0.0)
    exit_risk = float(getattr(radar, "exit_risk_score", 0.0) or 0.0)
    risk_modifier = -15.0 if exit_risk >= 70 else -8.0 if exit_risk >= 55 else 0.0
    factor_modifier, factor_components = _liangjia_factor_modifier(factors or {}, config)
    score = base + market_modifier + risk_modifier + factor_modifier
    return score, {
        "base_score": base,
        "structure": analysis.score.structure,
        "position": analysis.score.position,
        "signal": analysis.score.signal,
        "risk": analysis.score.risk,
        "market_state": market_state,
        "market_modifier": market_modifier,
        "exit_risk_modifier": risk_modifier,
        "factor_modifier": round(factor_modifier, 2),
        **factor_components,
    }


def _liangjia_factor_modifier(factors: dict[str, Any], config: FactorConfig) -> tuple[float, dict[str, Any]]:
    volume_position = _liangjia_volume_position_modifier(factors, config)
    money_flow = _liangjia_money_flow_modifier(factors, config)
    trend_participation = _liangjia_trend_participation_modifier(factors)
    position_compression = _liangjia_position_compression_modifier(factors, config)
    volatility_risk = _liangjia_volatility_modifier(factors, config)
    total = volume_position + money_flow + trend_participation + position_compression + volatility_risk
    total = _clamp(total, -18.0, 18.0)
    return total, {
        "volume_position_modifier": round(volume_position, 2),
        "money_flow_modifier": round(money_flow, 2),
        "trend_participation_modifier": round(trend_participation, 2),
        "position_compression_modifier": round(position_compression, 2),
        "volatility_modifier": round(volatility_risk, 2),
    }


def _liangjia_volume_position_modifier(factors: dict[str, Any], config: FactorConfig) -> float:
    volume_ratio = _num(factors.get("volume_ratio")) or _num(factors.get("volume_expand_ratio"))
    close_position = _num(factors.get("close_position_day"))
    price_position = _num(factors.get("price_position_20d"))
    ret20 = _num(factors.get("return_20d"))
    score = 0.0
    if volume_ratio is not None and close_position is not None:
        if volume_ratio >= config.volume_expand_good and close_position >= 0.65:
            score += 6.0
        elif volume_ratio >= config.volume_expand_good and close_position <= 0.35:
            score -= 8.0
        elif volume_ratio < 0.75 and close_position >= 0.55:
            score += 2.0
    if price_position is not None:
        if 0.45 <= price_position <= 0.90:
            score += 2.0
        elif price_position > 0.98 and ret20 is not None and ret20 > config.strong_return_20d:
            score -= 4.0
        elif price_position < 0.25:
            score -= 3.0
    return score


def _liangjia_money_flow_modifier(factors: dict[str, Any], config: FactorConfig) -> float:
    obv_change = _num(factors.get("obv_change_5d"))
    mfi = _num(factors.get("mfi14"))
    score = 0.0
    if factors.get("obv_above_ma20"):
        score += 3.0
    if obv_change is not None:
        if obv_change > 3:
            score += 4.0
        elif obv_change > 0:
            score += 2.0
        elif obv_change < -3:
            score -= 5.0
        elif obv_change < 0:
            score -= 2.0
    if mfi is not None:
        if 45 <= mfi <= 75:
            score += 2.0
        elif mfi > config.mfi_overheat:
            score -= 4.0
        elif mfi < config.mfi_weak:
            score -= 3.0
    return score


def _liangjia_trend_participation_modifier(factors: dict[str, Any]) -> float:
    adx = _num(factors.get("adx14"))
    plus_di = _num(factors.get("plus_di14"))
    minus_di = _num(factors.get("minus_di14"))
    score = 0.0
    if adx is None or plus_di is None or minus_di is None:
        return score
    if adx >= 20 and plus_di > minus_di:
        score += 4.0
    elif adx >= 20 and plus_di < minus_di:
        score -= 5.0
    elif plus_di > minus_di:
        score += 2.0
    return score


def _liangjia_position_compression_modifier(factors: dict[str, Any], config: FactorConfig) -> float:
    pct_b = _num(factors.get("boll_percent_b"))
    bandwidth_ratio = _num(factors.get("boll_bandwidth_ratio"))
    volatility_ratio = _num(factors.get("volatility_ratio"))
    score = 0.0
    if bandwidth_ratio is not None and bandwidth_ratio <= config.bollinger_squeeze_ratio and pct_b is not None and pct_b >= 0.50:
        score += 4.0
    if pct_b is not None:
        if 0.50 <= pct_b <= 0.95:
            score += 2.0
        elif pct_b > 1.20:
            score -= 5.0
        elif pct_b < 0.20:
            score -= 4.0
    if volatility_ratio is not None:
        if volatility_ratio <= config.volatility_contraction_good:
            score += 2.0
        elif volatility_ratio > 1.50:
            score -= 3.0
    return score


def _liangjia_volatility_modifier(factors: dict[str, Any], config: FactorConfig) -> float:
    atr_pct = _num(factors.get("atr_pct"))
    drawdown = _num(factors.get("short_max_drawdown"))
    score = 0.0
    if atr_pct is not None:
        if atr_pct > config.atr_pct_high:
            score -= 6.0
        elif 0.015 <= atr_pct <= 0.055:
            score += 2.0
    if drawdown is not None and drawdown < config.short_max_drawdown_limit:
        score -= 5.0
    return score


def _short_term_score(factors: dict[str, Any], config: FactorConfig) -> tuple[float, dict[str, Any]]:
    returns = [
        _score_short_return(factors.get("return_1d"), config.strong_return_5d / 5, config.overheat_return_20d / 20),
        _score_short_return(factors.get("return_3d"), config.strong_return_5d * 0.7, config.overheat_return_20d * 0.35),
        _score_short_return(factors.get("return_5d"), config.strong_return_5d, config.overheat_return_20d * 0.5),
        _score_short_return(factors.get("return_10d"), config.strong_return_20d * 0.7, config.overheat_return_20d * 0.75),
        _score_short_return(factors.get("return_20d"), config.strong_return_20d, config.overheat_return_20d),
    ]
    return_score = sum(score * weight for score, weight in zip(returns, config.short_return_weights)) / max(sum(config.short_return_weights), 1e-9)
    ma_score = _score_short_ma(factors, config)
    volume_score = _score_volume(factors, config)
    momentum_score, momentum_components = _score_momentum(factors, config)
    breakout_score = 85.0 if factors.get("new_high_60") else 75.0 if factors.get("new_high_20") else 50.0
    risk_score = _score_short_risk(factors, config)
    total = (
        return_score * 0.30
        + ma_score * config.short_ma_weight
        + volume_score * config.short_volume_weight
        + momentum_score * config.short_momentum_weight
        + breakout_score * config.short_breakout_weight
        + risk_score * config.short_risk_weight
    ) / max(0.30 + config.short_ma_weight + config.short_volume_weight + config.short_momentum_weight + config.short_breakout_weight + config.short_risk_weight, 1e-9)
    return total, {
        "return_score": round(return_score, 2),
        "ma_score": round(ma_score, 2),
        "volume_score": round(volume_score, 2),
        "momentum_score": round(momentum_score, 2),
        **momentum_components,
        "breakout_score": round(breakout_score, 2),
        "risk_score": round(risk_score, 2),
    }


def _long_term_score(factors: dict[str, Any], config: FactorConfig) -> tuple[float, dict[str, Any]]:
    trend_score = _score_long_trend(factors)
    return_score = _score_long_return(factors, config)
    quality_score = _score_quality(factors, config)
    growth_score = _score_growth(factors)
    valuation_score = _score_valuation(factors, config)
    risk_score = _score_financial_risk(factors, config)
    total = (
        trend_score * config.long_trend_weight
        + return_score * config.long_return_weight
        + quality_score * config.long_quality_weight
        + growth_score * config.long_growth_weight
        + valuation_score * config.long_valuation_weight
        + risk_score * config.long_risk_weight
    ) / max(
        config.long_trend_weight
        + config.long_return_weight
        + config.long_quality_weight
        + config.long_growth_weight
        + config.long_valuation_weight
        + config.long_risk_weight,
        1e-9,
    )
    return total, {
        "trend_score": round(trend_score, 2),
        "return_score": round(return_score, 2),
        "quality_score": round(quality_score, 2),
        "growth_score": round(growth_score, 2),
        "valuation_score": round(valuation_score, 2),
        "risk_score": round(risk_score, 2),
    }


def _risk_penalty(analysis: AnalysisResult, factors: FactorSnapshot, config: MultiFactorConfig) -> float:
    penalty = 0.0
    if any(signal.side.value in {"sell", "avoid"} for signal in analysis.signals):
        penalty += 60.0
    radar = getattr(analysis, "radar", None)
    exit_risk = float(getattr(radar, "exit_risk_score", 0.0) or 0.0)
    penalty += max(0.0, exit_risk - config.major_risk_exit_score)
    if (factors.long_term.get("debt_to_assets") or 0) > 0.80:
        penalty += 25.0
    return min(100.0, penalty)


def _score_short_return(value: Any, good: float, overheat: float) -> float:
    ret = _num(value)
    if ret is None:
        return 50.0
    if ret < -good:
        return 25.0
    if ret < 0:
        return 45.0
    if ret <= good:
        return 65.0
    if ret <= overheat:
        return 82.0
    return 58.0


def _score_short_ma(factors: dict[str, Any], config: FactorConfig) -> float:
    close_proxy = factors.get("ma5")
    ma5, ma10, ma20 = _num(factors.get("ma5")), _num(factors.get("ma10")), _num(factors.get("ma20"))
    if ma5 is None or ma10 is None or ma20 is None:
        return 50.0
    score = 50.0
    if ma5 > ma10 > ma20:
        score += 30.0
    elif ma5 > ma10 or ma10 > ma20:
        score += 15.0
    ret20 = _num(factors.get("return_20d"))
    if ret20 is not None and ret20 > config.overheat_return_20d:
        score -= 18.0
    return score


def _score_volume(factors: dict[str, Any], config: FactorConfig) -> float:
    ratio = _num(factors.get("volume_ratio")) or _num(factors.get("volume_expand_ratio"))
    if ratio is None:
        return 50.0
    if ratio >= config.volume_expand_good:
        return min(85.0, 60.0 + (ratio - config.volume_expand_good) * 20)
    if ratio >= 0.75:
        return 58.0
    return 45.0


def _score_momentum(factors: dict[str, Any], config: FactorConfig) -> tuple[float, dict[str, Any]]:
    rsi_score = _score_rsi(factors, config)
    macd_score = _score_macd(factors)
    kdj_score = _score_kdj(factors, config)
    oscillator_score = _score_oscillator(factors, config)
    bollinger_score = _score_bollinger(factors, config)
    obv_score = _score_obv(factors)
    trend_strength_score = _score_trend_strength(factors)
    rs_score = _score_short_relative_strength(factors, config)
    volatility_score = _score_volatility_quality(factors, config)
    total = (
        rsi_score * config.short_rsi_weight
        + macd_score * config.short_macd_weight
        + kdj_score * config.short_kdj_weight
        + oscillator_score * config.short_oscillator_weight
        + bollinger_score * config.short_bollinger_weight
        + obv_score * config.short_obv_weight
        + trend_strength_score * config.short_trend_strength_weight
        + rs_score * config.short_relative_strength_weight
        + volatility_score * config.short_volatility_weight
    ) / max(
        config.short_rsi_weight
        + config.short_macd_weight
        + config.short_kdj_weight
        + config.short_oscillator_weight
        + config.short_bollinger_weight
        + config.short_obv_weight
        + config.short_trend_strength_weight
        + config.short_relative_strength_weight
        + config.short_volatility_weight,
        1e-9,
    )
    return _clamp(total), {
        "rsi_score": round(rsi_score, 2),
        "macd_score": round(macd_score, 2),
        "kdj_score": round(kdj_score, 2),
        "oscillator_score": round(oscillator_score, 2),
        "bollinger_score": round(bollinger_score, 2),
        "obv_score": round(obv_score, 2),
        "trend_strength_score": round(trend_strength_score, 2),
        "relative_strength_score": round(rs_score, 2),
        "volatility_score": round(volatility_score, 2),
    }


def _score_rsi(factors: dict[str, Any], config: FactorConfig) -> float:
    rsi = _num(factors.get("rsi14")) or _num(factors.get("rsi"))
    rsi6 = _num(factors.get("rsi6"))
    score = 50.0
    if rsi is not None:
        if config.rsi_strong_low <= rsi <= config.rsi_strong_high:
            score += 24.0
        elif rsi > config.rsi_overheat:
            score -= 22.0
        elif rsi < config.rsi_weak:
            score -= 14.0
        elif rsi >= 45:
            score += 8.0
    if rsi6 is not None and rsi is not None:
        if config.rsi_strong_low <= rsi6 <= config.rsi_overheat and rsi6 >= rsi:
            score += 10.0
        elif rsi6 > config.rsi_overheat:
            score -= 8.0
        elif rsi6 < config.rsi_weak:
            score -= 8.0
    return _clamp(score)


def _score_macd(factors: dict[str, Any]) -> float:
    dif = _num(factors.get("macd_dif"))
    dea = _num(factors.get("macd_dea"))
    hist = _num(factors.get("macd_hist"))
    hist_prev = _num(factors.get("macd_hist_prev"))
    hist_delta_3d = _num(factors.get("macd_hist_delta_3d"))
    score = 50.0
    if hist is not None:
        if hist > 0:
            score += 14.0
        else:
            score -= 10.0
    if dif is not None and dea is not None:
        if dif > dea:
            score += 12.0
        else:
            score -= 8.0
        if dif > 0 and dea > 0:
            score += 8.0
    if hist_prev is not None and hist is not None:
        if hist > hist_prev:
            score += 8.0
        elif hist < hist_prev and hist < 0:
            score -= 8.0
    if hist_delta_3d is not None:
        if hist_delta_3d > 0:
            score += 8.0
        elif hist_delta_3d < 0:
            score -= 6.0
    if factors.get("macd_golden_cross"):
        score += 16.0
    if factors.get("macd_above_zero"):
        score += 6.0
    return _clamp(score)


def _score_kdj(factors: dict[str, Any], config: FactorConfig) -> float:
    k = _num(factors.get("kdj_k"))
    d = _num(factors.get("kdj_d"))
    j = _num(factors.get("kdj_j"))
    j_delta = _num(factors.get("kdj_j_delta_3d"))
    score = 50.0
    if k is not None and d is not None:
        if k > d:
            score += 12.0
        else:
            score -= 8.0
        if config.kdj_strong_low <= k <= config.kdj_strong_high:
            score += 14.0
        elif k > config.kdj_overheat:
            score -= 16.0
        elif k < 25:
            score -= 10.0
    if j is not None:
        if config.kdj_strong_low <= j <= config.kdj_overheat:
            score += 8.0
        elif j > 100:
            score -= 12.0
        elif j < 20:
            score -= 8.0
    if j_delta is not None:
        if j_delta > 0:
            score += 8.0
        elif j_delta < 0:
            score -= 6.0
    if factors.get("kdj_golden_cross"):
        score += 14.0
    if factors.get("kdj_dead_cross"):
        score -= 16.0
    return _clamp(score)


def _score_oscillator(factors: dict[str, Any], config: FactorConfig) -> float:
    cci = _num(factors.get("cci14"))
    wr = _num(factors.get("williams_r14"))
    mfi = _num(factors.get("mfi14"))
    score = 50.0
    if cci is not None:
        if 0 <= cci <= 150:
            score += 14.0
        elif cci > config.cci_overheat:
            score -= 14.0
        elif cci < config.cci_weak:
            score -= 12.0
    if wr is not None:
        if -50 <= wr <= -15:
            score += 12.0
        elif wr > -10:
            score -= 12.0
        elif wr < -80:
            score -= 10.0
    if mfi is not None:
        if 45 <= mfi <= 75:
            score += 14.0
        elif mfi > config.mfi_overheat:
            score -= 14.0
        elif mfi < config.mfi_weak:
            score -= 10.0
    return _clamp(score)


def _score_bollinger(factors: dict[str, Any], config: FactorConfig) -> float:
    pct_b = _num(factors.get("boll_percent_b"))
    bandwidth_ratio = _num(factors.get("boll_bandwidth_ratio"))
    score = 50.0
    if pct_b is not None:
        if 0.55 <= pct_b <= 0.95:
            score += 18.0
        elif 0.95 < pct_b <= 1.12:
            score += 10.0
        elif pct_b > 1.20:
            score -= 16.0
        elif pct_b < 0.25:
            score -= 14.0
    if bandwidth_ratio is not None:
        if bandwidth_ratio <= config.bollinger_squeeze_ratio and pct_b is not None and pct_b >= 0.50:
            score += 14.0
        elif bandwidth_ratio > 1.60 and pct_b is not None and pct_b > 1.0:
            score -= 8.0
    return _clamp(score)


def _score_obv(factors: dict[str, Any]) -> float:
    change = _num(factors.get("obv_change_5d"))
    score = 50.0
    if factors.get("obv_above_ma20"):
        score += 18.0
    if change is not None:
        if change > 3:
            score += 18.0
        elif change > 0:
            score += 10.0
        elif change < -3:
            score -= 18.0
        elif change < 0:
            score -= 10.0
    return _clamp(score)


def _score_trend_strength(factors: dict[str, Any]) -> float:
    adx = _num(factors.get("adx14"))
    plus_di = _num(factors.get("plus_di14"))
    minus_di = _num(factors.get("minus_di14"))
    price_position = _num(factors.get("price_position_20d"))
    close_position = _num(factors.get("close_position_day"))
    score = 50.0
    if adx is not None and plus_di is not None and minus_di is not None:
        if adx >= 20 and plus_di > minus_di:
            score += 22.0
        elif adx >= 20 and plus_di < minus_di:
            score -= 18.0
        elif plus_di > minus_di:
            score += 8.0
    if price_position is not None:
        if 0.55 <= price_position <= 0.95:
            score += 12.0
        elif price_position > 0.98:
            score -= 5.0
        elif price_position < 0.30:
            score -= 10.0
    if close_position is not None:
        if close_position >= 0.65:
            score += 8.0
        elif close_position <= 0.30:
            score -= 8.0
    return _clamp(score)


def _score_short_relative_strength(factors: dict[str, Any], config: FactorConfig) -> float:
    stock_vs_bench = _num(factors.get("stock_vs_benchmark_20d"))
    stock_vs_sector = _num(factors.get("stock_vs_sector_20d"))
    values = [value for value in [stock_vs_bench, stock_vs_sector] if value is not None]
    if not values:
        return 50.0
    avg = sum(values) / len(values)
    score = 50.0
    if avg >= config.relative_strength_good:
        score += 25.0
    elif avg >= 0:
        score += 10.0
    elif avg <= -config.relative_strength_good:
        score -= 20.0
    if stock_vs_bench is not None and stock_vs_bench >= config.relative_strength_good:
        score += 8.0
    if stock_vs_sector is not None and stock_vs_sector >= config.relative_strength_good:
        score += 8.0
    return _clamp(score)


def _score_volatility_quality(factors: dict[str, Any], config: FactorConfig) -> float:
    volatility_ratio = _num(factors.get("volatility_ratio"))
    atr_pct = _num(factors.get("atr_pct"))
    score = 60.0
    if volatility_ratio is not None:
        if volatility_ratio <= config.volatility_contraction_good:
            score += 18.0
        elif volatility_ratio > 1.50:
            score -= 14.0
    if atr_pct is not None:
        if 0.015 <= atr_pct <= 0.055:
            score += 10.0
        elif atr_pct > config.atr_pct_high:
            score -= 20.0
    return _clamp(score)


def _score_short_risk(factors: dict[str, Any], config: FactorConfig) -> float:
    drawdown = _num(factors.get("short_max_drawdown"))
    atr = _num(factors.get("atr"))
    atr_pct = _num(factors.get("atr_pct"))
    score = 70.0
    if drawdown is not None and drawdown < config.short_max_drawdown_limit:
        score -= 30.0
    if atr_pct is not None and atr_pct > config.atr_pct_high:
        score -= 18.0
    elif atr is not None and atr > 0:
        score += 5.0
    return _clamp(score)


def _score_long_trend(factors: dict[str, Any]) -> float:
    ma60, ma120, ma250 = _num(factors.get("ma60")), _num(factors.get("ma120")), _num(factors.get("ma250"))
    if ma60 is None or ma120 is None:
        return 50.0
    if ma250 is not None and ma60 > ma120 > ma250:
        return 88.0
    if ma60 > ma120:
        return 72.0
    return 38.0


def _score_long_return(factors: dict[str, Any], config: FactorConfig) -> float:
    ret60, ret120, ret250 = _num(factors.get("return_60d")), _num(factors.get("return_120d")), _num(factors.get("return_250d"))
    values = [value for value in [ret60, ret120, ret250] if value is not None]
    if not values:
        return 50.0
    avg = sum(values) / len(values)
    if avg >= config.long_return_good:
        return 82.0
    if avg >= 0:
        return 65.0
    if avg <= config.long_return_weak:
        return 35.0
    return 48.0


def _score_quality(factors: dict[str, Any], config: FactorConfig) -> float:
    metrics = [_num(factors.get(key)) for key in ["roe", "roa", "gross_margin", "net_margin", "ocf_to_profit"]]
    present = [value for value in metrics if value is not None]
    if not present:
        return 50.0
    score = 50.0
    roe = _num(factors.get("roe"))
    if roe is not None and roe >= config.min_roe:
        score += 20.0
    if (_num(factors.get("roa")) or 0) > 0.04:
        score += 10.0
    if (_num(factors.get("net_margin")) or 0) > 0.08:
        score += 10.0
    if (_num(factors.get("ocf_to_profit")) or 0) >= 1:
        score += 10.0
    return _clamp(score)


def _score_growth(factors: dict[str, Any]) -> float:
    growth = [_num(factors.get(key)) for key in ["revenue_growth", "net_profit_growth", "deducted_profit_growth"]]
    present = [value for value in growth if value is not None]
    if not present:
        return 50.0
    avg = sum(present) / len(present)
    if avg > 0.20:
        return 85.0
    if avg > 0.05:
        return 70.0
    if avg >= 0:
        return 55.0
    return 35.0


def _score_valuation(factors: dict[str, Any], config: FactorConfig) -> float:
    pe_pct, pb_pct = _num(factors.get("pe_percentile")), _num(factors.get("pb_percentile"))
    peg = _num(factors.get("peg"))
    values = [value for value in [pe_pct, pb_pct] if value is not None]
    if not values and peg is None:
        return 50.0
    score = 55.0
    if values:
        avg = sum(values) / len(values)
        if avg <= config.valuation_low_percentile:
            score += 18.0
        elif avg >= config.max_valuation_percentile:
            score -= 25.0
    if peg is not None:
        if peg <= 1.0:
            score += 15.0
        elif peg > 2.0:
            score -= 15.0
    return _clamp(score)


def _score_financial_risk(factors: dict[str, Any], config: FactorConfig) -> float:
    score = 70.0
    debt = _num(factors.get("debt_to_assets"))
    interest_debt = _num(factors.get("interest_debt_ratio"))
    drawdown = _num(factors.get("long_max_drawdown"))
    if debt is not None and debt > config.max_debt_to_assets:
        score -= 25.0
    if interest_debt is not None and interest_debt > 0.45:
        score -= 18.0
    if drawdown is not None and drawdown < config.long_max_drawdown_limit:
        score -= 20.0
    return _clamp(score)


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))
