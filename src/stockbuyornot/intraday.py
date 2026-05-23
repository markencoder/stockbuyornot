from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class IntradaySummary:
    symbol: str
    latest_time: pd.Timestamp
    latest_price: float
    session_return_pct: float
    momentum_30m_pct: float | None
    range_position_pct: float | None
    volume_ratio: float | None
    vwap: float | None
    vwap_gap_pct: float | None
    trend_label: str
    volume_label: str
    conclusion: str
    explanation: str


@dataclass(frozen=True)
class IntradayAdjustment:
    base_liangjia_score: float
    base_short_term_score: float
    liangjia_modifier: float
    short_term_modifier: float
    adjusted_liangjia_score: float
    adjusted_short_term_score: float
    action: str
    action_level: str
    support_flags: list[str]
    risk_flags: list[str]
    formula: str
    explanation: str


def summarize_intraday(df: pd.DataFrame, symbol: str = "") -> IntradaySummary:
    """Summarize the latest trading session from intraday OHLCV data."""
    if df.empty:
        raise ValueError("分时数据为空。")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)
    latest_day = data["date"].dt.date.iloc[-1]
    session = data[data["date"].dt.date == latest_day].copy()
    if session.empty:
        raise ValueError("分时数据缺少最新交易日。")

    first = session.iloc[0]
    last = session.iloc[-1]
    latest_price = float(last["close"])
    session_open = float(first["open"])
    session_high = float(session["high"].max())
    session_low = float(session["low"].min())
    session_return_pct = _pct_change(latest_price, session_open)
    range_position_pct = _range_position(latest_price, session_low, session_high)
    momentum_30m_pct = _recent_momentum(session)
    volume_ratio = _last_volume_ratio(data)
    vwap = _session_vwap(session)
    vwap_gap_pct = None if vwap is None else _pct_change(latest_price, vwap)

    trend_label = _trend_label(session_return_pct, momentum_30m_pct, range_position_pct, vwap_gap_pct)
    volume_label = _volume_label(volume_ratio)
    conclusion = _conclusion(trend_label, volume_label, session_return_pct, vwap_gap_pct)
    explanation = _explanation(conclusion, session_return_pct, momentum_30m_pct, range_position_pct, volume_ratio, vwap_gap_pct)

    return IntradaySummary(
        symbol=symbol or str(last.get("symbol", "")),
        latest_time=pd.to_datetime(last["date"]),
        latest_price=latest_price,
        session_return_pct=session_return_pct,
        momentum_30m_pct=momentum_30m_pct,
        range_position_pct=range_position_pct,
        volume_ratio=volume_ratio,
        vwap=vwap,
        vwap_gap_pct=vwap_gap_pct,
        trend_label=trend_label,
        volume_label=volume_label,
        conclusion=conclusion,
        explanation=explanation,
    )


def compute_intraday_adjustment(
    summary: IntradaySummary,
    base_liangjia_score: float,
    base_short_term_score: float,
    daily_advice: str = "",
    signal_direction: str = "",
) -> IntradayAdjustment:
    """Convert intraday confirmation into an execution-layer score adjustment."""
    support_flags: list[str] = []
    risk_flags: list[str] = []
    short_modifier = 0.0
    liangjia_modifier = 0.0

    if summary.session_return_pct >= 0.01:
        short_modifier += 4
        liangjia_modifier += 2
        support_flags.append("当日涨幅超过1%，盘中需求偏强")
    elif summary.session_return_pct > 0:
        short_modifier += 2
        liangjia_modifier += 1
        support_flags.append("当日小幅上涨")
    elif summary.session_return_pct <= -0.01:
        short_modifier -= 5
        liangjia_modifier -= 3
        risk_flags.append("当日跌幅超过1%，盘中卖压偏强")
    elif summary.session_return_pct < 0:
        short_modifier -= 2
        liangjia_modifier -= 1
        risk_flags.append("当日小幅走弱")

    if summary.momentum_30m_pct is not None:
        if summary.momentum_30m_pct >= 0.003:
            short_modifier += 3
            liangjia_modifier += 1
            support_flags.append("最近30分钟价格继续走强")
        elif summary.momentum_30m_pct <= -0.003:
            short_modifier -= 4
            liangjia_modifier -= 2
            risk_flags.append("最近30分钟价格转弱")

    if summary.range_position_pct is not None:
        if summary.range_position_pct >= 0.7:
            short_modifier += 3
            liangjia_modifier += 2
            support_flags.append("价格位于日内高位区，承接较好")
        elif summary.range_position_pct <= 0.3:
            short_modifier -= 4
            liangjia_modifier -= 3
            risk_flags.append("价格位于日内低位区，承接不足")

    if summary.vwap_gap_pct is not None:
        if summary.vwap_gap_pct >= 0.003:
            short_modifier += 4
            liangjia_modifier += 3
            support_flags.append("价格站在VWAP上方，盘中资金成本有支撑")
        elif summary.vwap_gap_pct < 0:
            short_modifier -= 4
            liangjia_modifier -= 3
            risk_flags.append("价格低于VWAP，盘中资金成本压制")

    if summary.volume_ratio is not None:
        if summary.volume_ratio >= 1.3 and summary.session_return_pct > 0:
            short_modifier += 3
            liangjia_modifier += 4
            support_flags.append("放量上涨，需求参与度提高")
        elif summary.volume_ratio >= 1.3 and summary.session_return_pct < 0:
            short_modifier -= 5
            liangjia_modifier -= 5
            risk_flags.append("放量下跌，供应压力提高")
        elif summary.volume_ratio <= 0.7 and summary.session_return_pct < 0:
            short_modifier -= 2
            risk_flags.append("缩量下跌但仍未收回，先等待确认")

    if summary.conclusion == "分时支持偏多":
        short_modifier += 3
        liangjia_modifier += 2
        support_flags.append("分时综合结论偏多")
    elif summary.conclusion in {"分时提示风险", "分时放量转弱"}:
        short_modifier -= 5
        liangjia_modifier -= 4
        risk_flags.append(summary.conclusion)

    if "偏空" in signal_direction and short_modifier > 0:
        short_modifier = min(short_modifier, 3)
        liangjia_modifier = min(liangjia_modifier, 2)
        risk_flags.append("日K量价方向偏空，分时反弹只作修复看待")
    if any(word in daily_advice for word in ["卖出", "减仓", "回避"]) and short_modifier > 0:
        short_modifier = min(short_modifier, 2)
        liangjia_modifier = min(liangjia_modifier, 1)
        risk_flags.append("日K建议偏防守，分时不能单独改成买入")

    short_modifier = _clamp(short_modifier, -20, 15)
    liangjia_modifier = _clamp(liangjia_modifier, -18, 12)
    adjusted_short = _clamp(base_short_term_score + short_modifier, 0, 100)
    adjusted_liangjia = _clamp(base_liangjia_score + liangjia_modifier, 0, 100)
    action, action_level = _intraday_action(
        adjusted_liangjia,
        adjusted_short,
        short_modifier,
        daily_advice,
        signal_direction,
        risk_flags,
        support_flags,
    )
    formula = (
        f"盘中短线参考分 = 日K短期分 {base_short_term_score:.0f} + 分时修正 {short_modifier:+.0f} = {adjusted_short:.0f}；"
        f"盘中量价参考分 = 日K量价分 {base_liangjia_score:.0f} + 分时修正 {liangjia_modifier:+.0f} = {adjusted_liangjia:.0f}。"
    )
    explanation = _adjustment_explanation(action, support_flags, risk_flags)
    return IntradayAdjustment(
        base_liangjia_score=base_liangjia_score,
        base_short_term_score=base_short_term_score,
        liangjia_modifier=liangjia_modifier,
        short_term_modifier=short_modifier,
        adjusted_liangjia_score=adjusted_liangjia,
        adjusted_short_term_score=adjusted_short,
        action=action,
        action_level=action_level,
        support_flags=support_flags,
        risk_flags=risk_flags,
        formula=formula,
        explanation=explanation,
    )


def _pct_change(current: float, base: float) -> float:
    if base == 0:
        return 0.0
    return current / base - 1.0


def _range_position(price: float, low: float, high: float) -> float | None:
    if high <= low:
        return None
    return (price - low) / (high - low)


def _recent_momentum(session: pd.DataFrame) -> float | None:
    if len(session) < 7:
        return None
    base = float(session["close"].iloc[-7])
    if base == 0:
        return None
    return float(session["close"].iloc[-1]) / base - 1.0


def _last_volume_ratio(data: pd.DataFrame) -> float | None:
    if len(data) < 21:
        return None
    prior_mean = pd.to_numeric(data["volume"], errors="coerce").shift(1).rolling(20, min_periods=10).mean().iloc[-1]
    last_volume = float(data["volume"].iloc[-1])
    if pd.isna(prior_mean) or prior_mean <= 0:
        return None
    return last_volume / float(prior_mean)


def _session_vwap(session: pd.DataFrame) -> float | None:
    if "amount" not in session.columns:
        return None
    amount = pd.to_numeric(session["amount"], errors="coerce").sum()
    volume = pd.to_numeric(session["volume"], errors="coerce").sum()
    if pd.isna(amount) or pd.isna(volume) or amount <= 0 or volume <= 0:
        return None
    vwap = float(amount / volume)
    latest = float(session["close"].iloc[-1])
    if latest > 0 and vwap > latest * 20:
        vwap = vwap / 100.0
    if latest > 0 and not (latest * 0.2 <= vwap <= latest * 5):
        return None
    return vwap


def _trend_label(
    session_return_pct: float,
    momentum_30m_pct: float | None,
    range_position_pct: float | None,
    vwap_gap_pct: float | None,
) -> str:
    upper_range = range_position_pct is not None and range_position_pct >= 0.65
    lower_range = range_position_pct is not None and range_position_pct <= 0.35
    above_vwap = vwap_gap_pct is not None and vwap_gap_pct >= 0
    below_vwap = vwap_gap_pct is not None and vwap_gap_pct < 0
    recent_up = momentum_30m_pct is not None and momentum_30m_pct >= 0
    recent_down = momentum_30m_pct is not None and momentum_30m_pct < 0

    if session_return_pct > 0 and upper_range and (above_vwap or recent_up):
        return "盘中趋势偏强"
    if session_return_pct < 0 and lower_range and (below_vwap or recent_down):
        return "盘中趋势偏弱"
    if upper_range and not below_vwap:
        return "盘中承接尚可"
    if lower_range and not above_vwap:
        return "盘中承接偏弱"
    return "盘中震荡平衡"


def _volume_label(volume_ratio: float | None) -> str:
    if volume_ratio is None:
        return "量能样本不足"
    if volume_ratio >= 2.0:
        return "尾盘明显放量"
    if volume_ratio >= 1.3:
        return "尾盘温和放量"
    if volume_ratio <= 0.7:
        return "尾盘缩量"
    return "尾盘量能正常"


def _conclusion(trend_label: str, volume_label: str, session_return_pct: float, vwap_gap_pct: float | None) -> str:
    above_vwap = vwap_gap_pct is not None and vwap_gap_pct >= 0
    below_vwap = vwap_gap_pct is not None and vwap_gap_pct < 0
    if trend_label == "盘中趋势偏强" and (above_vwap or session_return_pct > 0.01):
        return "分时支持偏多"
    if trend_label == "盘中趋势偏弱" and (below_vwap or session_return_pct < -0.01):
        return "分时提示风险"
    if volume_label == "尾盘明显放量" and session_return_pct < 0:
        return "分时放量转弱"
    if trend_label in {"盘中承接尚可", "盘中震荡平衡"}:
        return "分时中性观察"
    return "分时等待确认"


def _explanation(
    conclusion: str,
    session_return_pct: float,
    momentum_30m_pct: float | None,
    range_position_pct: float | None,
    volume_ratio: float | None,
    vwap_gap_pct: float | None,
) -> str:
    parts = [
        f"当日涨跌幅 {session_return_pct:.2%}",
        "30分钟动量 -" if momentum_30m_pct is None else f"30分钟动量 {momentum_30m_pct:.2%}",
        "日内位置 -" if range_position_pct is None else f"日内位置 {range_position_pct:.0%}",
        "尾盘量比 -" if volume_ratio is None else f"尾盘量比 {volume_ratio:.2f}",
        "相对VWAP -" if vwap_gap_pct is None else f"相对VWAP {vwap_gap_pct:.2%}",
    ]
    return f"{conclusion}：" + "，".join(parts) + "。"


def _intraday_action(
    adjusted_liangjia: float,
    adjusted_short: float,
    short_modifier: float,
    daily_advice: str,
    signal_direction: str,
    risk_flags: list[str],
    support_flags: list[str],
) -> tuple[str, str]:
    defensive_daily = any(word in daily_advice for word in ["卖出", "减仓", "回避"]) or "偏空" in signal_direction
    buy_like_daily = any(word in daily_advice for word in ["买入", "加仓"])
    serious_risk = any("放量下跌" in item or "分时提示风险" in item or "分时放量转弱" in item for item in risk_flags)

    if defensive_daily and serious_risk:
        return "盘中不支持买入，优先防守", "risk"
    if serious_risk or short_modifier <= -8 or adjusted_short < 50:
        return "盘中不支持买入，等待日K与分时重新确认", "risk"
    if defensive_daily:
        return "日K仍偏防守，分时只作为反弹观察", "risk"
    if buy_like_daily and adjusted_liangjia >= 70 and adjusted_short >= 70 and short_modifier >= 3:
        return "盘中支持当前日K买点，可按原计划执行", "support"
    if adjusted_liangjia >= 65 and adjusted_short >= 60 and support_flags:
        return "盘中略偏支持，但建议等回踩不破VWAP或关键价再执行", "watch"
    return "盘中证据不足，短线先等待确认", "neutral"


def _adjustment_explanation(action: str, support_flags: list[str], risk_flags: list[str]) -> str:
    support_text = "；".join(support_flags[:3]) if support_flags else "暂无明显分时加分项"
    risk_text = "；".join(risk_flags[:3]) if risk_flags else "暂无明显分时风险项"
    return f"{action}。加分依据：{support_text}。扣分/风险依据：{risk_text}。"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
