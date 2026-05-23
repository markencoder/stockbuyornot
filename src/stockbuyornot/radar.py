from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stockbuyornot.features import add_features
from stockbuyornot.models import AnalysisResult, SignalSide, Stage


ACTION_LABELS = {
    "watch": "\u53ef\u5173\u6ce8",
    "wait": "\u7b49\u5f85\u4e70\u70b9",
    "trade": "\u53ef\u4ea4\u6613",
    "avoid": "\u56de\u907f",
}


@dataclass(frozen=True)
class RadarDiagnosis:
    market_state: str
    sector_name: str
    sector_rs_rank: float | None
    stock_vs_sector_rs: float | None
    setup_score: float
    entry_quality_score: float
    exit_risk_score: float
    action_code: str
    expected_action: str
    reject_reason: str
    good_stock_conclusion: str
    entry_conclusion: str
    exit_conclusion: str
    stock_rs_20: float | None = None
    stock_rs_60: float | None = None
    reward_risk: float | None = None
    stop_distance_pct: float | None = None
    details: dict[str, Any] | None = None


def diagnose_radar(
    df: pd.DataFrame,
    analysis: AnalysisResult,
    benchmark: pd.DataFrame | None = None,
    sector: pd.DataFrame | None = None,
    sector_name: str = "",
    sector_rs_rank: float | None = None,
) -> RadarDiagnosis:
    featured = _ensure_featured(df)
    latest = featured.iloc[-1]
    close = float(latest["close"])

    market_state = market_state_from_benchmark(benchmark)
    stock_rs_20 = relative_strength(featured, benchmark, 20)
    stock_rs_60 = relative_strength(featured, benchmark, 60)
    stock_vs_sector_rs = relative_strength(featured, sector, 60)

    setup_score, setup_details = _setup_score(featured, analysis, benchmark, stock_rs_20, stock_rs_60, stock_vs_sector_rs)
    entry_score, entry_details = _entry_quality_score(featured, analysis)
    exit_score, exit_details = _exit_risk_score(featured, analysis, stock_rs_20, stock_vs_sector_rs, sector_rs_rank)

    reject_reasons = _reject_reasons(
        analysis=analysis,
        market_state=market_state,
        setup_score=setup_score,
        entry_score=entry_score,
        exit_score=exit_score,
        latest=latest,
        stock_rs_60=stock_rs_60,
    )
    action_code = _expected_action_code(market_state, analysis.structure.stage, setup_score, entry_score, exit_score)

    return RadarDiagnosis(
        market_state=market_state,
        sector_name=sector_name,
        sector_rs_rank=sector_rs_rank,
        stock_vs_sector_rs=stock_vs_sector_rs,
        setup_score=round(setup_score, 2),
        entry_quality_score=round(entry_score, 2),
        exit_risk_score=round(exit_score, 2),
        action_code=action_code,
        expected_action=ACTION_LABELS[action_code],
        reject_reason=";".join(reject_reasons),
        good_stock_conclusion=_good_stock_conclusion(setup_score, analysis.structure.stage, stock_rs_60),
        entry_conclusion=_entry_conclusion(entry_score, market_state, entry_details),
        exit_conclusion=_exit_conclusion(exit_score),
        stock_rs_20=stock_rs_20,
        stock_rs_60=stock_rs_60,
        reward_risk=entry_details.get("reward_risk"),
        stop_distance_pct=entry_details.get("stop_distance_pct"),
        details={**setup_details, **entry_details, **exit_details, "close": close},
    )


def market_state_from_benchmark(benchmark: pd.DataFrame | None) -> str:
    if benchmark is None or benchmark.empty or len(benchmark) < 60:
        return "unknown"
    data = _ensure_featured(benchmark).dropna(subset=["ma_slow"]).reset_index(drop=True)
    if data.empty:
        return "unknown"
    latest = data.iloc[-1]
    close = float(latest["close"])
    ma_fast = float(latest.get("ma_fast", close))
    ma_slow = float(latest.get("ma_slow", ma_fast))
    if close > ma_slow and ma_fast > ma_slow:
        return "strong"
    if close < ma_slow and ma_fast < ma_slow:
        return "weak"
    return "neutral"


def relative_strength(df: pd.DataFrame, reference: pd.DataFrame | None, window: int) -> float | None:
    if reference is None or reference.empty or len(df) < min(window, 20):
        return None
    stock = df.tail(window)
    ref = reference.copy()
    ref["date"] = pd.to_datetime(ref["date"])
    ref = ref[ref["date"].between(stock["date"].iloc[0], stock["date"].iloc[-1])]
    if len(stock) < min(window, 20) or len(ref) < min(window, 20):
        return None
    stock_return = float(stock["close"].iloc[-1] / stock["close"].iloc[0] - 1)
    ref_return = float(ref["close"].iloc[-1] / ref["close"].iloc[0] - 1)
    return stock_return - ref_return


def _setup_score(
    df: pd.DataFrame,
    analysis: AnalysisResult,
    benchmark: pd.DataFrame | None,
    rs20: float | None,
    rs60: float | None,
    stock_vs_sector_rs: float | None,
) -> tuple[float, dict[str, Any]]:
    latest = df.iloc[-1]
    close = float(latest["close"])
    ma_fast = float(latest.get("ma_fast", close))
    ma_slow = float(latest.get("ma_slow", ma_fast))
    ma_long = float(latest.get("ma_long", ma_slow))
    score = 0.0
    details: dict[str, Any] = {}

    stage = analysis.structure.stage
    if stage == Stage.MARKUP:
        score += 30
    elif stage in {Stage.ACCUMULATION, Stage.RANGE}:
        score += 18
    elif stage == Stage.UNKNOWN:
        score += 6
    elif stage == Stage.DISTRIBUTION:
        score -= 10
    elif stage == Stage.MARKDOWN:
        score -= 25

    if close > ma_fast > ma_slow:
        score += 12
    if close > ma_fast > ma_slow > ma_long:
        score += 8
    if float(latest.get("ma_fast_slope", 0) or 0) > 0:
        score += 6
    if float(latest.get("ma_slow_slope", 0) or 0) > 0:
        score += 6

    high60 = float(latest.get("rolling_high_60", close))
    near_stock_high = high60 > 0 and close >= high60 * 0.98
    details["near_60d_high"] = near_stock_high
    if near_stock_high:
        score += 10
        if not _reference_near_high(benchmark, 60):
            score += 6

    range_pct = float(latest.get("range_pct_60", 0) or 0)
    if 0 < range_pct <= 0.22:
        score += 8
    if _volume_dry_up(df):
        score += 8
    if _volatility_contracting(df):
        score += 6
    if _breakout_holding(df):
        score += 8

    score += _rs_points(rs60, positive_full=0.10, positive_mid=0.03, negative=-0.05, points=12)
    score += _rs_points(rs20, positive_full=0.06, positive_mid=0.00, negative=-0.03, points=8)
    score += _rs_points(stock_vs_sector_rs, positive_full=0.06, positive_mid=0.00, negative=-0.04, points=8)

    details.update(
        {
            "range_pct_60": range_pct,
            "volume_dry_up": _volume_dry_up(df),
            "volatility_contracting": _volatility_contracting(df),
            "breakout_holding": _breakout_holding(df),
        }
    )
    return _clamp(score), details


def _entry_quality_score(df: pd.DataFrame, analysis: AnalysisResult) -> tuple[float, dict[str, Any]]:
    latest = df.iloc[-1]
    close = float(latest["close"])
    score = 0.0
    details: dict[str, Any] = {}

    support = _nearest_support(close, analysis)
    support_distance = None if support is None else close / support.price - 1
    if support is not None:
        if 0.01 <= support_distance <= 0.04:
            score += 24
        elif -0.005 <= support_distance < 0.01:
            score += 14
        elif 0.04 < support_distance <= 0.07:
            score += 8
        if support.kind in {"pivot_low", "range_high", "high_volume_low"}:
            score += 8

    pullback = df.tail(8)
    peak = float(pullback["close"].max())
    pullback_depth = 1 - close / peak if peak > 0 else 0.0
    if 0.03 <= pullback_depth <= 0.10:
        score += 18
    elif 0.015 <= pullback_depth < 0.03:
        score += 8

    volume_shrink = float(pullback["volume"].tail(3).mean()) <= float(latest.get("vol_ma20", pullback["volume"].mean())) * 0.75
    range_contract = float(pullback["amplitude"].tail(3).mean()) <= float(df["amplitude"].tail(20).mean()) * 0.80
    support_held = support is None or float(latest["low"]) >= support.price * 0.985
    if volume_shrink:
        score += 14
    if range_contract:
        score += 12
    if support_held:
        score += 10

    stop = _best_stop(analysis, support, df)
    stop_distance = (close - stop) / close if stop and stop > 0 else None
    if stop_distance is not None:
        if 0.03 <= stop_distance <= 0.07:
            score += 16
        elif 0 < stop_distance < 0.03:
            score += 8
        elif stop_distance > 0.10:
            score -= 12

    reward_risk = _reward_risk(df, analysis, stop)
    if reward_risk is not None:
        if reward_risk >= 2.5:
            score += 12
        elif reward_risk >= 1.8:
            score += 8
        elif reward_risk < 1.2:
            score -= 10

    if close > float(latest.get("ma_fast", close)) * 1.12:
        score -= 18
    if _large_gap_up(df):
        score -= 10

    has_buy_signal = any(signal.side == SignalSide.BUY for signal in analysis.signals)
    if has_buy_signal:
        score += 8

    details.update(
        {
            "support_distance": support_distance,
            "support_kind": support.kind if support else "",
            "pullback_depth": pullback_depth,
            "volume_shrink": volume_shrink,
            "range_contract": range_contract,
            "support_held": support_held,
            "stop_distance_pct": stop_distance,
            "reward_risk": reward_risk,
            "large_gap_up": _large_gap_up(df),
        }
    )
    return _clamp(score), details


def _exit_risk_score(
    df: pd.DataFrame,
    analysis: AnalysisResult,
    rs20: float | None,
    stock_vs_sector_rs: float | None,
    sector_rs_rank: float | None,
) -> tuple[float, dict[str, Any]]:
    latest = df.iloc[-1]
    close = float(latest["close"])
    score = 0.0
    details: dict[str, Any] = {}

    if analysis.structure.stage == Stage.DISTRIBUTION:
        score += 22
    if analysis.structure.stage == Stage.MARKDOWN:
        score += 35
    if any(signal.side == SignalSide.SELL for signal in analysis.signals):
        score += 28
    if any(signal.side == SignalSide.AVOID for signal in analysis.signals):
        score += 18

    recent = df.tail(5)
    long_upper_count = int((recent["upper_shadow_ratio"] >= 0.45).sum())
    high_volume_stall = bool(
        (recent["vol_ratio"].tail(3) >= 1.5).any()
        and close <= float(df["close"].tail(10).max()) * 0.985
    )
    if long_upper_count >= 2:
        score += 18
    if high_volume_stall:
        score += 16
    if close < float(latest.get("ma_fast", close)):
        score += 15
    if rs20 is not None and rs20 < -0.03:
        score += 12
    if stock_vs_sector_rs is not None and stock_vs_sector_rs < -0.04:
        score += 10
    if sector_rs_rank is not None and sector_rs_rank < 0.30:
        score += 8

    details.update(
        {
            "long_upper_count_5": long_upper_count,
            "high_volume_stall": high_volume_stall,
            "below_ma20": close < float(latest.get("ma_fast", close)),
        }
    )
    return _clamp(score), details


def _reject_reasons(
    analysis: AnalysisResult,
    market_state: str,
    setup_score: float,
    entry_score: float,
    exit_score: float,
    latest: pd.Series,
    stock_rs_60: float | None,
) -> list[str]:
    reasons: list[str] = []
    if market_state == "weak":
        reasons.append("weak_market")
    if analysis.structure.stage in {Stage.MARKDOWN, Stage.DISTRIBUTION}:
        reasons.append("weak_or_late_structure")
    if stock_rs_60 is not None and stock_rs_60 < -0.03:
        reasons.append("weak_relative_strength")
    if setup_score < 45:
        reasons.append("low_setup_score")
    if entry_score < 50:
        reasons.append("poor_entry_quality")
    if exit_score >= 65:
        reasons.append("high_exit_risk")
    close = float(latest["close"])
    ma_fast = float(latest.get("ma_fast", close))
    if close > ma_fast * 1.12:
        reasons.append("overextended_from_ma20")
    return reasons


def _expected_action_code(market_state: str, stage: Stage, setup_score: float, entry_score: float, exit_score: float) -> str:
    if exit_score >= 70 or stage == Stage.MARKDOWN:
        return "avoid"
    if market_state == "weak":
        return "watch" if setup_score >= 65 else "avoid"
    if setup_score >= 65 and entry_score >= 75 and exit_score < 50:
        return "trade"
    if setup_score >= 70:
        return "wait"
    if setup_score >= 50:
        return "watch"
    return "avoid"


def _good_stock_conclusion(setup_score: float, stage: Stage, rs60: float | None) -> str:
    if setup_score >= 75:
        return "结构和强度较好，属于重点观察对象。"
    if setup_score >= 60:
        return "结构正在变好，可放入观察池等待买点。"
    if stage == Stage.MARKDOWN:
        return "仍在弱势结构中，优先回避。"
    if rs60 is not None and rs60 < -0.03:
        return "相对强度偏弱，暂不属于优先候选。"
    return "优势不够清晰，先观察。"


def _entry_conclusion(entry_score: float, market_state: str, details: dict[str, Any]) -> str:
    if market_state == "weak":
        return "大盘弱势，原则上不主动开新仓。"
    if entry_score >= 75:
        return "买点质量较好，可结合次日成交规则观察。"
    if entry_score >= 55:
        return "接近可用买点，但仍需要缩量、支撑或止跌确认。"
    if details.get("large_gap_up"):
        return "短线跳空或追高风险偏大，等待回踩更合适。"
    return "当前不是理想买点。"


def _exit_conclusion(exit_score: float) -> str:
    if exit_score >= 70:
        return "卖出风险较高，优先保护利润或回避。"
    if exit_score >= 45:
        return "风险正在累积，关注跌破MA20、关键阳线低点或连续放量滞涨。"
    return "暂未看到明显卖出风险，按止损和移动止盈管理。"


def _ensure_featured(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if {"ma_fast", "ma_slow", "ma_long", "vol_ma20", "amplitude"}.issubset(df.columns):
        return df
    return add_features(df).reset_index(drop=True)


def _reference_near_high(reference: pd.DataFrame | None, window: int) -> bool:
    if reference is None or reference.empty or len(reference) < 20:
        return False
    ref = _ensure_featured(reference).tail(window)
    if ref.empty:
        return False
    close = float(ref["close"].iloc[-1])
    high = float(ref["high"].max())
    return high > 0 and close >= high * 0.98


def _volume_dry_up(df: pd.DataFrame) -> bool:
    latest = df.iloc[-1]
    base = float(latest.get("vol_ma20", df["volume"].tail(20).mean()) or 0)
    if base <= 0:
        return False
    return float(df["volume"].tail(5).mean()) <= base * 0.80


def _volatility_contracting(df: pd.DataFrame) -> bool:
    if len(df) < 45:
        return False
    recent = float(df["amplitude"].tail(10).mean())
    prior = float(df["amplitude"].iloc[-45:-15].mean())
    return prior > 0 and recent <= prior * 0.80


def _breakout_holding(df: pd.DataFrame) -> bool:
    if len(df) < 25:
        return False
    prev_high = float(df["high"].iloc[-25:-1].max())
    latest = df.iloc[-1]
    return prev_high > 0 and float(latest["close"]) >= prev_high * 0.99 and float(latest.get("close_position", 0.5)) >= 0.55


def _nearest_support(close: float, analysis: AnalysisResult):
    supports = [level for level in analysis.support_levels if level.price > 0 and level.price <= close * 1.02]
    if not supports:
        return None
    return min(supports, key=lambda level: abs(close - level.price) / level.price)


def _best_stop(analysis: AnalysisResult, support: Any, df: pd.DataFrame) -> float | None:
    stops = [float(signal.stop_loss) for signal in analysis.signals if signal.stop_loss and signal.stop_loss > 0]
    if stops:
        return max(stop for stop in stops if stop < float(df.iloc[-1]["close"])) if any(stop < float(df.iloc[-1]["close"]) for stop in stops) else None
    if support is not None:
        return support.price * 0.99
    return float(df["low"].tail(5).min()) * 0.99


def _reward_risk(df: pd.DataFrame, analysis: AnalysisResult, stop: float | None) -> float | None:
    close = float(df.iloc[-1]["close"])
    if stop is None or stop <= 0 or stop >= close:
        return None
    risk = close - stop
    targets = [float(level.price) for level in analysis.resistance_levels if level.price > close]
    targets.extend([float(df["high"].tail(20).max()), float(df["high"].tail(60).max())])
    targets = [target for target in targets if target > close]
    if not targets:
        return None
    return (min(targets) - close) / risk


def _large_gap_up(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev_close = float(df["close"].iloc[-2])
    if prev_close <= 0:
        return False
    return float(df["open"].iloc[-1]) / prev_close - 1 >= 0.05


def _rs_points(value: float | None, positive_full: float, positive_mid: float, negative: float, points: float) -> float:
    if value is None:
        return 0.0
    if value >= positive_full:
        return points
    if value >= positive_mid:
        return points * 0.55
    if value <= negative:
        return -points * 0.65
    return 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _good_stock_conclusion(setup_score: float, stage: Stage, rs60: float | None) -> str:
    if setup_score >= 75:
        return "\u7ed3\u6784\u548c\u5f3a\u5ea6\u8f83\u597d\uff0c\u5c5e\u4e8e\u91cd\u70b9\u89c2\u5bdf\u5bf9\u8c61\u3002"
    if setup_score >= 60:
        return "\u7ed3\u6784\u6b63\u5728\u53d8\u597d\uff0c\u53ef\u653e\u5165\u89c2\u5bdf\u6c60\u7b49\u5f85\u4e70\u70b9\u3002"
    if stage == Stage.MARKDOWN:
        return "\u4ecd\u5728\u5f31\u52bf\u7ed3\u6784\u4e2d\uff0c\u4f18\u5148\u56de\u907f\u3002"
    if rs60 is not None and rs60 < -0.03:
        return "\u76f8\u5bf9\u5f3a\u5ea6\u504f\u5f31\uff0c\u6682\u4e0d\u5c5e\u4e8e\u4f18\u5148\u5019\u9009\u3002"
    return "\u4f18\u52bf\u4e0d\u591f\u6e05\u6670\uff0c\u5148\u89c2\u5bdf\u3002"


def _entry_conclusion(entry_score: float, market_state: str, details: dict[str, Any]) -> str:
    if market_state == "weak":
        return "\u5927\u76d8\u5f31\u52bf\uff0c\u539f\u5219\u4e0a\u4e0d\u4e3b\u52a8\u5f00\u65b0\u4ed3\u3002"
    if entry_score >= 75:
        return "\u4e70\u70b9\u8d28\u91cf\u8f83\u597d\uff0c\u53ef\u7ed3\u5408\u6b21\u65e5\u6210\u4ea4\u89c4\u5219\u89c2\u5bdf\u3002"
    if entry_score >= 55:
        return "\u63a5\u8fd1\u53ef\u7528\u4e70\u70b9\uff0c\u4f46\u4ecd\u9700\u8981\u7f29\u91cf\u3001\u652f\u6491\u6216\u6b62\u8dcc\u786e\u8ba4\u3002"
    if details.get("large_gap_up"):
        return "\u77ed\u7ebf\u8df3\u7a7a\u6216\u8ffd\u9ad8\u98ce\u9669\u504f\u5927\uff0c\u7b49\u5f85\u56de\u8e29\u66f4\u5408\u9002\u3002"
    return "\u5f53\u524d\u4e0d\u662f\u7406\u60f3\u4e70\u70b9\u3002"


def _exit_conclusion(exit_score: float) -> str:
    if exit_score >= 70:
        return "\u5356\u51fa\u98ce\u9669\u8f83\u9ad8\uff0c\u4f18\u5148\u4fdd\u62a4\u5229\u6da6\u6216\u56de\u907f\u3002"
    if exit_score >= 45:
        return "\u98ce\u9669\u6b63\u5728\u7d2f\u79ef\uff0c\u5173\u6ce8\u8dcc\u7834MA20\u3001\u5173\u952e\u9633\u7ebf\u4f4e\u70b9\u6216\u8fde\u7eed\u653e\u91cf\u6ede\u6da8\u3002"
    return "\u6682\u672a\u770b\u5230\u660e\u663e\u5356\u51fa\u98ce\u9669\uff0c\u6309\u6b62\u635f\u548c\u79fb\u52a8\u6b62\u76c8\u7ba1\u7406\u3002"
