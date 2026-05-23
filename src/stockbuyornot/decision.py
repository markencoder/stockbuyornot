from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stockbuyornot.models import AnalysisResult, SignalSide, Stage


DECISION_LABELS = {
    "buy": "\u53ef\u4e70\u5165",
    "wait": "\u7b49\u5f85\u4e70\u70b9",
    "watch": "\u53ef\u5173\u6ce8",
    "sell": "\u51cf\u4ed3/\u5356\u51fa",
    "avoid": "\u56de\u907f",
}


@dataclass(frozen=True)
class UnifiedDecision:
    action_code: str
    action_label: str
    candidate_score: float
    confidence: float
    primary_basis: str
    conflict: str
    reason: str
    stock_quality: str
    entry_quality: str
    risk_state: str


def make_unified_decision(analysis: AnalysisResult) -> UnifiedDecision:
    radar = getattr(analysis, "radar", None)
    signal_state = _signal_state(analysis)
    stage = analysis.structure.stage
    setup_score = float(getattr(radar, "setup_score", 0.0) or 0.0)
    entry_score = float(getattr(radar, "entry_quality_score", 0.0) or 0.0)
    exit_score = float(getattr(radar, "exit_risk_score", 0.0) or 0.0)
    market_state = str(getattr(radar, "market_state", "unknown") or "unknown")
    neutral_signal_score = float(analysis.score.total + max(0, 5 - analysis.score.relative_strength))

    if signal_state in {"sell", "avoid"}:
        action = "sell" if signal_state == "sell" else "avoid"
        return _decision(
            action,
            score=0.0,
            confidence=90.0 if signal_state == "sell" else 85.0,
            primary_basis="volume_price_exit",
            conflict=_conflict(signal_state, radar),
            reason="\u91cf\u4ef7\u5df2\u7ecf\u51fa\u73b0\u5356\u51fa\u6216\u56de\u907f\u4fe1\u53f7\uff0c\u8bf4\u660e\u4f9b\u7ed9\u5360\u4f18\u6216\u8d8b\u52bf\u8f6c\u5f31\uff0c\u4f18\u5148\u670d\u4ece\u4ef7\u683c\u548c\u6210\u4ea4\u91cf\u3002",
            radar=radar,
        )

    if stage == Stage.MARKDOWN:
        return _decision(
            "avoid",
            score=0.0,
            confidence=85.0,
            primary_basis="markdown_structure",
            conflict=_conflict(signal_state, radar),
            reason="\u7ed3\u6784\u5904\u4e8e\u4e3b\u8dcc\u9636\u6bb5\uff0c\u7f29\u91cf\u53cd\u5f39\u6216\u77ed\u7ebf\u6b62\u8dcc\u4e0d\u80fd\u5f53\u4f5c\u4e70\u70b9\u3002",
            radar=radar,
        )

    if exit_score >= 70:
        return _decision(
            "sell" if stage in {Stage.MARKUP, Stage.DISTRIBUTION} else "avoid",
            score=0.0,
            confidence=min(95.0, exit_score),
            primary_basis="risk_accumulation",
            conflict=_conflict(signal_state, radar),
            reason="\u5356\u51fa\u98ce\u9669\u8f83\u9ad8\uff0c\u53ef\u80fd\u5df2\u51fa\u73b0\u9ad8\u4f4d\u4f9b\u7ed9\u3001\u8dcc\u7834\u5747\u7ebf\u6216\u76f8\u5bf9\u5f3a\u5ea6\u8f6c\u5f31\u3002",
            radar=radar,
        )

    if signal_state == "buy":
        if market_state == "weak":
            return _decision(
                "wait",
                score=min(69.0, max(neutral_signal_score, entry_score)),
                confidence=65.0,
                primary_basis="buy_signal_weak_market",
                conflict=_conflict(signal_state, radar),
                reason="\u4e2a\u80a1\u6709\u91cf\u4ef7\u4e70\u70b9\uff0c\u4f46\u5927\u76d8\u504f\u5f31\uff0c\u53ea\u80fd\u89c2\u5bdf\u6216\u7b49\u5f85\u786e\u8ba4\uff0c\u4e0d\u4e3b\u52a8\u5f00\u65b0\u4ed3\u3002",
                radar=radar,
            )
        if entry_score >= 65 and exit_score < 55:
            score = max(neutral_signal_score, setup_score, entry_score)
            return _decision(
                "buy",
                score=min(100.0, score),
                confidence=min(95.0, max(72.0, entry_score)),
                primary_basis="confirmed_volume_price_buy",
                conflict=_conflict(signal_state, radar),
                reason="\u91cf\u4ef7\u4e70\u70b9\u6210\u7acb\uff0c\u4e14\u4e70\u70b9\u8d28\u91cf\u548c\u98ce\u9669\u72b6\u6001\u6ca1\u6709\u5426\u5b9a\u5b83\u3002",
                radar=radar,
            )
        return _decision(
            "wait",
            score=min(79.0, max(neutral_signal_score, setup_score)),
            confidence=62.0,
            primary_basis="buy_signal_needs_quality",
            conflict=_conflict(signal_state, radar),
            reason="\u6709\u91cf\u4ef7\u4e70\u70b9\uff0c\u4f46\u652f\u6491\u3001\u7f29\u91cf\u3001\u76c8\u4e8f\u6bd4\u6216\u98ce\u9669\u6761\u4ef6\u4e0d\u591f\u5b8c\u6574\uff0c\u5148\u7b49\u5f85\u786e\u8ba4\u3002",
            radar=radar,
        )

    if signal_state == "watch":
        if stage in {Stage.ACCUMULATION, Stage.RANGE, Stage.MARKUP} and setup_score >= 50 and exit_score < 60:
            action = "wait" if setup_score >= 65 or entry_score >= 55 else "watch"
            return _decision(
                action,
                score=min(79.0 if action == "wait" else 64.0, max(neutral_signal_score, setup_score)),
                confidence=60.0,
                primary_basis="watch_signal",
                conflict=_conflict(signal_state, radar),
                reason="\u91cf\u4ef7\u5904\u4e8e\u89c2\u5bdf\u4fe1\u53f7\uff0c\u8bf4\u660e\u4f9b\u9700\u53ef\u80fd\u6539\u5584\uff0c\u4f46\u5c1a\u672a\u5f62\u6210\u53ef\u76f4\u63a5\u4e70\u5165\u7684\u786e\u8ba4\u3002",
                radar=radar,
            )

    if setup_score >= 70 and stage in {Stage.ACCUMULATION, Stage.RANGE, Stage.MARKUP} and exit_score < 55:
        return _decision(
            "watch",
            score=min(64.0, setup_score),
            confidence=55.0,
            primary_basis="radar_setup_only",
            conflict=_conflict(signal_state, radar),
            reason="\u7ed3\u6784\u548c\u76f8\u5bf9\u5f3a\u5ea6\u8f83\u597d\uff0c\u4f46\u5c1a\u65e0\u660e\u786e\u91cf\u4ef7\u4e70\u70b9\uff0c\u53ea\u80fd\u8fdb\u5165\u89c2\u5bdf\u6c60\u3002",
            radar=radar,
        )

    return _decision(
        "avoid",
        score=0.0,
        confidence=70.0,
        primary_basis="no_actionable_volume_price_edge",
        conflict=_conflict(signal_state, radar),
        reason="\u6ca1\u6709\u5f62\u6210\u6e05\u6670\u91cf\u4ef7\u4f18\u52bf\uff0c\u6216\u5019\u9009\u8d28\u91cf\u4e0d\u8db3\uff0c\u6682\u4e0d\u8fdb\u5165\u9009\u80a1\u7ed3\u679c\u3002",
        radar=radar,
    )


def _signal_state(analysis: AnalysisResult) -> str:
    sides = {signal.side for signal in analysis.signals}
    if SignalSide.SELL in sides:
        return "sell"
    if SignalSide.AVOID in sides:
        return "avoid"
    if SignalSide.BUY in sides:
        return "buy"
    if SignalSide.WATCH in sides:
        return "watch"
    return "none"


def _conflict(signal_state: str, radar: Any) -> str:
    radar_action = str(getattr(radar, "action_code", "") or "")
    if not radar_action:
        return ""
    if signal_state in {"sell", "avoid"} and radar_action in {"watch", "wait", "trade"}:
        return "radar_positive_but_volume_price_exit"
    if signal_state == "buy" and radar_action == "avoid":
        return "volume_price_buy_but_radar_filter_negative"
    if signal_state in {"none", "watch"} and radar_action == "trade":
        return "radar_trade_without_volume_price_buy"
    return ""


def _decision(
    action: str,
    score: float,
    confidence: float,
    primary_basis: str,
    conflict: str,
    reason: str,
    radar: Any,
) -> UnifiedDecision:
    return UnifiedDecision(
        action_code=action,
        action_label=DECISION_LABELS[action],
        candidate_score=round(max(0.0, min(100.0, score)), 2),
        confidence=round(max(0.0, min(100.0, confidence)), 2),
        primary_basis=primary_basis,
        conflict=conflict,
        reason=reason,
        stock_quality=str(getattr(radar, "good_stock_conclusion", "")),
        entry_quality=str(getattr(radar, "entry_conclusion", "")),
        risk_state=str(getattr(radar, "exit_conclusion", "")),
    )
