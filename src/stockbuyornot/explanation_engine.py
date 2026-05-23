from __future__ import annotations

from dataclasses import asdict
from typing import Any

from stockbuyornot.models import AnalysisResult
from stockbuyornot.view_engine import stage_label


def build_multifactor_output(analysis: AnalysisResult, stock_name: str = "") -> dict[str, Any]:
    scores = analysis.factor_scores
    factors = analysis.factors
    long_view = analysis.long_term_view
    short_view = analysis.short_term_view
    if scores is None or factors is None or long_view is None or short_view is None:
        return {}

    short_payload = asdict(short_view)
    if short_view.entry_zone is not None:
        short_payload["entry_zone"] = [short_view.entry_zone[0], short_view.entry_zone[1]]

    return {
        "股票代码": analysis.symbol,
        "股票名称": stock_name or "",
        "交易日期": analysis.as_of.strftime("%Y-%m-%d"),
        "structure": {
            "trend": analysis.structure.trend,
            "description": analysis.structure.description,
            "confidence": analysis.structure.confidence,
            "metrics": analysis.structure.metrics,
        },
        "stage": stage_label(analysis.structure.stage),
        "long_term_view": asdict(long_view),
        "short_term_view": short_payload,
        "score_reference": {
            "liangjia_score": scores.liangjia_score,
            "short_term_score": scores.short_term_score,
            "long_term_score": scores.long_term_score,
            "overall_score": scores.overall_score,
            "components": scores.components,
        },
        "short_term_factors": factors.short_term,
        "long_term_factors": factors.long_term,
        "key_signals": [signal.name for signal in analysis.signals],
        "risk_flags": list(dict.fromkeys(short_view.risk_warnings + long_view.risk_warnings)),
        "data_quality": factors.data_quality,
        "explanation": {
            "long_term": long_view.explanation,
            "short_term": short_view.explanation,
            "liangjia_score": analysis.score.explanation,
        },
    }
