from __future__ import annotations

import pandas as pd

from stockbuyornot.classification_engine import classify_stock
from stockbuyornot.config import AppConfig, FactorConfig, MultiFactorConfig
from stockbuyornot.decision import make_unified_decision
from stockbuyornot.explanation_engine import build_multifactor_output
from stockbuyornot.factor_engine import compute_factors
from stockbuyornot.features import add_features
from stockbuyornot.levels import identify_levels
from stockbuyornot.models import AnalysisResult, SignalSide
from stockbuyornot.radar import diagnose_radar
from stockbuyornot.scoring import score_analysis
from stockbuyornot.scoring_engine import compute_multi_factor_scores
from stockbuyornot.signals import detect_signals
from stockbuyornot.structure import classify_structure
from stockbuyornot.view_engine import build_investment_views


def analyze_ohlcv(
    df: pd.DataFrame,
    symbol: str = "",
    benchmark: pd.DataFrame | None = None,
    sector: pd.DataFrame | None = None,
    sector_name: str = "",
    sector_rs_rank: float | None = None,
    fundamentals: dict | pd.DataFrame | None = None,
    stock_name: str = "",
    config: AppConfig = AppConfig(),
) -> AnalysisResult:
    if len(df) < 60:
        raise ValueError("At least 60 trading days are required for a stable analysis.")

    featured = add_features(df, config.features).dropna(subset=["ma_slow"]).reset_index(drop=True)
    if len(featured) < 20:
        raise ValueError("Not enough rows after feature calculation.")

    structure = classify_structure(featured)
    supports, resistances = identify_levels(featured, config.levels)
    signals = detect_signals(featured, structure, supports, resistances, config.signals)
    latest = featured.iloc[-1]
    relative_strength = _relative_strength(featured, benchmark)
    score = score_analysis(float(latest["close"]), structure, supports, resistances, signals, relative_strength, config.scoring)
    suggestion = _suggestion(score.total, signals)

    result = AnalysisResult(
        symbol=symbol or str(latest.get("symbol", "")),
        as_of=latest["date"],
        close=float(latest["close"]),
        structure=structure,
        support_levels=supports,
        resistance_levels=resistances,
        signals=signals,
        score=score,
        suggestion=suggestion,
    )
    radar = diagnose_radar(
        featured,
        result,
        benchmark=benchmark,
        sector=sector,
        sector_name=sector_name,
        sector_rs_rank=sector_rs_rank,
    )
    result_with_radar = AnalysisResult(
        symbol=result.symbol,
        as_of=result.as_of,
        close=result.close,
        structure=result.structure,
        support_levels=result.support_levels,
        resistance_levels=result.resistance_levels,
        signals=result.signals,
        score=result.score,
        suggestion=result.suggestion,
        radar=radar,
    )
    decision = make_unified_decision(result_with_radar)
    result_with_decision = AnalysisResult(
        symbol=result_with_radar.symbol,
        as_of=result_with_radar.as_of,
        close=result_with_radar.close,
        structure=result_with_radar.structure,
        support_levels=result_with_radar.support_levels,
        resistance_levels=result_with_radar.resistance_levels,
        signals=result_with_radar.signals,
        score=result_with_radar.score,
        suggestion=decision.action_label,
        radar=result_with_radar.radar,
        decision=decision,
    )
    factor_config = getattr(config, "factors", FactorConfig())
    multi_factor_config = getattr(config, "multi_factor", MultiFactorConfig())
    factors = compute_factors(featured, benchmark=benchmark, sector=sector, fundamentals=fundamentals, config=factor_config)
    factor_scores = compute_multi_factor_scores(result_with_decision, factors, factor_config, multi_factor_config)
    classification = classify_stock(result_with_decision, factors, factor_scores, multi_factor_config)
    long_term_view, short_term_view = build_investment_views(
        result_with_decision,
        factors,
        factor_scores,
        multi_factor_config,
    )
    result = AnalysisResult(
        symbol=result_with_decision.symbol,
        as_of=result_with_decision.as_of,
        close=result_with_decision.close,
        structure=result_with_decision.structure,
        support_levels=result_with_decision.support_levels,
        resistance_levels=result_with_decision.resistance_levels,
        signals=result_with_decision.signals,
        score=result_with_decision.score,
        suggestion=short_term_view.advice,
        radar=result_with_decision.radar,
        decision=result_with_decision.decision,
        factors=factors,
        factor_scores=factor_scores,
        classification=classification,
        long_term_view=long_term_view,
        short_term_view=short_term_view,
    )
    return AnalysisResult(
        symbol=result.symbol,
        as_of=result.as_of,
        close=result.close,
        structure=result.structure,
        support_levels=result.support_levels,
        resistance_levels=result.resistance_levels,
        signals=result.signals,
        score=result.score,
        suggestion=result.suggestion,
        radar=result.radar,
        decision=result.decision,
        factors=result.factors,
        factor_scores=result.factor_scores,
        classification=result.classification,
        long_term_view=result.long_term_view,
        short_term_view=result.short_term_view,
        output=build_multifactor_output(result, stock_name=stock_name),
    )


def _relative_strength(df: pd.DataFrame, benchmark: pd.DataFrame | None) -> float | None:
    if benchmark is None or benchmark.empty or len(df) < 60:
        return None
    stock_return = df["close"].iloc[-1] / df["close"].iloc[-60] - 1
    bench = benchmark.copy()
    bench["date"] = pd.to_datetime(bench["date"])
    aligned = bench[bench["date"].between(df["date"].iloc[-60], df["date"].iloc[-1])]
    if len(aligned) < 20:
        return None
    bench_return = aligned["close"].iloc[-1] / aligned["close"].iloc[0] - 1
    return float(stock_return - bench_return)


def _suggestion(total: int, signals: list) -> str:
    if any(signal.name in {"向下反转卖点", "向下突破卖点"} for signal in signals):
        return "卖出/强风险"
    if any(signal.side == SignalSide.AVOID for signal in signals):
        return "回避/剔除"
    if any(signal.side == SignalSide.SELL for signal in signals):
        return "减仓/卖出风险"
    if total >= 80 and any(signal.side == SignalSide.BUY for signal in signals):
        return "强信号，进入重点买点池"
    if total >= 60:
        return "观察信号，等待确认"
    if total >= 40:
        return "平衡信号，暂不操作"
    return "剔除或回避"
