import pandas as pd

from stockbuyornot.models import FactorSnapshot, MarketStructure, ScoreBreakdown, Stage
from stockbuyornot.models import AnalysisResult
from stockbuyornot.scoring_engine import compute_multi_factor_scores


def _analysis() -> AnalysisResult:
    return AnalysisResult(
        symbol="TEST",
        as_of=pd.Timestamp("2026-01-01"),
        close=10.0,
        structure=MarketStructure(stage=Stage.MARKUP, trend="up", description="test", confidence=0.8),
        support_levels=[],
        resistance_levels=[],
        signals=[],
        score=ScoreBreakdown(total=72, structure=30, position=15, signal=20, risk=5, relative_strength=2, explanation=[]),
        suggestion="watch",
    )


def test_short_term_score_uses_rsi_macd_and_kdj_quality():
    base_long = {"debt_to_assets": 0.35, "interest_debt_ratio": 0.10}
    healthy = FactorSnapshot(
        short_term={
            "return_1d": 0.01,
            "return_3d": 0.025,
            "return_5d": 0.04,
            "return_10d": 0.06,
            "return_20d": 0.09,
            "ma5": 12.0,
            "ma10": 11.5,
            "ma20": 11.0,
            "volume_ratio": 1.35,
            "rsi14": 62.0,
            "rsi6": 66.0,
            "macd_dif": 0.20,
            "macd_dea": 0.12,
            "macd_hist": 0.16,
            "macd_hist_prev": 0.08,
            "macd_hist_delta_3d": 0.12,
            "macd_golden_cross": True,
            "macd_above_zero": True,
            "kdj_k": 68.0,
            "kdj_d": 55.0,
            "kdj_j": 94.0,
            "kdj_j_delta_3d": 18.0,
            "kdj_golden_cross": True,
            "cci14": 85.0,
            "williams_r14": -28.0,
            "mfi14": 62.0,
            "boll_percent_b": 0.78,
            "boll_bandwidth_ratio": 0.72,
            "obv_above_ma20": True,
            "obv_change_5d": 4.0,
            "adx14": 26.0,
            "plus_di14": 32.0,
            "minus_di14": 18.0,
            "price_position_20d": 0.78,
            "close_position_day": 0.74,
            "stock_vs_benchmark_20d": 0.05,
            "stock_vs_sector_20d": 0.04,
            "new_high_20": True,
            "short_max_drawdown": -0.04,
            "atr": 0.30,
            "atr_pct": 0.03,
            "volatility_ratio": 0.70,
        },
        long_term=base_long,
        data_quality={},
    )
    weak = FactorSnapshot(
        short_term={
            **healthy.short_term,
            "rsi14": 82.0,
            "rsi6": 88.0,
            "macd_dif": -0.08,
            "macd_dea": 0.04,
            "macd_hist": -0.12,
            "macd_hist_prev": -0.05,
            "macd_hist_delta_3d": -0.10,
            "macd_golden_cross": False,
            "macd_above_zero": False,
            "kdj_k": 92.0,
            "kdj_d": 96.0,
            "kdj_j": 112.0,
            "kdj_j_delta_3d": -20.0,
            "kdj_golden_cross": False,
            "kdj_dead_cross": True,
            "cci14": 230.0,
            "williams_r14": -5.0,
            "mfi14": 90.0,
            "boll_percent_b": 1.25,
            "boll_bandwidth_ratio": 1.80,
            "obv_above_ma20": False,
            "obv_change_5d": -4.0,
            "adx14": 28.0,
            "plus_di14": 16.0,
            "minus_di14": 30.0,
            "price_position_20d": 0.99,
            "close_position_day": 0.25,
            "stock_vs_benchmark_20d": -0.05,
            "stock_vs_sector_20d": -0.04,
            "atr_pct": 0.09,
            "volatility_ratio": 1.70,
        },
        long_term=base_long,
        data_quality={},
    )

    healthy_scores = compute_multi_factor_scores(_analysis(), healthy)
    weak_scores = compute_multi_factor_scores(_analysis(), weak)

    assert healthy_scores.short_term_score > weak_scores.short_term_score + 8
    assert healthy_scores.components["short_term"]["macd_score"] > weak_scores.components["short_term"]["macd_score"]
    assert healthy_scores.components["short_term"]["kdj_score"] > weak_scores.components["short_term"]["kdj_score"]
    assert healthy_scores.components["short_term"]["rsi_score"] > weak_scores.components["short_term"]["rsi_score"]


def test_liangjia_score_uses_supply_demand_confirmation_factors():
    base_long = {"debt_to_assets": 0.35, "interest_debt_ratio": 0.10}
    demand_confirmed = FactorSnapshot(
        short_term={
            "volume_ratio": 1.55,
            "close_position_day": 0.78,
            "price_position_20d": 0.72,
            "obv_above_ma20": True,
            "obv_change_5d": 4.5,
            "mfi14": 62.0,
            "adx14": 25.0,
            "plus_di14": 32.0,
            "minus_di14": 18.0,
            "boll_percent_b": 0.72,
            "boll_bandwidth_ratio": 0.70,
            "volatility_ratio": 0.75,
            "atr_pct": 0.03,
            "short_max_drawdown": -0.04,
        },
        long_term=base_long,
        data_quality={},
    )
    supply_warning = FactorSnapshot(
        short_term={
            "volume_ratio": 1.60,
            "close_position_day": 0.22,
            "price_position_20d": 0.99,
            "return_20d": 0.16,
            "obv_above_ma20": False,
            "obv_change_5d": -4.5,
            "mfi14": 90.0,
            "adx14": 26.0,
            "plus_di14": 14.0,
            "minus_di14": 30.0,
            "boll_percent_b": 1.25,
            "boll_bandwidth_ratio": 1.70,
            "volatility_ratio": 1.70,
            "atr_pct": 0.09,
            "short_max_drawdown": -0.15,
        },
        long_term=base_long,
        data_quality={},
    )

    demand_scores = compute_multi_factor_scores(_analysis(), demand_confirmed)
    supply_scores = compute_multi_factor_scores(_analysis(), supply_warning)

    assert demand_scores.liangjia_score > supply_scores.liangjia_score + 20
    assert demand_scores.components["liangjia"]["factor_modifier"] > 0
    assert supply_scores.components["liangjia"]["factor_modifier"] < 0
    assert demand_scores.components["liangjia"]["money_flow_modifier"] > supply_scores.components["liangjia"]["money_flow_modifier"]
