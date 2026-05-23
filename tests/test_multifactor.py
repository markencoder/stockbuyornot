import numpy as np
import pandas as pd

from stockbuyornot.analysis import analyze_ohlcv
from stockbuyornot.classification_engine import classify_stock
from stockbuyornot.models import (
    AnalysisResult,
    FactorSnapshot,
    KeyLevel,
    MarketStructure,
    MultiFactorScores,
    ScoreBreakdown,
    SignalSide,
    Stage,
    TradeSignal,
)
from stockbuyornot.scoring_engine import compute_multi_factor_scores


def _ohlcv(rows: int = 320) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = np.linspace(10, 18, rows) + np.sin(np.arange(rows) / 8) * 0.2
    open_ = close * 0.998
    high = close * 1.012
    low = close * 0.988
    volume = np.linspace(100000, 180000, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": volume * close,
            "symbol": "TEST",
        }
    )


def _analysis(signals: list[TradeSignal] | None = None, stage: Stage = Stage.MARKUP) -> AnalysisResult:
    return AnalysisResult(
        symbol="TEST",
        as_of=pd.Timestamp("2026-01-01"),
        close=10.0,
        structure=MarketStructure(stage=stage, trend="up", description="test", confidence=0.8),
        support_levels=[KeyLevel(name="support", price=9.5, kind="support")],
        resistance_levels=[],
        signals=signals or [],
        score=ScoreBreakdown(total=72, structure=30, position=15, signal=20, risk=5, relative_strength=2, explanation=[]),
        suggestion="watch",
    )


def _factors() -> FactorSnapshot:
    return FactorSnapshot(
        short_term={},
        long_term={"debt_to_assets": 0.35, "interest_debt_ratio": 0.10},
        data_quality={"price_rows": 320},
    )


def _balanced_short_factors() -> dict:
    return {
        "close": 10.0,
        "ma20": 9.7,
        "ma20_gap_pct": 0.031,
        "pullback_from_20d_high": -0.06,
        "volume_ratio": 0.78,
        "return_1d": 0.005,
        "return_3d": 0.015,
        "return_5d": 0.025,
        "return_10d": 0.04,
        "return_20d": 0.08,
    }


def test_analyze_outputs_split_view_json_shape_with_missing_fundamentals():
    result = analyze_ohlcv(_ohlcv(), symbol="TEST")

    assert result.factors is not None
    assert result.factor_scores is not None
    assert result.classification is not None
    assert result.long_term_view is not None
    assert result.short_term_view is not None
    assert result.output is not None
    assert result.short_term_view.short_term_score == result.factor_scores.short_term_score
    for key in [
        "股票代码",
        "交易日期",
        "long_term_view",
        "short_term_view",
        "score_reference",
        "short_term_factors",
        "long_term_factors",
        "risk_flags",
        "data_quality",
        "explanation",
    ]:
        assert key in result.output
    assert "final_decision" not in result.output
    assert result.output["long_term_view"]["advice"].startswith("长期")
    assert result.output["short_term_view"]["advice"].startswith("短期")
    assert result.output["short_term_view"]["short_term_score"] == result.factor_scores.short_term_score


def test_long_term_good_without_liangjia_buy_cannot_buy():
    scores = MultiFactorScores(liangjia_score=60, short_term_score=55, long_term_score=88, overall_score=75)
    classification = classify_stock(_analysis(signals=[]), _factors(), scores)

    assert classification.category == "长期好股"
    assert classification.action in {"WAIT", "WATCH"}


def test_risk_signal_overrides_high_long_term_score():
    sell = TradeSignal(name="向下反转卖点", side=SignalSide.SELL, strength=90, logic="", evidence=[])
    scores = MultiFactorScores(liangjia_score=85, short_term_score=80, long_term_score=90, overall_score=88)
    classification = classify_stock(_analysis(signals=[sell]), _factors(), scores)

    assert classification.category == "剔除池/回避池"
    assert classification.action == "SELL"
    assert classification.risk_flags


def test_buy_or_add_requires_original_liangjia_buy_signal():
    buy = TradeSignal(
        name="上涨中继买点",
        side=SignalSide.BUY,
        strength=85,
        logic="",
        evidence=[],
        stop_loss=9.5,
        entry_zone=(10.0, 10.3),
    )
    scores = MultiFactorScores(liangjia_score=86, short_term_score=82, long_term_score=84, overall_score=85)
    classification = classify_stock(_analysis(signals=[buy]), _factors(), scores)

    assert classification.category == "短期与长期兼具"
    assert classification.action in {"BUY", "ADD"}


def test_execution_window_blocks_chasing_buy_signal():
    buy = TradeSignal(
        name="上涨中继买点",
        side=SignalSide.BUY,
        strength=88,
        logic="",
        evidence=[],
        stop_loss=9.5,
        entry_zone=(10.0, 10.3),
    )
    scores = MultiFactorScores(
        liangjia_score=86,
        short_term_score=86,
        long_term_score=70,
        overall_score=82,
        components={
            "execution_score": 48,
            "execution_window": {"score": 48, "flags": ["价格远离MA20，追高风险较大"]},
        },
    )

    classification = classify_stock(_analysis(signals=[buy]), _factors(), scores)

    assert classification.action == "WAIT"
    assert any("执行窗口分不足" in item for item in classification.explanation)


def test_execution_window_score_has_better_distribution_for_ideal_vs_chase():
    buy = TradeSignal(name="上涨中继买点", side=SignalSide.BUY, strength=86, logic="", evidence=[], stop_loss=9.4)
    balanced = FactorSnapshot(short_term=_balanced_short_factors(), long_term={}, data_quality={})
    chase_factors = _balanced_short_factors()
    chase_factors.update({"ma20_gap_pct": 0.16, "pullback_from_20d_high": -0.005, "volume_ratio": 1.8})
    chase = FactorSnapshot(short_term=chase_factors, long_term={}, data_quality={})

    ideal_scores = compute_multi_factor_scores(_analysis(signals=[buy]), balanced)
    chase_scores = compute_multi_factor_scores(_analysis(signals=[buy]), chase)

    ideal_execution = ideal_scores.components["execution_score"]
    chase_execution = chase_scores.components["execution_score"]

    assert 60 <= ideal_execution < 95
    assert chase_execution < ideal_execution - 20
    assert chase_execution < 65


def test_split_views_do_not_convert_long_term_good_into_short_term_buy():
    result = analyze_ohlcv(_ohlcv(), symbol="TEST")

    assert result.long_term_view is not None
    assert result.short_term_view is not None
    has_buy_signal = any(signal.side == SignalSide.BUY for signal in result.signals)
    if not has_buy_signal:
        assert result.short_term_view.advice not in {"短期买入", "短期加仓"}
