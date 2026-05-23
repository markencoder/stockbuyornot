from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    volume_window: int = 20
    volume_short_window: int = 5
    volume_mid_window: int = 10
    volume_long_window: int = 60
    volume_percentile_window: int = 120
    trend_fast_window: int = 20
    trend_slow_window: int = 60
    long_window: int = 120
    volatility_window: int = 20


@dataclass(frozen=True)
class LevelConfig:
    range_window: int = 60
    pivot_window: int = 5
    high_volume_ratio: float = 1.8
    low_volume_ratio: float = 0.55
    near_level_pct: float = 0.03


@dataclass(frozen=True)
class SignalConfig:
    breakout_pct: float = 0.01
    breakdown_pct: float = 0.01
    volume_expand_ratio: float = 1.5
    climax_volume_ratio: float = 2.0
    dry_volume_ratio: float = 0.7
    strong_body_ratio: float = 0.60
    small_body_ratio: float = 0.35
    close_near_high: float = 0.7
    close_near_low: float = 0.3
    long_shadow_ratio: float = 0.45
    consolidation_max_range_pct: float = 0.35
    consolidation_min_days: int = 35
    high_gain_pct: float = 0.45
    stop_buffer_pct: float = 0.01
    invalidation_base_buffer_pct: float = 0.015
    invalidation_max_buffer_pct: float = 0.04
    small_change_pct: float = 0.015
    support_tolerance_pct: float = 0.015
    max_buy_stop_distance_pct: float = 0.08


@dataclass(frozen=True)
class ScoringConfig:
    structure_weight: int = 30
    position_weight: int = 20
    signal_weight: int = 30
    risk_weight: int = 10
    relative_strength_weight: int = 10
    max_stop_distance_pct: float = 0.08


@dataclass(frozen=True)
class FactorConfig:
    short_return_weights: tuple[float, float, float, float, float] = (0.08, 0.12, 0.15, 0.15, 0.10)
    short_ma_weight: float = 0.12
    short_volume_weight: float = 0.12
    short_momentum_weight: float = 0.12
    short_breakout_weight: float = 0.08
    short_risk_weight: float = 0.06
    short_rsi_weight: float = 0.14
    short_macd_weight: float = 0.20
    short_kdj_weight: float = 0.14
    short_oscillator_weight: float = 0.10
    short_bollinger_weight: float = 0.10
    short_obv_weight: float = 0.10
    short_trend_strength_weight: float = 0.08
    short_relative_strength_weight: float = 0.10
    short_volatility_weight: float = 0.04
    long_trend_weight: float = 0.35
    long_return_weight: float = 0.20
    long_quality_weight: float = 0.20
    long_growth_weight: float = 0.10
    long_valuation_weight: float = 0.10
    long_risk_weight: float = 0.05
    rsi_overheat: float = 78.0
    rsi_strong_low: float = 50.0
    rsi_strong_high: float = 72.0
    rsi_weak: float = 42.0
    kdj_strong_low: float = 45.0
    kdj_strong_high: float = 82.0
    kdj_overheat: float = 90.0
    cci_overheat: float = 200.0
    cci_weak: float = -100.0
    mfi_overheat: float = 85.0
    mfi_weak: float = 30.0
    bollinger_squeeze_ratio: float = 0.80
    atr_pct_high: float = 0.08
    volatility_contraction_good: float = 0.85
    short_max_drawdown_limit: float = -0.12
    long_max_drawdown_limit: float = -0.35
    min_roe: float = 0.10
    max_debt_to_assets: float = 0.70
    max_valuation_percentile: float = 0.85
    strong_return_5d: float = 0.05
    strong_return_20d: float = 0.12
    overheat_return_20d: float = 0.30
    volume_expand_good: float = 1.30
    relative_strength_good: float = 0.03
    ma_gap_overheat: float = 0.15
    ideal_short_score_low: float = 55.0
    ideal_short_score_high: float = 75.0
    chase_ma20_gap_pct: float = 0.08
    severe_chase_ma20_gap_pct: float = 0.12
    ideal_pullback_min_pct: float = -0.10
    ideal_pullback_max_pct: float = -0.03
    long_return_good: float = 0.15
    long_return_weak: float = -0.10
    valuation_low_percentile: float = 0.35


@dataclass(frozen=True)
class MultiFactorConfig:
    liangjia_weight: float = 0.45
    short_term_weight: float = 0.25
    long_term_weight: float = 0.25
    risk_penalty_weight: float = 0.05
    short_good_liangjia_min: float = 65.0
    short_good_short_min: float = 65.0
    long_good_long_min: float = 68.0
    both_good_overall_min: float = 72.0
    watch_min_score: float = 50.0
    execution_good_min: float = 65.0
    major_risk_exit_score: float = 70.0
    max_buy_risk_pct: float = 0.10


@dataclass(frozen=True)
class AppConfig:
    features: FeatureConfig = FeatureConfig()
    levels: LevelConfig = LevelConfig()
    signals: SignalConfig = SignalConfig()
    scoring: ScoringConfig = ScoringConfig()
    factors: FactorConfig = FactorConfig()
    multi_factor: MultiFactorConfig = MultiFactorConfig()
