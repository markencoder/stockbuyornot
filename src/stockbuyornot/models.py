from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Stage(str, Enum):
    ACCUMULATION = "第一阶段-筑底"
    MARKUP = "第二阶段-主升"
    DISTRIBUTION = "第三阶段-筑顶/高位震荡"
    MARKDOWN = "第四阶段-主跌"
    RANGE = "区间震荡"
    UNKNOWN = "结构不明"


class SignalSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    WATCH = "watch"
    AVOID = "avoid"


@dataclass(frozen=True)
class KeyLevel:
    name: str
    price: float
    kind: str
    date: pd.Timestamp | None = None
    weight: float = 1.0


@dataclass(frozen=True)
class MarketStructure:
    stage: Stage
    trend: str
    description: str
    confidence: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeSignal:
    name: str
    side: SignalSide
    strength: int
    logic: str
    evidence: list[str]
    stop_loss: float | None = None
    entry_zone: tuple[float, float] | None = None
    invalidation: str | None = None
    level: KeyLevel | None = None
    trigger_level: float | None = None
    invalidation_price: float | None = None
    risk_buffer_pct: float | None = None
    invalidation_basis: str | None = None


@dataclass(frozen=True)
class ScoreBreakdown:
    total: int
    structure: int
    position: int
    signal: int
    risk: int
    relative_strength: int
    explanation: list[str]


@dataclass(frozen=True)
class FactorSnapshot:
    short_term: dict[str, Any]
    long_term: dict[str, Any]
    data_quality: dict[str, Any]


@dataclass(frozen=True)
class MultiFactorScores:
    liangjia_score: float
    short_term_score: float
    long_term_score: float
    overall_score: float
    components: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StockClassification:
    category: str
    action: str
    priority: int
    main_signal: str
    buy_point_type: str
    entry_zone: tuple[float, float] | None
    stop_loss: float | None
    risk_pct: float | None
    key_signals: list[str]
    risk_flags: list[str]
    explanation: list[str]


@dataclass(frozen=True)
class LongTermView:
    score: float
    rating: str
    advice: str
    key_factors: list[str]
    risk_warnings: list[str]
    explanation: str


@dataclass(frozen=True)
class ShortTermView:
    liangjia_score: float
    short_term_score: float
    signal_type: str
    signal_strength: int
    signal_direction: str
    advice: str
    action_code: str
    entry_zone: tuple[float, float] | None
    stop_loss: float | None
    risk_pct: float | None
    key_factors: list[str]
    risk_warnings: list[str]
    explanation: str


@dataclass(frozen=True)
class AnalysisResult:
    symbol: str
    as_of: pd.Timestamp
    close: float
    structure: MarketStructure
    support_levels: list[KeyLevel]
    resistance_levels: list[KeyLevel]
    signals: list[TradeSignal]
    score: ScoreBreakdown
    suggestion: str
    radar: Any | None = None
    decision: Any | None = None
    factors: FactorSnapshot | None = None
    factor_scores: MultiFactorScores | None = None
    classification: StockClassification | None = None
    long_term_view: LongTermView | None = None
    short_term_view: ShortTermView | None = None
    output: dict[str, Any] | None = None
