from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import pandas as pd

from stockbuyornot.analysis import analyze_ohlcv
from stockbuyornot.config import AppConfig
from stockbuyornot.features import add_features
from stockbuyornot.models import AnalysisResult, SignalSide, Stage, TradeSignal


PULLBACK_SIGNAL_NAME = "上涨中继买点"
IMMEDIATE_EXIT_SIGNAL_NAMES = {"向下反转卖点", "向下突破卖点"}
CONFIRMED_EXIT_SIGNAL_NAMES = {"放量滞涨卖点"}


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    max_positions: int = 5
    neutral_max_positions: int = 2
    initial_cash: float = 100000.0
    fee_rate: float = 0.0003
    stamp_tax: float = 0.0005
    slippage: float = 0.001
    min_avg_amount_20: float = 50_000_000.0
    min_stop_distance_pct: float = 0.03
    max_stop_distance_pct: float = 0.07
    min_relative_strength: float = 0.03
    min_short_relative_strength: float = -0.03
    min_reward_risk: float = 1.8
    trailing_stop_pct: float = 0.10
    max_holding_days: int = 45
    stale_days: int = 12
    breakeven_r: float = 1.0
    trail_start_r: float = 2.0
    min_score: int = 75
    analysis_lookback_days: int = 180
    market_mode: str = "balanced"


@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    signal_date: pd.Timestamp
    stop_loss: float
    initial_stop: float
    initial_risk: float
    high_close: float
    highest_date: pd.Timestamp
    entry_score: float
    reward_risk: float
    days_held: int = 0


@dataclass(frozen=True)
class TrendCandidate:
    symbol: str
    analysis: AnalysisResult
    signal: TradeSignal
    ranking_score: float
    stop_distance_pct: float
    relative_strength: float
    short_relative_strength: float
    avg_amount_20: float
    trend_score: float
    rs_score: float
    pullback_score: float
    reward_risk_score: float
    liquidity_score: float
    reward_risk: float
    market_state: str = ""
    reject_reason: str = ""


@dataclass(frozen=True)
class TrendEvaluation:
    candidate: TrendCandidate | None
    reject_reason: str = ""
    log_rejection: bool = False
    signal: TradeSignal | None = None
    trend_score: float = 0.0
    rs_score: float = 0.0
    pullback_score: float = 0.0
    reward_risk_score: float = 0.0
    liquidity_score: float = 0.0
    reward_risk: float = 0.0
    stop_distance_pct: float = 0.0
    relative_strength: float = 0.0
    short_relative_strength: float = 0.0
    avg_amount_20: float = 0.0


@dataclass(frozen=True)
class PortfolioBacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    candidates: pd.DataFrame
    metrics: dict[str, float]

    def write_outputs(self, output: str | Path) -> None:
        base = Path(output)
        base.parent.mkdir(parents=True, exist_ok=True)
        stem = base.with_suffix("")
        self.equity.to_csv(stem.with_name(stem.name + "_equity.csv"), index=False, encoding="utf-8-sig")
        self.trades.to_csv(stem.with_name(stem.name + "_trades.csv"), index=False, encoding="utf-8-sig")
        self.positions.to_csv(stem.with_name(stem.name + "_positions.csv"), index=False, encoding="utf-8-sig")
        self.candidates.to_csv(stem.with_name(stem.name + "_candidates.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame([self.metrics]).to_json(
            stem.with_name(stem.name + "_metrics.json"),
            orient="records",
            force_ascii=False,
            indent=2,
        )


class TrendPullbackStrategy:
    def __init__(self, config: PortfolioBacktestConfig) -> None:
        self.config = config

    def evaluate(
        self,
        symbol: str,
        analysis: AnalysisResult,
        history: pd.DataFrame,
        relative_strength: float,
    ) -> TrendCandidate | None:
        evaluation = self.diagnose(
            symbol=symbol,
            analysis=analysis,
            history=history,
            benchmark=pd.DataFrame(),
            signal_date=pd.Timestamp(history.iloc[-1]["date"]),
            market_state="strong",
            relative_strength_60=relative_strength,
            relative_strength_20=0.0,
        )
        return evaluation.candidate

    def diagnose(
        self,
        symbol: str,
        analysis: AnalysisResult,
        history: pd.DataFrame,
        benchmark: pd.DataFrame,
        signal_date: pd.Timestamp,
        market_state: str,
        relative_strength_60: float | None = None,
        relative_strength_20: float | None = None,
    ) -> TrendEvaluation:
        signal = _find_signal(analysis, PULLBACK_SIGNAL_NAME)
        if signal is None or signal.stop_loss is None:
            return TrendEvaluation(None, "no_pullback_signal", False)

        if analysis.structure.stage != Stage.MARKUP:
            return TrendEvaluation(None, "not_markup_stage", True, signal=signal)
        if analysis.score.total < self.config.min_score:
            return TrendEvaluation(None, "low_signal_score", True, signal=signal)

        history = _ensure_featured(history)
        latest = history.iloc[-1]
        close = float(latest["close"])
        if close <= 0:
            return TrendEvaluation(None, "invalid_close", True, signal=signal)

        trend_score = _trend_quality_score(history)
        if trend_score < 70:
            return TrendEvaluation(None, "weak_trend_quality", True, signal=signal, trend_score=trend_score)
        if close > float(latest["ma_fast"]) * 1.12:
            return TrendEvaluation(None, "overextended_from_ma20", True, signal=signal, trend_score=trend_score)

        pullback_score, pullback_reason = _pullback_quality_score(history, signal, self.config)
        if pullback_score < 70:
            return TrendEvaluation(
                None,
                pullback_reason,
                True,
                signal=signal,
                trend_score=trend_score,
                pullback_score=pullback_score,
            )

        stop = _initial_stop(signal, history, self.config)
        stop_distance = (close - stop) / close
        if stop_distance <= 0:
            return TrendEvaluation(None, "invalid_stop", True, signal=signal, stop_distance_pct=stop_distance)
        if stop_distance < self.config.min_stop_distance_pct and not _confirmed_stop_bar(latest):
            return TrendEvaluation(None, "stop_too_close_without_confirmation", True, signal=signal, stop_distance_pct=stop_distance)
        if stop_distance > self.config.max_stop_distance_pct:
            return TrendEvaluation(None, "stop_distance_too_wide", True, signal=signal, stop_distance_pct=stop_distance)

        rs60 = relative_strength_60 if relative_strength_60 is not None else _relative_strength(history, benchmark, signal_date, 60)
        rs20 = relative_strength_20 if relative_strength_20 is not None else _relative_strength(history, benchmark, signal_date, 20)
        if rs60 < self.config.min_relative_strength:
            return TrendEvaluation(None, "weak_60d_relative_strength", True, signal=signal, relative_strength=rs60)
        if rs20 < self.config.min_short_relative_strength:
            return TrendEvaluation(
                None,
                "weak_20d_relative_strength",
                True,
                signal=signal,
                relative_strength=rs60,
                short_relative_strength=rs20,
            )
        rs_score = _relative_strength_rank_score(rs60, rs20)

        avg_amount = _avg_amount_20(history)
        if avg_amount < self.config.min_avg_amount_20:
            return TrendEvaluation(
                None,
                "insufficient_liquidity",
                True,
                signal=signal,
                avg_amount_20=avg_amount,
                relative_strength=rs60,
                short_relative_strength=rs20,
            )
        liquidity_score = min(100.0, avg_amount / self.config.min_avg_amount_20 * 70.0)

        reward_risk = _reward_risk_ratio(history, signal, stop)
        if reward_risk < self.config.min_reward_risk:
            return TrendEvaluation(
                None,
                "reward_risk_too_low",
                True,
                signal=signal,
                reward_risk=reward_risk,
                stop_distance_pct=stop_distance,
                relative_strength=rs60,
                short_relative_strength=rs20,
                avg_amount_20=avg_amount,
            )
        reward_risk_score = min(100.0, reward_risk / 3.0 * 100.0)

        ranking_score = (
            trend_score * 0.30
            + rs_score * 0.25
            + pullback_score * 0.20
            + reward_risk_score * 0.15
            + liquidity_score * 0.10
        )
        candidate = TrendCandidate(
            symbol=symbol,
            analysis=analysis,
            signal=TradeSignal(
                signal.name,
                signal.side,
                signal.strength,
                signal.logic,
                signal.evidence,
                stop_loss=round(stop, 3),
                entry_zone=signal.entry_zone,
                invalidation=signal.invalidation,
                level=signal.level,
                trigger_level=signal.trigger_level,
                invalidation_price=signal.invalidation_price,
                risk_buffer_pct=signal.risk_buffer_pct,
                invalidation_basis=signal.invalidation_basis,
            ),
            ranking_score=ranking_score,
            stop_distance_pct=stop_distance,
            relative_strength=rs60,
            short_relative_strength=rs20,
            avg_amount_20=avg_amount,
            trend_score=trend_score,
            rs_score=rs_score,
            pullback_score=pullback_score,
            reward_risk_score=reward_risk_score,
            liquidity_score=liquidity_score,
            reward_risk=reward_risk,
            market_state=market_state,
        )
        return TrendEvaluation(
            candidate,
            signal=signal,
            trend_score=trend_score,
            rs_score=rs_score,
            pullback_score=pullback_score,
            reward_risk_score=reward_risk_score,
            liquidity_score=liquidity_score,
            reward_risk=reward_risk,
            stop_distance_pct=stop_distance,
            relative_strength=rs60,
            short_relative_strength=rs20,
            avg_amount_20=avg_amount,
        )


def portfolio_backtest(
    data_by_symbol: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    start: str,
    end: str,
    portfolio_config: PortfolioBacktestConfig = PortfolioBacktestConfig(),
    app_config: AppConfig = AppConfig(),
    progress_callback: Callable[[str], None] | None = None,
) -> PortfolioBacktestResult:
    prepared = {symbol: _prepare_ohlcv(df) for symbol, df in data_by_symbol.items() if not df.empty}
    benchmark_prepared = _prepare_ohlcv(benchmark)
    benchmark_featured = add_features(benchmark_prepared, app_config.features).reset_index(drop=True)
    trade_dates = _trade_dates(benchmark_featured, start, end)
    strategy = TrendPullbackStrategy(portfolio_config)

    cash = portfolio_config.initial_cash
    positions: dict[str, Position] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    position_rows: list[dict] = []
    candidate_rows: list[dict] = []
    started = time.time()
    steps = max(len(trade_dates) - 1, 0)

    for step, (signal_date, execution_date) in enumerate(zip(trade_dates[:-1], trade_dates[1:]), start=1):
        market_state = _market_state(benchmark_featured, signal_date)
        target_position_count = _target_position_count(portfolio_config, market_state)

        exits: list[tuple[str, str]] = []
        for symbol, position in list(positions.items()):
            history = _history_until(prepared[symbol], signal_date)
            if len(history) < 120:
                continue
            close = float(history.iloc[-1]["close"])
            position.high_close = max(position.high_close, close)
            if close >= position.high_close:
                position.highest_date = signal_date
            position.days_held += 1

            reason = _exit_reason(symbol, _analysis_window(history, portfolio_config), position, portfolio_config, app_config)
            if reason:
                exits.append((symbol, reason))

        sold_symbols: set[str] = set()
        for symbol, reason in exits:
            row = _row_on_date(prepared[symbol], execution_date)
            if row is None:
                continue
            position = positions.pop(symbol)
            exit_price = float(row["open"]) * (1 - portfolio_config.slippage)
            proceeds = position.shares * exit_price * (1 - portfolio_config.fee_rate - portfolio_config.stamp_tax)
            cash += proceeds
            trades.append(_trade_row(execution_date, symbol, "sell", exit_price, position.shares, reason, cash, position))
            sold_symbols.add(symbol)

        candidates: list[TrendCandidate] = []
        if target_position_count > len(positions):
            for symbol, data in prepared.items():
                if symbol in positions or symbol in sold_symbols:
                    continue
                history = _history_until(data, signal_date)
                if len(history) < 120:
                    continue
                analysis_history = _analysis_window(history, portfolio_config)
                try:
                    analysis = analyze_ohlcv(analysis_history, symbol=symbol, benchmark=benchmark_prepared, config=app_config)
                except ValueError:
                    continue
                evaluation = strategy.diagnose(
                    symbol=symbol,
                    analysis=analysis,
                    history=analysis_history,
                    benchmark=benchmark_prepared,
                    signal_date=signal_date,
                    market_state=market_state,
                    relative_strength_60=_relative_strength(history, benchmark_prepared, signal_date, 60),
                    relative_strength_20=_relative_strength(history, benchmark_prepared, signal_date, 20),
                )
                if evaluation.candidate:
                    candidates.append(evaluation.candidate)
                elif evaluation.log_rejection:
                    candidate_rows.append(_candidate_row(signal_date, None, symbol, analysis, evaluation, market_state, prepared[symbol]))

        candidates.sort(key=lambda candidate: candidate.ranking_score, reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            candidate_rows.append(
                _candidate_row(signal_date, rank, candidate.symbol, candidate.analysis, candidate, market_state, prepared[candidate.symbol])
            )

        slots = max(0, target_position_count - len(positions))
        if slots > 0:
            equity_before_buy = cash + _positions_value(positions, prepared, signal_date)
            target_value = equity_before_buy / max(portfolio_config.max_positions, 1)
            for candidate in candidates[:slots]:
                row = _row_on_date(prepared[candidate.symbol], execution_date)
                if row is None:
                    continue
                entry_price = float(row["open"]) * (1 + portfolio_config.slippage)
                shares = int(min(target_value, cash) / entry_price // 100 * 100)
                if shares <= 0:
                    continue
                cost = shares * entry_price * (1 + portfolio_config.fee_rate)
                if cost > cash:
                    continue
                initial_stop = float(candidate.signal.stop_loss or 0.0)
                initial_risk = entry_price - initial_stop
                if initial_risk <= 0:
                    continue
                cash -= cost
                positions[candidate.symbol] = Position(
                    symbol=candidate.symbol,
                    shares=shares,
                    entry_price=entry_price,
                    entry_date=execution_date,
                    signal_date=signal_date,
                    stop_loss=initial_stop,
                    initial_stop=initial_stop,
                    initial_risk=initial_risk,
                    high_close=float(row["close"]),
                    highest_date=execution_date,
                    entry_score=candidate.ranking_score,
                    reward_risk=candidate.reward_risk,
                )
                trades.append(
                    _trade_row(
                        execution_date,
                        candidate.symbol,
                        "buy",
                        entry_price,
                        shares,
                        candidate.signal.name,
                        cash,
                        positions[candidate.symbol],
                    )
                )

        total_equity = cash + _positions_value(positions, prepared, execution_date)
        equity_rows.append(
            {
                "date": execution_date,
                "cash": cash,
                "positions_value": total_equity - cash,
                "total_equity": total_equity,
                "market_state": market_state,
                "market_weak": market_state == "weak",
                "target_positions": target_position_count,
                "positions": len(positions),
            }
        )
        for position in positions.values():
            latest_close = _close_on_or_before(prepared[position.symbol], execution_date)
            position_rows.append(
                {
                    "date": execution_date,
                    "symbol": position.symbol,
                    "shares": position.shares,
                    "entry_price": position.entry_price,
                    "stop_loss": position.stop_loss,
                    "initial_stop": position.initial_stop,
                    "high_close": position.high_close,
                    "close": latest_close,
                    "days_held": position.days_held,
                    "entry_score": position.entry_score,
                    "reward_risk": position.reward_risk,
                }
            )
        if progress_callback:
            progress_callback(
                _portfolio_progress_line(
                    step=step,
                    total=steps,
                    signal_date=signal_date,
                    execution_date=execution_date,
                    market_state=market_state,
                    candidates=len(candidates),
                    positions=len(positions),
                    started=started,
                )
            )

    final_date = trade_dates[-1] if trade_dates else pd.to_datetime(end)
    for symbol, position in list(positions.items()):
        row = _row_on_date(prepared[symbol], final_date)
        if row is None:
            continue
        exit_price = float(row["close"]) * (1 - portfolio_config.slippage)
        cash += position.shares * exit_price * (1 - portfolio_config.fee_rate - portfolio_config.stamp_tax)
        trades.append(_trade_row(final_date, symbol, "sell", exit_price, position.shares, "end_of_data", cash, position))
        positions.pop(symbol)

    if trade_dates:
        final_market_state = _market_state(benchmark_featured, final_date)
        final_total_equity = cash + _positions_value(positions, prepared, final_date)
        final_row = {
            "date": final_date,
            "cash": cash,
            "positions_value": final_total_equity - cash,
            "total_equity": final_total_equity,
            "market_state": final_market_state,
            "market_weak": final_market_state == "weak",
            "target_positions": _target_position_count(portfolio_config, final_market_state),
            "positions": len(positions),
        }
        if equity_rows and pd.Timestamp(equity_rows[-1]["date"]) == final_date:
            equity_rows[-1] = final_row
        else:
            equity_rows.append(final_row)

    equity = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)
    positions_df = pd.DataFrame(position_rows)
    candidates_df = pd.DataFrame(candidate_rows)
    metrics = _portfolio_metrics(equity, trades_df, benchmark_prepared, start, end, portfolio_config.initial_cash)
    metrics.update(_signal_health_metrics(candidates_df))
    return PortfolioBacktestResult(equity, trades_df, positions_df, candidates_df, metrics)


def _exit_reason(
    symbol: str,
    history: pd.DataFrame,
    position: Position,
    config: PortfolioBacktestConfig,
    app_config: AppConfig,
) -> str | None:
    featured = add_features(history, app_config.features).dropna(subset=["ma_slow"]).reset_index(drop=True)
    if featured.empty:
        return None
    latest = featured.iloc[-1]
    close = float(latest["close"])
    profit_r = (position.high_close - position.entry_price) / position.initial_risk

    if profit_r >= config.breakeven_r:
        position.stop_loss = max(position.stop_loss, position.entry_price)
    if profit_r >= config.trail_start_r or position.high_close >= position.entry_price * 1.12:
        trend_stop = max(float(latest["ma_fast"]) * 0.98, position.high_close * (1 - config.trailing_stop_pct))
        position.stop_loss = max(position.stop_loss, trend_stop)

    if close <= position.stop_loss:
        if position.stop_loss <= position.initial_stop * 1.001:
            return "stop_loss"
        if position.stop_loss <= position.entry_price * 1.001:
            return "breakeven_stop"
        return "trailing_stop"

    try:
        analysis = analyze_ohlcv(history, symbol=symbol, config=app_config)
    except ValueError:
        return None
    signal_names = {signal.name for signal in analysis.signals}
    if signal_names & IMMEDIATE_EXIT_SIGNAL_NAMES:
        return "sell_signal"
    if signal_names & CONFIRMED_EXIT_SIGNAL_NAMES:
        previous_confirmed = _previous_has_signal(history, symbol, CONFIRMED_EXIT_SIGNAL_NAMES, app_config)
        if previous_confirmed or close < float(latest["ma_fast"]):
            return "confirmed_stall_sell"
    if position.days_held >= config.stale_days and position.high_close < position.entry_price + position.initial_risk:
        return "stale_no_1r"
    if position.days_held >= config.max_holding_days and close < position.high_close * 0.995:
        return "max_holding_days"
    return None


def _previous_has_signal(history: pd.DataFrame, symbol: str, signal_names: set[str], app_config: AppConfig) -> bool:
    if len(history) < 121:
        return False
    try:
        previous = analyze_ohlcv(history.iloc[:-1].copy(), symbol=symbol, config=app_config)
    except ValueError:
        return False
    return any(signal.name in signal_names for signal in previous.signals)


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _trade_dates(benchmark: pd.DataFrame, start: str, end: str) -> list[pd.Timestamp]:
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    dates = benchmark.loc[benchmark["date"].between(start_ts, end_ts), "date"]
    return [pd.Timestamp(date) for date in dates.drop_duplicates().sort_values()]


def _history_until(data: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    return data[data["date"] <= date].copy().reset_index(drop=True)


def _analysis_window(data: pd.DataFrame, config: PortfolioBacktestConfig) -> pd.DataFrame:
    return data.tail(config.analysis_lookback_days).copy().reset_index(drop=True)


def _ensure_featured(data: pd.DataFrame) -> pd.DataFrame:
    if {"ma_fast", "ma_slow", "ma_long", "vol_ma20", "amplitude"}.issubset(data.columns):
        return data.copy().reset_index(drop=True)
    return add_features(data).reset_index(drop=True)


def _row_on_date(data: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    rows = data[data["date"] == date]
    if rows.empty:
        return None
    return rows.iloc[0]


def _close_on_or_before(data: pd.DataFrame, date: pd.Timestamp) -> float:
    history = data[data["date"] <= date]
    if history.empty:
        return 0.0
    return float(history.iloc[-1]["close"])


def _positions_value(positions: dict[str, Position], data_by_symbol: dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
    value = 0.0
    for symbol, position in positions.items():
        value += position.shares * _close_on_or_before(data_by_symbol[symbol], date)
    return value


def _avg_amount_20(history: pd.DataFrame) -> float:
    latest = history.tail(20).copy()
    if "amount" in latest.columns and latest["amount"].notna().any():
        amount = pd.to_numeric(latest["amount"], errors="coerce")
    else:
        amount = latest["close"] * latest["volume"]
    return float(amount.dropna().mean() or 0.0)


def _relative_strength(history: pd.DataFrame, benchmark: pd.DataFrame, date: pd.Timestamp, window: int = 60) -> float:
    stock = history[history["date"] <= date].tail(window)
    bench = benchmark[benchmark["date"] <= date].tail(window)
    min_rows = min(20, window)
    if len(stock) < min_rows or len(bench) < min_rows:
        return 0.0
    stock_return = float(stock["close"].iloc[-1] / stock["close"].iloc[0] - 1)
    bench_return = float(bench["close"].iloc[-1] / bench["close"].iloc[0] - 1)
    return stock_return - bench_return


def _market_state(benchmark_featured: pd.DataFrame, date: pd.Timestamp) -> str:
    rows = benchmark_featured[benchmark_featured["date"] <= date]
    if rows.empty:
        return "weak"
    latest = rows.iloc[-1]
    ma_slow = latest.get("ma_slow")
    ma_fast = latest.get("ma_fast")
    if pd.isna(ma_slow) or pd.isna(ma_fast):
        return "weak"
    if latest["close"] > ma_slow and ma_fast > ma_slow:
        return "strong"
    if latest["close"] < ma_slow and ma_fast < ma_slow:
        return "weak"
    return "neutral"


def _target_position_count(config: PortfolioBacktestConfig, market_state: str) -> int:
    mode = config.market_mode.lower()
    if market_state == "weak":
        return 0
    if market_state == "strong":
        return config.max_positions
    if mode == "strict":
        return 0
    if mode == "aggressive":
        return config.max_positions
    return min(config.neutral_max_positions, config.max_positions)


def _find_signal(analysis: AnalysisResult, name: str) -> TradeSignal | None:
    return next((signal for signal in analysis.signals if signal.name == name and signal.side == SignalSide.BUY), None)


def _trend_quality_score(history: pd.DataFrame) -> float:
    latest = history.iloc[-1]
    close = float(latest["close"])
    ma_fast = float(latest.get("ma_fast", close))
    ma_slow = float(latest.get("ma_slow", ma_fast))
    ma_long = float(latest.get("ma_long", ma_slow))
    ma_fast_slope = float(latest.get("ma_fast_slope", 0) or 0)
    ma_slow_slope = float(latest.get("ma_slow_slope", 0) or 0)
    if not (close > ma_fast > ma_slow > ma_long and ma_fast_slope > 0 and ma_slow_slope > 0):
        return 0.0
    prior = history.iloc[-21] if len(history) >= 21 else history.iloc[0]
    highs_up = float(latest.get("rolling_high_20", close)) >= float(prior.get("rolling_high_20", close))
    lows_up = float(latest.get("rolling_low_20", close)) >= float(prior.get("rolling_low_20", close))
    red_fat = bool(latest.get("red_fat_green_thin", False))
    score = 70.0
    if highs_up:
        score += 10.0
    if lows_up:
        score += 10.0
    if red_fat:
        score += 10.0
    return min(100.0, score)


def _pullback_quality_score(history: pd.DataFrame, signal: TradeSignal, config: PortfolioBacktestConfig) -> tuple[float, str]:
    latest = history.iloc[-1]
    pullback = history.tail(8)
    peak = float(pullback["close"].max())
    close = float(latest["close"])
    pullback_depth = 1 - close / peak if peak > 0 else 0.0
    if pullback_depth < 0.03:
        return 0.0, "pullback_too_shallow"
    if pullback_depth > 0.10:
        return 0.0, "pullback_too_deep"

    support_distance = _support_distance(close, signal)
    if support_distance is not None and support_distance > 0.04:
        return 0.0, "too_far_above_support"
    if support_distance is not None and support_distance < -0.005:
        return 0.0, "support_broken"

    volume_shrink = float(pullback["volume"].tail(3).mean()) <= float(history["vol_ma20"].iloc[-1]) * 0.75
    range_contract = float(pullback["amplitude"].tail(3).mean()) <= float(history["amplitude"].tail(20).mean()) * 0.80
    holds_support = signal.level is None or float(latest["low"]) >= float(signal.level.price) * 0.985
    if not volume_shrink:
        return 40.0, "pullback_volume_not_dry"
    if not range_contract:
        return 55.0, "pullback_range_not_contracting"
    if not holds_support:
        return 55.0, "support_intraday_broken"

    score = 70.0
    score += min(15.0, (pullback_depth - 0.03) / 0.07 * 15.0)
    if support_distance is not None and 0.0 <= support_distance <= 0.04:
        score += 10.0
    if bool(latest.get("narrow_range", False)) or float(latest.get("close_position", 0.5)) >= 0.50:
        score += 5.0
    return min(100.0, score), ""


def _support_distance(close: float, signal: TradeSignal) -> float | None:
    if signal.level is None or signal.level.price <= 0:
        return None
    return close / signal.level.price - 1


def _initial_stop(signal: TradeSignal, history: pd.DataFrame, config: PortfolioBacktestConfig) -> float:
    if signal.stop_loss is not None:
        return float(signal.stop_loss)
    pullback_low = float(history["low"].tail(5).min())
    level_stop = float(signal.level.price) if signal.level is not None else pullback_low
    return min(level_stop, pullback_low) * 0.99


def _confirmed_stop_bar(latest: pd.Series) -> bool:
    return bool(latest["close"] >= latest["open"] and latest.get("close_position", 0.0) >= 0.55)


def _relative_strength_rank_score(relative_strength_60: float, relative_strength_20: float) -> float:
    score = 50.0
    score += max(-25.0, min(35.0, relative_strength_60 * 350.0))
    score += max(-15.0, min(15.0, relative_strength_20 * 250.0))
    return max(0.0, min(100.0, score))


def _reward_risk_ratio(history: pd.DataFrame, signal: TradeSignal, stop: float) -> float:
    latest = history.iloc[-1]
    close = float(latest["close"])
    risk = close - stop
    if risk <= 0:
        return 0.0
    resistance_targets = [float(level.price) for level in history.attrs.get("resistances", []) if level.price > close]
    technical_targets = [
        float(history["high"].tail(20).max()),
        float(history["high"].tail(60).max()),
    ]
    if signal.level is not None:
        technical_targets.append(float(signal.level.price) * 1.08)
    targets = [target for target in resistance_targets + technical_targets if target > close]
    if not targets:
        return 0.0
    reward = min(targets) - close
    return reward / risk


def _candidate_row(
    date: pd.Timestamp,
    rank: int | None,
    symbol: str,
    analysis: AnalysisResult,
    item: TrendCandidate | TrendEvaluation,
    market_state: str,
    future_data: pd.DataFrame | None = None,
) -> dict:
    candidate = item if isinstance(item, TrendCandidate) else item.candidate
    source = candidate or item
    signal = candidate.signal.name if candidate else (item.signal.name if isinstance(item, TrendEvaluation) and item.signal else "")
    row = {
        "date": date,
        "rank": rank,
        "symbol": symbol,
        "score": analysis.score.total,
        "ranking_score": candidate.ranking_score if candidate else 0.0,
        "trend_score": source.trend_score,
        "rs_score": source.rs_score,
        "pullback_score": source.pullback_score,
        "reward_risk_score": source.reward_risk_score,
        "liquidity_score": source.liquidity_score,
        "relative_strength": source.relative_strength,
        "short_relative_strength": source.short_relative_strength,
        "stop_distance_pct": source.stop_distance_pct,
        "avg_amount_20": source.avg_amount_20,
        "reward_risk": source.reward_risk,
        "market_state": market_state,
        "signal": signal,
        "reject_reason": "" if candidate else item.reject_reason,
    }
    radar = getattr(analysis, "radar", None)
    if radar is not None:
        row.update(
            {
                "sector_name": radar.sector_name,
                "sector_rs_rank": radar.sector_rs_rank,
                "stock_vs_sector_rs": radar.stock_vs_sector_rs,
                "setup_score": radar.setup_score,
                "entry_quality_score": radar.entry_quality_score,
                "exit_risk_score": radar.exit_risk_score,
                "expected_action": radar.expected_action,
                "radar_reject_reason": radar.reject_reason,
            }
        )
    row.update(_forward_returns(future_data, date))
    return row


def _trade_row(
    date: pd.Timestamp,
    symbol: str,
    side: str,
    price: float,
    shares: int,
    reason: str,
    cash: float,
    position: Position,
) -> dict:
    if side == "sell":
        r_multiple = (price - position.entry_price) / position.initial_risk
    else:
        r_multiple = 0.0
    return {
        "date": date,
        "symbol": symbol,
        "side": side,
        "price": price,
        "shares": shares,
        "reason": reason,
        "exit_reason": reason if side == "sell" else "",
        "cash": cash,
        "holding_days": position.days_held,
        "entry_score": position.entry_score,
        "initial_stop": position.initial_stop,
        "r_multiple": r_multiple,
    }


def _forward_returns(data: pd.DataFrame | None, signal_date: pd.Timestamp) -> dict[str, float | None]:
    empty = {f"fwd_return_{days}d": None for days in [5, 10, 20, 45]}
    empty["fwd_max_gain_45d"] = None
    empty["fwd_max_drawdown_45d"] = None
    if data is None or data.empty:
        return empty
    rows = data[data["date"] > signal_date].reset_index(drop=True)
    signal_row = data[data["date"] <= signal_date].tail(1)
    if rows.empty or signal_row.empty:
        return empty
    base = float(signal_row.iloc[-1]["close"])
    result: dict[str, float | None] = {}
    for days in [5, 10, 20, 45]:
        if len(rows) >= days and base > 0:
            result[f"fwd_return_{days}d"] = float(rows.iloc[days - 1]["close"] / base - 1)
        else:
            result[f"fwd_return_{days}d"] = None
    horizon = rows.head(45)
    if horizon.empty or base <= 0:
        result["fwd_max_gain_45d"] = None
        result["fwd_max_drawdown_45d"] = None
    else:
        result["fwd_max_gain_45d"] = float(horizon["high"].max() / base - 1)
        result["fwd_max_drawdown_45d"] = float(horizon["low"].min() / base - 1)
    return result


def _signal_health_metrics(candidates: pd.DataFrame) -> dict[str, float]:
    if candidates.empty:
        return {}
    if "rank" not in candidates.columns:
        return {}
    reject_reason = candidates["reject_reason"] if "reject_reason" in candidates.columns else ""
    actionable = candidates[(candidates["rank"].notna()) & (reject_reason == "")]
    metrics: dict[str, float] = {}
    for days in [5, 10, 20, 45]:
        column = f"fwd_return_{days}d"
        if column not in actionable.columns:
            continue
        values = pd.to_numeric(actionable[column], errors="coerce").dropna()
        metrics[f"signal_{days}d_count"] = float(len(values))
        metrics[f"signal_{days}d_avg_return"] = float(values.mean()) if not values.empty else 0.0
        metrics[f"signal_{days}d_win_rate"] = float((values > 0).mean()) if not values.empty else 0.0
    for bucket, frame in _candidate_buckets(actionable):
        returns = pd.to_numeric(frame.get("fwd_return_20d"), errors="coerce").dropna()
        if returns.empty:
            continue
        prefix = f"signal_bucket_{bucket}"
        metrics[f"{prefix}_count"] = float(len(returns))
        metrics[f"{prefix}_20d_avg_return"] = float(returns.mean())
        metrics[f"{prefix}_20d_win_rate"] = float((returns > 0).mean())
    return metrics


def _candidate_buckets(candidates: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    buckets: list[tuple[str, pd.DataFrame]] = []
    if "setup_score" in candidates.columns:
        buckets.append(("setup_70_plus", candidates[pd.to_numeric(candidates["setup_score"], errors="coerce") >= 70]))
    if "stock_vs_sector_rs" in candidates.columns:
        buckets.append(("strong_vs_sector", candidates[pd.to_numeric(candidates["stock_vs_sector_rs"], errors="coerce") >= 0.03]))
    if "stop_distance_pct" in candidates.columns:
        stop = pd.to_numeric(candidates["stop_distance_pct"], errors="coerce")
        buckets.append(("stop_3_7pct", candidates[(stop >= 0.03) & (stop <= 0.07)]))
    return buckets


def _portfolio_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark: pd.DataFrame,
    start: str,
    end: str,
    initial_cash: float,
) -> dict[str, float]:
    if equity.empty:
        final_equity = initial_cash
        total_return = 0.0
        max_drawdown = 0.0
        annual_return = 0.0
    else:
        final_equity = float(equity["total_equity"].iloc[-1])
        total_return = final_equity / initial_cash - 1
        curve = equity["total_equity"]
        drawdown = curve / curve.cummax() - 1
        max_drawdown = float(drawdown.min())
        days = max(1, (pd.to_datetime(equity["date"].iloc[-1]) - pd.to_datetime(equity["date"].iloc[0])).days)
        annual_return = (final_equity / initial_cash) ** (365 / days) - 1 if final_equity > 0 else -1.0

    trade_pairs = []
    open_trades: dict[str, pd.Series] = {}
    if not trades.empty:
        for _, trade in trades.iterrows():
            if trade["side"] == "buy":
                open_trades[trade["symbol"]] = trade
            elif trade["side"] == "sell" and trade["symbol"] in open_trades:
                buy = open_trades.pop(trade["symbol"])
                trade_pairs.append(
                    {
                        "return": float(trade["price"] / buy["price"] - 1),
                        "holding_days": float(trade.get("holding_days", 0)),
                        "exit_reason": str(trade.get("exit_reason") or trade.get("reason") or ""),
                    }
                )
    wins = [pair for pair in trade_pairs if pair["return"] > 0]
    avg_trade_return = sum(pair["return"] for pair in trade_pairs) / len(trade_pairs) if trade_pairs else 0.0
    avg_holding_days = sum(pair["holding_days"] for pair in trade_pairs) / len(trade_pairs) if trade_pairs else 0.0
    benchmark_return = _benchmark_return(benchmark, start, end)
    return_drawdown_ratio = total_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    metrics: dict[str, float] = {
        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "return_drawdown_ratio": return_drawdown_ratio,
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "trade_count": len(trade_pairs),
        "win_rate": len(wins) / len(trade_pairs) if trade_pairs else 0.0,
        "avg_trade_return": avg_trade_return,
        "avg_holding_days": avg_holding_days,
    }
    for reason in sorted({pair["exit_reason"] for pair in trade_pairs if pair["exit_reason"]}):
        pairs = [pair for pair in trade_pairs if pair["exit_reason"] == reason]
        reason_wins = [pair for pair in pairs if pair["return"] > 0]
        prefix = f"exit_{reason}"
        metrics[f"{prefix}_count"] = float(len(pairs))
        metrics[f"{prefix}_win_rate"] = len(reason_wins) / len(pairs) if pairs else 0.0
        metrics[f"{prefix}_avg_return"] = sum(pair["return"] for pair in pairs) / len(pairs) if pairs else 0.0
        metrics[f"{prefix}_avg_holding_days"] = sum(pair["holding_days"] for pair in pairs) / len(pairs) if pairs else 0.0
    return metrics


def _benchmark_return(benchmark: pd.DataFrame, start: str, end: str) -> float:
    rows = benchmark[benchmark["date"].between(pd.to_datetime(start), pd.to_datetime(end))]
    if len(rows) < 2:
        return 0.0
    return float(rows["close"].iloc[-1] / rows["close"].iloc[0] - 1)


def _portfolio_progress_line(
    step: int,
    total: int,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    market_state: str,
    candidates: int,
    positions: int,
    started: float,
) -> str:
    elapsed = time.time() - started
    avg = elapsed / max(step, 1)
    remaining = avg * max(total - step, 0)
    percent = step / max(total, 1) * 100
    return (
        f"[{step}/{total} {percent:5.1f}%] signal={signal_date.date()} execute={execution_date.date()} "
        f"market={market_state} candidates={candidates} positions={positions} "
        f"elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(remaining)}"
    )


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minute:02d}m"
    if minute:
        return f"{minute}m{sec:02d}s"
    return f"{sec}s"
