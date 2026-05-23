from argparse import Namespace

import numpy as np
import pandas as pd

from stockbuyornot.analysis import analyze_ohlcv
from stockbuyornot.cli import _load_portfolio_data
from stockbuyornot.data.providers import AkshareProvider
from stockbuyornot.models import SignalSide, TradeSignal
from stockbuyornot.portfolio import (
    PortfolioBacktestConfig,
    Position,
    TrendPullbackStrategy,
    _exit_reason,
    _target_position_count,
    portfolio_backtest,
)


def _benchmark_fixture(rows: int = 220, weak: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = np.linspace(100, 82 if weak else 130, rows)
    open_ = close * 0.998
    high = close * 1.01
    low = close * 0.99
    volume = np.full(rows, 1_000_000)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": volume * close,
            "symbol": "000300",
        }
    )


def _trend_pullback_fixture(rows: int = 220, amount_scale: float = 80_000_000) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = np.linspace(10, 22, rows) + np.sin(np.arange(rows) / 5) * 0.12
    open_ = close * 0.997
    high = np.maximum(open_, close) * 1.012
    low = np.minimum(open_, close) * 0.988
    volume = np.linspace(5_000_000, 6_000_000, rows)

    close[-7:] = [23.0, 22.6, 22.25, 22.05, 21.95, 21.90, 22.10]
    open_[-7:] = [23.1, 22.7, 22.32, 22.08, 21.98, 21.92, 21.95]
    high[-7:] = np.maximum(open_[-7:], close[-7:]) * 1.004
    low[-7:] = np.minimum(open_[-7:], close[-7:]) * 0.996
    volume[-7:] = [5_200_000, 4_400_000, 3_600_000, 2_800_000, 2_200_000, 1_800_000, 1_600_000]

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount_scale,
            "symbol": "UP",
        }
    )


def _downtrend_fixture(rows: int = 220) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = np.linspace(25, 12, rows)
    open_ = close * 1.002
    high = open_ * 1.01
    low = close * 0.99
    volume = np.linspace(6_000_000, 3_000_000, rows)
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": 80_000_000,
            "symbol": "DOWN",
        }
    )


def _position() -> Position:
    return Position(
        symbol="UP",
        shares=1000,
        entry_price=20.0,
        entry_date=pd.Timestamp("2024-10-01"),
        signal_date=pd.Timestamp("2024-09-30"),
        stop_loss=19.0,
        initial_stop=19.0,
        initial_risk=1.0,
        high_close=21.2,
        highest_date=pd.Timestamp("2024-10-10"),
        entry_score=88.0,
        reward_risk=2.5,
        days_held=8,
    )


def test_trend_pullback_candidate_enters_strategy():
    df = _trend_pullback_fixture()
    benchmark = _benchmark_fixture()
    analysis = analyze_ohlcv(df, symbol="UP", benchmark=benchmark)
    candidate = TrendPullbackStrategy(PortfolioBacktestConfig(min_score=70, min_avg_amount_20=1)).evaluate("UP", analysis, df, 0.08)

    assert candidate is not None
    assert candidate.signal.name == "上涨中继买点"
    assert candidate.stop_distance_pct <= 0.07
    assert candidate.reward_risk >= 1.8


def test_downtrend_is_not_candidate():
    df = _downtrend_fixture()
    analysis = analyze_ohlcv(df, symbol="DOWN", benchmark=_benchmark_fixture())
    candidate = TrendPullbackStrategy(PortfolioBacktestConfig(min_score=0)).evaluate("DOWN", analysis, df, 0.0)

    assert candidate is None


def test_stop_distance_over_limit_is_filtered():
    df = _trend_pullback_fixture()
    analysis = analyze_ohlcv(df, symbol="UP", benchmark=_benchmark_fixture())
    candidate = TrendPullbackStrategy(
        PortfolioBacktestConfig(max_stop_distance_pct=0.001, min_score=0, min_avg_amount_20=1, min_reward_risk=0)
    ).evaluate("UP", analysis, df, 0.08)

    assert candidate is None


def test_weak_relative_strength_is_filtered():
    df = _trend_pullback_fixture()
    benchmark = _benchmark_fixture()
    analysis = analyze_ohlcv(df, symbol="UP", benchmark=benchmark)
    evaluation = TrendPullbackStrategy(PortfolioBacktestConfig(min_score=0, min_avg_amount_20=1)).diagnose(
        "UP",
        analysis,
        df,
        benchmark,
        df["date"].iloc[-1],
        "strong",
        relative_strength_60=-0.01,
        relative_strength_20=0.0,
    )

    assert evaluation.candidate is None
    assert evaluation.reject_reason == "weak_60d_relative_strength"


def test_overextended_markup_is_filtered():
    df = _trend_pullback_fixture()
    df.loc[df.index[-1], "close"] = df["close"].iloc[-1] * 1.18
    df.loc[df.index[-1], "high"] = df["close"].iloc[-1] * 1.01
    analysis = analyze_ohlcv(df, symbol="UP", benchmark=_benchmark_fixture())
    candidate = TrendPullbackStrategy(PortfolioBacktestConfig(min_score=0, min_avg_amount_20=1, min_reward_risk=0)).evaluate(
        "UP",
        analysis,
        df,
        0.08,
    )

    assert candidate is None


def test_weak_market_opens_no_new_positions():
    up = _trend_pullback_fixture()
    benchmark = _benchmark_fixture(weak=True)
    result = portfolio_backtest(
        {"UP": up},
        benchmark,
        start=str(up["date"].iloc[140].date()).replace("-", ""),
        end=str(up["date"].iloc[-1].date()).replace("-", ""),
        portfolio_config=PortfolioBacktestConfig(min_score=0, min_avg_amount_20=1, min_relative_strength=-1, min_reward_risk=0),
    )

    assert result.trades.empty
    assert result.metrics["trade_count"] == 0


def test_neutral_market_limits_new_positions():
    config = PortfolioBacktestConfig(max_positions=5, neutral_max_positions=2)

    assert _target_position_count(config, "neutral") == 2
    assert _target_position_count(config, "weak") == 0
    assert _target_position_count(config, "strong") == 5


def test_portfolio_backtest_generates_diagnostic_outputs():
    up = _trend_pullback_fixture()
    benchmark = _benchmark_fixture(weak=False)
    result = portfolio_backtest(
        {"UP": up},
        benchmark,
        start=str(up["date"].iloc[140].date()).replace("-", ""),
        end=str(up["date"].iloc[-1].date()).replace("-", ""),
        portfolio_config=PortfolioBacktestConfig(
            min_score=0,
            min_avg_amount_20=1,
            max_positions=1,
            min_relative_strength=-1,
            min_reward_risk=0,
        ),
    )

    assert not result.equity.empty
    assert not result.candidates.empty
    assert set(result.trades["side"]).issubset({"buy", "sell"})
    assert {"entry_score", "initial_stop", "exit_reason", "r_multiple"}.issubset(result.trades.columns)
    assert {"trend_score", "rs_score", "pullback_score", "reward_risk", "market_state", "reject_reason"}.issubset(
        result.candidates.columns
    )


def test_breakeven_stop_moves_after_1r():
    df = _trend_pullback_fixture()
    position = _position()

    _exit_reason("UP", df, position, PortfolioBacktestConfig(min_score=0), AppConfigForTest())

    assert position.stop_loss >= position.entry_price


def test_single_volume_stall_does_not_force_exit(monkeypatch):
    df = _trend_pullback_fixture()
    position = _position()
    position.high_close = 20.4

    def fake_analyze(*args, **kwargs):
        signals = [TradeSignal("放量滞涨卖点", SignalSide.SELL, 80, "", [])] if len(args[0]) == len(df) else []
        return type(
            "FakeAnalysis",
            (),
            {"signals": signals},
        )()

    monkeypatch.setattr("stockbuyornot.portfolio.analyze_ohlcv", fake_analyze)

    assert _exit_reason("UP", df, position, PortfolioBacktestConfig(min_score=0), AppConfigForTest()) is None


def test_immediate_sell_signal_has_priority(monkeypatch):
    df = _trend_pullback_fixture()
    position = _position()
    position.high_close = 20.4

    def fake_analyze(*args, **kwargs):
        return type(
            "FakeAnalysis",
            (),
            {"signals": [TradeSignal("向下反转卖点", SignalSide.SELL, 90, "", [])]},
        )()

    monkeypatch.setattr("stockbuyornot.portfolio.analyze_ohlcv", fake_analyze)

    assert _exit_reason("UP", df, position, PortfolioBacktestConfig(min_score=0), AppConfigForTest()) == "sell_signal"


def test_portfolio_csv_dir_excludes_benchmark_csv(tmp_path):
    stock = _trend_pullback_fixture()
    benchmark = _benchmark_fixture()
    stock.to_csv(tmp_path / "000001.csv", index=False)
    benchmark.to_csv(tmp_path / "benchmark.csv", index=False)

    args = Namespace(
        csv_dir=str(tmp_path),
        benchmark_csv=str(tmp_path / "benchmark.csv"),
        limit=0,
        end="20241031",
        adjust="qfq",
    )

    data = _load_portfolio_data(args, AkshareProvider(), "20240101")

    assert list(data) == ["000001"]


class AppConfigForTest:
    from stockbuyornot.config import FeatureConfig, LevelConfig, ScoringConfig, SignalConfig

    features = FeatureConfig()
    levels = LevelConfig()
    signals = SignalConfig()
    scoring = ScoringConfig()
