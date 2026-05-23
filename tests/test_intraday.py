import pandas as pd

from stockbuyornot.data.providers import _normalize_intraday
from stockbuyornot.intraday import IntradaySummary, compute_intraday_adjustment, summarize_intraday
from stockbuyornot.ui.streamlit_app import short_error


def test_normalize_intraday_akshare_columns():
    raw = pd.DataFrame(
        {
            "时间": ["2026-05-21 09:35:00", "2026-05-21 09:40:00"],
            "开盘": [10.0, 10.1],
            "收盘": [10.1, 10.2],
            "最高": [10.2, 10.3],
            "最低": [9.9, 10.0],
            "成交量": [1000, 1200],
            "成交额": [1010000, 1224000],
        }
    )

    result = _normalize_intraday(raw, symbol="000001")

    assert list(result.columns) == ["date", "open", "high", "low", "close", "volume", "amount", "symbol"]
    assert result.loc[0, "symbol"] == "000001"
    assert result.loc[1, "close"] == 10.2
    assert result.loc[1, "amount"] == 1224000


def test_summarize_intraday_returns_confirmation_labels():
    times = pd.date_range("2026-05-21 09:35:00", periods=30, freq="5min")
    closes = [10 + i * 0.02 for i in range(30)]
    data = pd.DataFrame(
        {
            "date": times,
            "open": closes,
            "high": [price + 0.03 for price in closes],
            "low": [price - 0.03 for price in closes],
            "close": closes,
            "volume": [1000 + i * 10 for i in range(30)],
            "amount": [(1000 + i * 10) * closes[i] for i in range(30)],
            "symbol": "000001",
        }
    )

    summary = summarize_intraday(data, symbol="000001")

    assert summary.symbol == "000001"
    assert summary.latest_price > 10
    assert summary.session_return_pct > 0
    assert summary.conclusion in {"分时支持偏多", "分时中性观察", "分时等待确认"}


def test_empty_intraday_error_is_user_friendly():
    assert "没有读取到可用分时数据" in short_error(ValueError("Intraday data is empty."))
    assert "没有读取到可用行情数据" in short_error(ValueError("OHLCV data is empty."))


def test_intraday_adjustment_supports_strong_confirmation():
    summary = IntradaySummary(
        symbol="000001",
        latest_time=pd.Timestamp("2026-05-21 15:00:00"),
        latest_price=10.8,
        session_return_pct=0.018,
        momentum_30m_pct=0.006,
        range_position_pct=0.86,
        volume_ratio=1.6,
        vwap=10.65,
        vwap_gap_pct=0.014,
        trend_label="盘中趋势偏强",
        volume_label="尾盘温和放量",
        conclusion="分时支持偏多",
        explanation="",
    )

    adjustment = compute_intraday_adjustment(summary, 72, 68, daily_advice="短期买入", signal_direction="偏多")

    assert adjustment.adjusted_short_term_score > 68
    assert adjustment.adjusted_liangjia_score > 72
    assert adjustment.action_level == "support"


def test_intraday_adjustment_blocks_weak_confirmation():
    summary = IntradaySummary(
        symbol="000001",
        latest_time=pd.Timestamp("2026-05-21 15:00:00"),
        latest_price=10.1,
        session_return_pct=-0.015,
        momentum_30m_pct=-0.007,
        range_position_pct=0.12,
        volume_ratio=1.8,
        vwap=10.25,
        vwap_gap_pct=-0.015,
        trend_label="盘中趋势偏弱",
        volume_label="尾盘温和放量",
        conclusion="分时提示风险",
        explanation="",
    )

    adjustment = compute_intraday_adjustment(summary, 76, 74, daily_advice="短期买入", signal_direction="偏多")

    assert adjustment.adjusted_short_term_score < 74
    assert adjustment.adjusted_liangjia_score < 76
    assert adjustment.action_level == "risk"
    assert "不支持买入" in adjustment.action


def test_intraday_adjustment_does_not_override_bearish_daily_signal():
    summary = IntradaySummary(
        symbol="000001",
        latest_time=pd.Timestamp("2026-05-21 15:00:00"),
        latest_price=10.8,
        session_return_pct=0.018,
        momentum_30m_pct=0.006,
        range_position_pct=0.86,
        volume_ratio=1.6,
        vwap=10.65,
        vwap_gap_pct=0.014,
        trend_label="盘中趋势偏强",
        volume_label="尾盘温和放量",
        conclusion="分时支持偏多",
        explanation="",
    )

    adjustment = compute_intraday_adjustment(summary, 72, 68, daily_advice="短期卖出", signal_direction="偏空")

    assert adjustment.short_term_modifier <= 3
    assert adjustment.action_level == "risk"
    assert "分时不能单独改成买入" in "；".join(adjustment.risk_flags)
