import pandas as pd

from stockbuyornot.ui import streamlit_app as app
from stockbuyornot.ui.streamlit_app import (
    compact_scan_display,
    compute_market_status_snapshot,
    filter_scan_display_by_min_score,
    filter_scan_results_by_display,
    factor_score_badges,
    execution_window_from_scores,
    execution_window_label,
    execution_window_tooltip,
    localize_scan_table,
    market_status_card,
    metric_card,
    metric_score_tooltip,
    parse_watchlist_price_updates,
    parse_entry_zone,
    price_text,
    purchased_pnl_pct,
    purchased_risk_status,
    result_to_scan_row,
    scan_candidate_score,
    score_badge_tooltip,
    scan_save_sort_score,
    sort_purchased_records,
    watchlist_detail_plan_text,
    watchlist_execution_status,
    watchlist_intraday_rule,
    watchlist_plan_row,
    watchlist_trade_tier,
    update_watchlist_manual_prices,
)
from stockbuyornot.models import Stage


def _record(price: float | None) -> dict:
    return {
        "manual_price": price,
        "signals": [
            {
                "买入区间": "11.00 - 12.00",
                "操作失效价": 10.50,
            }
        ],
    }


def test_compute_market_status_snapshot_identifies_strong_market():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=90, freq="D"),
            "close": [3000 + i * 8 for i in range(90)],
        }
    )

    snapshot = compute_market_status_snapshot(frame)

    assert snapshot["state"] == "strong"
    assert snapshot["state_label"] == "强市"
    assert snapshot["ret20"] > 0
    assert snapshot["close_vs_ma60"] > 0
    assert snapshot["tomorrow_label"] in {"偏强", "略偏强"}
    assert snapshot["tomorrow_score"] > 55


def test_market_status_card_contains_operation_hint():
    html = market_status_card(
        {
            "state": "weak",
            "state_label": "弱市",
            "as_of": "2026-05-21",
            "close": 4.808,
            "ret20": -0.03,
            "ret60": -0.08,
            "close_vs_ma60": -0.04,
            "tomorrow_label": "略偏弱",
            "tomorrow_score": 42,
            "tomorrow_hint": "明日倾向：略偏弱。",
            "hint": "弱市：谨慎开新仓。",
        }
    )

    assert "沪深300" in html
    assert "弱市" in html
    assert "明日倾向" in html
    assert "42分" in html
    assert "4.808" in html
    assert "-3.0%" in html


def test_watchlist_execution_status_uses_invalidation_before_entry_zone():
    assert watchlist_execution_status(_record(10.49)) == "invalid"
    assert watchlist_execution_status(_record(10.80)) == "waiting"
    assert watchlist_execution_status(_record(11.50)) == "actionable"
    assert watchlist_execution_status(_record(12.50)) == "extended"


def test_watchlist_execution_status_requires_price():
    assert watchlist_execution_status(_record(None)) == "unpriced"


def test_parse_entry_zone_accepts_common_separators():
    assert parse_entry_zone("11.00 - 12.00") == (11.0, 12.0)
    assert parse_entry_zone("12.00~11.00") == (11.0, 12.0)


def test_scan_candidate_score_blocks_avoid_action():
    score = type("Score", (), {"total": 82, "relative_strength": 5})()
    decision = type("Decision", (), {"action_code": "avoid", "candidate_score": 0.0})()
    result = type("Result", (), {"score": score, "decision": decision})()

    assert scan_candidate_score(result) == 0.0


def test_scan_candidate_score_uses_setup_for_non_avoid_action():
    score = type("Score", (), {"total": 52, "relative_strength": 2})()
    decision = type("Decision", (), {"action_code": "wait", "candidate_score": 72.0})()
    result = type("Result", (), {"score": score, "decision": decision})()

    assert scan_candidate_score(result) == 72.0


def test_localize_scan_table_translates_decision_columns():
    frame = pd.DataFrame(
        [
            {
                "candidate_score": 72,
                "final_action": "wait",
                "primary_basis": "buy_signal_needs_quality",
                "decision_conflict": "",
                "market_state": "weak",
                "reject_reason": "weak_market;poor_entry_quality",
            }
        ]
    )

    localized = localize_scan_table(frame)

    assert "\u5019\u9009\u5206" in localized.columns
    assert "candidate_score" not in localized.columns
    assert localized.loc[0, "\u6700\u7ec8\u52a8\u4f5c"] == "\u7b49\u5f85\u4e70\u70b9"
    assert localized.loc[0, "\u4e3b\u8981\u4f9d\u636e"] == "\u6709\u4e70\u70b9\u4f46\u8d28\u91cf\u4e0d\u8db3"
    assert localized.loc[0, "\u5e02\u573a\u72b6\u6001"] == "\u5f31\u5e02"
    assert localized.loc[0, "\u96f7\u8fbe\u6392\u9664\u539f\u56e0"] == "\u5927\u76d8\u504f\u5f31\u3001\u4e70\u70b9\u8d28\u91cf\u4e0d\u8db3"


def test_scan_display_filters_by_current_min_score():
    frame = pd.DataFrame(
        [
            {"\u4ee3\u7801": "000001", "\u5019\u9009\u5206": 59, "\u91cf\u4ef7\u5206": 80},
            {"\u4ee3\u7801": "000002", "\u5019\u9009\u5206": 60, "\u91cf\u4ef7\u5206": 61},
            {"\u4ee3\u7801": "000003", "\u9519\u8bef": "fetch failed"},
        ]
    )

    filtered = filter_scan_display_by_min_score(frame, 60)

    assert filtered["\u4ee3\u7801"].tolist() == ["000002"]


def test_compact_scan_display_keeps_readable_columns():
    frame = pd.DataFrame(
        [
            {
                "\u4ee3\u7801": "000002",
                "\u5019\u9009\u5206": 82.4,
                "\u6700\u7ec8\u52a8\u4f5c": "\u53ef\u4e70\u5165",
                "\u91cf\u4ef7\u5206": 76.2,
                "\u7edf\u4e00\u8bf4\u660e": "\u8fd9\u4e00\u5217\u4e0d\u5e94\u8be5\u5728\u7d27\u51d1\u8868\u683c\u91cc\u5360\u5f88\u591a\u5bbd\u5ea6",
            }
        ]
    )

    compact = compact_scan_display(frame)

    assert "\u7edf\u4e00\u8bf4\u660e" not in compact.columns
    assert compact.loc[0, "\u5019\u9009\u5206"] == 82


def test_scan_save_buttons_follow_visible_display_symbols():
    visible = pd.DataFrame({"\u4ee3\u7801": ["000002"]})
    results = [
        type("Result", (), {"symbol": "000001"})(),
        type("Result", (), {"symbol": "000002"})(),
    ]

    filtered = filter_scan_results_by_display(results, visible)

    assert [result.symbol for result in filtered] == ["000002"]


def test_scan_save_sort_uses_candidate_score_and_badges_show_all_scores():
    scores = type(
        "Scores",
        (),
        {
            "overall_score": 91.0,
            "liangjia_score": 80.0,
            "short_term_score": 75.0,
            "long_term_score": 88.0,
            "components": {"execution_score": 70.0, "execution_window": {"score": 70.0, "flags": []}},
        },
    )()
    result = type("Result", (), {"score": type("Score", (), {"total": 55})(), "factor_scores": scores})()

    assert scan_save_sort_score(result) == scan_candidate_score(result)
    assert scan_save_sort_score(result) < 91.0
    html = factor_score_badges(result)
    assert "\u5019\u9009" in html
    assert "\u7efc\u5408" in html
    assert "\u91cf\u4ef7" in html
    assert "\u77ed\u671f" in html
    assert "\u957f\u671f" in html
    assert "title=" in html
    assert "\u5019\u9009\u5206" in score_badge_tooltip(result, "\u5019\u9009")
    assert "\u7efc\u5408\u5206" in score_badge_tooltip(result, "\u7efc\u5408")
    assert "\u77ed\u671f\u5206" in score_badge_tooltip(result, "\u77ed\u671f")
    assert "\u91cf\u4ef7\u5206" in metric_score_tooltip(result, "\u91cf\u4ef7\u5206")
    assert "title=" in metric_card("\u7efc\u5408\u5206", "91", tooltip="\u6253\u5206\u903b\u8f91")


def test_candidate_score_is_not_pinned_by_single_high_score():
    scores = type(
        "Scores",
        (),
        {
            "overall_score": 96.0,
            "liangjia_score": 92.0,
            "short_term_score": 95.0,
            "long_term_score": 70.0,
            "components": {
                "execution_score": 52.0,
                "execution_window": {"score": 52.0, "flags": ["价格远离MA20，追高风险较大"]},
            },
        },
    )()
    short_view = type("ShortView", (), {"liangjia_score": 92.0})()
    long_view = type("LongView", (), {"score": 70.0})()
    result = type("Result", (), {"score": type("Score", (), {"total": 88})(), "factor_scores": scores, "short_term_view": short_view, "long_term_view": long_view})()

    assert scan_candidate_score(result) < 90
    assert scan_candidate_score(result) >= 52


def test_scan_table_scores_match_factor_badges_source():
    scores = type(
        "Scores",
        (),
        {
            "overall_score": 91.0,
            "liangjia_score": 73.0,
            "short_term_score": 68.0,
            "long_term_score": 82.0,
            "components": {"execution_score": 64.0, "execution_window": {"score": 64.0, "flags": ["等待确认"]}},
        },
    )()
    result = type(
        "Result",
        (),
        {
            "symbol": "000001",
            "as_of": pd.Timestamp("2026-05-23"),
            "close": 12.34,
            "structure": type("Structure", (), {"stage": Stage.MARKUP})(),
            "score": type("Score", (), {"total": 55})(),
            "signals": [],
            "factor_scores": scores,
            "long_term_view": None,
            "short_term_view": None,
            "classification": None,
            "radar": None,
        },
    )()

    row = result_to_scan_row(result)

    assert row["候选分"] == scan_candidate_score(result)
    assert row["综合分"] == scores.overall_score
    assert row["量价分"] == scores.liangjia_score
    assert row["短期分"] == scores.short_term_score
    assert row["长期分"] == scores.long_term_score
    assert row["执行窗口分"] == 64.0


def test_watchlist_sort_score_uses_persisted_candidate_score():
    record = {
        "candidate_score": 67.0,
        "factor_scores": {
            "overall_score": 95.0,
            "liangjia_score": 95.0,
            "short_term_score": 95.0,
            "long_term_score": 95.0,
            "execution_window_score": 95.0,
        },
        "score": {"total": 95},
    }

    assert app.watchlist_sort_score(record) == 67.0


def test_watchlist_sort_score_recomputes_scan_formula_for_legacy_record():
    record = {
        "factor_scores": {
            "overall_score": 91.0,
            "liangjia_score": 73.0,
            "short_term_score": 68.0,
            "long_term_score": 82.0,
            "execution_window_score": 64.0,
        },
        "short_term_view": {"liangjia_score": 73.0},
        "long_term_view": {"score": 82.0},
        "score": {"total": 55},
    }
    scores = type(
        "Scores",
        (),
        {
            "overall_score": 91.0,
            "liangjia_score": 73.0,
            "short_term_score": 68.0,
            "long_term_score": 82.0,
            "components": {"execution_window": {"score": 64.0, "flags": []}},
        },
    )()
    short_view = type("ShortView", (), {"liangjia_score": 73.0})()
    result = type("Result", (), {"factor_scores": scores, "short_term_view": short_view})()

    assert app.watchlist_sort_score(record) == scan_candidate_score(result)


def test_score_tooltip_handles_numeric_component_values():
    scores = type(
        "Scores",
        (),
        {
            "overall_score": 80.0,
            "liangjia_score": 76.0,
            "short_term_score": 72.0,
            "long_term_score": 70.0,
            "components": {
                "risk_penalty": 6.0,
                "liangjia": {"base_score": 74.0, "market_modifier": 1.0, "exit_risk_modifier": 0.0},
                "short_term": {"return_score": 70.0, "ma_score": 75.0, "volume_score": 68.0},
            },
        },
    )()
    result = type("Result", (), {"score": type("Score", (), {"total": 55})(), "factor_scores": scores})()

    html = factor_score_badges(result)

    assert "title=" in html
    assert "\u7efc\u5408\u5206" in score_badge_tooltip(result, "\u7efc\u5408")


def test_execution_window_tooltip_explains_score_bands():
    scores = type(
        "Scores",
        (),
        {
            "components": {
                "execution_score": 68,
                "execution_window": {"score": 68, "flags": ["价格贴近MA20，未明显追高"]},
            }
        },
    )()

    score, flags = execution_window_from_scores(scores)
    tooltip = execution_window_tooltip(score, flags)

    assert score == 68
    assert execution_window_label(score) == "观察执行"
    assert "75分以上" in tooltip
    assert "55分以下" in tooltip
    assert "价格贴近MA20" in tooltip


def test_price_text_keeps_dataframe_column_string_like():
    assert price_text(None) == ""
    assert price_text("") == ""
    assert price_text(12) == "12.00"
    assert price_text("12.345") == "12.35"


def test_watchlist_summary_price_column_is_arrow_friendly_text():
    empty_row = app.watchlist_summary_row({"symbol": "000001", "manual_price": None})
    priced_row = app.watchlist_summary_row({"symbol": "000002", "manual_price": 12.34})

    assert empty_row["实时/对照价"] == ""
    assert priced_row["实时/对照价"] == "12.34"
    assert isinstance(empty_row["实时/对照价"], str)
    assert isinstance(priced_row["实时/对照价"], str)


def test_purchased_position_uses_saved_risk_levels():
    record = {
        "buy_price": 10.0,
        "current_price": 11.0,
        "snapshot": {
            "signals": [
                {
                    "操作失效价": 9.50,
                    "止损": 9.80,
                }
            ]
        },
    }

    assert round(purchased_pnl_pct(record), 4) == 0.1
    assert purchased_risk_status(record) == "holding"

    record["current_price"] = 9.7
    assert purchased_risk_status(record) == "stop_risk"

    record["current_price"] = 9.4
    assert purchased_risk_status(record) == "invalid"


def test_sort_purchased_records_by_score_then_time():
    records = [
        {"symbol": "000001", "purchased_at": "2026-01-01 10:00:00", "snapshot": {"score": {"total": 70}}},
        {"symbol": "000002", "purchased_at": "2026-01-02 10:00:00", "snapshot": {"score": {"total": 70}}},
        {"symbol": "000003", "purchased_at": "2026-01-01 10:00:00", "snapshot": {"score": {"total": 82}}},
    ]

    assert [record["symbol"] for record in sort_purchased_records(records)] == ["000003", "000002", "000001"]


def test_watchlist_refresh_preserves_manual_tracking_fields(monkeypatch):
    old_record = {
        "symbol": "000001",
        "source": "单票诊断",
        "saved_at": "2026-01-01 09:30:00",
        "operation_status": "valid",
        "status_updated_at": "2026-01-02 10:00:00",
        "manual_price": 12.34,
        "operator_note": "重点跟踪",
        "execution_status": "actionable",
        "refresh_error": "old error",
    }

    def fake_record(result, source):
        return {
            "symbol": result.symbol,
            "source": source,
            "saved_at": "new save time",
            "operation_status": "unmarked",
            "manual_price": None,
            "operator_note": "",
            "score": {"total": 88},
        }

    monkeypatch.setattr(app, "analysis_result_to_watchlist_record", fake_record)
    result = type("Result", (), {"symbol": "000001"})()

    refreshed = app.merge_watchlist_refresh_record(old_record, result, "2026-05-21 10:00:00")

    assert refreshed["score"]["total"] == 88
    assert refreshed["saved_at"] == old_record["saved_at"]
    assert refreshed["operation_status"] == old_record["operation_status"]
    assert refreshed["manual_price"] == old_record["manual_price"]
    assert refreshed["operator_note"] == old_record["operator_note"]
    assert refreshed["last_refreshed_at"] == "2026-05-21 10:00:00"
    assert refreshed["refresh_error"] == ""


def test_watchlist_trade_plan_classifies_actionable_buy_candidate():
    record = {
        "symbol": "000001",
        "manual_price": 11.5,
        "score": {"total": 76},
        "short_term_view": {"advice": "短期买入", "liangjia_score": 78},
        "structure": {"stage": "第二阶段-主升"},
        "signals": [{"买入区间": "11.00 - 12.00", "止损": 10.6, "操作失效价": 10.5}],
        "radar": {"买点质量分": 72, "卖出风险分": 20},
    }

    assert watchlist_trade_tier(record) == "A-明日可执行"
    assert "VWAP" in watchlist_intraday_rule(record)
    row = watchlist_plan_row(record)
    assert row["交易分层"] == "A-明日可执行"
    assert row["盘前动作"] == "开盘前保留重点盯盘，不集合竞价追入"
    assert "交易分层：A-明日可执行" in watchlist_detail_plan_text(record)


def test_watchlist_trade_plan_blocks_high_risk_candidate():
    record = {
        "symbol": "000002",
        "manual_price": 9.5,
        "short_term_view": {"advice": "短期买入"},
        "structure": {"stage": "第二阶段-主升"},
        "signals": [{"买入区间": "11.00 - 12.00", "止损": 10.6, "操作失效价": 10.5}],
        "radar": {"买点质量分": 80, "卖出风险分": 20},
    }

    assert watchlist_trade_tier(record) == "C-先剔除"
    assert "分时不能单独改成买入" in watchlist_intraday_rule(record)


def test_parse_watchlist_price_updates_accepts_common_formats():
    updates, errors = parse_watchlist_price_updates("000001 12.34\n600519,1688.80\n300750\t201.5\nbad-line\n000002 -1")

    assert updates == {"000001": 12.34, "600519": 1688.8, "300750": 201.5}
    assert errors == ["bad-line", "000002 -1"]


def test_update_watchlist_manual_prices_preserves_unmatched_records(monkeypatch):
    records = [
        {"symbol": "000001", "manual_price": None},
        {"symbol": "000002", "manual_price": 8.8},
    ]
    saved = {}
    monkeypatch.setattr(app, "load_watchlist_records", lambda: records)
    monkeypatch.setattr(app, "save_watchlist_records", lambda value: saved.setdefault("records", value))

    updated = update_watchlist_manual_prices({"000001": 12.34, "999999": 1.0})

    assert updated == 1
    assert saved["records"][0]["manual_price"] == 12.34
    assert saved["records"][0]["status_updated_at"]
    assert saved["records"][1]["manual_price"] == 8.8
