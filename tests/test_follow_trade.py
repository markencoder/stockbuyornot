import pandas as pd

from stockbuyornot.follow_trade import build_follow_candidates, foreign_watchlist, moves_from_frame, parse_foreign_moves, selected_moves_from_pool


def test_parse_foreign_moves_from_text():
    moves = parse_foreign_moves("NVDA 3.5\n000660.KS,2.8\nTSLA 1.2")

    assert [move.symbol for move in moves] == ["NVDA", "000660.KS", "TSLA"]
    assert moves[0].name == "英伟达"
    assert moves[1].market == "韩股"


def test_build_follow_candidates_filters_small_moves():
    moves = parse_foreign_moves("NVDA 1.5\nTSLA 2.5")
    result = build_follow_candidates(moves, min_up_pct=2.0)

    assert not result.empty
    assert "宁德时代" in result["A股名称"].tolist()
    assert "工业富联" not in result["A股名称"].tolist()


def test_build_follow_candidates_aggregates_multiple_triggers():
    moves = parse_foreign_moves("NVDA 3.0\nAMD 3.0")
    result = build_follow_candidates(moves, min_up_pct=2.0)
    row = result[result["A股代码"] == "601138"].iloc[0]

    assert "英伟达" in row["触发源"]
    assert "超微半导体" in row["触发源"]
    assert row["跟随分"] > 90


def test_moves_from_frame_accepts_chinese_columns():
    frame = pd.DataFrame([{"市场": "美股", "代码": "AAPL", "名称": "苹果", "昨夜涨跌幅%": "2.3%"}])
    moves = moves_from_frame(frame)

    assert len(moves) == 1
    assert moves[0].symbol == "AAPL"
    assert moves[0].pct_change == 2.3


def test_foreign_watchlist_contains_selectable_mapped_symbols():
    watchlist = foreign_watchlist(["美股"])

    assert {"选择", "市场", "代码", "名称", "最近涨幅%", "前一交易日涨跌幅%"}.issubset(watchlist.columns)
    assert "NVDA" in watchlist["代码"].tolist()


def test_selected_moves_from_pool_uses_selected_rows():
    frame = pd.DataFrame(
        [
            {"选择": True, "市场": "美股", "代码": "NVDA", "名称": "英伟达", "前一交易日涨跌幅%": 3.2, "最近涨幅%": 8.1},
            {"选择": False, "市场": "美股", "代码": "AAPL", "名称": "苹果", "前一交易日涨跌幅%": 2.4, "最近涨幅%": 5.0},
        ]
    )

    moves = selected_moves_from_pool(frame)

    assert len(moves) == 1
    assert moves[0].symbol == "NVDA"
    assert moves[0].pct_change == 3.2


def test_amd_mapping_includes_packaging_and_server_chain():
    result = build_follow_candidates(parse_foreign_moves("超微半导体 3.0"), min_up_pct=2.0)

    assert {"通富微电", "工业富联", "沪电股份"}.issubset(set(result["A股名称"]))


def test_micron_mapping_includes_storage_chain():
    result = build_follow_candidates(parse_foreign_moves("美光科技 3.0"), min_up_pct=2.0)

    assert {"江波龙", "佰维存储", "德明利", "香农芯创"}.issubset(set(result["A股名称"]))
