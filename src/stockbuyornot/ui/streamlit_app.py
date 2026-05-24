from __future__ import annotations

import json
import html
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import altair as alt
import pandas as pd
import streamlit as st

from stockbuyornot.analysis import analyze_ohlcv
from stockbuyornot.auth import (
    User,
    authenticate_user,
    create_user,
    get_user_by_email,
    get_user_by_id,
    initialize_auth_db,
    is_admin_user,
    list_users,
    sanitize_user_key,
    subscription_is_active,
    update_subscription_status,
)
from stockbuyornot.backtest import backtest_single
from stockbuyornot.data.providers import AkshareProvider, CsvProvider
from stockbuyornot.decision import make_unified_decision
from stockbuyornot.forecasting import (
    forecast_chart_frame,
    forecast_points_frame,
    forecast_symbol,
)
from stockbuyornot.follow_trade import (
    build_follow_candidates,
    build_foreign_strength_pool,
    moves_from_frame,
    parse_foreign_moves,
    sample_moves_frame,
    selected_moves_from_pool,
)
from stockbuyornot.intraday import IntradayAdjustment, IntradaySummary, compute_intraday_adjustment, summarize_intraday
from stockbuyornot.models import AnalysisResult, SignalSide
from stockbuyornot.payment import payment_config_from_env, payment_reference
from stockbuyornot.portfolio import PortfolioBacktestConfig, portfolio_backtest
from stockbuyornot.radar import market_state_from_benchmark
from stockbuyornot.view_engine import stage_label


WATCHLIST_PATH = Path("data/watchlist.json")
PURCHASED_PATH = Path("data/purchased.json")
MARKET_INDEX_OPTIONS = {
    "000300": "沪深300",
    "000001": "上证指数",
    "399001": "深证成指",
    "399006": "创业板指",
    "000905": "中证500",
    "000016": "上证50",
    "000688": "科创50",
}


st.set_page_config(page_title="A股量价工作台", page_icon="📈", layout="wide")


def main() -> None:
    inject_css()
    user = render_auth_gate()
    if user is None:
        return

    settings = sidebar_settings(user)
    provider = AkshareProvider(
        request_timeout=settings["request_timeout"],
        request_retries=settings["request_retries"],
    )

    st.title("A股量价信号工作台")
    _, market_col = st.columns([0.58, 0.42], gap="large")
    with market_col:
        render_market_status_widget(settings)

    tab_analyze, tab_scan, tab_watchlist, tab_purchased, tab_backtest, tab_follow, tab_data, tab_quick_usage = st.tabs(
        ["单票诊断", "股票池扫描", "我的备选池", "已购买", "策略回测", "跟随交易", "数据工具", "简明用法"]
    )
    with tab_analyze:
        render_analyze_tab(settings, provider)
    with tab_scan:
        if billing_allows_feature(user, "股票池扫描"):
            render_scan_tab(settings, provider)
    with tab_watchlist:
        render_watchlist_tab(settings, provider)
    with tab_purchased:
        render_purchased_tab()
    with tab_backtest:
        if billing_allows_feature(user, "策略回测"):
            render_backtest_tab(settings, provider)
    with tab_follow:
        if billing_allows_feature(user, "跟随交易"):
            render_follow_trade_tab(settings, provider)
    with tab_data:
        render_data_tab(settings, provider)
    with tab_quick_usage:
        render_quick_usage_tab()


def render_auth_gate() -> User | None:
    initialize_auth_db()
    user_id = st.session_state.get("auth_user_id")
    if user_id:
        user = get_user_by_id(int(user_id))
        if user is not None:
            return user
        st.session_state.pop("auth_user_id", None)

    _, auth_col, _ = st.columns([1, 0.72, 1])
    with auth_col:
        st.markdown(
            """
            <div class="auth-heading">
                <h1>A股量价信号工作台</h1>
                <p>登录后进入你的个人工作台</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            login_tab, register_tab = st.tabs(["登录", "注册"])

            with login_tab:
                with st.form("login_form"):
                    email = st.text_input("邮箱", key="login_email")
                    password = st.text_input("密码", type="password", key="login_password")
                    submitted = st.form_submit_button("登录", type="primary", use_container_width=True)
                if submitted:
                    user = authenticate_user(email, password)
                    if user is None:
                        st.error("邮箱或密码不正确。")
                    else:
                        st.session_state["auth_user_id"] = user.id
                        st.rerun()

            with register_tab:
                with st.form("register_form"):
                    display_name = st.text_input("昵称", key="register_display_name")
                    email = st.text_input("邮箱", key="register_email")
                    password = st.text_input("密码", type="password", key="register_password")
                    submitted = st.form_submit_button("创建账号", type="primary", use_container_width=True)
                if submitted:
                    try:
                        user = create_user(email, password, display_name)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["auth_user_id"] = user.id
                        st.success("账号已创建。")
                        st.rerun()

    return None


def render_account_panel(user: User) -> None:
    config = payment_config_from_env()
    status_label = {
        "free": "免费版",
        "trial": "试用中",
        "active": "会员有效",
        "admin": "管理员",
        "expired": "已过期",
    }.get(user.subscription_status, user.subscription_status)
    if is_admin_user(user):
        status_label = "管理员"
    st.markdown(f"**{user.display_name}**")
    st.caption(f"{user.email} | {status_label}")
    if user.subscription_expires_at:
        st.caption(f"到期时间：{user.subscription_expires_at}")
    if st.button("退出登录", use_container_width=True):
        st.session_state.pop("auth_user_id", None)
        st.rerun()

    with st.expander("会员/付费入口", expanded=not subscription_is_active(user)):
        st.write(f"套餐金额：{config.amount_cny} 元")
        st.code(payment_reference(user.id, user.email), language=None)
        if config.qr_image_url:
            st.image(config.qr_image_url, caption=f"{config.provider} 收款码")
        else:
            st.info("收款码暂未配置。上线收费时设置 STOCKBUYORNOT_PAYMENT_QR_URL 即可显示二维码。")
        if config.support_contact:
            st.caption(f"付款后联系：{config.support_contact}")
        st.caption("正式接入微信/支付宝后，支付回调只需要把用户 subscription_status 更新为 active。")


def render_admin_panel() -> None:
    status_options = ["free", "trial", "active", "expired", "admin"]
    with st.expander("管理员后台", expanded=False):
        st.caption("按邮箱开通、取消或调整会员状态。")
        with st.form("admin_subscription_form"):
            email = st.text_input("用户邮箱", key="admin_user_email")
            status = st.selectbox("会员状态", status_options, index=2, key="admin_status")
            expires_at = st.text_input("到期时间", value="2026-12-31", help="可留空；建议格式 YYYY-MM-DD")
            submitted = st.form_submit_button("保存会员状态", type="primary", use_container_width=True)
        if submitted:
            target = get_user_by_email(email)
            if target is None:
                st.error("没有找到这个用户，请确认邮箱已经注册。")
            else:
                update_subscription_status(target.id, status, expires_at.strip() or None)
                st.success(f"已更新 {target.email} 为 {status}。")
                if target.id == st.session_state.get("auth_user_id"):
                    st.rerun()

        users = list_users(limit=30)
        if users:
            frame = pd.DataFrame(
                [
                    {
                        "邮箱": item.email,
                        "昵称": item.display_name,
                        "状态": "admin" if is_admin_user(item) else item.subscription_status,
                        "到期时间": item.subscription_expires_at or "",
                    }
                    for item in users
                ]
            )
            st.dataframe(frame, use_container_width=True, hide_index=True)
        else:
            st.info("暂无注册用户。")


def current_user_data_path(filename: str) -> Path:
    user_id = st.session_state.get("auth_user_id")
    if not user_id:
        return Path("data") / filename
    user = get_user_by_id(int(user_id))
    if user is None:
        return Path("data") / filename
    return Path("data/users") / sanitize_user_key(user.email) / filename


def billing_allows_feature(user: User, feature_name: str) -> bool:
    config = payment_config_from_env()
    if not config.billing_enforced or subscription_is_active(user):
        return True
    st.warning(f"{feature_name} 是会员功能。请在侧边栏完成付费后继续使用。")
    return False


def sidebar_settings(user: User | None = None) -> dict:
    with st.sidebar:
        if user is not None:
            render_account_panel(user)
            if is_admin_user(user):
                render_admin_panel()
            st.divider()
        st.header("全局参数")
        start = st.text_input("开始日期", value="20240101", help="格式：YYYYMMDD")
        end = st.text_input("结束日期", value=pd.Timestamp.today().strftime("%Y%m%d"), help="格式：YYYYMMDD")
        adjust_label = st.selectbox("复权方式", ["前复权", "后复权", "不复权"], index=0)
        adjust = {"前复权": "qfq", "后复权": "hfq", "不复权": ""}[adjust_label]

        st.divider()
        st.header("扫描与网络")
        min_score = st.slider("最低入选分", 0, 100, 60, 5)
        request_timeout = st.slider("单次请求超时/秒", 3.0, 20.0, 8.0, 1.0)
        request_retries = st.slider("数据源重试次数", 1, 3, 1, 1)

        st.divider()
        st.caption("提示：当前工作台仅做研究与复盘辅助，不构成投资建议。")

    return {
        "start": start,
        "end": end,
        "adjust": adjust,
        "min_score": min_score,
        "request_timeout": request_timeout,
        "request_retries": request_retries,
    }


def render_analyze_tab(settings: dict, provider: AkshareProvider) -> None:
    st.subheader("单票诊断")
    with st.container(border=True):
        data_source = st.radio("数据源", ["在线A股", "上传CSV", "本地CSV路径"], horizontal=True, key="single_source")
        input_cols = st.columns([0.8, 1.4, 0.8], gap="medium")
        symbol = input_cols[0].text_input("股票代码", value="000001", key="single_symbol")
        uploaded = None
        csv_path = ""
        if data_source == "上传CSV":
            uploaded = input_cols[1].file_uploader("上传日线CSV", type=["csv"], key="single_upload")
        elif data_source == "本地CSV路径":
            csv_path = input_cols[1].text_input("CSV路径", value="", key="single_csv_path")
        else:
            input_cols[1].text_input("数据源", value="腾讯/东方财富/AKShare 在线接口", disabled=True, key="single_online_source_label")
        run = input_cols[2].button("开始诊断", type="primary", use_container_width=True)

    if run:
        try:
            df = load_single_data(data_source, symbol, settings, provider, uploaded, csv_path)
            benchmark = load_benchmark_for_ui(provider, settings)
            result = analyze_for_ui(df, symbol=symbol or infer_symbol_from_data(df, "CSV"), benchmark=benchmark)
        except Exception as exc:
            st.error(short_error(exc))
            return
        st.session_state["single_analysis_df"] = df
        st.session_state["single_analysis_result"] = result
    elif "single_analysis_result" in st.session_state and "single_analysis_df" in st.session_state:
        df = st.session_state["single_analysis_df"]
        result = st.session_state["single_analysis_result"]
    else:
        return

    render_result_summary(result)
    render_vwap_explanation_note()
    if data_source == "在线A股":
        render_intraday_confirm_panel(result, provider)
    save_cols = st.columns([0.72, 0.28], gap="medium")
    save_cols[0].caption("可以把当前诊断快照保存到我的备选池，后续继续跟踪评分、量价信号和关键价位。")
    if save_cols[1].button("加入我的备选池", type="primary", use_container_width=True, key=f"save_single_{result.symbol}"):
        save_watchlist_result(result, source="单票诊断")
        st.success(f"{result.symbol} 已保存到我的备选池。")
    render_price_volume(df)
    render_result_detail(result)


def render_vwap_explanation_note() -> None:
    with st.expander("VWAP说明", expanded=False):
        st.markdown(
            """
            VWAP 是盘中成交量加权平均价，可以理解为当日市场的平均成交成本。

            - 价格在 VWAP 上方并能回踩不破，通常说明盘中承接较强，可作为买点确认条件之一。
            - 价格跌破 VWAP 后反抽无力，通常说明盘中资金成本形成压制，应降低追买意愿。
            - VWAP 只用于盘中执行辅助，不能单独替代日K结构、量价信号、买入区间和止损规则。
            - 高开过大时不建议直接追，优先等待回落不破 VWAP、开盘价或关键支撑后再判断。
            """
        )


def render_scan_tab(settings: dict, provider: AkshareProvider) -> None:
    st.subheader("股票池扫描")
    controls, output = st.columns([0.9, 1.1], gap="large")

    with controls:
        source = st.radio(
            "股票池来源",
            ["手动代码", "内置股票池", "行业板块", "概念板块", "本地CSV文件夹"],
            key="scan_source",
        )
        limit = st.number_input("最多扫描数量", min_value=0, value=20, step=10, help="0 表示不限制。")
        include_errors = st.checkbox("显示失败项", value=False)

        symbols_text = ""
        pool = "沪深300"
        board_name = ""
        csv_dir = ""
        if source == "手动代码":
            symbols_text = st.text_area("股票代码", value="000001 600519 300750", help="可用空格、逗号、换行分隔。")
        elif source == "内置股票池":
            pool = st.selectbox("股票池", ["全部A股", "上证50", "沪深300", "中证500", "上交所", "深交所", "创业板", "科创板"])
        elif source == "行业板块":
            board_name = st.text_input("行业名称", value="银行")
        elif source == "概念板块":
            board_name = st.text_input("概念名称", value="人工智能")
        else:
            csv_dir = st.text_input("CSV文件夹路径", value="data/daily")

        run_scan = st.button("开始扫描", type="primary", use_container_width=True)

    if not run_scan:
        with output:
            cached_table = st.session_state.get("scan_table")
            cached_results = st.session_state.get("scan_results", [])
            if cached_table is None:
                st.info("配置股票池后点击开始扫描。扫描结果会按评分从高到低展示。")
            else:
                render_scan_output(cached_table, cached_results, settings["min_score"])
        return

    try:
        sources = resolve_scan_sources(source, symbols_text, pool, board_name, csv_dir, provider)
    except Exception as exc:
        st.error(short_error(exc))
        return

    if limit:
        sources = sources[: int(limit)]

    benchmark = load_benchmark_for_ui(provider, settings)
    sector_benchmark = load_board_for_ui(provider, board_name, settings)
    sector_rs_rank = sector_rank_hint(sector_benchmark, benchmark)
    rows: list[dict] = []
    scan_results: list[AnalysisResult] = []
    progress = st.progress(0)
    status = st.empty()

    for index, (symbol, source_ref) in enumerate(sources, start=1):
        status.write(f"正在扫描 {index}/{len(sources)}：{symbol}")
        try:
            df = load_scan_data(symbol, source_ref, settings, provider)
            result = analyze_for_ui(
                df,
                symbol=symbol,
                benchmark=benchmark,
                sector=sector_benchmark,
                sector_name=board_name,
                sector_rs_rank=sector_rs_rank,
            )
            if scan_candidate_score(result) >= settings["min_score"] and scan_action_code(result) in {"buy", "add", "wait", "watch"}:
                row = result_to_scan_row(result)
                row["候选分"] = scan_candidate_score(result)
                rows.append(row)
                scan_results.append(result)
        except Exception as exc:
            if include_errors:
                rows.append({"代码": symbol, "错误": short_error(exc)})
        progress.progress(index / max(len(sources), 1))

    progress.empty()
    status.empty()
    table = pd.DataFrame(rows)
    st.session_state["scan_table"] = table
    st.session_state["scan_results"] = scan_results
    with output:
        render_scan_output(table, scan_results, settings["min_score"])


def render_scan_output(table: pd.DataFrame, scan_results: list[AnalysisResult], min_score: int = 0) -> None:
    if table.empty:
        st.warning("没有命中候选。可以降低最低入选分，或换一个股票池。")
        return
    display = localize_scan_table(table.copy())
    display = filter_scan_display_by_min_score(display, min_score)
    if display.empty:
        st.warning("没有命中候选。可以降低最低入选分，或换一个股票池。")
        return
    if "候选分" in display.columns:
        display = display.sort_values("候选分", ascending=False)
    elif "candidate_score" in display.columns:
        display = display.sort_values("candidate_score", ascending=False)
    compact = compact_scan_display(display)
    st.caption(f"当前只显示候选分不低于 {min_score} 分的股票；下载 CSV 会保留完整诊断字段。")
    st.dataframe(compact, use_container_width=True, hide_index=True, column_config=scan_column_config(compact))
    st.download_button(
        "下载扫描结果CSV",
        data=display.to_csv(index=False, encoding="utf-8-sig"),
        file_name="candidates.csv",
        mime="text/csv",
        use_container_width=True,
    )
    render_scan_save_buttons(filter_scan_results_by_display(scan_results, display))


def render_backtest_tab(settings: dict, provider: AkshareProvider) -> None:
    st.subheader("策略回测")

    with st.container(border=True):
        row1 = st.columns([0.9, 0.9, 1.2, 0.8], gap="medium")
        source = row1[0].radio("数据源", ["在线A股", "上传CSV", "本地CSV路径"], horizontal=True, key="backtest_source")
        symbol = row1[1].text_input("股票代码", value="000001", key="backtest_symbol")
        uploaded = None
        csv_path = ""
        if source == "上传CSV":
            uploaded = row1[2].file_uploader("上传日线CSV", type=["csv"], key="backtest_upload")
        elif source == "本地CSV路径":
            csv_path = row1[2].text_input("CSV路径", value="", key="backtest_csv_path")
        else:
            row1[2].text_input("数据源", value="腾讯/东方财富/AKShare 在线接口", disabled=True, key="backtest_online_source_label")
        run = row1[3].button("运行回测", type="primary", use_container_width=True)

        st.markdown("**交易与风控参数**")
        row2 = st.columns(6, gap="medium")
        min_score = row2[0].slider("最低买入分", 40, 95, 75, 5)
        initial_cash = row2[1].number_input("初始资金", min_value=10000.0, value=100000.0, step=10000.0)
        risk_per_trade = row2[2].slider("单笔风险", 0.002, 0.05, 0.01, 0.002, format="%.3f")
        fee_rate = row2[3].number_input("佣金率", min_value=0.0, value=0.0003, step=0.0001, format="%.4f")
        stamp_tax = row2[4].number_input("印花税", min_value=0.0, value=0.0005, step=0.0001, format="%.4f")
        slippage = row2[5].number_input("滑点", min_value=0.0, value=0.0010, step=0.0005, format="%.4f")

    if not run:
        st.info("设置参数后运行回测。回测使用信号日之后的价格，结果用于验证信号，不代表未来收益。")
        return

    try:
        df = load_single_data(source, symbol, settings, provider, uploaded, csv_path)
        result = backtest_single(
            df,
            symbol=symbol or infer_symbol_from_data(df, "CSV"),
            min_score=min_score,
            initial_cash=initial_cash,
            risk_per_trade=risk_per_trade,
            fee_rate=fee_rate,
            stamp_tax=stamp_tax,
            slippage=slippage,
        )
    except Exception as exc:
        st.error(short_error(exc))
        return

    render_backtest_metrics(result.metrics)
    equity = equity_curve_from_trades(result.trades, initial_cash)
    if not equity.empty:
        render_equity_chart(equity)

    pairs = trade_pairs(result.trades)
    table_tabs = st.tabs(["交易流水", "配对盈亏", "回测说明"])
    with table_tabs[0]:
        if result.trades.empty:
            st.info("回测区间内没有触发交易。")
        else:
            trades = result.trades.copy()
            trades["date"] = pd.to_datetime(trades["date"]).dt.date
            st.dataframe(trades, use_container_width=True, hide_index=True)
            st.download_button(
                "下载交易流水CSV",
                data=trades.to_csv(index=False, encoding="utf-8-sig"),
                file_name="backtest_trades.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with table_tabs[1]:
        if pairs.empty:
            st.info("没有可配对的完整买卖交易。")
        else:
            st.dataframe(pairs, use_container_width=True, hide_index=True)
            st.download_button(
                "下载配对盈亏CSV",
                data=pairs.to_csv(index=False, encoding="utf-8-sig"),
                file_name="backtest_pairs.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with table_tabs[2]:
        st.markdown(
            """
            - 买入条件：出现买入信号，且评分不低于最低买入分。
            - 成交价格：信号日之后的下一个交易日开盘价，并计入滑点。
            - 仓位计算：按单笔风险比例和止损距离倒推买入股数。
            - 卖出条件：出现卖出信号、触发止损，或数据结束时平仓。
            - 成本模型：买入计佣金，卖出计佣金和印花税。
            """
        )


def render_follow_trade_tab(settings: dict, provider: AkshareProvider) -> None:
    st.subheader("跟随交易")
    st.caption(
        "根据美股、韩股前一晚的大涨信号，映射到A股上游企业。这个页面用于开盘前生成观察清单，"
        "不自动下单，也不替代开盘竞价和量价确认。"
    )

    controls, output = st.columns([1.0, 1.25], gap="large")
    with controls:
        input_mode = st.radio("海外触发源选择方式", ["自动选强势股", "表格输入", "文本粘贴"], horizontal=True)
        min_up_pct = st.slider("海外标的最小涨幅", 0.5, 10.0, 2.0, 0.5, format="%.1f%%")
        top_n = st.slider("最多输出A股候选", 5, 80, 30, 5)
        enrich_analysis = st.checkbox("叠加A股量价诊断", value=True)
        min_a_score = st.slider("A股量价最低分", 0, 100, 50, 5, disabled=not enrich_analysis)
        analyze_limit = st.slider("最多诊断候选数", 5, 50, 20, 5, disabled=not enrich_analysis)

        if input_mode == "自动选强势股":
            market_scope = st.multiselect("海外市场", ["美股", "韩股"], default=["美股", "韩股"])
            lookback_days = st.slider("近期强势观察天数", 2, 20, 5, 1)
            min_recent_pct = st.slider("近期最小涨幅", -10.0, 30.0, 0.0, 0.5, format="%.1f%%")
            pct_basis = st.radio("生成跟随信号使用", ["前一交易日涨跌幅%", "最近涨幅%"], index=1, horizontal=True)
            if st.button("刷新海外强势股", use_container_width=True):
                st.session_state["follow_strength_pool"] = build_foreign_strength_pool(
                    markets=market_scope,
                    lookback_days=lookback_days,
                    min_recent_pct=min_recent_pct,
                )
            pool = st.session_state.get("follow_strength_pool")
            if pool is None:
                pool = build_foreign_strength_pool(markets=market_scope, lookback_days=lookback_days, min_recent_pct=min_recent_pct)
                st.session_state["follow_strength_pool"] = pool
            edited = st.data_editor(
                pool,
                num_rows="fixed",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "选择": st.column_config.CheckboxColumn("选择"),
                    "最近涨幅%": st.column_config.NumberColumn("最近涨幅%", step=0.1, format="%.2f"),
                    "前一交易日涨跌幅%": st.column_config.NumberColumn("前一交易日涨跌幅%", step=0.1, format="%.2f"),
                },
                disabled=["市场", "代码", "名称", "产业主题", "映射A股数", "数据状态"],
                key="follow_strength_editor",
            )
            moves = selected_moves_from_pool(edited, pct_column=pct_basis)
        elif input_mode == "表格输入":
            edited = st.data_editor(
                sample_moves_frame(),
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "昨夜涨跌幅%": st.column_config.NumberColumn("昨夜涨跌幅%", step=0.1, format="%.2f"),
                },
            )
            moves = moves_from_frame(edited)
        else:
            text = st.text_area(
                "每行一个海外标的",
                value="NVDA 3.5\n000660.KS 2.8\nTSLA 1.2",
                help="格式示例：NVDA 3.5；000660.KS 2.8；AAPL -1.5。代码和涨跌幅之间可用空格、逗号或Tab。",
                height=150,
            )
            moves = parse_foreign_moves(text)

        run = st.button("生成跟随交易清单", type="primary", use_container_width=True)

    if not run:
        with output:
            st.info("先在左侧选择近期强势海外股，或输入昨夜美股/韩股涨跌幅，再点击生成。")
            st.markdown(
                """
                **使用方式建议**

                - 先刷新近期强势股，勾选真正有产业链映射的海外标的。
                - 美股会自动尝试拉取近期涨幅；韩股当前作为重点观察池，可手动修改涨跌幅。
                - 若按“前一交易日涨跌幅”生成，适合隔夜跟随；若按“最近涨幅”生成，适合主题发酵观察。
                - 再看A股是否位于明确上游环节，例如设备、材料、零部件、光模块、PCB。
                - 开盘若直接高开过大，不追；等待回落不破开盘价或VWAP再考虑。
                - 如果叠加量价诊断，优先看评分更高且没有卖出/回避信号的标的。
                """
            )
        return

    candidates = build_follow_candidates(moves, min_up_pct=min_up_pct)
    if candidates.empty:
        with output:
            st.warning("没有生成候选。可以降低海外涨幅阈值，或确认输入代码是否在内置映射表中。")
        return

    candidates = candidates.head(top_n).copy()
    if enrich_analysis:
        candidates = enrich_follow_candidates_with_analysis(
            candidates,
            settings=settings,
            provider=provider,
            min_a_score=min_a_score,
            analyze_limit=analyze_limit,
        )

    with output:
        render_follow_trade_result(candidates)


def enrich_follow_candidates_with_analysis(
    candidates: pd.DataFrame,
    settings: dict,
    provider: AkshareProvider,
    min_a_score: int,
    analyze_limit: int,
) -> pd.DataFrame:
    rows = []
    progress = st.progress(0.0)
    status = st.empty()
    scan_count = min(len(candidates), analyze_limit)
    for index, (_, row) in enumerate(candidates.iterrows(), start=1):
        enriched = row.to_dict()
        if index <= analyze_limit:
            symbol = str(row["A股代码"]).zfill(6)
            status.write(f"正在诊断A股量价 {index}/{scan_count}：{symbol}")
            try:
                df = provider.daily(symbol, settings["start"], settings["end"], settings["adjust"])
                result = analyze_ohlcv(df, symbol=symbol)
                signals = "、".join(signal.name for signal in result.signals)
                enriched.update(
                    {
                        "量价评分": result.score.total,
                        "当前结构": result.structure.stage.value,
                        "量价信号": signals,
                        "量价建议": result.suggestion,
                        "最终分": round(float(row["跟随分"]) + result.score.total * 0.6, 2),
                    }
                )
            except Exception as exc:
                enriched.update({"量价评分": None, "当前结构": "", "量价信号": "", "量价建议": short_error(exc), "最终分": row["跟随分"]})
            progress.progress(index / max(scan_count, 1))
        else:
            enriched.update({"量价评分": None, "当前结构": "", "量价信号": "", "量价建议": "未诊断", "最终分": row["跟随分"]})
        rows.append(enriched)
    status.empty()
    progress.empty()
    output = pd.DataFrame(rows)
    if "量价评分" in output.columns:
        output = output[(output["量价评分"].isna()) | (output["量价评分"] >= min_a_score)].copy()
    sort_col = "最终分" if "最终分" in output.columns else "跟随分"
    return output.sort_values(sort_col, ascending=False).reset_index(drop=True)


def render_follow_trade_result(candidates: pd.DataFrame) -> None:
    st.markdown("**跟随交易候选**")
    st.dataframe(candidates, use_container_width=True, hide_index=True)
    st.download_button(
        "下载跟随交易清单CSV",
        data=candidates.to_csv(index=False, encoding="utf-8-sig"),
        file_name="follow_trade_candidates.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if not candidates.empty:
        top = candidates.iloc[0]
        st.markdown(
            f"""
            **开盘执行提示**

            当前排序第一：`{top['A股代码']} {top['A股名称']}`，触发源：{top['触发源']}。
            建议开盘前加入观察，不建议无条件集合竞价追入。若高开过大、开盘后放量下杀或量价诊断出现卖出/回避信号，应放弃跟随。
            """
        )


def render_data_tab(settings: dict, provider: AkshareProvider) -> None:
    st.subheader("数据工具")
    col1, col2 = st.columns([0.9, 1.1], gap="large")

    with col1:
        symbol = st.text_input("预览股票代码", value="000001", key="data_symbol")
        data_kind = st.radio("数据类型", ["日K", "分时"], horizontal=True, key="data_kind")
        period = "5"
        lookback_days = 10
        if data_kind == "分时":
            period = st.selectbox("分时周期", ["1", "5", "15", "30", "60"], index=1, format_func=lambda item: f"{item}分钟")
            lookback_days = st.slider("分时回看天数", 1, 30, 10, 1)
        rows = st.slider("显示最近行数", 20, 300, 80, 20)
        if st.button("读取数据", type="primary", use_container_width=True):
            try:
                if data_kind == "分时":
                    df = load_intraday_data(provider, symbol, period, lookback_days)
                    st.session_state["data_preview_kind"] = "分时"
                    st.session_state["data_preview_period"] = period
                else:
                    df = provider.daily(symbol, settings["start"], settings["end"], settings["adjust"])
                    st.session_state["data_preview_kind"] = "日K"
                    st.session_state["data_preview_period"] = ""
            except Exception as exc:
                st.error(short_error(exc))
                return
            st.session_state["data_preview"] = df

    with col2:
        df = st.session_state.get("data_preview")
        preview_kind = st.session_state.get("data_preview_kind", "日K")
        preview_period = st.session_state.get("data_preview_period", "")
        if df is None:
            st.info("这里用于检查在线数据源是否正常。日K用于主策略，分时用于盘中确认。")
        else:
            if preview_kind == "分时":
                summary = summarize_intraday(df, symbol=infer_symbol_from_data(df, symbol))
                render_intraday_summary(summary)
                render_intraday_chart(df, title=f"{summary.symbol} {preview_period}分钟分时")
            else:
                st.markdown(metric_card("数据行数", str(len(df))), unsafe_allow_html=True)
            st.dataframe(df.tail(rows), use_container_width=True, hide_index=True)
            st.download_button(
                "下载当前数据CSV",
                data=df.to_csv(index=False, encoding="utf-8-sig"),
                file_name=f"{infer_symbol_from_data(df, 'data')}_{preview_kind}.csv",
                mime="text/csv",
                use_container_width=True,
            )


def load_intraday_data(provider: AkshareProvider, symbol: str, period: str = "5", lookback_days: int = 10) -> pd.DataFrame:
    if not symbol:
        raise ValueError("请输入股票代码。")
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=int(lookback_days))
    try:
        return provider.intraday(symbol=symbol, period=period, start=f"{start:%Y%m%d}", end=f"{end:%Y%m%d}", adjust="")
    except Exception:
        if int(lookback_days) >= 30:
            raise
        fallback_start = end - pd.Timedelta(days=30)
        return provider.intraday(symbol=symbol, period=period, start=f"{fallback_start:%Y%m%d}", end=f"{end:%Y%m%d}", adjust="")


def render_intraday_confirm_panel(result: AnalysisResult, provider: AkshareProvider) -> None:
    symbol = result.symbol
    with st.expander("盘中分时确认（不改变日K主结论）", expanded=False):
        st.caption("分时数据用于修正盘中执行参考，不改变上方日K主结论；真正的BUY/ADD仍必须来自日K量价买点。")
        cols = st.columns([0.22, 0.22, 0.28, 0.28], gap="medium")
        period = cols[0].selectbox("周期", ["1", "5", "15", "30", "60"], index=1, format_func=lambda item: f"{item}分钟", key=f"intraday_period_{symbol}")
        lookback_days = cols[1].slider("回看天数", 1, 30, 10, 1, key=f"intraday_days_{symbol}")
        run = cols[2].button("读取分时确认", use_container_width=True, key=f"intraday_run_{symbol}")
        cols[3].markdown(metric_card("用途", "盘中确认", "辅助层"), unsafe_allow_html=True)

        state_key = f"intraday_confirm_df_{symbol}"
        if run:
            try:
                st.session_state[state_key] = load_intraday_data(provider, symbol, period, lookback_days)
                st.session_state[f"{state_key}_period"] = period
            except Exception as exc:
                st.error(short_error(exc))
                return

        df = st.session_state.get(state_key)
        if df is None:
            st.info("点击读取后，会显示最新交易日的分时趋势、尾盘量比、相对VWAP和简要结论。")
            return

        summary = summarize_intraday(df, symbol=symbol)
        render_intraday_summary(summary)
        adjustment = build_intraday_adjustment(result, summary)
        render_intraday_adjustment(adjustment, result)
        render_intraday_chart(df, title=f"{symbol} {st.session_state.get(f'{state_key}_period', period)}分钟分时")


def build_intraday_adjustment(result: AnalysisResult, summary: IntradaySummary) -> IntradayAdjustment:
    short_view = getattr(result, "short_term_view", None)
    scores = getattr(result, "factor_scores", None)
    base_liangjia = float(
        getattr(short_view, "liangjia_score", None)
        or getattr(scores, "liangjia_score", None)
        or getattr(result.score, "total", 0)
    )
    base_short = float(
        getattr(short_view, "short_term_score", None)
        or getattr(scores, "short_term_score", None)
        or getattr(result.score, "total", 0)
    )
    daily_advice = str(getattr(short_view, "advice", "") or getattr(result, "suggestion", ""))
    signal_direction = str(getattr(short_view, "signal_direction", ""))
    return compute_intraday_adjustment(
        summary,
        base_liangjia_score=base_liangjia,
        base_short_term_score=base_short,
        daily_advice=daily_advice,
        signal_direction=signal_direction,
    )


def render_intraday_adjustment(adjustment: IntradayAdjustment, result: AnalysisResult | None = None) -> None:
    st.markdown("**盘中修正后的短线/量价参考**")
    cols = st.columns(6, gap="medium")
    execution_score = None
    execution_flags: list[str] = []
    if result is not None and getattr(result, "factor_scores", None) is not None:
        execution_score, execution_flags = execution_window_from_scores(result.factor_scores)
    cols[0].markdown(
        metric_card(
            "日K执行窗口",
            "-" if execution_score is None else f"{execution_score:.0f}",
            "是否适合执行",
            execution_window_tooltip(execution_score, execution_flags),
        ),
        unsafe_allow_html=True,
    )
    cols[1].markdown(metric_card("日K短期分", f"{adjustment.base_short_term_score:.0f}", "原始短期分"), unsafe_allow_html=True)
    cols[2].markdown(
        metric_card("分时修正", f"{adjustment.short_term_modifier:+.0f}", "修正短期交易机会"),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        metric_card("盘中短线参考", f"{adjustment.adjusted_short_term_score:.0f}", "短期分+分时修正"),
        unsafe_allow_html=True,
    )
    cols[4].markdown(
        metric_card("盘中量价参考", f"{adjustment.adjusted_liangjia_score:.0f}", f"量价修正 {adjustment.liangjia_modifier:+.0f}"),
        unsafe_allow_html=True,
    )
    cols[5].markdown(metric_card("执行提示", adjustment.action_level_label if hasattr(adjustment, "action_level_label") else _intraday_level_label(adjustment.action_level)), unsafe_allow_html=True)

    if adjustment.action_level == "support":
        st.success(adjustment.explanation)
    elif adjustment.action_level == "risk":
        st.warning(adjustment.explanation)
    else:
        st.info(adjustment.explanation)
    st.caption(adjustment.formula)
    with st.container(border=True):
        left, right = st.columns(2, gap="large")
        left.markdown("**分时加分依据**")
        left.write("；".join(adjustment.support_flags) if adjustment.support_flags else "暂无明显加分项。")
        right.markdown("**分时扣分/风险依据**")
        right.write("；".join(adjustment.risk_flags) if adjustment.risk_flags else "暂无明显风险项。")
        st.markdown(f"**盘中执行建议：{adjustment.action}**")


def _intraday_level_label(level: str) -> str:
    return {"support": "支持", "watch": "观察", "risk": "防守", "neutral": "等待"}.get(level, "等待")


def render_intraday_summary(summary: IntradaySummary) -> None:
    cols = st.columns(5, gap="medium")
    cols[0].markdown(metric_card("最新价", f"{summary.latest_price:.2f}", summary.latest_time.strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
    cols[1].markdown(metric_card("当日涨跌", f"{summary.session_return_pct:.2%}", summary.trend_label), unsafe_allow_html=True)
    cols[2].markdown(
        metric_card("30分钟动量", "-" if summary.momentum_30m_pct is None else f"{summary.momentum_30m_pct:.2%}"),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        metric_card("日内位置", "-" if summary.range_position_pct is None else f"{summary.range_position_pct:.0%}", "越靠近100%越接近日内高位"),
        unsafe_allow_html=True,
    )
    cols[4].markdown(
        metric_card("尾盘量比", "-" if summary.volume_ratio is None else f"{summary.volume_ratio:.2f}", summary.volume_label),
        unsafe_allow_html=True,
    )
    vwap_text = "-" if summary.vwap is None else f"{summary.vwap:.2f}"
    gap_text = "-" if summary.vwap_gap_pct is None else f"{summary.vwap_gap_pct:.2%}"
    st.info(f"{summary.explanation} VWAP参考价 {vwap_text}，当前相对VWAP {gap_text}。")


def render_intraday_chart(df: pd.DataFrame, title: str = "分时走势") -> None:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    latest_day = data["date"].dt.date.iloc[-1]
    data = data[data["date"].dt.date == latest_day].copy()
    if data.empty:
        return

    data["type"] = "分时收盘价"
    chart_rows = [data[["date", "close", "type"]].rename(columns={"close": "value"})]
    if "amount" in data.columns and data["amount"].notna().any():
        cum_volume = pd.to_numeric(data["volume"], errors="coerce").cumsum()
        cum_amount = pd.to_numeric(data["amount"], errors="coerce").cumsum()
        vwap = cum_amount / cum_volume.replace(0, pd.NA)
        if vwap.dropna().iloc[-1] > float(data["close"].iloc[-1]) * 20:
            vwap = vwap / 100.0
        vwap_frame = pd.DataFrame({"date": data["date"], "value": vwap, "type": "VWAP"})
        chart_rows.append(vwap_frame)
    chart_data = pd.concat(chart_rows, ignore_index=True)
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=False)
        .encode(
            x=alt.X("date:T", title="时间", axis=alt.Axis(format="%H:%M", labelAngle=0)),
            y=alt.Y("value:Q", title="价格", scale=alt.Scale(zero=False)),
            color=alt.Color("type:N", title=""),
            tooltip=[
                alt.Tooltip("date:T", title="时间", format="%H:%M"),
                alt.Tooltip("type:N", title="类型"),
                alt.Tooltip("value:Q", title="价格", format=".2f"),
            ],
        )
        .properties(title=title, height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def render_quick_usage_tab() -> None:
    st.subheader("量价交易工作台简明用法")
    st.caption("先计划，后执行；先风控，后买入。系统用于研究、复盘和辅助决策，不替代你的独立判断。")

    st.markdown(
        """
        <style>
        .usage-card {
            background: #f8fafc;
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            padding: 18px 20px;
            min-height: 176px;
            margin-bottom: 12px;
        }
        .usage-step {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            border-radius: 999px;
            background: #0f766e;
            color: #ffffff;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .usage-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: #10223d;
            margin-bottom: 8px;
        }
        .usage-text {
            color: #334155;
            line-height: 1.65;
            font-size: 0.98rem;
        }
        .usage-rule {
            background: #ecfdf5;
            border: 1px solid #99f6e4;
            border-radius: 8px;
            padding: 18px 20px;
            color: #134e4a;
            line-height: 1.65;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    row1 = st.columns(3, gap="medium")
    cards = [
        (
            "1",
            "前一晚选股",
            "先扫描上证50、沪深300、中证500、创业板等股票池，优先保留候选分高、结构较好、量价分和短期分同步转强的股票。",
        ),
        (
            "2",
            "单票诊断",
            "对候选股逐只查看结构阶段、量价信号、买入区间、止损位、失效价和卖出风险；没有明确买点的股票只放观察，不直接买入。",
        ),
        (
            "3",
            "放入备选池",
            "把可跟踪的股票加入备选池，第二天开盘前先一键更新，再看系统自动分出的A类可执行、B类观察、C类剔除/防守。",
        ),
    ]
    for col, (step, title, text) in zip(row1, cards):
        col.markdown(usage_card(step, title, text), unsafe_allow_html=True)

    row2 = st.columns(3, gap="medium")
    cards = [
        (
            "4",
            "开盘后盯盘",
            "先处理已有持仓风险，再看A类股票。高开过大不追，跌破开盘价、VWAP或操作失效价时放弃买入或转为防守。",
        ),
        (
            "5",
            "盘中执行",
            "批量粘贴实时价格更新备选池；只有价格仍在买入区间、分时承接较好、止损距离可控时，才按计划分批执行。",
        ),
        (
            "6",
            "收盘复盘",
            "收盘后复查买卖是否遵守计划，把已买入股票加入已购买重点观察，并记录买入价、止损、备注和后续移动止盈条件。",
        ),
    ]
    for col, (step, title, text) in zip(row2, cards):
        col.markdown(usage_card(step, title, text), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="usage-rule">
            <b>核心规则：</b>不要机械执行“上午卖出、下午买入”。卖出服从风险信号，买入服从日K量价买点和盘中确认。
            短期买入或加仓必须有明确量价买点；长期分高只代表值得关注，不能单独变成短线买入理由。
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_usage_reference_section()


def render_usage_reference_section() -> None:
    st.divider()
    st.subheader("完整使用说明")

    with st.expander("工作台功能说明", expanded=True):
        st.markdown(
            """
            **单票诊断**：读取一只股票的日线数据，输出结构阶段、关键支撑/压力、量价信号、评分和解释。

            **股票池扫描**：支持手动代码、内置股票池、行业/概念板块、本地 CSV 文件夹，批量筛选分数达标的候选。

            **我的备选池**：保存候选股票的诊断快照、买入区间、失效价、止损参考和操作状态，支持批量更新对照价和快速删除。

            **已购买**：保存已买入股票的买入价、股数、当前价、持仓备注和买入时诊断快照，用于重点观察和风险复盘。

            **策略回测**：使用当前买卖信号和风控模型，在单只股票或组合股票池上做历史回测，展示资金曲线、交易流水和配对盈亏。

            **跟随交易**：根据美股、韩股前一晚涨跌，映射到A股上游企业，生成开盘前跟随观察清单，并可叠加A股量价诊断。

            **数据工具**：快速检查数据源、预览日K或1/5/15/30/60分钟分时 OHLCV 数据，并导出 CSV。

            程序的核心定位是“量价信号解释器”，不是涨跌预测器。真实交易前仍需人工复盘、参数校准和风险控制。
            """
        )

    with st.expander("本地启动与常用命令", expanded=False):
        st.markdown(
            """
            当前推荐环境：

            ```text
            C:\\ProgramData\\Anaconda3\\envs\\tower312
            ```

            在项目目录启动工作台：

            ```powershell
            cd C:\\Users\\pro6a\\Documents\\stockbuyornot
            .\\workbench.cmd
            ```

            打开地址：

            ```text
            http://127.0.0.1:8501
            ```

            如果打不开，或浏览器提示 `127.0.0.1 拒绝连接`：

            ```powershell
            .\\restart_workbench.cmd
            ```

            命令行单票分析：

            ```powershell
            .\\stockbuyornot.cmd analyze --symbol 000001 --start 20240101 --end 20260517
            ```

            命令行股票扫描：

            ```powershell
            .\\stockbuyornot.cmd scan --symbols 000001 600519 300750 --start 20240101 --end 20260517 --min-score 60
            ```

            导出扫描结果：

            ```powershell
            .\\stockbuyornot.cmd scan --symbols 000001 600519 300750 --start 20240101 --end 20260517 --min-score 60 --output candidates.csv
            ```
            """
        )

    with st.expander("本地 CSV 数据", expanded=False):
        st.markdown(
            """
            单个 CSV 文件分析：

            ```powershell
            .\\stockbuyornot.cmd analyze --csv data\\daily\\000001.csv
            ```

            文件夹扫描：

            ```powershell
            .\\stockbuyornot.cmd scan --csv-dir data\\daily --start 20240101 --end 20260517 --min-score 70 --output candidates.csv
            ```

            CSV 至少包含：

            ```text
            date,open,high,low,close,volume
            ```

            推荐包含：

            ```text
            amount,symbol
            ```
            """
        )

    with st.expander("回测与组合回测", expanded=False):
        st.markdown(
            """
            单票回测：

            ```powershell
            .\\stockbuyornot.cmd backtest --symbol 000001 --start 20200101 --end 20260517
            ```

            组合回测入口主要用于评估“第二阶段主升 + 缩量回踩 + 上涨中继买点”的股票池级别收益。

            小股票池测试：

            ```powershell
            .\\stockbuyornot.cmd portfolio-backtest --symbols 000001 600519 300750 --start 20240101 --end 20260517 --output candidate\\trend_pullback_test.csv
            ```

            指数股票池：

            ```powershell
            .\\stockbuyornot.cmd portfolio-backtest --pool sse50 --start 20240101 --end 20260517 --max-positions 5 --output candidate\\sse50_trend_pullback.csv
            .\\stockbuyornot.cmd portfolio-backtest --pool csi300 --start 20240101 --end 20260517 --max-positions 5 --output candidate\\csi300_trend_pullback.csv
            .\\stockbuyornot.cmd portfolio-backtest --pool csi500 --start 20240101 --end 20260517 --max-positions 5 --output candidate\\csi500_trend_pullback.csv
            .\\stockbuyornot.cmd portfolio-backtest --pool chinext --start 20240101 --end 20260517 --max-positions 5 --output candidate\\chinext_trend_pullback.csv
            ```

            常用参数：

            ```powershell
            --market-mode balanced
            --max-positions 5
            --neutral-max-positions 2
            --min-score 75
            --min-avg-amount 50000000
            --max-stop-distance 0.07
            --min-relative-strength 0.03
            --min-reward-risk 1.8
            --breakeven-r 1.0
            --trail-start-r 2.0
            --trail-pct 0.10
            --stale-days 12
            --max-holding-days 45
            ```
            """
        )

    with st.expander("跟随交易", expanded=False):
        st.markdown(
            """
            跟随交易页用于开盘前根据美股、韩股前一晚涨跌，映射 A 股上游企业。

            使用方式：

            ```text
            1. 选择“自动选强势股”，刷新近期美股/韩股观察池。
            2. 勾选要跟随的海外标的；美股会自动计算近期涨幅和前一交易日涨跌幅。
            3. 韩股当前作为重点观察池展示，可在表格里手动补充涨跌幅。
            4. 选择用“前一交易日涨跌幅”或“最近涨幅”生成跟随信号。
            5. 程序根据内置产业链映射生成 A 股候选。
            6. 可勾选“叠加A股量价诊断”，过滤掉量价结构较弱的标的。
            7. 下载 follow_trade_candidates.csv 作为开盘观察清单。
            ```

            注意：

            ```text
            跟随交易只生成观察清单，不自动下单。
            若A股高开超过5%不追；开盘后跌破开盘价且放量，应放弃跟随。
            映射表是研究辅助，后续需要定期维护产业链关系。
            ```
            """
        )


def usage_card(step: str, title: str, text: str) -> str:
    return f"""
    <div class="usage-card">
        <div class="usage-step">{html.escape(step)}</div>
        <div class="usage-title">{html.escape(title)}</div>
        <div class="usage-text">{html.escape(text)}</div>
    </div>
    """


def load_single_data(data_source: str, symbol: str, settings: dict, provider: AkshareProvider, uploaded, csv_path: str) -> pd.DataFrame:
    if data_source == "在线A股":
        if not symbol:
            raise ValueError("请输入股票代码。")
        return provider.daily(symbol, settings["start"], settings["end"], settings["adjust"])
    if data_source == "上传CSV":
        if uploaded is None:
            raise ValueError("请先上传 CSV。")
        return read_uploaded_csv(uploaded, symbol, settings)
    if not csv_path:
        raise ValueError("请输入 CSV 路径。")
    return CsvProvider(csv_path).daily(symbol=symbol, start=settings["start"], end=settings["end"], adjust=settings["adjust"])


def read_uploaded_csv(uploaded, symbol: str, settings: dict) -> pd.DataFrame:
    with NamedTemporaryFile(delete=False, suffix=".csv") as temp:
        temp.write(uploaded.getvalue())
        path = temp.name
    return CsvProvider(path).daily(symbol=symbol, start=settings["start"], end=settings["end"], adjust=settings["adjust"])


def resolve_scan_sources(source: str, symbols_text: str, pool: str, board_name: str, csv_dir: str, provider: AkshareProvider) -> list[tuple[str, str | Path]]:
    if source == "手动代码":
        symbols = parse_symbols(symbols_text)
    elif source == "内置股票池":
        symbols = provider.symbols_for_pool(pool_alias(pool))
    elif source == "行业板块":
        symbols = provider.industry_symbols(board_name)
    elif source == "概念板块":
        symbols = provider.concept_symbols(board_name)
    else:
        paths = sorted(Path(csv_dir).glob("*.csv"))
        if not paths:
            raise ValueError(f"没有在 {csv_dir} 找到 CSV 文件。")
        return [(path.stem, path) for path in paths]

    if not symbols:
        raise ValueError("股票池为空。")
    return [(symbol, "online") for symbol in symbols]


def load_scan_data(symbol: str, source_ref: str | Path, settings: dict, provider: AkshareProvider) -> pd.DataFrame:
    if isinstance(source_ref, Path):
        return CsvProvider(source_ref).daily(symbol=symbol, start=settings["start"], end=settings["end"], adjust=settings["adjust"])
    return provider.daily(symbol, settings["start"], settings["end"], settings["adjust"])


def parse_symbols(text: str) -> list[str]:
    cleaned = text.replace(",", " ").replace("，", " ").split()
    return [item.zfill(6) for item in cleaned if item.strip()]


def pool_alias(label: str) -> str:
    return {
        "全部A股": "all",
        "上证50": "sse50",
        "沪深300": "csi300",
        "中证500": "csi500",
        "上交所": "sse",
        "深交所": "szse",
        "创业板": "chinext",
        "科创板": "star",
    }[label]


def result_to_scan_row(result: AnalysisResult) -> dict:
    sides = {"buy": "买", "sell": "卖", "watch": "观察", "avoid": "回避"}
    return {
        "代码": result.symbol,
        "日期": result.as_of.date(),
        "收盘": round(result.close, 2),
        "结构": result.structure.stage.value,
        "评分": result.score.total,
        "建议": result.suggestion,
        "信号": "、".join(signal.name for signal in result.signals),
        "方向": "、".join(sides.get(signal.side.value, signal.side.value) for signal in result.signals),
    }


def render_result_summary(result: AnalysisResult) -> None:
    score_delta = "强信号" if result.score.total >= 80 else "观察" if result.score.total >= 60 else "中性/回避"
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(metric_card("收盘", f"{result.close:.2f}"), unsafe_allow_html=True)
    col2.markdown(metric_card("结构", result.structure.stage.value), unsafe_allow_html=True)
    col3.markdown(metric_card("评分", str(result.score.total), score_delta), unsafe_allow_html=True)
    col4.markdown(metric_card("建议", result.suggestion), unsafe_allow_html=True)


def render_price_volume(df: pd.DataFrame) -> None:
    st.subheader("价格与成交量")
    chart_data = df.copy()
    chart_data["date"] = pd.to_datetime(chart_data["date"])
    chart_data = chart_data.set_index("date").tail(220).reset_index()
    chart_data["ma20"] = chart_data["close"].rolling(20, min_periods=5).mean()
    chart_data["ma60"] = chart_data["close"].rolling(60, min_periods=20).mean()
    chart_data["volume_ma20"] = chart_data["volume"].rolling(20, min_periods=5).mean()
    chart_data["color"] = chart_data.apply(lambda row: "#d94b45" if row["close"] >= row["open"] else "#2f9e77", axis=1)

    base = alt.Chart(chart_data).encode(x=alt.X("date:T", axis=alt.Axis(title=None, labelAngle=0)))
    wick = base.mark_rule(size=1).encode(
        y=alt.Y("low:Q", title="价格", scale=alt.Scale(zero=False)),
        y2="high:Q",
        color=alt.Color("color:N", scale=None),
        tooltip=price_tooltip(),
    )
    body = base.mark_bar(size=5).encode(y="open:Q", y2="close:Q", color=alt.Color("color:N", scale=None), tooltip=price_tooltip())
    ma20 = base.mark_line(color="#2b6cb0", strokeWidth=1.6).encode(y="ma20:Q")
    ma60 = base.mark_line(color="#805ad5", strokeWidth=1.6).encode(y="ma60:Q")
    price_chart = (wick + body + ma20 + ma60).properties(height=330)

    volume_bar = base.mark_bar(opacity=0.55).encode(
        y=alt.Y("volume:Q", title="成交量"),
        color=alt.Color("color:N", scale=None),
        tooltip=[alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"), alt.Tooltip("volume:Q", title="成交量", format=",.0f")],
    )
    volume_ma = base.mark_line(color="#64748b", strokeWidth=1.3).encode(y="volume_ma20:Q")
    st.altair_chart(alt.vconcat(price_chart, (volume_bar + volume_ma).properties(height=105)).resolve_scale(x="shared").configure_view(strokeWidth=0), use_container_width=True)


def render_result_detail(result: AnalysisResult) -> None:
    st.subheader("诊断详情")
    col1, col2, col3 = st.columns([1, 1, 1.2], gap="large")
    with col1:
        st.markdown("**关键支撑**")
        st.dataframe(levels_to_frame(result.support_levels), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**关键压力**")
        st.dataframe(levels_to_frame(result.resistance_levels), use_container_width=True, hide_index=True)
    with col3:
        st.markdown("**评分拆解**")
        score = result.score
        st.dataframe(
            pd.DataFrame(
                [
                    {"模块": "结构", "得分": score.structure},
                    {"模块": "位置", "得分": score.position},
                    {"模块": "量价信号", "得分": score.signal},
                    {"模块": "风险", "得分": score.risk},
                    {"模块": "相对强度", "得分": score.relative_strength},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**量价信号**")
    if not result.signals:
        st.info("暂无明确关键信号。")
    else:
        for signal in result.signals:
            render_signal_card(signal)

    st.markdown("**解释**")
    for item in result.score.explanation:
        st.write(f"- {item}")


def render_signal_card(signal) -> None:
    side_label = {SignalSide.BUY: "买入", SignalSide.SELL: "卖出", SignalSide.WATCH: "观察", SignalSide.AVOID: "回避"}.get(signal.side, signal.side.value)
    with st.container(border=True):
        header_cols = st.columns([1.2, 0.4, 0.4])
        header_cols[0].markdown(f"**{signal.name}**")
        header_cols[1].markdown(metric_card("方向", side_label), unsafe_allow_html=True)
        header_cols[2].markdown(metric_card("强度", str(signal.strength)), unsafe_allow_html=True)
        st.write(signal.logic)
        if signal.entry_zone:
            st.write(f"参考区间：{signal.entry_zone[0]:.2f} - {signal.entry_zone[1]:.2f}")
        if signal.stop_loss:
            st.write(f"止损位：{signal.stop_loss:.2f}")
        trigger_level = getattr(signal, "trigger_level", None)
        invalidation_price = getattr(signal, "invalidation_price", None)
        risk_buffer_pct = getattr(signal, "risk_buffer_pct", None)
        if trigger_level is not None and invalidation_price is not None:
            st.write(
                f"关键位：{trigger_level:.2f}；实时失效价：{invalidation_price:.2f}；"
                f"缓冲：{(risk_buffer_pct or 0) * 100:.1f}%"
            )
        if signal.invalidation:
            st.write(f"失效条件：{signal.invalidation}")
        if signal.evidence:
            st.write("证据：" + "；".join(signal.evidence))


def levels_to_frame(levels: Iterable) -> pd.DataFrame:
    rows = []
    for level in list(levels)[:8]:
        rows.append({"名称": level.name, "价格": round(level.price, 2), "类型": level.kind, "日期": "" if level.date is None else pd.to_datetime(level.date).date()})
    return pd.DataFrame(rows)


def render_backtest_metrics(metrics: dict[str, float]) -> None:
    cols = st.columns(6)
    cols[0].markdown(metric_card("初始资金", f"{metrics.get('initial_cash', 0):,.0f}"), unsafe_allow_html=True)
    cols[1].markdown(metric_card("期末资金", f"{metrics.get('final_cash', 0):,.0f}"), unsafe_allow_html=True)
    cols[2].markdown(metric_card("总收益", f"{metrics.get('total_return', 0):.2%}"), unsafe_allow_html=True)
    cols[3].markdown(metric_card("交易次数", str(int(metrics.get("trade_count", 0)))), unsafe_allow_html=True)
    cols[4].markdown(metric_card("胜率", f"{metrics.get('win_rate', 0):.2%}"), unsafe_allow_html=True)
    cols[5].markdown(metric_card("平均单笔", f"{metrics.get('avg_trade_return', 0):.2%}"), unsafe_allow_html=True)


def equity_curve_from_trades(trades: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    equity = trades[["date", "cash"]].copy()
    equity["date"] = pd.to_datetime(equity["date"])
    equity = pd.concat([pd.DataFrame([{"date": equity["date"].min(), "cash": initial_cash}]), equity], ignore_index=True)
    equity = equity.sort_values("date").reset_index(drop=True)
    equity["return"] = equity["cash"] / initial_cash - 1
    return equity


def render_equity_chart(equity: pd.DataFrame) -> None:
    st.subheader("资金曲线")
    chart = (
        alt.Chart(equity)
        .mark_line(color="#2b6cb0", strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("cash:Q", title="资金", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                alt.Tooltip("cash:Q", title="资金", format=",.2f"),
                alt.Tooltip("return:Q", title="收益率", format=".2%"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)


def trade_pairs(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    buy = None
    for _, trade in trades.iterrows():
        if trade["side"] == "buy":
            buy = trade
        elif trade["side"] == "sell" and buy is not None:
            pnl_pct = trade["price"] / buy["price"] - 1
            rows.append(
                {
                    "买入日期": pd.to_datetime(buy["date"]).date(),
                    "卖出日期": pd.to_datetime(trade["date"]).date(),
                    "买入价": round(float(buy["price"]), 3),
                    "卖出价": round(float(trade["price"]), 3),
                    "股数": int(buy["shares"]),
                    "收益率": f"{pnl_pct:.2%}",
                    "买入原因": buy["reason"],
                    "卖出原因": trade["reason"],
                }
            )
            buy = None
    return pd.DataFrame(rows)


def metric_card(label: str, value: str, note: str | None = None, tooltip: str | None = None) -> str:
    note_html = f'<div class="metric-note">{note}</div>' if note else ""
    title_attr = f' title="{html.escape(tooltip, quote=True)}"' if tooltip else ""
    cursor_style = ' style="cursor:help;"' if tooltip else ""
    return f"""
    <div class="metric-card"{title_attr}{cursor_style}>
        <div class="metric-label">{label}</div>
        <div class="metric-value" title="{value}">{value}</div>
        {note_html}
    </div>
    """


def render_market_status_widget(settings: dict) -> None:
    token_key = "market_status_refresh_token"
    st.session_state.setdefault(token_key, 0)
    button_cols = st.columns([0.54, 0.46], gap="small")
    button_cols[0].caption("大盘状态")
    if button_cols[1].button("刷新大盘", key="refresh_market_status", use_container_width=True):
        st.session_state[token_key] += 1
    index_symbol = st.selectbox(
        "参考指数",
        list(MARKET_INDEX_OPTIONS.keys()),
        format_func=lambda value: MARKET_INDEX_OPTIONS.get(value, value),
        key="market_status_index_symbol",
        label_visibility="collapsed",
    )

    try:
        snapshot = load_market_status_for_widget(
            index_symbol,
            settings["end"],
            float(settings["request_timeout"]),
            int(settings["request_retries"]),
            int(st.session_state[token_key]),
        )
    except Exception as exc:
        snapshot = {
            "state": "unknown",
            "state_label": "未知",
            "as_of": "-",
            "close": None,
            "ret20": None,
            "ret60": None,
            "close_vs_ma60": None,
            "index_name": MARKET_INDEX_OPTIONS.get(index_symbol, index_symbol),
            "tomorrow_label": "无法预测",
            "tomorrow_score": None,
            "hint": f"大盘数据暂时不可用：{exc}",
        }
    st.markdown(market_status_card(snapshot), unsafe_allow_html=True)


@st.cache_data(ttl=900, show_spinner=False)
def load_market_status_for_widget(
    index_symbol: str,
    end: str,
    request_timeout: float,
    request_retries: int,
    refresh_token: int,
) -> dict:
    _ = refresh_token
    provider = AkshareProvider(request_timeout=request_timeout, request_retries=request_retries)
    data = provider.index_daily(index_symbol, market_status_start(end), end)
    return compute_market_status_snapshot(data, index_name=MARKET_INDEX_OPTIONS.get(index_symbol, index_symbol))


def compute_market_status_snapshot(data: pd.DataFrame | None, index_name: str = "沪深300") -> dict:
    if data is None or data.empty:
        return _unknown_market_status(f"没有取到{index_name}数据", index_name)
    frame = data.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"]).reset_index(drop=True)
    if len(frame) < 60:
        return _unknown_market_status(f"{index_name}样本不足，暂不能判断强弱", index_name)

    close = frame["close"]
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    latest_close = float(close.iloc[-1])
    latest_ma60 = _safe_float(ma60.iloc[-1])
    try:
        state = market_state_from_benchmark(frame)
    except Exception:
        state = _market_state_from_ma(latest_close, _safe_float(ma20.iloc[-1]), latest_ma60)
    as_of = "-"
    if "date" in frame.columns and pd.notna(frame["date"].iloc[-1]):
        as_of = pd.Timestamp(frame["date"].iloc[-1]).strftime("%Y-%m-%d")
    tomorrow = market_tomorrow_forecast(frame, state, _safe_float(ma20.iloc[-1]), latest_ma60)

    return {
        "state": state,
        "state_label": market_state_label(state),
        "as_of": as_of,
        "index_name": index_name,
        "close": latest_close,
        "ma20": _safe_float(ma20.iloc[-1]),
        "ma60": latest_ma60,
        "ret20": _window_return(close, 20),
        "ret60": _window_return(close, 60),
        "close_vs_ma60": (latest_close / latest_ma60 - 1.0) if latest_ma60 else None,
        "tomorrow_label": tomorrow["label"],
        "tomorrow_score": tomorrow["score"],
        "tomorrow_hint": tomorrow["hint"],
        "hint": market_state_hint(state),
    }


def market_status_card(snapshot: dict) -> str:
    state = str(snapshot.get("state") or "unknown")
    state_label = html.escape(str(snapshot.get("state_label") or market_state_label(state)))
    index_name = html.escape(str(snapshot.get("index_name") or "沪深300"))
    as_of = html.escape(str(snapshot.get("as_of") or "-"))
    close_text = _market_number(snapshot.get("close"))
    ret20_text = _market_pct(snapshot.get("ret20"))
    ret60_text = _market_pct(snapshot.get("ret60"))
    ma60_gap_text = _market_pct(snapshot.get("close_vs_ma60"))
    tomorrow_label = html.escape(str(snapshot.get("tomorrow_label") or "无法预测"))
    tomorrow_score = _market_score(snapshot.get("tomorrow_score"))
    tomorrow_hint = html.escape(str(snapshot.get("tomorrow_hint") or "明日倾向仅为统计辅助。"))
    hint = html.escape(str(snapshot.get("hint") or ""))
    title = html.escape("；".join(part for part in [str(snapshot.get("hint") or ""), str(snapshot.get("tomorrow_hint") or "")] if part), quote=True)
    return f"""
    <div class="market-status-card market-status-{html.escape(state)}" title="{title}">
        <div class="market-status-top">
            <span class="market-status-name">{index_name}</span>
            <span class="market-status-pill">{state_label}</span>
        </div>
        <div class="market-status-main">
            <span>{close_text}</span>
            <small>{as_of}</small>
        </div>
        <div class="market-status-grid">
            <span>20日 {ret20_text}</span>
            <span>60日 {ret60_text}</span>
            <span>距MA60 {ma60_gap_text}</span>
        </div>
        <div class="market-status-forecast">
            <span>明日倾向</span>
            <b>{tomorrow_label}</b>
            <small>{tomorrow_score}</small>
        </div>
        <div class="market-status-hint">{hint}</div>
        <div class="market-status-hint">{tomorrow_hint}</div>
    </div>
    """


def market_status_start(end: str) -> str:
    end_ts = pd.to_datetime(end, format="%Y%m%d", errors="coerce")
    if pd.isna(end_ts):
        end_ts = pd.Timestamp.today()
    return (end_ts - pd.Timedelta(days=430)).strftime("%Y%m%d")


def market_state_label(state: str) -> str:
    return {"strong": "强市", "neutral": "震荡", "weak": "弱市", "unknown": "未知"}.get(str(state), str(state))


def market_state_hint(state: str) -> str:
    if state == "strong":
        return "强市：可正常筛选，优先执行窗口高且量价确认的股票。"
    if state == "neutral":
        return "震荡：控制仓位，等待缩量回踩和盘中确认。"
    if state == "weak":
        return "弱市：谨慎开新仓，只考虑独立强势且风险距离小的股票。"
    return "状态未知：先检查数据源，再降低仓位假设。"


def market_tomorrow_forecast(frame: pd.DataFrame, state: str, ma20_latest: float | None, ma60_latest: float | None) -> dict:
    close = pd.to_numeric(frame["close"], errors="coerce").dropna().reset_index(drop=True)
    if len(close) < 60:
        return {"score": None, "label": "无法预测", "hint": "样本不足，明日倾向暂不判断。"}

    latest = _safe_float(close.iloc[-1])
    if latest is None:
        return {"score": None, "label": "无法预测", "hint": "收盘价缺失，明日倾向暂不判断。"}

    score = 50.0
    reasons: list[str] = []

    state_delta = {"strong": 8.0, "neutral": 0.0, "weak": -8.0}.get(state, 0.0)
    score += state_delta
    if state_delta > 0:
        reasons.append("中期结构偏强")
    elif state_delta < 0:
        reasons.append("中期结构偏弱")

    ret1 = _window_return(close, 1)
    ret3 = _window_return(close, 3)
    ret5 = _window_return(close, 5)
    ret20 = _window_return(close, 20)
    ma20_slope = _series_slope(close.rolling(20).mean(), 5)
    ma60_slope = _series_slope(close.rolling(60).mean(), 10)

    if ma20_latest:
        gap20 = latest / ma20_latest - 1.0
        if 0 <= gap20 <= 0.04:
            score += 8
            reasons.append("站上MA20且不过热")
        elif -0.03 <= gap20 < 0:
            score += 2
            reasons.append("贴近MA20震荡")
        elif gap20 > 0.08:
            score -= 7
            reasons.append("短期偏离MA20较远")
        elif gap20 < -0.04:
            score -= 7
            reasons.append("跌离MA20")

    if ma60_latest:
        gap60 = latest / ma60_latest - 1.0
        if gap60 > 0:
            score += 5
        else:
            score -= 5

    if ma20_slope is not None:
        score += 7 if ma20_slope > 0 else -7
        reasons.append("MA20上行" if ma20_slope > 0 else "MA20下行")
    if ma60_slope is not None:
        score += 4 if ma60_slope > 0 else -4

    if ret5 is not None:
        if 0 < ret5 <= 0.05:
            score += 5
            reasons.append("5日动量温和")
        elif ret5 > 0.08:
            score -= 5
            reasons.append("5日涨幅偏急")
        elif ret5 < -0.04:
            score -= 5
            reasons.append("5日动量转弱")

    if ret20 is not None:
        if ret20 > 0:
            score += 3
        else:
            score -= 3

    if ret1 is not None:
        if 0 < ret1 <= 0.02:
            score += 3
        elif ret1 > 0.035:
            score -= 2
            reasons.append("单日拉升后有震荡消化需求")
        elif ret1 < -0.02:
            score -= 4
            reasons.append("当日下跌压力较大")
    if ret3 is not None and ret3 < -0.04:
        score -= 4

    if "volume" in frame.columns:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        vol_ma20 = _safe_float(volume.rolling(20).mean().iloc[-1])
        latest_vol = _safe_float(volume.iloc[-1])
        if vol_ma20 and latest_vol:
            vol_ratio = latest_vol / vol_ma20
            if ret1 is not None and ret1 > 0 and 1.05 <= vol_ratio <= 2.2:
                score += 4
                reasons.append("上涨有成交量配合")
            elif ret1 is not None and ret1 < 0 and vol_ratio >= 1.3:
                score -= 5
                reasons.append("下跌放量")
            elif vol_ratio < 0.75:
                score -= 2

    score = max(0.0, min(100.0, score))
    label = market_tomorrow_label(score)
    if not reasons:
        reasons.append("趋势与动量信号不突出")
    hint = f"明日倾向：{label}，统计分{score:.0f}；依据：" + "、".join(reasons[:3]) + "。"
    return {"score": score, "label": label, "hint": hint}


def market_tomorrow_label(score: float | None) -> str:
    if score is None:
        return "无法预测"
    if score >= 68:
        return "偏强"
    if score >= 56:
        return "略偏强"
    if score >= 45:
        return "中性震荡"
    if score >= 35:
        return "略偏弱"
    return "偏弱"


def _market_state_from_ma(close: float, ma20: float | None, ma60: float | None) -> str:
    if ma20 is None or ma60 is None:
        return "unknown"
    if close > ma60 and ma20 > ma60:
        return "strong"
    if close < ma60 and ma20 < ma60:
        return "weak"
    return "neutral"


def _unknown_market_status(reason: str, index_name: str = "沪深300") -> dict:
    return {
        "state": "unknown",
        "state_label": "未知",
        "as_of": "-",
        "index_name": index_name,
        "close": None,
        "ret20": None,
        "ret60": None,
        "close_vs_ma60": None,
        "tomorrow_label": "无法预测",
        "tomorrow_score": None,
        "tomorrow_hint": "数据不足，明日倾向暂不判断。",
        "hint": reason,
    }


def _window_return(close: pd.Series, window: int) -> float | None:
    if len(close) <= window:
        return None
    base = _safe_float(close.iloc[-window - 1])
    latest = _safe_float(close.iloc[-1])
    if not base:
        return None
    return latest / base - 1.0


def _safe_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed


def _series_slope(series: pd.Series, window: int) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= window:
        return None
    base = _safe_float(clean.iloc[-window - 1])
    latest = _safe_float(clean.iloc[-1])
    if not base:
        return None
    return latest / base - 1.0


def _market_pct(value) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "-"
    return f"{parsed * 100:+.1f}%"


def _market_score(value) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "-"
    return f"{parsed:.0f}分"


def _market_number(value) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "-"
    if abs(parsed) < 100:
        return f"{parsed:.3f}"
    return f"{parsed:.0f}"


def compact_score_badge(score: int | float) -> str:
    value = f"{float(score):.0f}"
    return f"""
    <div style="
        display:inline-flex;
        align-items:center;
        gap:6px;
        padding:8px 12px;
        border:1px solid #dbe3ef;
        border-radius:8px;
        background:#f8fafc;
        white-space:nowrap;
        min-width:86px;
        justify-content:center;
    ">
        <span style="font-size:0.86rem;color:#475569;">评分</span>
        <span style="font-size:1.15rem;font-weight:800;color:#0f172a;line-height:1;">{value}</span>
    </div>
    """


def factor_score_badges(result: AnalysisResult) -> str:
    scores = getattr(result, "factor_scores", None)
    if scores is None:
        items = [("\u8bc4\u5206", float(result.score.total))]
    else:
        items = [
            ("\u5019\u9009", scan_candidate_score(result)),
            ("\u7efc\u5408", scores.overall_score),
            ("\u91cf\u4ef7", scores.liangjia_score),
            ("\u77ed\u671f", scores.short_term_score),
            ("\u957f\u671f", scores.long_term_score),
        ]
    spans = "".join(
        (
            '<span style="display:inline-flex;align-items:center;gap:4px;padding:6px 9px;'
            'border:1px solid #dbe3ef;border-radius:8px;background:#f8fafc;white-space:nowrap;cursor:help;" '
            f'title="{html.escape(score_badge_tooltip(result, label), quote=True)}">'
            f'<span style="font-size:0.78rem;color:#475569;">{label}</span>'
            f'<span style="font-size:0.98rem;font-weight:800;color:#0f172a;line-height:1;">{float(value):.0f}</span>'
            "</span>"
        )
        for label, value in items
    )
    return f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">{spans}</div>'


def score_badge_tooltip(result: AnalysisResult, label: str) -> str:
    scores = getattr(result, "factor_scores", None)
    if scores is None:
        return "旧版评分：主要由结构、位置、量价信号、止损风险和相对强度组成。"
    components = getattr(scores, "components", {}) or {}
    if label == "\u7efc\u5408":
        return (
            "综合分：量价分权重最高，再叠加短期确认和长期分层；"
            "风险惩罚会扣分。当前约为 "
            f"量价{scores.liangjia_score:.0f}、短期{scores.short_term_score:.0f}、长期{scores.long_term_score:.0f} 的加权结果。"
        )
    if label == "\u91cf\u4ef7":
        item = components.get("liangjia", {})
        return (
            "量价分：保留原供需主线，先看结构，再看位置，再看量价信号和止损风险；"
            f"基础分{item.get('base_score', scores.liangjia_score):.0f}，"
            f"市场修正{item.get('market_modifier', 0):+.0f}，"
            f"卖出风险修正{item.get('exit_risk_modifier', 0):+.0f}。"
        )
    if label == "\u77ed\u671f":
        item = components.get("short_term", {})
        return (
            "短期分：用于确认近期走势和交易机会，主要看1/3/5/10/20日动量、短均线、量比、RSI/MACD/KDJ、突破和短期回撤；"
            f"动量{item.get('return_score', 0):.0f}，均线{item.get('ma_score', 0):.0f}，"
            f"量能{item.get('volume_score', 0):.0f}，风险{item.get('risk_score', 0):.0f}。"
        )
    if label == "\u957f\u671f":
        item = components.get("long_term", {})
        return (
            "长期分：用于判断长期趋势、基本面质量、成长、估值和财务风险；没有基本面数据时按中性处理，不能单独触发买入；"
            f"趋势{item.get('trend_score', 0):.0f}，收益{item.get('return_score', 0):.0f}，"
            f"质量{item.get('quality_score', 0):.0f}，估值{item.get('valuation_score', 0):.0f}。"
        )
    return "该分数用于辅助排序，最终买入仍必须服从量价买点和风险过滤。"


def scan_save_sort_score(result: AnalysisResult) -> float:
    return float(scan_candidate_score(result))


def metric_score_tooltip(result: AnalysisResult, label: str) -> str:
    mapping = {
        "\u91cf\u4ef7\u5206": "\u91cf\u4ef7",
        "\u7efc\u5408\u5206": "\u7efc\u5408",
        "\u77ed\u671f\u5206": "\u77ed\u671f",
        "\u957f\u671f\u5206": "\u957f\u671f",
    }
    if label in mapping:
        return score_badge_tooltip(result, mapping[label])
    if label == "\u5019\u9009\u5206":
        return "\u5019\u9009\u5206\uff1a\u7528\u4e8e\u626b\u63cf\u5165\u9009\u548c\u6392\u5e8f\u3002\u65b0\u7248\u4e0d\u518d\u5408\u6210\u5355\u4e00\u6700\u7ec8\u7ed3\u8bba\uff0c\u800c\u662f\u540c\u65f6\u53c2\u8003\u957f\u671f\u5206\u3001\u77ed\u671f\u5206\u548c\u91cf\u4ef7\u5206\uff1b\u77ed\u671f\u4e70\u5165\u4ecd\u5fc5\u987b\u7531\u91cf\u4ef7\u4e70\u70b9\u89e6\u53d1\u3002"
    if label == "\u5f3a\u80a1\u8bbe\u7f6e":
        return "\u5f3a\u80a1\u8bbe\u7f6e\u5206\uff1a\u8861\u91cf\u5e02\u573a\u3001\u677f\u5757\u3001\u4e2a\u80a1\u76f8\u5bf9\u5f3a\u5ea6\u548c\u7ed3\u6784\u662f\u5426\u6b63\u5728\u53d8\u597d\uff1b\u5b83\u53ea\u51b3\u5b9a\u5173\u6ce8\u4f18\u5148\u7ea7\uff0c\u4e0d\u5355\u72ec\u89e6\u53d1\u4e70\u5165\u3002"
    if label == "\u4e70\u70b9\u8d28\u91cf":
        return "\u4e70\u70b9\u8d28\u91cf\u5206\uff1a\u8861\u91cf\u652f\u6491\u6765\u6e90\u3001\u7f29\u91cf\u56de\u8e29\u3001\u6b62\u635f\u8ddd\u79bb\u3001\u76c8\u4e8f\u6bd4\u548c\u662f\u5426\u8dcc\u7834\u5173\u952e\u4f4d\uff1b\u7528\u4e8e\u786e\u8ba4\u91cf\u4ef7\u4e70\u70b9\u662f\u5426\u503c\u5f97\u6267\u884c\u3002"
    if label == "\u5356\u51fa\u98ce\u9669":
        return "\u5356\u51fa\u98ce\u9669\u5206\uff1a\u8861\u91cf\u9ad8\u4f4d\u653e\u91cf\u6ede\u6da8\u3001\u5411\u4e0b\u53cd\u8f6c\u3001\u8dcc\u7834\u5173\u952e\u4f4d\u3001\u76f8\u5bf9\u5f3a\u5ea6\u8f6c\u5f31\u7b49\u98ce\u9669\uff1b\u98ce\u9669\u4fe1\u53f7\u4f18\u5148\u7ea7\u9ad8\u4e8e\u673a\u4f1a\u4fe1\u53f7\u3002"
    return "\u8be5\u5206\u6570\u7528\u4e8e\u8f85\u52a9\u5224\u65ad\u3002\u957f\u671f\u548c\u77ed\u671f\u5206\u5f00\u89e3\u91ca\uff1b\u77ed\u671f\u52a8\u4f5c\u4ecd\u9075\u5b88\u5148\u7ed3\u6784\u3001\u540e\u4f4d\u7f6e\u3001\u518d\u91cf\u4ef7\u4fe1\u53f7\u7684\u4e3b\u6d41\u7a0b\u3002"


def price_tooltip() -> list:
    return [
        alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
        alt.Tooltip("open:Q", title="开盘", format=".2f"),
        alt.Tooltip("high:Q", title="最高", format=".2f"),
        alt.Tooltip("low:Q", title="最低", format=".2f"),
        alt.Tooltip("close:Q", title="收盘", format=".2f"),
        alt.Tooltip("volume:Q", title="成交量", format=",.0f"),
    ]


def infer_symbol_from_data(df: pd.DataFrame, default: str) -> str:
    if "symbol" in df.columns and not df["symbol"].dropna().empty:
        symbol = str(df["symbol"].dropna().iloc[-1])
        return symbol if symbol else default
    return default


def short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    if "At least 60 trading days" in message:
        return "数据不足：至少需要 60 个交易日，推荐 120 个交易日以上。"
    if "OHLCV data is empty" in message or "Intraday data is empty" in message or "未读取到" in message:
        if "分钟分时数据" in message or "Intraday" in message:
            return "没有读取到可用分时数据：请确认股票代码正确，并尝试把回看天数调大、切换5分钟/15分钟周期，或稍后重试。"
        return "没有读取到可用行情数据：请确认股票代码、日期区间和数据源是否正常。"
    if "Unable to fetch" in message or "Connection" in message or "timeout" in message.lower():
        return "在线数据读取失败：请稍后重试，或改用本地 CSV 数据。"
    return message or exc.__class__.__name__


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1480px;
            padding-top: 1.2rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
            padding-bottom: 2rem;
        }
        h1 { font-size: 2.25rem; line-height: 1.15; margin-bottom: 0.75rem; }
        h3 { margin-top: 0.7rem; margin-bottom: 0.75rem; }
        .auth-heading {
            margin: 8vh 0 1rem;
            text-align: center;
        }
        .auth-heading h1 {
            margin: 0 0 0.35rem;
            color: #0f172a;
            font-size: 1.75rem;
            line-height: 1.2;
            letter-spacing: 0;
        }
        .auth-heading p {
            margin: 0;
            color: #64748b;
            font-size: 0.95rem;
        }
        .metric-card {
            min-height: 112px;
            background: #f8fafc;
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            padding: 16px 18px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 8px;
            overflow: hidden;
        }
        .metric-label { color: #10223d; font-size: 1rem; line-height: 1.25; white-space: nowrap; }
        .metric-value {
            color: #182235;
            font-size: clamp(1.45rem, 1.9vw, 2.25rem);
            font-weight: 650;
            line-height: 1.08;
            letter-spacing: 0;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .metric-note { color: #2f9e44; font-size: 0.95rem; line-height: 1.2; white-space: normal; }
        .market-status-card {
            margin-top: 0.15rem;
            min-height: 126px;
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            background: #f8fafc;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            overflow: hidden;
        }
        .market-status-strong { border-color: #86efac; background: #f0fdf4; }
        .market-status-neutral { border-color: #fde68a; background: #fffbeb; }
        .market-status-weak { border-color: #fecaca; background: #fef2f2; }
        .market-status-unknown { border-color: #cbd5e1; background: #f8fafc; }
        .market-status-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            white-space: nowrap;
        }
        .market-status-name { color: #334155; font-size: 0.9rem; font-weight: 650; }
        .market-status-pill {
            border-radius: 999px;
            background: #ffffff;
            border: 1px solid #dbe3ef;
            padding: 2px 8px;
            color: #0f172a;
            font-size: 0.84rem;
            font-weight: 750;
        }
        .market-status-main {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 10px;
            color: #0f172a;
        }
        .market-status-main span { font-size: 1.42rem; font-weight: 800; line-height: 1; }
        .market-status-main small { color: #64748b; font-size: 0.78rem; white-space: nowrap; }
        .market-status-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 5px;
            color: #334155;
            font-size: 0.78rem;
            line-height: 1.2;
        }
        .market-status-grid span {
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .market-status-forecast {
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 8px;
            align-items: center;
            border-top: 1px solid rgba(148, 163, 184, 0.35);
            padding-top: 7px;
            color: #334155;
            font-size: 0.82rem;
            line-height: 1.2;
        }
        .market-status-forecast b {
            color: #0f172a;
            font-size: 0.94rem;
            text-align: center;
        }
        .market-status-forecast small {
            color: #64748b;
            text-align: right;
            white-space: nowrap;
        }
        .market-status-hint {
            color: #475569;
            font-size: 0.78rem;
            line-height: 1.25;
        }
        div[data-testid="stHorizontalBlock"] { gap: 1rem; }
        section[data-testid="stSidebar"] { border-right: 1px solid #e2e8f0; }
        div[data-testid="stTabs"] button p { font-size: 0.98rem; }
        div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 8px; }
        @media (max-width: 1100px) {
            .block-container { padding-left: 1rem; padding-right: 1rem; }
            .metric-card { min-height: 92px; padding: 12px 14px; }
            .metric-value { font-size: 1.45rem; }
            .market-status-card { min-height: auto; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_scan_save_buttons(results: list[AnalysisResult]) -> None:
    if not results:
        return
    st.markdown("**加入备选池**")
    saved_symbols = {item.get("symbol", "") for item in load_watchlist_records()}
    ordered = sorted(results, key=scan_save_sort_score, reverse=True)
    for result in ordered:
        with st.container(border=True):
            cols = st.columns([0.12, 0.34, 0.20, 0.20, 0.14], gap="medium")
            cols[0].markdown(f"**{result.symbol}**")
            cols[1].markdown(factor_score_badges(result), unsafe_allow_html=True)
            cols[2].write(result.structure.stage.value)
            cols[3].write("、".join(signal.name for signal in result.signals) or "暂无明确信号")
            already_saved = result.symbol in saved_symbols
            label = "更新备选池" if already_saved else "加入备选池"
            if cols[4].button(label, key=f"save_scan_{result.symbol}_{result.as_of.date()}", use_container_width=True):
                save_watchlist_result(result, source="股票池扫描")
                st.success(f"{result.symbol} 已保存到我的备选池。")


def render_watchlist_tab(settings: dict, provider: AkshareProvider) -> None:
    st.subheader("我的备选池")
    records = load_watchlist_records()
    if not records:
        st.info("备选池还是空的。可以在单票诊断或股票池扫描结果里点击“加入我的备选池”。")
        return

    refresh_message = st.session_state.pop("watchlist_refresh_message", "")
    if refresh_message:
        st.success(refresh_message)
    refresh_warning = st.session_state.pop("watchlist_refresh_warning", "")
    if refresh_warning:
        st.warning(refresh_warning)

    refresh_cols = st.columns([0.28, 0.72], gap="medium")
    if refresh_cols[0].button("一键更新备选池", type="primary", use_container_width=True):
        updated_count, failed_items = refresh_watchlist_records(settings, provider)
        st.session_state["watchlist_refresh_message"] = f"已更新 {updated_count} 只备选股票的最新评分、信号和关键位。"
        if failed_items:
            failed_text = "、".join(f"{symbol}（{reason}）" for symbol, reason in failed_items[:5])
            more_text = "" if len(failed_items) <= 5 else f" 等 {len(failed_items)} 只"
            st.session_state["watchlist_refresh_warning"] = f"有 {len(failed_items)} 只更新失败：{failed_text}{more_text}。"
        st.rerun()
    refresh_cols[1].caption("批量重新拉取日线数据并运行当前诊断算法；会保留你的人工标记、备注和对照价。")

    records = load_watchlist_records()
    sorted_records = sort_watchlist_records(records)
    render_watchlist_trade_plan(sorted_records)
    render_watchlist_bulk_price_update(sorted_records)
    render_watchlist_quick_delete(sorted_records)
    summary = pd.DataFrame([watchlist_summary_row(record) for record in sorted_records])
    st.dataframe(style_watchlist_summary(summary), use_container_width=True, hide_index=True)

    export_cols = st.columns([0.5, 0.5], gap="medium")
    export_cols[0].download_button(
        "下载备选池CSV",
        data=summary.to_csv(index=False, encoding="utf-8-sig"),
        file_name="watchlist.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "下载完整快照JSON",
        data=json.dumps(records, ensure_ascii=False, indent=2),
        file_name="watchlist.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("**备选详情**")
    for record in sorted_records:
        status_label = operation_status_label(watchlist_display_status(record))
        execution_label = execution_status_label(watchlist_execution_status(record))
        with st.expander(f"{record.get('symbol', '')} | {execution_label} | {record.get('score', {}).get('total', 0)}分 | {record.get('suggestion', '')}"):
            cols = st.columns([0.18, 0.18, 0.18, 0.18, 0.28], gap="medium")
            cols[0].markdown(metric_card("收盘", f"{float(record.get('close', 0)):.2f}"), unsafe_allow_html=True)
            cols[1].markdown(metric_card("结构", record.get("structure", {}).get("stage", "")), unsafe_allow_html=True)
            cols[2].markdown(metric_card("评分", str(record.get("score", {}).get("total", 0))), unsafe_allow_html=True)
            cols[3].markdown(metric_card("操作状态", execution_label, status_label), unsafe_allow_html=True)
            cols[4].markdown(metric_card("加入时间", record.get("saved_at", "-")), unsafe_allow_html=True)
            st.info(watchlist_detail_plan_text(record))

            with st.form(key=f"watchlist_status_{record.get('symbol', '')}"):
                status_options = ["未标记", "仍有效", "已失效"]
                current_status = operation_status_label(record.get("operation_status", "unmarked"))
                form_cols = st.columns([0.24, 0.2, 0.36, 0.2], gap="medium")
                chosen_status = form_cols[0].radio(
                    "操作标记",
                    status_options,
                    index=status_options.index(current_status) if current_status in status_options else 0,
                    horizontal=True,
                )
                current_price = form_cols[1].number_input(
                    "实时/对照价",
                    min_value=0.0,
                    value=0.0 if record.get("manual_price") in [None, ""] else float(record.get("manual_price", 0)),
                    step=0.01,
                    format="%.2f",
                )
                note = form_cols[2].text_input("操作备注", value=record.get("operator_note", ""))
                submitted = form_cols[3].form_submit_button("保存标记", use_container_width=True)
                if submitted:
                    update_watchlist_status(record.get("symbol", ""), operation_status_key(chosen_status), current_price, note)
                    st.rerun()

            status_updated_at = record.get("status_updated_at", "")
            if status_updated_at:
                st.caption(f"状态更新时间：{status_updated_at}")

            risk_cols = st.columns(5, gap="medium")
            signals_for_risk = record.get("signals", [])
            risk_cols[0].markdown(metric_card("买入区间", watchlist_entry_zone_text(signals_for_risk) or "-"), unsafe_allow_html=True)
            risk_cols[1].markdown(metric_card("关键位", watchlist_trigger_text(signals_for_risk) or "-"), unsafe_allow_html=True)
            risk_cols[2].markdown(metric_card("操作失效价", watchlist_invalidation_price_text(signals_for_risk) or "-"), unsafe_allow_html=True)
            risk_cols[3].markdown(metric_card("止损参考", watchlist_stop_text(signals_for_risk) or "-"), unsafe_allow_html=True)
            risk_cols[4].markdown(metric_card("实时口径", "区间内可执行"), unsafe_allow_html=True)

            radar = record.get("radar", {}) or {}
            if radar:
                radar_cols = st.columns(4, gap="medium")
                radar_cols[0].markdown(metric_card("雷达动作", str(radar.get("预期动作", "-"))), unsafe_allow_html=True)
                radar_cols[1].markdown(metric_card("强股设置", str(radar.get("强股设置分", "-"))), unsafe_allow_html=True)
                radar_cols[2].markdown(metric_card("买点质量", str(radar.get("买点质量分", "-"))), unsafe_allow_html=True)
                radar_cols[3].markdown(metric_card("卖出风险", str(radar.get("卖出风险分", "-"))), unsafe_allow_html=True)
                st.write(f"买点结论：{radar.get('买点结论', '-')}")

            signal_rows = record.get("signals", [])
            if signal_rows:
                st.markdown("**量价信号**")
                st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)

            render_watchlist_forecast_panel(record, settings)

            level_cols = st.columns(2, gap="large")
            with level_cols[0]:
                st.markdown("**关键支撑**")
                st.dataframe(pd.DataFrame(record.get("support_levels", [])), use_container_width=True, hide_index=True)
            with level_cols[1]:
                st.markdown("**关键压力**")
                st.dataframe(pd.DataFrame(record.get("resistance_levels", [])), use_container_width=True, hide_index=True)

            explanations = record.get("score", {}).get("explanation", [])
            if explanations:
                st.markdown("**评分解释**")
                for item in explanations:
                    st.write(f"- {item}")

            action_cols = st.columns(2, gap="medium")
            if action_cols[0].button("加入已购买重点观察", key=f"buy_watchlist_{record.get('symbol', '')}", use_container_width=True):
                add_purchased_from_watchlist(record)
                st.success(f"{record.get('symbol', '')} 已加入已购买重点观察。")

            if action_cols[1].button("从备选池移除", key=f"remove_watchlist_{record.get('symbol', '')}", use_container_width=True):
                remove_watchlist_symbol(record.get("symbol", ""))
                st.rerun()


def render_watchlist_quick_delete(records: list[dict]) -> None:
    if not records:
        return

    with st.container(border=True):
        st.markdown("**快速删除备选股票**")
        symbol_options = [str(record.get("symbol", "")) for record in records if record.get("symbol")]
        label_map = {
            str(record.get("symbol", "")): (
                f"{record.get('symbol', '')} | {record.get('score', {}).get('total', 0)}分 | "
                f"{execution_status_label(watchlist_execution_status(record))} | {record.get('suggestion', '')}"
            )
            for record in records
            if record.get("symbol")
        }
        quick_cols = st.columns([0.72, 0.28], gap="medium")
        selected_symbol = quick_cols[0].selectbox(
            "选择要移除的股票",
            options=symbol_options,
            format_func=lambda symbol: label_map.get(symbol, symbol),
            key="watchlist_quick_delete_symbol",
        )
        if quick_cols[1].button("从备选池移除", key="watchlist_quick_delete_button", use_container_width=True):
            remove_watchlist_symbol(selected_symbol)
            st.success(f"{selected_symbol} 已从备选池移除。")
            st.rerun()


def render_watchlist_forecast_panel(record: dict, settings: dict) -> None:
    symbol = str(record.get("symbol", "")).strip()
    if not symbol:
        return

    st.markdown("**股价预测辅助**")
    intro_cols = st.columns([0.26, 0.74], gap="medium")
    opened_key = f"forecast_watchlist_open_{symbol}"
    if intro_cols[0].button("预测未来10日", key=f"forecast_watchlist_{symbol}", use_container_width=True):
        st.session_state[opened_key] = True
    intro_cols[1].caption("使用轻量多变量时间序列模型预测未来10个交易日，只做统计辅助，不替代量价信号和止损。")

    if not st.session_state.get(opened_key, False):
        return

    stop_loss = watchlist_best_stop_price(record.get("signals", [])) or watchlist_best_invalidation_price(record.get("signals", []))
    action = str(record.get("suggestion", ""))
    try:
        with st.spinner("正在训练轻量多变量模型并生成预测..."):
            result = cached_watchlist_forecast(
                symbol=symbol,
                end=str(settings.get("end", "")),
                adjust=str(settings.get("adjust", "qfq")),
                request_timeout=float(settings.get("request_timeout", 8.0)),
                request_retries=int(settings.get("request_retries", 1)),
                volume_price_action=action,
                stop_loss=None if stop_loss is None else float(stop_loss),
            )
    except Exception as exc:
        st.warning(short_error(exc))
        return

    metric_cols = st.columns(5, gap="medium")
    metric_cols[0].markdown(metric_card("未来10日预期涨跌", f"{result.expected_return_10d:.2%}"), unsafe_allow_html=True)
    metric_cols[1].markdown(metric_card("上涨概率", f"{result.up_probability:.0%}"), unsafe_allow_html=True)
    metric_cols[2].markdown(metric_card("模型可信度", f"{result.confidence:.0f}"), unsafe_allow_html=True)
    metric_cols[3].markdown(metric_card("模型", str(result.diagnostics.get("model_used", "-"))), unsafe_allow_html=True)
    metric_cols[4].markdown(metric_card("预测结论", result.conclusion), unsafe_allow_html=True)

    st.caption(str(result.diagnostics.get("disclaimer", "预测是统计辅助，不构成投资建议。")))
    chart_data = forecast_chart_frame(result)
    forecast_data = chart_data[chart_data["type"] == "预测中位"]
    band = (
        alt.Chart(forecast_data)
        .mark_area(opacity=0.18, color="#2563eb")
        .encode(x=alt.X("date:T", title="日期"), y=alt.Y("lower:Q", title="收盘价"), y2="upper:Q")
    )
    line = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="日期"),
            y=alt.Y("value:Q", title="收盘价"),
            color=alt.Color("type:N", title="类型"),
            tooltip=["date:T", "type:N", "value:Q", "lower:Q", "upper:Q"],
        )
    )
    st.altair_chart(band + line, use_container_width=True)
    st.dataframe(forecast_points_frame(result), use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_watchlist_forecast(
    symbol: str,
    end: str,
    adjust: str,
    request_timeout: float,
    request_retries: int,
    volume_price_action: str,
    stop_loss: float | None,
):
    provider = AkshareProvider(request_timeout=request_timeout, request_retries=request_retries)
    forecast_settings = {
        "end": end or pd.Timestamp.today().strftime("%Y%m%d"),
        "adjust": adjust or "qfq",
    }
    return forecast_symbol(
        provider,
        symbol=symbol,
        settings=forecast_settings,
        volume_price_action=volume_price_action,
        stop_loss=stop_loss,
    )


def render_purchased_tab() -> None:
    st.subheader("已购买股票")
    records = load_purchased_records()
    if not records:
        st.info("这里用于重点观察已经买入的股票。可以先在备选池详情里点击“加入已购买重点观察”。")
        return

    sorted_records = sort_purchased_records(records)
    summary = pd.DataFrame([purchased_summary_row(record) for record in sorted_records])
    st.dataframe(style_purchased_summary(summary), use_container_width=True, hide_index=True)

    export_cols = st.columns([0.5, 0.5], gap="medium")
    export_cols[0].download_button(
        "下载已购买CSV",
        data=summary.to_csv(index=False, encoding="utf-8-sig"),
        file_name="purchased.csv",
        mime="text/csv",
        use_container_width=True,
    )
    export_cols[1].download_button(
        "下载完整持仓快照JSON",
        data=json.dumps(records, ensure_ascii=False, indent=2),
        file_name="purchased.json",
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("**持仓观察详情**")
    for record in sorted_records:
        snapshot = record.get("snapshot", {}) or {}
        symbol = record.get("symbol", "")
        score = (snapshot.get("score") or {}).get("total", 0)
        status = purchased_risk_status(record)
        with st.expander(f"{symbol} | {purchased_status_label(status)} | {score}分 | {snapshot.get('suggestion', '')}"):
            render_purchased_editor(record)
            render_purchased_snapshot_detail(record)


def render_purchased_editor(record: dict) -> None:
    symbol = record.get("symbol", "")
    snapshot = record.get("snapshot", {}) or {}
    buy_price = float(record.get("buy_price") or 0)
    current_price = float(record.get("current_price") or 0)
    pnl_pct = purchased_pnl_pct(record)
    status = purchased_risk_status(record)

    cols = st.columns(5, gap="medium")
    cols[0].markdown(metric_card("买入价", "-" if buy_price <= 0 else f"{buy_price:.2f}"), unsafe_allow_html=True)
    cols[1].markdown(metric_card("当前价", "-" if current_price <= 0 else f"{current_price:.2f}"), unsafe_allow_html=True)
    cols[2].markdown(metric_card("浮盈亏", "-" if pnl_pct is None else f"{pnl_pct:.2%}"), unsafe_allow_html=True)
    cols[3].markdown(metric_card("风险状态", purchased_status_label(status)), unsafe_allow_html=True)
    cols[4].markdown(metric_card("评分", str((snapshot.get("score") or {}).get("total", 0))), unsafe_allow_html=True)

    with st.form(key=f"purchased_form_{symbol}"):
        form_cols = st.columns([0.18, 0.18, 0.16, 0.16, 0.22, 0.10], gap="medium")
        buy_date = form_cols[0].text_input("买入日期", value=record.get("buy_date", ""))
        new_buy_price = form_cols[1].number_input("买入价", min_value=0.0, value=buy_price, step=0.01, format="%.2f")
        shares = form_cols[2].number_input("股数", min_value=0, value=int(record.get("shares") or 0), step=100)
        new_current_price = form_cols[3].number_input("当前价", min_value=0.0, value=current_price, step=0.01, format="%.2f")
        note = form_cols[4].text_input("持仓备注", value=record.get("position_note", ""))
        submitted = form_cols[5].form_submit_button("保存", use_container_width=True)
        if submitted:
            update_purchased_record(symbol, buy_date, new_buy_price, shares, new_current_price, note)
            st.rerun()

    st.caption(f"加入已购买时间：{record.get('purchased_at', '-')}")


def render_purchased_snapshot_detail(record: dict) -> None:
    snapshot = record.get("snapshot", {}) or {}
    signals = snapshot.get("signals", [])
    radar = snapshot.get("radar", {}) or {}

    risk_cols = st.columns(5, gap="medium")
    risk_cols[0].markdown(metric_card("买入区间", watchlist_entry_zone_text(signals) or "-"), unsafe_allow_html=True)
    risk_cols[1].markdown(metric_card("关键位", watchlist_trigger_text(signals) or "-"), unsafe_allow_html=True)
    risk_cols[2].markdown(metric_card("操作失效价", watchlist_invalidation_price_text(signals) or "-"), unsafe_allow_html=True)
    risk_cols[3].markdown(metric_card("止损参考", watchlist_stop_text(signals) or "-"), unsafe_allow_html=True)
    risk_cols[4].markdown(metric_card("诊断日期", snapshot.get("as_of", "-")), unsafe_allow_html=True)

    if radar:
        radar_cols = st.columns(4, gap="medium")
        radar_cols[0].markdown(metric_card("雷达动作", str(radar.get("预期动作", "-"))), unsafe_allow_html=True)
        radar_cols[1].markdown(metric_card("强股设置", str(radar.get("强股设置分", "-"))), unsafe_allow_html=True)
        radar_cols[2].markdown(metric_card("买点质量", str(radar.get("买点质量分", "-"))), unsafe_allow_html=True)
        radar_cols[3].markdown(metric_card("卖出风险", str(radar.get("卖出风险分", "-"))), unsafe_allow_html=True)
        st.write(f"买点结论：{radar.get('买点结论', '-')}")
        st.write(f"卖出结论：{radar.get('卖出结论', '-')}")

    if signals:
        st.markdown("**买入时保存的量价信号**")
        st.dataframe(pd.DataFrame(signals), use_container_width=True, hide_index=True)

    level_cols = st.columns(2, gap="large")
    with level_cols[0]:
        st.markdown("**买入时关键支撑**")
        st.dataframe(pd.DataFrame(snapshot.get("support_levels", [])), use_container_width=True, hide_index=True)
    with level_cols[1]:
        st.markdown("**买入时关键压力**")
        st.dataframe(pd.DataFrame(snapshot.get("resistance_levels", [])), use_container_width=True, hide_index=True)

    explanations = (snapshot.get("score") or {}).get("explanation", [])
    if explanations:
        st.markdown("**买入时评分解释**")
        for item in explanations:
            st.write(f"- {item}")

    if st.button("从已购买移除", key=f"remove_purchased_{record.get('symbol', '')}", use_container_width=True):
        remove_purchased_symbol(record.get("symbol", ""))
        st.rerun()


def load_watchlist_records() -> list[dict]:
    path = current_user_data_path("watchlist.json")
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def save_watchlist_records(records: list[dict]) -> None:
    path = current_user_data_path("watchlist.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def load_purchased_records() -> list[dict]:
    path = current_user_data_path("purchased.json")
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def save_purchased_records(records: list[dict]) -> None:
    path = current_user_data_path("purchased.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def add_purchased_from_watchlist(record: dict) -> None:
    symbol = record.get("symbol", "")
    existing_records = load_purchased_records()
    existing = next((item for item in existing_records if item.get("symbol") == symbol), {})
    purchased = {
        "symbol": symbol,
        "purchased_at": existing.get("purchased_at") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "buy_date": existing.get("buy_date") or pd.Timestamp.today().strftime("%Y-%m-%d"),
        "buy_price": existing.get("buy_price") or record.get("manual_price") or record.get("close") or 0,
        "shares": existing.get("shares") or 0,
        "current_price": existing.get("current_price") or record.get("manual_price") or record.get("close") or 0,
        "position_note": existing.get("position_note", ""),
        "snapshot": record,
    }
    records = [item for item in existing_records if item.get("symbol") != symbol]
    records.append(purchased)
    save_purchased_records(records)


def update_purchased_record(symbol: str, buy_date: str, buy_price: float, shares: int, current_price: float, note: str) -> None:
    records = load_purchased_records()
    for record in records:
        if record.get("symbol") == symbol:
            record["buy_date"] = buy_date.strip()
            record["buy_price"] = float(buy_price)
            record["shares"] = int(shares)
            record["current_price"] = float(current_price)
            record["position_note"] = note.strip()
            record["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            break
    save_purchased_records(records)


def remove_purchased_symbol(symbol: str) -> None:
    records = [item for item in load_purchased_records() if item.get("symbol") != symbol]
    save_purchased_records(records)


def sort_purchased_records(records: list[dict]) -> list[dict]:
    return sorted(
        records,
        key=lambda item: (
            int(((item.get("snapshot") or {}).get("score") or {}).get("total", 0)),
            item.get("purchased_at", ""),
            item.get("symbol", ""),
        ),
        reverse=True,
    )


def save_watchlist_result(result: AnalysisResult, source: str) -> None:
    record = analysis_result_to_watchlist_record(result, source)
    existing_records = load_watchlist_records()
    existing = next((item for item in existing_records if item.get("symbol") == result.symbol), {})
    for key in ["operation_status", "status_updated_at", "manual_price", "operator_note"]:
        if key in existing:
            record[key] = existing[key]
    records = [item for item in existing_records if item.get("symbol") != result.symbol]
    records.append(record)
    save_watchlist_records(records)


def remove_watchlist_symbol(symbol: str) -> None:
    records = [item for item in load_watchlist_records() if item.get("symbol") != symbol]
    save_watchlist_records(records)


def refresh_watchlist_records(settings: dict, provider: AkshareProvider) -> tuple[int, list[tuple[str, str]]]:
    records = load_watchlist_records()
    if not records:
        return 0, []

    benchmark = load_benchmark_for_ui(provider, settings)
    refreshed_records: list[dict] = []
    failed_items: list[tuple[str, str]] = []
    progress = st.progress(0)
    status = st.empty()
    fetch_start = warmup_start(str(settings.get("start", "")))
    end = str(settings.get("end", ""))
    adjust = str(settings.get("adjust", "qfq"))
    now_text = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    for index, record in enumerate(records, start=1):
        symbol = str(record.get("symbol", "")).strip()
        status.write(f"正在更新备选池 {index}/{len(records)}：{symbol}")
        if not symbol:
            failed_items.append(("-", "缺少股票代码"))
            refreshed_records.append(record)
            progress.progress(index / max(len(records), 1))
            continue
        try:
            df = provider.daily(symbol, fetch_start, end, adjust)
            result = analyze_for_ui(df, symbol=symbol, benchmark=benchmark)
            refreshed = merge_watchlist_refresh_record(record, result, now_text)
            refreshed_records.append(refreshed)
        except Exception as exc:
            record["refresh_error"] = short_error(exc)
            record["last_refreshed_at"] = now_text
            refreshed_records.append(record)
            failed_items.append((symbol, short_error(exc)))
        progress.progress(index / max(len(records), 1))

    save_watchlist_records(refreshed_records)
    progress.empty()
    status.empty()
    return len(records) - len(failed_items), failed_items


def merge_watchlist_refresh_record(old_record: dict, result: AnalysisResult, refreshed_at: str) -> dict:
    refreshed = analysis_result_to_watchlist_record(result, source=str(old_record.get("source", "备选池一键更新") or "备选池一键更新"))
    for key in [
        "saved_at",
        "operation_status",
        "status_updated_at",
        "manual_price",
        "operator_note",
        "execution_status",
    ]:
        if key in old_record:
            refreshed[key] = old_record.get(key)
    refreshed["last_refreshed_at"] = refreshed_at
    refreshed["refresh_error"] = ""
    return refreshed


def sort_watchlist_records(records: list[dict]) -> list[dict]:
    return sorted(
        records,
        key=lambda item: (
            int((item.get("score") or {}).get("total", 0)),
            item.get("saved_at", ""),
            item.get("symbol", ""),
        ),
        reverse=True,
    )


def purchased_summary_row(record: dict) -> dict:
    snapshot = record.get("snapshot", {}) or {}
    score = snapshot.get("score", {}) or {}
    pnl_pct = purchased_pnl_pct(record)
    return {
        "代码": record.get("symbol", ""),
        "评分": score.get("total", 0),
        "加入时间": record.get("purchased_at", ""),
        "买入日期": record.get("buy_date", ""),
        "买入价": "" if not record.get("buy_price") else round(float(record.get("buy_price", 0)), 2),
        "当前价": "" if not record.get("current_price") else round(float(record.get("current_price", 0)), 2),
        "股数": int(record.get("shares") or 0),
        "浮盈亏": "" if pnl_pct is None else f"{pnl_pct:.2%}",
        "风险状态": purchased_status_label(purchased_risk_status(record)),
        "操作失效价": watchlist_invalidation_price_text(snapshot.get("signals", [])),
        "止损参考": watchlist_stop_text(snapshot.get("signals", [])),
        "雷达动作": (snapshot.get("radar") or {}).get("预期动作", ""),
        "建议": snapshot.get("suggestion", ""),
        "持仓备注": record.get("position_note", ""),
    }


def purchased_pnl_pct(record: dict) -> float | None:
    buy_price = float(record.get("buy_price") or 0)
    current_price = float(record.get("current_price") or 0)
    if buy_price <= 0 or current_price <= 0:
        return None
    return current_price / buy_price - 1


def purchased_risk_status(record: dict) -> str:
    current_price = record.get("current_price")
    if current_price in [None, ""] or float(current_price) <= 0:
        return "unpriced"
    snapshot = record.get("snapshot", {}) or {}
    invalidation_price = watchlist_best_invalidation_price(snapshot.get("signals", []))
    stop_price = watchlist_best_stop_price(snapshot.get("signals", []))
    price = float(current_price)
    if invalidation_price is not None and price < invalidation_price:
        return "invalid"
    if stop_price is not None and price < stop_price:
        return "stop_risk"
    return "holding"


def purchased_status_label(status: str) -> str:
    return {
        "invalid": "信号失效",
        "stop_risk": "跌破止损",
        "holding": "重点观察",
        "unpriced": "未填现价",
    }.get(status or "unpriced", "未填现价")


def style_purchased_summary(summary: pd.DataFrame):
    def style_row(row) -> list[str]:
        status = row.get("风险状态", "")
        if status in {"信号失效", "跌破止损"}:
            return ["background-color: #fee2e2; color: #7f1d1d"] * len(row)
        if status == "重点观察":
            return ["background-color: #dcfce7; color: #14532d"] * len(row)
        return ["background-color: #f8fafc; color: #334155"] * len(row)

    return summary.style.apply(style_row, axis=1)


def update_watchlist_status(symbol: str, status: str, manual_price: float | None, note: str) -> None:
    records = load_watchlist_records()
    for record in records:
        if record.get("symbol") == symbol:
            record["operation_status"] = status
            record["status_updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            record["manual_price"] = None if manual_price is None or manual_price <= 0 else float(manual_price)
            record["operator_note"] = note.strip()
            break
    save_watchlist_records(records)


def analysis_result_to_watchlist_record(result: AnalysisResult, source: str) -> dict:
    long_view = getattr(result, "long_term_view", None)
    short_view = getattr(result, "short_term_view", None)
    factor_scores = getattr(result, "factor_scores", None)
    return {
        "symbol": result.symbol,
        "source": source,
        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation_status": "unmarked",
        "status_updated_at": "",
        "manual_price": None,
        "operator_note": "",
        "execution_status": "unpriced",
        "as_of": result.as_of.strftime("%Y-%m-%d"),
        "close": float(result.close),
        "suggestion": result.suggestion,
        "long_term_view": {} if long_view is None else {
            "score": long_view.score,
            "rating": long_view.rating,
            "advice": long_view.advice,
            "key_factors": long_view.key_factors,
            "risk_warnings": long_view.risk_warnings,
            "explanation": long_view.explanation,
        },
        "short_term_view": {} if short_view is None else {
            "liangjia_score": short_view.liangjia_score,
            "short_term_score": short_view.short_term_score,
            "signal_type": short_view.signal_type,
            "signal_strength": short_view.signal_strength,
            "signal_direction": short_view.signal_direction,
            "advice": short_view.advice,
            "action_code": short_view.action_code,
            "entry_zone": short_view.entry_zone,
            "stop_loss": short_view.stop_loss,
            "risk_pct": short_view.risk_pct,
            "key_factors": short_view.key_factors,
            "risk_warnings": short_view.risk_warnings,
            "explanation": short_view.explanation,
        },
        "factor_scores": {} if factor_scores is None else {
            "overall_score": factor_scores.overall_score,
            "liangjia_score": factor_scores.liangjia_score,
            "short_term_score": factor_scores.short_term_score,
            "long_term_score": factor_scores.long_term_score,
        },
        "structure": {
            "stage": stage_label(result.structure.stage),
            "trend": result.structure.trend,
            "description": result.structure.description,
            "confidence": float(result.structure.confidence),
            "metrics": result.structure.metrics,
        },
        "score": {
            "total": int(result.score.total),
            "structure": int(result.score.structure),
            "position": int(result.score.position),
            "signal": int(result.score.signal),
            "risk": int(result.score.risk),
            "relative_strength": int(result.score.relative_strength),
            "explanation": result.score.explanation,
        },
        "signals": [signal_to_record(signal) for signal in result.signals],
        "radar": radar_to_record(getattr(result, "radar", None)),
        "support_levels": [level_to_record(level) for level in result.support_levels],
        "resistance_levels": [level_to_record(level) for level in result.resistance_levels],
    }


def signal_to_record(signal) -> dict:
    trigger_level = getattr(signal, "trigger_level", None)
    invalidation_price = getattr(signal, "invalidation_price", None)
    risk_buffer_pct = getattr(signal, "risk_buffer_pct", None)
    invalidation_basis = getattr(signal, "invalidation_basis", None)
    return {
        "名称": signal.name,
        "方向": signal.side.value,
        "强度": int(signal.strength),
        "逻辑": signal.logic,
        "证据": "；".join(signal.evidence),
        "止损": None if signal.stop_loss is None else round(float(signal.stop_loss), 3),
        "买入区间": "" if signal.entry_zone is None else f"{signal.entry_zone[0]:.2f} - {signal.entry_zone[1]:.2f}",
        "失效条件": signal.invalidation or "",
        "关键位": None if trigger_level is None else round(float(trigger_level), 3),
        "操作失效价": None if invalidation_price is None else round(float(invalidation_price), 3),
        "风险缓冲": None if risk_buffer_pct is None else round(float(risk_buffer_pct), 4),
        "失效依据": invalidation_basis or "",
    }


def radar_to_record(radar) -> dict:
    if radar is None:
        return {}
    return {
        "市场状态": radar.market_state,
        "板块名称": radar.sector_name,
        "板块强度分位": radar.sector_rs_rank,
        "个股相对板块强度": radar.stock_vs_sector_rs,
        "强股设置分": radar.setup_score,
        "买点质量分": radar.entry_quality_score,
        "卖出风险分": radar.exit_risk_score,
        "预期动作": radar.expected_action,
        "排除原因": radar.reject_reason,
        "好股票结论": radar.good_stock_conclusion,
        "买点结论": radar.entry_conclusion,
        "卖出结论": radar.exit_conclusion,
        "20日相对强度": radar.stock_rs_20,
        "60日相对强度": radar.stock_rs_60,
        "盈亏比": radar.reward_risk,
        "止损距离": radar.stop_distance_pct,
    }


def level_to_record(level) -> dict:
    return {
        "名称": level.name,
        "价格": round(float(level.price), 3),
        "类型": level.kind,
        "日期": "" if level.date is None else pd.to_datetime(level.date).strftime("%Y-%m-%d"),
        "权重": float(level.weight),
    }


def watchlist_summary_row(record: dict) -> dict:
    score = record.get("score", {})
    structure = record.get("structure", {})
    signals = record.get("signals", [])
    status = watchlist_display_status(record)
    execution = watchlist_execution_status(record)
    radar = record.get("radar", {}) or {}
    return {
        "代码": record.get("symbol", ""),
        "评分": score.get("total", 0),
        "加入时间": record.get("saved_at", ""),
        "操作状态": execution_status_label(execution),
        "风控状态": operation_status_label(status),
        "实时/对照价": price_text(record.get("manual_price")),
        "买入区间": watchlist_entry_zone_text(signals),
        "关键位": watchlist_trigger_text(signals),
        "操作失效价": watchlist_invalidation_price_text(signals),
        "止损参考": watchlist_stop_text(signals),
        "雷达动作": radar.get("预期动作", ""),
        "买点质量": radar.get("买点质量分", ""),
        "失效条件": watchlist_invalidation_text(signals),
        "状态更新时间": record.get("status_updated_at", ""),
        "信号日期": record.get("as_of", ""),
        "收盘": round(float(record.get("close", 0)), 2),
        "结构": structure.get("stage", ""),
        "建议": record.get("suggestion", ""),
        "信号": "、".join(item.get("名称", "") for item in signals),
        "来源": record.get("source", ""),
    }


def watchlist_display_status(record: dict) -> str:
    manual_status = record.get("operation_status", "unmarked") or "unmarked"
    if manual_status != "unmarked":
        return manual_status
    manual_price = record.get("manual_price")
    invalidation_price = watchlist_best_invalidation_price(record.get("signals", []))
    if manual_price in [None, ""] or invalidation_price is None:
        return "unmarked"
    return "invalid" if float(manual_price) < invalidation_price else "valid"


def watchlist_execution_status(record: dict) -> str:
    manual_price = record.get("manual_price")
    if manual_price in [None, ""]:
        return "unpriced"
    price = float(manual_price)
    invalidation_price = watchlist_best_invalidation_price(record.get("signals", []))
    if invalidation_price is not None and price < invalidation_price:
        return "invalid"
    zone = watchlist_best_entry_zone(record.get("signals", []))
    if zone is None:
        return "valid_no_zone" if invalidation_price is not None else "unpriced"
    low, high = zone
    if price < low:
        return "waiting"
    if price <= high:
        return "actionable"
    return "extended"


def execution_status_label(status: str) -> str:
    return {
        "invalid": "已失效",
        "waiting": "未到买点",
        "actionable": "可执行",
        "extended": "偏离买点",
        "valid_no_zone": "仍有效",
        "unpriced": "未填价格",
    }.get(status or "unpriced", "未填价格")


def operation_status_label(status: str) -> str:
    return {"valid": "仍有效", "invalid": "已失效", "unmarked": "未标记"}.get(status or "unmarked", "未标记")


def operation_status_key(label: str) -> str:
    return {"仍有效": "valid", "已失效": "invalid", "未标记": "unmarked"}.get(label, "unmarked")


def style_watchlist_summary(summary: pd.DataFrame):
    def style_row(row) -> list[str]:
        status = row.get("操作状态", "")
        if status == "已失效":
            return ["background-color: #fee2e2; color: #7f1d1d"] * len(row)
        if status in {"可执行", "仍有效"}:
            return ["background-color: #dcfce7; color: #14532d"] * len(row)
        if status == "未到买点":
            return ["background-color: #fef9c3; color: #713f12"] * len(row)
        if status == "偏离买点":
            return ["background-color: #ffedd5; color: #7c2d12"] * len(row)
        return ["background-color: #f8fafc; color: #334155"] * len(row)

    return summary.style.apply(style_row, axis=1)


def watchlist_stop_text(signals: list[dict]) -> str:
    stops = []
    for signal in signals:
        stop = signal.get("止损")
        if stop not in [None, ""]:
            stops.append(str(stop))
    return " / ".join(stops[:2])


def watchlist_entry_zone_text(signals: list[dict]) -> str:
    zones = []
    for signal in signals:
        zone = signal.get("买入区间")
        if zone:
            zones.append(str(zone))
    return " / ".join(zones[:2])


def watchlist_best_entry_zone(signals: list[dict]) -> tuple[float, float] | None:
    zones = []
    for signal in signals:
        zone = signal.get("买入区间")
        parsed = parse_entry_zone(zone)
        if parsed is not None:
            zones.append(parsed)
    if not zones:
        return None
    return min(zones, key=lambda item: item[1])


def parse_entry_zone(value) -> tuple[float, float] | None:
    if not value:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]), float(value[1])
    text = str(value).replace("－", "-").replace("—", "-").replace("~", "-")
    parts = [part.strip() for part in text.split("-")]
    if len(parts) != 2:
        return None
    try:
        low = float(parts[0])
        high = float(parts[1])
    except ValueError:
        return None
    return (min(low, high), max(low, high))


def watchlist_trigger_text(signals: list[dict]) -> str:
    items = []
    for signal in signals:
        value = signal.get("关键位")
        if value not in [None, ""]:
            items.append(f"{float(value):.2f}")
    return " / ".join(items[:2])


def watchlist_invalidation_price_text(signals: list[dict]) -> str:
    items = []
    for signal in signals:
        value = signal.get("操作失效价")
        if value not in [None, ""]:
            items.append(f"{float(value):.2f}")
    return " / ".join(items[:2])


def watchlist_best_invalidation_price(signals: list[dict]) -> float | None:
    prices = []
    for signal in signals:
        value = signal.get("操作失效价")
        if value not in [None, ""]:
            prices.append(float(value))
    return max(prices) if prices else None


def watchlist_best_stop_price(signals: list[dict]) -> float | None:
    prices = []
    for signal in signals:
        value = signal.get("止损")
        if value not in [None, ""]:
            prices.append(float(value))
    return max(prices) if prices else None


def watchlist_invalidation_text(signals: list[dict]) -> str:
    items = []
    for signal in signals:
        invalidation = signal.get("失效条件", "")
        if invalidation:
            items.append(str(invalidation))
    return "；".join(items[:2])


_render_single_backtest_tab = render_backtest_tab


def render_backtest_tab(settings: dict, provider: AkshareProvider) -> None:
    single_tab, portfolio_tab = st.tabs(["单票回测", "组合回测"])
    with single_tab:
        _render_single_backtest_tab(settings, provider)
    with portfolio_tab:
        render_portfolio_backtest_panel(settings, provider)


def render_portfolio_backtest_panel(settings: dict, provider: AkshareProvider) -> None:
    st.subheader("组合回测")
    st.caption("组合回测使用优化后的趋势回踩算法：先看市场状态，再从股票池里筛选候选、控制持仓数量，并按止损/移动止盈/信号退出。")

    with st.container(border=True):
        left, right = st.columns([1.0, 1.2], gap="large")

        with left:
            source = st.radio(
                "股票池来源",
                ["手动代码", "内置股票池", "行业板块", "概念板块", "本地CSV文件夹"],
                key="portfolio_source",
            )
            limit = st.number_input("最多回测标的数", min_value=0, value=20, step=10, help="0 表示不限制；在线数据建议先用 10-50 只验证。")
            symbols_text = ""
            pool = "沪深300"
            board_name = ""
            csv_dir = ""
            if source == "手动代码":
                symbols_text = st.text_area("股票代码", value="000001 600519 300750", help="可用空格、逗号、换行分隔。", key="portfolio_symbols")
            elif source == "内置股票池":
                pool = st.selectbox("股票池", ["全部A股", "上证50", "沪深300", "中证500", "上交所", "深交所", "创业板", "科创板"], index=2, key="portfolio_pool")
            elif source == "行业板块":
                board_name = st.text_input("行业名称", value="银行", key="portfolio_industry")
            elif source == "概念板块":
                board_name = st.text_input("概念名称", value="人工智能", key="portfolio_concept")
            else:
                csv_dir = st.text_input("CSV文件夹路径", value="data/daily", key="portfolio_csv_dir")

            benchmark_symbol = st.text_input("基准指数代码", value=default_benchmark_symbol(pool), help="常用：000300 沪深300，000016 上证50，000905 中证500，399006 创业板指。")
            benchmark_csv = st.text_input("基准CSV路径（可选）", value="", help="留空时使用在线指数数据。")

        with right:
            row1 = st.columns(4, gap="medium")
            max_positions = row1[0].number_input("最大持仓", min_value=1, max_value=30, value=5, step=1)
            neutral_max_positions = row1[1].number_input("中性市持仓", min_value=0, max_value=30, value=2, step=1)
            initial_cash = row1[2].number_input("初始资金", min_value=10000.0, value=100000.0, step=10000.0, key="portfolio_cash")
            min_score = row1[3].slider("最低候选分", 40, 95, 75, 5, key="portfolio_min_score")

            row2 = st.columns(4, gap="medium")
            market_mode_label = row2[0].selectbox("市场过滤", ["均衡", "严格", "积极"], index=0)
            min_avg_amount = row2[1].number_input("20日最低成交额", min_value=0.0, value=50_000_000.0, step=10_000_000.0, format="%.0f")
            min_rs = row2[2].slider("60日最小相对强度", -0.20, 0.30, 0.03, 0.01, format="%.2f")
            min_reward_risk = row2[3].slider("最小盈亏比", 1.0, 4.0, 1.8, 0.1, format="%.1f")

            row3 = st.columns(4, gap="medium")
            max_stop_distance = row3[0].slider("最大止损距离", 0.03, 0.15, 0.07, 0.01, format="%.2f")
            breakeven_r = row3[1].slider("保本止损R", 0.5, 3.0, 1.0, 0.1, format="%.1f")
            trail_start_r = row3[2].slider("移动止盈R", 1.0, 5.0, 2.0, 0.1, format="%.1f")
            trail_pct = row3[3].slider("回撤止盈比例", 0.03, 0.25, 0.10, 0.01, format="%.2f")

            row4 = st.columns(5, gap="medium")
            stale_days = row4[0].number_input("无效持仓天数", min_value=3, max_value=60, value=12, step=1)
            max_holding_days = row4[1].number_input("最长持仓天数", min_value=5, max_value=250, value=45, step=5)
            fee_rate = row4[2].number_input("佣金率", min_value=0.0, value=0.0003, step=0.0001, format="%.4f", key="portfolio_fee")
            stamp_tax = row4[3].number_input("印花税", min_value=0.0, value=0.0005, step=0.0001, format="%.4f", key="portfolio_tax")
            slippage = row4[4].number_input("滑点", min_value=0.0, value=0.0010, step=0.0005, format="%.4f", key="portfolio_slippage")

        run = st.button("运行组合回测", type="primary", use_container_width=True)

    if not run:
        st.info("建议先用手动代码或较小股票池验证参数。组合回测会逐日评估候选，标的越多、区间越长，耗时越明显。")
        return

    market_mode = {"严格": "strict", "均衡": "balanced", "积极": "aggressive"}[market_mode_label]
    config = PortfolioBacktestConfig(
        max_positions=int(max_positions),
        neutral_max_positions=int(neutral_max_positions),
        initial_cash=float(initial_cash),
        fee_rate=float(fee_rate),
        stamp_tax=float(stamp_tax),
        slippage=float(slippage),
        min_avg_amount_20=float(min_avg_amount),
        max_stop_distance_pct=float(max_stop_distance),
        min_relative_strength=float(min_rs),
        min_reward_risk=float(min_reward_risk),
        trailing_stop_pct=float(trail_pct),
        stale_days=int(stale_days),
        max_holding_days=int(max_holding_days),
        breakeven_r=float(breakeven_r),
        trail_start_r=float(trail_start_r),
        min_score=int(min_score),
        market_mode=market_mode,
    )

    progress = st.progress(0.0)
    status = st.empty()
    progress_lines: list[str] = []

    try:
        fetch_start = warmup_start(settings["start"])
        data_by_symbol = load_portfolio_data_for_ui(
            source=source,
            symbols_text=symbols_text,
            pool=pool,
            board_name=board_name,
            csv_dir=csv_dir,
            limit=int(limit),
            settings=settings,
            provider=provider,
            fetch_start=fetch_start,
            status=status,
            progress=progress,
        )
        if not data_by_symbol:
            raise ValueError("没有可用于组合回测的股票数据。")

        benchmark = load_portfolio_benchmark_for_ui(
            benchmark_symbol=benchmark_symbol,
            benchmark_csv=benchmark_csv,
            fetch_start=fetch_start,
            settings=settings,
            provider=provider,
        )

        def on_progress(message: str) -> None:
            progress_lines.append(message)
            if len(progress_lines) > 6:
                progress_lines.pop(0)
            status.write("\n\n".join(progress_lines))

        result = portfolio_backtest(
            data_by_symbol,
            benchmark,
            settings["start"],
            settings["end"],
            portfolio_config=config,
            progress_callback=on_progress,
        )
        progress.progress(1.0)
        status.success("组合回测完成。")
    except Exception as exc:
        st.error(short_error(exc))
        return

    render_portfolio_result(result, initial_cash)


def load_portfolio_data_for_ui(
    source: str,
    symbols_text: str,
    pool: str,
    board_name: str,
    csv_dir: str,
    limit: int,
    settings: dict,
    provider: AkshareProvider,
    fetch_start: str,
    status,
    progress,
) -> dict[str, pd.DataFrame]:
    sources = resolve_scan_sources(source, symbols_text, pool, board_name, csv_dir, provider)
    if limit:
        sources = sources[:limit]

    data_by_symbol: dict[str, pd.DataFrame] = {}
    total = len(sources)
    for index, (symbol, source_ref) in enumerate(sources, start=1):
        status.write(f"正在加载 {index}/{total}：{symbol}")
        try:
            if isinstance(source_ref, Path):
                data = CsvProvider(source_ref).daily(symbol=symbol, start=fetch_start, end=settings["end"], adjust=settings["adjust"])
            else:
                data = provider.daily(symbol, fetch_start, settings["end"], settings["adjust"])
            if not data.empty:
                data_by_symbol[symbol] = data
        except Exception as exc:
            status.write(f"{symbol} 跳过：{short_error(exc)}")
        progress.progress(index / max(total, 1) * 0.35)
    return data_by_symbol


def load_portfolio_benchmark_for_ui(
    benchmark_symbol: str,
    benchmark_csv: str,
    fetch_start: str,
    settings: dict,
    provider: AkshareProvider,
) -> pd.DataFrame:
    symbol = (benchmark_symbol or "000300").strip()
    if benchmark_csv.strip():
        return CsvProvider(benchmark_csv.strip()).daily(symbol=symbol, start=fetch_start, end=settings["end"], adjust=settings["adjust"])
    return provider.index_daily(symbol, fetch_start, settings["end"])


def render_portfolio_result(result, initial_cash: float) -> None:
    render_portfolio_metrics(result.metrics, initial_cash)
    if not result.equity.empty:
        render_portfolio_equity_chart(result.equity, initial_cash)

    detail_tabs = st.tabs(["交易记录", "每日持仓", "候选诊断", "净值数据", "指标明细"])
    with detail_tabs[0]:
        render_dataframe_download(result.trades, "组合交易记录", "portfolio_trades.csv")
    with detail_tabs[1]:
        render_dataframe_download(result.positions, "每日持仓", "portfolio_positions.csv")
    with detail_tabs[2]:
        candidates = result.candidates.copy()
        if not candidates.empty:
            sort_cols = [col for col in ["date", "rank", "ranking_score"] if col in candidates.columns]
            if sort_cols:
                candidates = candidates.sort_values(sort_cols, ascending=[False, True, False][: len(sort_cols)])
        render_dataframe_download(candidates, "候选诊断", "portfolio_candidates.csv")
    with detail_tabs[3]:
        render_dataframe_download(result.equity, "净值数据", "portfolio_equity.csv")
    with detail_tabs[4]:
        metrics_frame = pd.DataFrame([result.metrics])
        st.dataframe(metrics_frame, use_container_width=True, hide_index=True)
        st.download_button(
            "下载指标CSV",
            data=metrics_frame.to_csv(index=False, encoding="utf-8-sig"),
            file_name="portfolio_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_portfolio_metrics(metrics: dict[str, float], initial_cash: float) -> None:
    final_equity = metrics.get("final_equity", metrics.get("final_cash", metrics.get("ending_equity", 0)))
    total_return = metrics.get("total_return", final_equity / initial_cash - 1 if initial_cash and final_equity else 0)
    max_drawdown = metrics.get("max_drawdown", 0)
    trade_count = metrics.get("trade_count", metrics.get("trades", 0))
    win_rate = metrics.get("win_rate", 0)
    avg_r = metrics.get("avg_r_multiple", metrics.get("avg_trade_return", 0))

    cols = st.columns(6)
    cols[0].markdown(metric_card("期末权益", f"{final_equity:,.0f}"), unsafe_allow_html=True)
    cols[1].markdown(metric_card("总收益", f"{total_return:.2%}"), unsafe_allow_html=True)
    cols[2].markdown(metric_card("最大回撤", f"{max_drawdown:.2%}"), unsafe_allow_html=True)
    cols[3].markdown(metric_card("交易次数", str(int(trade_count))), unsafe_allow_html=True)
    cols[4].markdown(metric_card("胜率", f"{win_rate:.2%}"), unsafe_allow_html=True)
    cols[5].markdown(metric_card("平均R/单笔", f"{avg_r:.2f}"), unsafe_allow_html=True)


def render_portfolio_equity_chart(equity: pd.DataFrame, initial_cash: float) -> None:
    st.subheader("组合净值曲线")
    chart_data = equity.copy()
    chart_data["date"] = pd.to_datetime(chart_data["date"])
    value_col = "total_equity" if "total_equity" in chart_data.columns else "cash"
    chart_data["return"] = chart_data[value_col] / initial_cash - 1
    chart_data["drawdown"] = chart_data[value_col] / chart_data[value_col].cummax() - 1

    line = (
        alt.Chart(chart_data)
        .mark_line(color="#2563eb", strokeWidth=2.2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y(f"{value_col}:Q", title="权益", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                alt.Tooltip(f"{value_col}:Q", title="权益", format=",.2f"),
                alt.Tooltip("return:Q", title="收益率", format=".2%"),
                alt.Tooltip("drawdown:Q", title="回撤", format=".2%"),
                alt.Tooltip("positions:Q", title="持仓数"),
                alt.Tooltip("market_state:N", title="市场状态"),
            ],
        )
        .properties(height=300)
    )
    drawdown = (
        alt.Chart(chart_data)
        .mark_area(color="#94a3b8", opacity=0.35)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("drawdown:Q", title="回撤", axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("date:T", title="日期", format="%Y-%m-%d"),
                alt.Tooltip("drawdown:Q", title="回撤", format=".2%"),
            ],
        )
        .properties(height=120)
    )
    st.altair_chart(alt.vconcat(line, drawdown).resolve_scale(x="shared").configure_view(strokeWidth=0), use_container_width=True)


def render_dataframe_download(frame: pd.DataFrame, label: str, filename: str) -> None:
    if frame.empty:
        st.info(f"暂无{label}。")
        return
    display = frame.copy()
    if "date" in display.columns:
        display["date"] = pd.to_datetime(display["date"]).dt.date
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button(
        f"下载{label}CSV",
        data=display.to_csv(index=False, encoding="utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def warmup_start(start: str) -> str:
    try:
        start_date = pd.to_datetime(start, format="%Y%m%d")
    except ValueError:
        start_date = pd.to_datetime(start)
    return (start_date - pd.Timedelta(days=420)).strftime("%Y%m%d")


def default_benchmark_symbol(pool: str) -> str:
    alias = pool_alias(pool) if pool else ""
    if alias == "sse50":
        return "000016"
    if alias == "csi500":
        return "000905"
    if alias == "chinext":
        return "399006"
    return "000300"


def load_benchmark_for_ui(provider: AkshareProvider, settings: dict) -> pd.DataFrame | None:
    try:
        return provider.index_daily("000300", warmup_start(settings["start"]), settings["end"])
    except Exception:
        return None


def load_board_for_ui(provider: AkshareProvider, board_name: str, settings: dict) -> pd.DataFrame | None:
    if not board_name:
        return None
    try:
        return provider.board_daily(board_name, warmup_start(settings["start"]), settings["end"])
    except Exception:
        return None


def sector_rank_hint(sector: pd.DataFrame | None, benchmark: pd.DataFrame | None) -> float | None:
    if sector is None or benchmark is None or sector.empty or benchmark.empty or len(sector) < 60 or len(benchmark) < 60:
        return None
    sector_return = float(sector["close"].iloc[-1] / sector["close"].iloc[-60] - 1)
    benchmark_return = float(benchmark["close"].iloc[-1] / benchmark["close"].iloc[-60] - 1)
    spread = sector_return - benchmark_return
    if spread >= 0.10:
        return 0.90
    if spread >= 0.03:
        return 0.70
    if spread >= -0.03:
        return 0.50
    return 0.25


# Final Chinese rendering overrides for the single-stock long/short summary.
def _render_split_conclusion_summary(result: AnalysisResult) -> None:
    long_view = getattr(result, "long_term_view", None)
    short_view = getattr(result, "short_term_view", None)
    if long_view is None and short_view is None:
        return

    st.markdown("**长期与短期结论总结**")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown(f"**长期投资价值：{long_view.advice if long_view else '-'}**")
            if long_view is None:
                st.write("长期数据不足，暂时无法形成中长期判断。")
            else:
                for line in _long_conclusion_lines(result, long_view):
                    st.write(line)
    with right:
        with st.container(border=True):
            st.markdown(f"**短期交易机会：{short_view.advice if short_view else '-'}**")
            if short_view is None:
                st.write("短期量价数据不足，暂时无法形成交易判断。")
            else:
                for line in _short_conclusion_lines(result, short_view):
                    st.write(line)


def _long_conclusion_lines(result: AnalysisResult, long_view) -> list[str]:
    factors = getattr(result, "factors", None)
    long_factors = getattr(factors, "long_term", {}) or {}
    lines = [
        f"- 评分与评级：长期分 {long_view.score:.0f}，评级为「{long_view.rating}」。",
        f"- 核心判断：{_long_action_sentence(long_view.advice)}",
    ]
    key_text = _join_limited(long_view.key_factors, 3)
    if key_text:
        lines.append(f"- 主要依据：{key_text}。")
    factor_text = _long_factor_snapshot(long_factors)
    if factor_text:
        lines.append(f"- 指标参考：{factor_text}。")
    if long_view.risk_warnings:
        lines.append(f"- 风险提示：{_join_limited(long_view.risk_warnings, 3)}。")
    else:
        lines.append("- 风险提示：当前长期模块没有识别到特别突出的长期风险，但仍需结合财报和行业变化复核。")
    lines.append(
        "- 操作边界：长期判断只回答“是否值得中长期关注”，不代表今天短期可以买入。"
        "短期买卖仍以量价信号、买点质量和止损条件为准。"
    )
    return lines


def _short_conclusion_lines(result: AnalysisResult, short_view) -> list[str]:
    lines = [
        f"- 评分与方向：短期分 {short_view.short_term_score:.0f}，量价分 {short_view.liangjia_score:.0f}，"
        f"信号方向「{short_view.signal_direction}」。",
        f"- 核心判断：{_short_action_sentence(short_view)}",
    ]
    if short_view.signal_type:
        lines.append(f"- 当前信号：{short_view.signal_type}，信号强度 {short_view.signal_strength}。")
    if short_view.entry_zone or short_view.stop_loss or short_view.risk_pct is not None:
        stop_text = "-" if short_view.stop_loss is None else f"{short_view.stop_loss:.2f}"
        risk_text = "-" if short_view.risk_pct is None else f"{short_view.risk_pct:.1%}"
        lines.append(
            f"- 交易边界：买入区间 {_entry_zone_text(short_view.entry_zone)}，止损位 {stop_text}，风险比例 {risk_text}。"
        )
    key_text = _join_limited(short_view.key_factors, 3)
    if key_text:
        lines.append(f"- 主要依据：{key_text}。")
    if short_view.risk_warnings:
        lines.append(f"- 风险提示：{_join_limited(short_view.risk_warnings, 3)}。")
    else:
        lines.append("- 风险提示：当前短期模块没有识别到强卖出或回避风险，但若跌破关键位或放量转弱，应重新评估。")
    lines.append(
        "- 执行原则：短期买入或加仓必须来自量价买点；如果只有短期分高但量价买点不明确，优先等待回踩或突破确认。"
    )
    return lines


def _long_action_sentence(advice: str) -> str:
    if advice == "长期可关注":
        return "长期趋势、质量或估值条件相对较好，可以进入中长期跟踪池，但仍要等待合适短期买点。"
    if advice == "长期谨慎关注":
        return "有一定长期观察价值，但优势不够充分，适合小仓位观察或继续等待基本面和趋势确认。"
    if advice == "长期暂不关注":
        return "长期吸引力不足，暂时不适合作为核心配置对象。"
    if advice == "长期回避":
        return "长期风险或弱势特征较明显，不建议作为中长期配置对象。"
    return "长期结论偏中性，需要继续结合财务、估值和行业趋势观察。"


def _short_action_sentence(short_view) -> str:
    advice = short_view.advice
    if advice == "短期买入":
        return "当前已有量价买点，若价格仍在计划区间内，可按止损约束考虑执行。"
    if advice == "短期加仓":
        return "当前处于较强量价买点或趋势延续状态，已有仓位可关注加仓机会，但不能放松止损。"
    if advice == "短期持有":
        return "短期结构尚可，但新买点不够明确；已有仓位可持有观察，新仓不宜追高。"
    if advice == "短期等待":
        return "当前不具备足够清晰的短线买点，建议等待回踩缩量、突破确认或风险释放。"
    if advice == "短期减仓":
        return "短期供应压力开始上升，已有仓位应降低风险敞口。"
    if advice == "短期卖出":
        return "短期出现明确卖出风险，优先保护本金和已有利润。"
    if advice == "短期回避":
        return "短期结构或风险信号偏弱，不适合开新仓。"
    return "短期结论偏中性，等待更明确的量价信号。"


def _long_factor_snapshot(factors: dict) -> str:
    items = []
    mapping = [
        ("return_60d", "60日收益", "{:.1%}"),
        ("return_120d", "120日收益", "{:.1%}"),
        ("return_250d", "250日收益", "{:.1%}"),
        ("roe", "ROE", "{:.1%}"),
        ("gross_margin", "毛利率", "{:.1%}"),
        ("net_margin", "净利率", "{:.1%}"),
        ("pe", "PE", "{:.1f}"),
        ("pb", "PB", "{:.1f}"),
        ("debt_to_assets", "资产负债率", "{:.1%}"),
    ]
    for key, label, fmt in mapping:
        value = factors.get(key)
        try:
            if value is not None:
                items.append(f"{label}{fmt.format(float(value))}")
        except (TypeError, ValueError):
            pass
        if len(items) >= 5:
            break
    return "；".join(items)


def _join_limited(items: list[str], limit: int) -> str:
    return "；".join(str(item) for item in items[:limit] if item)


def scan_candidate_score(result: AnalysisResult) -> float:
    long_view = getattr(result, "long_term_view", None)
    short_view = getattr(result, "short_term_view", None)
    score_options: list[float] = []
    if long_view is not None:
        score_options.append(float(long_view.score))
    if short_view is not None:
        score_options.append(float(short_view.liangjia_score))
    scores = getattr(result, "factor_scores", None)
    if scores is not None:
        score_options.extend([float(scores.short_term_score), float(scores.long_term_score)])
    if score_options:
        return max(score_options)
    decision = getattr(result, "decision", None)
    if decision is not None:
        return float(decision.candidate_score)
    return float(result.score.total)


def scan_action_code(result: AnalysisResult) -> str:
    short_view = getattr(result, "short_term_view", None)
    long_view = getattr(result, "long_term_view", None)
    if short_view is not None:
        code = str(short_view.action_code)
        if code in {"buy", "add", "hold", "wait"}:
            return "watch" if code == "hold" else code
        if long_view is not None and long_view.advice in {"\u957f\u671f\u53ef\u5173\u6ce8", "\u957f\u671f\u8c28\u614e\u5173\u6ce8"} and code != "avoid":
            return "watch"
        return code
    classification = getattr(result, "classification", None)
    if classification is not None:
        return str(classification.action).lower()
    decision = getattr(result, "decision", None)
    if decision is not None:
        return str(decision.action_code)
    return ""


def analyze_for_ui(
    df: pd.DataFrame,
    symbol: str = "",
    benchmark: pd.DataFrame | None = None,
    sector: pd.DataFrame | None = None,
    sector_name: str = "",
    sector_rs_rank: float | None = None,
) -> AnalysisResult:
    try:
        result = analyze_ohlcv(
            df,
            symbol=symbol,
            benchmark=benchmark,
            sector=sector,
            sector_name=sector_name,
            sector_rs_rank=sector_rs_rank,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'sector'" not in str(exc):
            raise
        result = analyze_ohlcv(df, symbol=symbol, benchmark=benchmark)

    if getattr(result, "radar", None) is not None:
        if getattr(result, "decision", None) is not None:
            return result
        return attach_decision(result)

    try:
        from types import SimpleNamespace

        from stockbuyornot.radar import diagnose_radar

        radar = diagnose_radar(
            df,
            result,
            benchmark=benchmark,
            sector=sector,
            sector_name=sector_name,
            sector_rs_rank=sector_rs_rank,
        )
        data = dict(getattr(result, "__dict__", {}))
        data["radar"] = radar
        result = SimpleNamespace(**data)
        return attach_decision(result)
    except Exception:
        return result


def attach_decision(result):
    try:
        from types import SimpleNamespace

        decision = make_unified_decision(result)
        data = dict(getattr(result, "__dict__", {}))
        data["decision"] = decision
        data["suggestion"] = decision.action_label
        return SimpleNamespace(**data)
    except Exception:
        return result


def result_to_scan_row(result: AnalysisResult) -> dict:
    sides = {"buy": "买", "sell": "卖", "watch": "观察", "avoid": "回避"}
    row = {
        "代码": result.symbol,
        "日期": result.as_of.date(),
        "收盘": round(result.close, 2),
        "结构": result.structure.stage.value,
        "评分": result.score.total,
        "建议": result.suggestion,
        "信号": "、".join(signal.name for signal in result.signals),
        "方向": "、".join(sides.get(signal.side.value, signal.side.value) for signal in result.signals),
    }
    decision = getattr(result, "decision", None)
    if decision is not None:
        st.markdown("**最终统一结论**")
        d1, d2, d3, d4 = st.columns(4)
        d1.markdown(metric_card("最终动作", decision.action_label, decision.action_code), unsafe_allow_html=True)
        d2.markdown(metric_card("候选分", f"{decision.candidate_score:.0f}", f"信心 {decision.confidence:.0f}"), unsafe_allow_html=True)
        d3.markdown(metric_card("主依据", decision.primary_basis, decision.conflict or "无冲突"), unsafe_allow_html=True)
        d4.markdown(metric_card("选股口径", "量价优先", "雷达只做过滤/分层"), unsafe_allow_html=True)
        st.write(decision.reason)

    radar = getattr(result, "radar", None)
    if radar is not None:
        row.update(
            {
                "market_state": radar.market_state,
                "sector_name": radar.sector_name,
                "sector_rs_rank": radar.sector_rs_rank,
                "stock_vs_sector_rs": radar.stock_vs_sector_rs,
                "setup_score": radar.setup_score,
                "entry_quality_score": radar.entry_quality_score,
                "exit_risk_score": radar.exit_risk_score,
                "action_code": radar.action_code,
                "expected_action": radar.expected_action,
                "reject_reason": radar.reject_reason,
            }
        )
    decision = getattr(result, "decision", None)
    if decision is not None:
        row.update(
            {
                "final_action": decision.action_code,
                "final_action_label": decision.action_label,
                "candidate_score": decision.candidate_score,
                "decision_confidence": decision.confidence,
                "primary_basis": decision.primary_basis,
                "decision_conflict": decision.conflict,
                "decision_reason": decision.reason,
            }
        )
    return row


def render_result_summary(result: AnalysisResult) -> None:
    score_delta = "强信号" if result.score.total >= 80 else "观察" if result.score.total >= 60 else "中性/回避"
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(metric_card("收盘", f"{result.close:.2f}"), unsafe_allow_html=True)
    col2.markdown(metric_card("结构", result.structure.stage.value), unsafe_allow_html=True)
    col3.markdown(metric_card("评分", str(result.score.total), score_delta), unsafe_allow_html=True)
    col4.markdown(metric_card("建议", result.suggestion), unsafe_allow_html=True)

    radar = getattr(result, "radar", None)
    if radar is None:
        return
    st.markdown("**强股雷达诊断**")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("强股设置", f"{radar.setup_score:.0f}", radar.expected_action), unsafe_allow_html=True)
    c2.markdown(metric_card("买点质量", f"{radar.entry_quality_score:.0f}", radar.entry_conclusion), unsafe_allow_html=True)
    c3.markdown(metric_card("卖出风险", f"{radar.exit_risk_score:.0f}", radar.exit_conclusion), unsafe_allow_html=True)
    c4.markdown(metric_card("市场状态", radar.market_state, radar.reject_reason or "暂无主要排除项"), unsafe_allow_html=True)
    st.write(f"现在是不是好股票：{radar.good_stock_conclusion}")
    st.write(f"现在是不是好买点：{radar.entry_conclusion}")
    st.write(f"如果持有，什么时候该卖：{radar.exit_conclusion}")


ACTION_TEXT = {
    "buy": "\u53ef\u4e70\u5165",
    "wait": "\u7b49\u5f85\u4e70\u70b9",
    "watch": "\u53ef\u5173\u6ce8",
    "sell": "\u51cf\u4ed3/\u5356\u51fa",
    "avoid": "\u56de\u907f",
    "trade": "\u53ef\u4ea4\u6613",
}

BASIS_TEXT = {
    "volume_price_exit": "\u91cf\u4ef7\u5356\u51fa/\u56de\u907f\u4fe1\u53f7",
    "markdown_structure": "\u4e3b\u8dcc\u7ed3\u6784",
    "risk_accumulation": "\u98ce\u9669\u7d2f\u79ef",
    "buy_signal_weak_market": "\u6709\u4e70\u70b9\u4f46\u5927\u76d8\u504f\u5f31",
    "confirmed_volume_price_buy": "\u91cf\u4ef7\u4e70\u70b9\u786e\u8ba4",
    "buy_signal_needs_quality": "\u6709\u4e70\u70b9\u4f46\u8d28\u91cf\u4e0d\u8db3",
    "watch_signal": "\u91cf\u4ef7\u89c2\u5bdf\u4fe1\u53f7",
    "radar_setup_only": "\u4ec5\u5f3a\u80a1\u8bbe\u7f6e\u8f83\u597d",
    "no_actionable_volume_price_edge": "\u6ca1\u6709\u660e\u786e\u91cf\u4ef7\u4f18\u52bf",
}

CONFLICT_TEXT = {
    "": "\u65e0\u51b2\u7a81",
    "radar_positive_but_volume_price_exit": "\u96f7\u8fbe\u504f\u6b63\u9762\uff0c\u4f46\u91cf\u4ef7\u5df2\u8f6c\u5f31\uff0c\u4ee5\u91cf\u4ef7\u4e3a\u51c6",
    "volume_price_buy_but_radar_filter_negative": "\u6709\u91cf\u4ef7\u4e70\u70b9\uff0c\u4f46\u73af\u5883\u6216\u8d28\u91cf\u4e0d\u8fbe\u6807",
    "radar_trade_without_volume_price_buy": "\u96f7\u8fbe\u8f83\u597d\uff0c\u4f46\u5c1a\u65e0\u91cf\u4ef7\u4e70\u70b9",
}

MARKET_TEXT = {
    "strong": "\u5f3a\u5e02",
    "neutral": "\u9707\u8361\u5e02",
    "weak": "\u5f31\u5e02",
    "unknown": "\u672a\u77e5",
}

REJECT_TEXT = {
    "weak_market": "\u5927\u76d8\u504f\u5f31",
    "weak_or_late_structure": "\u7ed3\u6784\u504f\u5f31\u6216\u504f\u540e\u671f",
    "weak_relative_strength": "\u76f8\u5bf9\u5f3a\u5ea6\u504f\u5f31",
    "low_setup_score": "\u5f3a\u80a1\u8bbe\u7f6e\u5206\u4e0d\u8db3",
    "poor_entry_quality": "\u4e70\u70b9\u8d28\u91cf\u4e0d\u8db3",
    "high_exit_risk": "\u5356\u51fa\u98ce\u9669\u504f\u9ad8",
    "overextended_from_ma20": "\u8ddd\u79bbMA20\u8fc7\u8fdc\uff0c\u6709\u8ffd\u9ad8\u98ce\u9669",
}


def scan_candidate_score(result: AnalysisResult) -> float:
    decision = getattr(result, "decision", None)
    if decision is not None:
        return float(decision.candidate_score)
    return float(result.score.total)


def scan_action_code(result: AnalysisResult) -> str:
    decision = getattr(result, "decision", None)
    if decision is not None:
        return str(decision.action_code)
    return ""


def translate_reject_reason(value: str | None) -> str:
    if not value:
        return "\u65e0"
    parts = [part for part in str(value).split(";") if part]
    return "\u3001".join(REJECT_TEXT.get(part, part) for part in parts) if parts else "\u65e0"


def format_pct(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def localize_scan_table(display: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "candidate_score": "\u5019\u9009\u5206",
        "final_action": "\u6700\u7ec8\u52a8\u4f5c",
        "final_action_label": "\u6700\u7ec8\u52a8\u4f5c",
        "decision_confidence": "\u51b3\u7b56\u4fe1\u5fc3",
        "primary_basis": "\u4e3b\u8981\u4f9d\u636e",
        "decision_conflict": "\u51b2\u7a81\u63d0\u793a",
        "decision_reason": "\u7edf\u4e00\u8bf4\u660e",
        "market_state": "\u5e02\u573a\u72b6\u6001",
        "sector_name": "\u677f\u5757\u540d\u79f0",
        "sector_rs_rank": "\u677f\u5757\u5f3a\u5ea6\u5206\u4f4d",
        "stock_vs_sector_rs": "\u4e2a\u80a1\u5f3a\u4e8e\u677f\u5757",
        "setup_score": "\u5f3a\u80a1\u8bbe\u7f6e\u5206",
        "entry_quality_score": "\u4e70\u70b9\u8d28\u91cf\u5206",
        "exit_risk_score": "\u5356\u51fa\u98ce\u9669\u5206",
        "radar_action_code": "\u96f7\u8fbe\u5206\u7c7b",
        "radar_action": "\u96f7\u8fbe\u5206\u7c7b",
        "radar_reject_reason": "\u96f7\u8fbe\u6392\u9664\u539f\u56e0",
        "reject_reason": "\u96f7\u8fbe\u6392\u9664\u539f\u56e0",
    }
    display = display.rename(columns={key: value for key, value in rename.items() if key in display.columns})
    if "\u6700\u7ec8\u52a8\u4f5c" in display.columns:
        display["\u6700\u7ec8\u52a8\u4f5c"] = display["\u6700\u7ec8\u52a8\u4f5c"].map(lambda value: ACTION_TEXT.get(str(value), value))
    if "\u5e02\u573a\u72b6\u6001" in display.columns:
        display["\u5e02\u573a\u72b6\u6001"] = display["\u5e02\u573a\u72b6\u6001"].map(lambda value: MARKET_TEXT.get(str(value), value))
    if "\u96f7\u8fbe\u5206\u7c7b" in display.columns:
        display["\u96f7\u8fbe\u5206\u7c7b"] = display["\u96f7\u8fbe\u5206\u7c7b"].map(lambda value: ACTION_TEXT.get(str(value), value))
    if "\u4e3b\u8981\u4f9d\u636e" in display.columns:
        display["\u4e3b\u8981\u4f9d\u636e"] = display["\u4e3b\u8981\u4f9d\u636e"].map(lambda value: BASIS_TEXT.get(str(value), value))
    if "\u51b2\u7a81\u63d0\u793a" in display.columns:
        display["\u51b2\u7a81\u63d0\u793a"] = display["\u51b2\u7a81\u63d0\u793a"].map(lambda value: CONFLICT_TEXT.get(str(value), value))
    if "\u96f7\u8fbe\u6392\u9664\u539f\u56e0" in display.columns:
        display["\u96f7\u8fbe\u6392\u9664\u539f\u56e0"] = display["\u96f7\u8fbe\u6392\u9664\u539f\u56e0"].map(translate_reject_reason)
    return display


def filter_scan_display_by_min_score(display: pd.DataFrame, min_score: int) -> pd.DataFrame:
    score_column = scan_score_column(display)
    if not score_column:
        return display.iloc[0:0].copy()
    filtered = display.copy()
    scores = pd.to_numeric(filtered[score_column], errors="coerce")
    return filtered[scores >= float(min_score)].copy()


def scan_score_column(display: pd.DataFrame) -> str:
    for column in ["\u5019\u9009\u5206", "candidate_score", "\u91cf\u4ef7\u5206", "\u8bc4\u5206"]:
        if column in display.columns:
            return column
    return ""


def compact_scan_display(display: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "\u4ee3\u7801",
        "\u5019\u9009\u5206",
        "\u957f\u671f\u6295\u8d44\u4ef7\u503c",
        "\u957f\u671f\u5206",
        "\u77ed\u671f\u4ea4\u6613\u673a\u4f1a",
        "\u91cf\u4ef7\u5206",
        "\u77ed\u671f\u5206",
        "\u6267\u884c\u7a97\u53e3\u5206",
        "\u6267\u884c\u7a97\u53e3",
        "\u4fe1\u53f7\u65b9\u5411",
        "\u4fe1\u53f7\u5f3a\u5ea6",
        "\u91cf\u4ef7\u4fe1\u53f7\u7c7b\u578b",
        "\u7efc\u5408\u5206",
        "\u6536\u76d8",
        "\u7ed3\u6784",
        "\u5e02\u573a\u72b6\u6001",
        "\u5f3a\u80a1\u8bbe\u7f6e\u5206",
        "\u4e70\u70b9\u8d28\u91cf\u5206",
        "\u5356\u51fa\u98ce\u9669\u5206",
    ]
    columns = [column for column in preferred if column in display.columns]
    compact = display[columns].copy() if columns else display.copy()
    score_columns = [
        "\u5019\u9009\u5206",
        "\u7efc\u5408\u5206",
        "\u77ed\u671f\u5206",
        "\u957f\u671f\u5206",
        "\u91cf\u4ef7\u5206",
        "\u6267\u884c\u7a97\u53e3\u5206",
        "\u4fe1\u53f7\u5f3a\u5ea6",
        "\u5f3a\u80a1\u8bbe\u7f6e\u5206",
        "\u4e70\u70b9\u8d28\u91cf\u5206",
        "\u5356\u51fa\u98ce\u9669\u5206",
    ]
    for column in score_columns:
        if column in compact.columns:
            compact[column] = pd.to_numeric(compact[column], errors="coerce").round(0).astype("Int64")
    for column in ["\u91cf\u4ef7\u4fe1\u53f7", "\u91cf\u4ef7\u4fe1\u53f7\u7c7b\u578b", "\u7ed3\u6784", "\u957f\u671f\u6295\u8d44\u4ef7\u503c", "\u77ed\u671f\u4ea4\u6613\u673a\u4f1a", "\u6267\u884c\u7a97\u53e3"]:
        if column in compact.columns:
            compact[column] = compact[column].map(lambda value: _short_cell(value, 18))
    return compact


def filter_scan_results_by_display(scan_results: list[AnalysisResult], display: pd.DataFrame) -> list[AnalysisResult]:
    if "\u4ee3\u7801" not in display.columns:
        return scan_results
    visible_symbols = {str(value) for value in display["\u4ee3\u7801"].dropna().tolist()}
    return [result for result in scan_results if str(result.symbol) in visible_symbols]


def _short_cell(value, max_chars: int) -> str:
    if value in [None, ""]:
        return ""
    text = str(value)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "\u2026"


def scan_column_config(display: pd.DataFrame) -> dict:
    config = {}
    small_numbers = [
        "\u5019\u9009\u5206",
        "\u7efc\u5408\u5206",
        "\u77ed\u671f\u5206",
        "\u957f\u671f\u5206",
        "\u91cf\u4ef7\u5206",
        "\u6267\u884c\u7a97\u53e3\u5206",
        "\u4fe1\u53f7\u5f3a\u5ea6",
        "\u6536\u76d8",
        "\u5f3a\u80a1\u8bbe\u7f6e\u5206",
        "\u4e70\u70b9\u8d28\u91cf\u5206",
        "\u5356\u51fa\u98ce\u9669\u5206",
    ]
    help_text = {
        "\u5019\u9009\u5206": "\u5019\u9009\u5206\uff1a\u5f53\u524d\u626b\u63cf\u7528\u4e8e\u5165\u9009\u548c\u6392\u5e8f\u7684\u4e3b\u5206\uff0c\u65b0\u7248\u4f18\u5148\u7b49\u4e8e\u7efc\u5408\u5206\u3002",
        "\u7efc\u5408\u5206": "\u7efc\u5408\u5206\uff1a\u91cf\u4ef7\u5206\u6743\u91cd\u6700\u9ad8\uff0c\u518d\u53e0\u52a0\u77ed\u671f\u786e\u8ba4\u548c\u957f\u671f\u5206\u5c42\uff0c\u5e76\u6263\u9664\u98ce\u9669\u60e9\u7f5a\u3002",
        "\u91cf\u4ef7\u5206": "\u91cf\u4ef7\u5206\uff1a\u4fdd\u7559\u539f\u6709\u4f9b\u9700\u4e3b\u7ebf\uff0c\u6309\u7ed3\u6784\u3001\u4f4d\u7f6e\u3001\u4fe1\u53f7\u3001\u98ce\u9669\u548c\u5e02\u573a\u73af\u5883\u8bc4\u5206\u3002",
        "\u77ed\u671f\u5206": "\u77ed\u671f\u5206\uff1a\u770b\u77ed\u671f\u6da8\u8dcc\u5e45\u3001\u77ed\u5747\u7ebf\u3001\u91cf\u6bd4\u3001RSI/MACD/KDJ\u3001\u65b0\u9ad8\u548c\u56de\u64a4\u3002",
        "\u6267\u884c\u7a97\u53e3\u5206": "\u6267\u884c\u7a97\u53e3\u5206\uff1a\u5e73\u8861\u91cf\u4ef7\u7ed3\u6784\u3001\u77ed\u671f\u786e\u8ba4\u3001MA20\u4e56\u79bb\u548c\u56de\u8e29\u5e45\u5ea6\uff1b\u5206\u9ad8\u624d\u8868\u793a\u66f4\u63a5\u8fd1\u53ef\u6267\u884c\u4e70\u70b9\uff0c\u5206\u4f4e\u5e94\u7b49\u5f85\u6216\u4e0d\u8ffd\u9ad8\u3002",
        "\u957f\u671f\u5206": "\u957f\u671f\u5206\uff1a\u770b\u957f\u671f\u8d8b\u52bf\u3001\u957f\u671f\u6536\u76ca\u3001\u57fa\u672c\u9762\u8d28\u91cf\u3001\u6210\u957f\u3001\u4f30\u503c\u548c\u8d22\u52a1\u98ce\u9669\u3002",
        "\u4fe1\u53f7\u5f3a\u5ea6": "\u4fe1\u53f7\u5f3a\u5ea6\uff1a\u5f53\u524d\u5177\u4f53\u91cf\u4ef7\u4ea4\u6613\u4fe1\u53f7\u7684\u5f3a\u5f31\uff0c\u548c\u6574\u4f53\u91cf\u4ef7\u5206\u4e0d\u662f\u540c\u4e00\u4e2a\u6982\u5ff5\u3002",
        "\u5f3a\u80a1\u8bbe\u7f6e\u5206": "\u5f3a\u80a1\u8bbe\u7f6e\u5206\uff1a\u5f3a\u80a1\u96f7\u8fbe\u5bf9\u5e02\u573a\u3001\u677f\u5757\u3001\u4e2a\u80a1\u5f3a\u5ea6\u548c\u7ed3\u6784\u6539\u5584\u7684\u8bc4\u4f30\u3002",
        "\u4e70\u70b9\u8d28\u91cf\u5206": "\u4e70\u70b9\u8d28\u91cf\u5206\uff1a\u770b\u652f\u6491\u6765\u6e90\u3001\u7f29\u91cf\u56de\u8e29\u3001\u6b62\u635f\u8ddd\u79bb\u548c\u76c8\u4e8f\u6bd4\u3002",
        "\u5356\u51fa\u98ce\u9669\u5206": "\u5356\u51fa\u98ce\u9669\u5206\uff1a\u770b\u9ad8\u4f4d\u653e\u91cf\u6ede\u6da8\u3001\u8dcc\u7834\u5173\u952e\u4f4d\u3001\u76f8\u5bf9\u5f3a\u5ea6\u8f6c\u5f31\u7b49\u98ce\u9669\u3002",
    }
    for column in small_numbers:
        if column in display.columns:
            config[column] = st.column_config.NumberColumn(column, width="small", help=help_text.get(column))
    for column in ["\u4ee3\u7801", "\u5e02\u573a\u72b6\u6001", "\u4fe1\u53f7\u65b9\u5411"]:
        if column in display.columns:
            config[column] = st.column_config.TextColumn(column, width="small")
    for column in ["\u957f\u671f\u6295\u8d44\u4ef7\u503c", "\u77ed\u671f\u4ea4\u6613\u673a\u4f1a"]:
        if column in display.columns:
            config[column] = st.column_config.TextColumn(column, width="small")
    for column in ["\u7ed3\u6784", "\u91cf\u4ef7\u4fe1\u53f7", "\u91cf\u4ef7\u4fe1\u53f7\u7c7b\u578b", "\u6267\u884c\u7a97\u53e3"]:
        if column in display.columns:
            config[column] = st.column_config.TextColumn(column, width="medium")
    return config


def result_to_scan_row(result: AnalysisResult) -> dict:
    sides = {"buy": "\u4e70", "sell": "\u5356", "watch": "\u89c2\u5bdf", "avoid": "\u56de\u907f"}
    row = {
        "\u4ee3\u7801": result.symbol,
        "\u65e5\u671f": result.as_of.date(),
        "\u6536\u76d8": round(result.close, 2),
        "\u7ed3\u6784": result.structure.stage.value,
        "\u91cf\u4ef7\u5206": result.score.total,
        "\u7edf\u4e00\u5efa\u8bae": result.suggestion,
        "\u91cf\u4ef7\u4fe1\u53f7": "\u3001".join(signal.name for signal in result.signals),
        "\u4fe1\u53f7\u65b9\u5411": "\u3001".join(sides.get(signal.side.value, signal.side.value) for signal in result.signals),
    }
    radar = getattr(result, "radar", None)
    if radar is not None:
        row.update(
            {
                "\u5e02\u573a\u72b6\u6001": MARKET_TEXT.get(radar.market_state, radar.market_state),
                "\u677f\u5757\u540d\u79f0": radar.sector_name,
                "\u677f\u5757\u5f3a\u5ea6\u5206\u4f4d": format_pct(radar.sector_rs_rank),
                "\u4e2a\u80a1\u5f3a\u4e8e\u677f\u5757": format_pct(radar.stock_vs_sector_rs),
                "\u5f3a\u80a1\u8bbe\u7f6e\u5206": radar.setup_score,
                "\u4e70\u70b9\u8d28\u91cf\u5206": radar.entry_quality_score,
                "\u5356\u51fa\u98ce\u9669\u5206": radar.exit_risk_score,
                "\u96f7\u8fbe\u5206\u7c7b": ACTION_TEXT.get(radar.action_code, radar.expected_action),
                "\u96f7\u8fbe\u6392\u9664\u539f\u56e0": translate_reject_reason(radar.reject_reason),
            }
        )
    decision = getattr(result, "decision", None)
    if decision is not None:
        row.update(
            {
                "\u6700\u7ec8\u52a8\u4f5c": ACTION_TEXT.get(decision.action_code, decision.action_label),
                "\u5019\u9009\u5206": decision.candidate_score,
                "\u51b3\u7b56\u4fe1\u5fc3": decision.confidence,
                "\u4e3b\u8981\u4f9d\u636e": BASIS_TEXT.get(decision.primary_basis, decision.primary_basis),
                "\u51b2\u7a81\u63d0\u793a": CONFLICT_TEXT.get(decision.conflict, decision.conflict),
                "\u7edf\u4e00\u8bf4\u660e": decision.reason,
            }
        )
    return row


def render_result_summary(result: AnalysisResult) -> None:
    long_view = getattr(result, "long_term_view", None)
    short_view = getattr(result, "short_term_view", None)
    scores = getattr(result, "factor_scores", None)
    score_delta = "\u5f3a\u4fe1\u53f7" if result.score.total >= 80 else "\u89c2\u5bdf" if result.score.total >= 60 else "\u4e2d\u6027/\u56de\u907f"
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(metric_card("\u6536\u76d8", f"{result.close:.2f}"), unsafe_allow_html=True)
    col2.markdown(metric_card("\u7ed3\u6784", stage_label(result.structure.stage)), unsafe_allow_html=True)
    if long_view is not None:
        col3.markdown(metric_card("\u957f\u671f\u6295\u8d44\u4ef7\u503c", long_view.advice, f"{long_view.score:.0f}\u5206", metric_score_tooltip(result, "\u957f\u671f\u5206")), unsafe_allow_html=True)
    else:
        col3.markdown(metric_card("\u957f\u671f\u6295\u8d44\u4ef7\u503c", "-"), unsafe_allow_html=True)
    if short_view is not None:
        execution_score, execution_flags = execution_window_from_scores(scores) if scores is not None else (None, [])
        col4.markdown(
            metric_card(
                "\u77ed\u671f\u4ea4\u6613\u673a\u4f1a",
                short_view.advice,
                "\u6267\u884c\u7a97\u53e3 -" if execution_score is None else f"\u6267\u884c\u7a97\u53e3 {execution_score:.0f}",
                execution_window_tooltip(execution_score, execution_flags),
            ),
            unsafe_allow_html=True,
        )
    else:
        col4.markdown(metric_card("\u77ed\u671f\u4ea4\u6613\u673a\u4f1a", result.suggestion), unsafe_allow_html=True)

    if long_view is not None and short_view is not None:
        st.markdown("**\u957f\u671f\u4e0e\u77ed\u671f\u5206\u5f00\u8bca\u65ad**")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("**\u957f\u671f\u6295\u8d44\u4ef7\u503c**")
            l1, l2, l3 = st.columns(3)
            l1.markdown(metric_card("\u957f\u671f\u5efa\u8bae", long_view.advice), unsafe_allow_html=True)
            l2.markdown(metric_card("\u957f\u671f\u5206", f"{long_view.score:.0f}", long_view.rating, metric_score_tooltip(result, "\u957f\u671f\u5206")), unsafe_allow_html=True)
            if scores is not None:
                l3.markdown(metric_card("\u957f\u671f\u53c2\u8003", f"{scores.long_term_score:.0f}", "\u4e0d\u4ee3\u8868\u77ed\u671f\u53ef\u4e70"), unsafe_allow_html=True)
            else:
                l3.markdown(metric_card("\u957f\u671f\u53c2\u8003", "-"), unsafe_allow_html=True)
            _render_view_points("\u957f\u671f\u5173\u952e\u56e0\u7d20", long_view.key_factors)
            _render_view_points("\u957f\u671f\u98ce\u9669\u63d0\u793a", long_view.risk_warnings)
            st.caption(long_view.explanation)
        with right:
            st.markdown("**\u77ed\u671f\u4ea4\u6613\u673a\u4f1a**")
            execution_score, execution_flags = execution_window_from_scores(scores) if scores is not None else (None, [])
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.markdown(metric_card("\u77ed\u671f\u5efa\u8bae", short_view.advice, short_view.signal_direction), unsafe_allow_html=True)
            s2.markdown(
                metric_card(
                    "\u4e70\u70b9\u6267\u884c\u7a97\u53e3",
                    "-" if execution_score is None else f"{execution_score:.0f}",
                    execution_window_label(execution_score),
                    execution_window_tooltip(execution_score, execution_flags),
                ),
                unsafe_allow_html=True,
            )
            s3.markdown(metric_card("\u77ed\u671f\u5206", f"{short_view.short_term_score:.0f}", "\u77ed\u671f\u52a8\u91cf/\u5747\u7ebf/\u91cf\u80fd", metric_score_tooltip(result, "\u77ed\u671f\u5206")), unsafe_allow_html=True)
            s4.markdown(metric_card("\u91cf\u4ef7\u5206", f"{short_view.liangjia_score:.0f}", score_delta, metric_score_tooltip(result, "\u91cf\u4ef7\u5206")), unsafe_allow_html=True)
            s5.markdown(metric_card("\u4fe1\u53f7\u5f3a\u5ea6", str(short_view.signal_strength), short_view.signal_type), unsafe_allow_html=True)
            trade_cols = st.columns(3)
            trade_cols[0].markdown(metric_card("\u4e70\u5165\u533a\u95f4", _entry_zone_text(short_view.entry_zone)), unsafe_allow_html=True)
            trade_cols[1].markdown(metric_card("\u6b62\u635f\u4f4d", "-" if short_view.stop_loss is None else f"{short_view.stop_loss:.2f}"), unsafe_allow_html=True)
            trade_cols[2].markdown(metric_card("\u98ce\u9669\u6bd4\u4f8b", "-" if short_view.risk_pct is None else f"{short_view.risk_pct:.1%}"), unsafe_allow_html=True)
            _render_view_points("\u77ed\u671f\u5173\u952e\u56e0\u7d20", short_view.key_factors)
            _render_view_points("\u77ed\u671f\u98ce\u9669\u63d0\u793a", short_view.risk_warnings)
            st.caption(short_view.explanation)

    radar = getattr(result, "radar", None)
    if radar is not None:
        st.markdown("**\u5f3a\u80a1\u96f7\u8fbe\u8bca\u65ad\uff08\u8f85\u52a9\uff09**")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("\u5f3a\u80a1\u8bbe\u7f6e", f"{radar.setup_score:.0f}", ACTION_TEXT.get(radar.action_code, radar.expected_action), metric_score_tooltip(result, "\u5f3a\u80a1\u8bbe\u7f6e")), unsafe_allow_html=True)
        c2.markdown(metric_card("\u4e70\u70b9\u8d28\u91cf", f"{radar.entry_quality_score:.0f}", radar.entry_conclusion, metric_score_tooltip(result, "\u4e70\u70b9\u8d28\u91cf")), unsafe_allow_html=True)
        c3.markdown(metric_card("\u5356\u51fa\u98ce\u9669", f"{radar.exit_risk_score:.0f}", radar.exit_conclusion, metric_score_tooltip(result, "\u5356\u51fa\u98ce\u9669")), unsafe_allow_html=True)
        c4.markdown(metric_card("\u5e02\u573a\u72b6\u6001", MARKET_TEXT.get(radar.market_state, radar.market_state), translate_reject_reason(radar.reject_reason)), unsafe_allow_html=True)
        st.write(f"\u73b0\u5728\u662f\u4e0d\u662f\u597d\u80a1\u7968\uff1a{radar.good_stock_conclusion}")
        st.write(f"\u73b0\u5728\u662f\u4e0d\u662f\u597d\u4e70\u70b9\uff1a{radar.entry_conclusion}")
        st.write(f"\u5982\u679c\u6301\u6709\uff0c\u4ec0\u4e48\u65f6\u5019\u8be5\u5356\uff1a{radar.exit_conclusion}")
    _render_split_conclusion_summary(result)


def _render_view_points(title: str, items: list[str]) -> None:
    st.markdown(f"**{title}**")
    if not items:
        st.write("- \u6682\u65e0")
        return
    for item in items[:8]:
        st.write(f"- {item}")


def _entry_zone_text(zone) -> str:
    if not zone:
        return "-"
    return f"{zone[0]:.2f} - {zone[1]:.2f}"


def _render_split_conclusion_summary(result: AnalysisResult) -> None:
    long_view = getattr(result, "long_term_view", None)
    short_view = getattr(result, "short_term_view", None)
    if long_view is None and short_view is None:
        return

    st.markdown("**长期与短期结论总结**")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown(f"**长期投资价值：{long_view.advice if long_view else '-'}**")
            if long_view is None:
                st.write("长期数据不足，暂时无法形成中长期判断。")
            else:
                for line in _long_conclusion_lines(result, long_view):
                    st.write(line)
    with right:
        with st.container(border=True):
            st.markdown(f"**短期交易机会：{short_view.advice if short_view else '-'}**")
            if short_view is None:
                st.write("短期量价数据不足，暂时无法形成交易判断。")
            else:
                for line in _short_conclusion_lines(result, short_view):
                    st.write(line)


def _long_conclusion_lines(result: AnalysisResult, long_view) -> list[str]:
    factors = getattr(result, "factors", None)
    long_factors = getattr(factors, "long_term", {}) or {}
    lines = [
        f"- 评分与评级：长期分 {long_view.score:.0f}，评级为「{long_view.rating}」。",
        f"- 核心判断：{_long_action_sentence(long_view.advice)}",
    ]
    key_text = _join_limited(long_view.key_factors, 3)
    if key_text:
        lines.append(f"- 主要依据：{key_text}。")
    factor_text = _long_factor_snapshot(long_factors)
    if factor_text:
        lines.append(f"- 指标参考：{factor_text}。")
    if long_view.risk_warnings:
        lines.append(f"- 风险提示：{_join_limited(long_view.risk_warnings, 3)}。")
    else:
        lines.append("- 风险提示：当前长期模块没有识别到特别突出的长期风险，但仍需结合财报和行业变化复核。")
    lines.append("- 操作边界：长期判断只回答“是否值得中长期关注”，不代表今天短期可以买入。短期买卖仍以量价信号、买点质量和止损条件为准。")
    return lines


def _short_conclusion_lines(result: AnalysisResult, short_view) -> list[str]:
    lines = [
        f"- 评分与方向：短期分 {short_view.short_term_score:.0f}，量价分 {short_view.liangjia_score:.0f}，信号方向「{short_view.signal_direction}」。",
        f"- 核心判断：{_short_action_sentence(short_view)}",
    ]
    if short_view.signal_type:
        lines.append(f"- 当前信号：{short_view.signal_type}，信号强度 {short_view.signal_strength}。")
    if short_view.entry_zone or short_view.stop_loss or short_view.risk_pct is not None:
        lines.append(
            "- 交易边界："
            f"买入区间 {_entry_zone_text(short_view.entry_zone)}；"
            f"止损位 {'-' if short_view.stop_loss is None else f'{short_view.stop_loss:.2f}'}；"
            f"风险比例 {'-' if short_view.risk_pct is None else f'{short_view.risk_pct:.1%}'}。"
        )
    key_text = _join_limited(short_view.key_factors, 3)
    if key_text:
        lines.append(f"- 主要依据：{key_text}。")
    if short_view.risk_warnings:
        lines.append(f"- 风险提示：{_join_limited(short_view.risk_warnings, 3)}。")
    else:
        lines.append("- 风险提示：当前短期模块没有识别到强卖出/回避风险，但若跌破关键位或放量转弱，应重新评估。")
    lines.append("- 执行原则：短期买入/加仓必须来自量价买点；若只有短期分高但量价买点不明确，优先等待回踩或突破确认。")
    return lines


def _long_action_sentence(advice: str) -> str:
    if advice == "长期可关注":
        return "长期趋势、质量或估值条件相对较好，可以进入中长期跟踪池，但仍要等待合适短期买点。"
    if advice == "长期谨慎关注":
        return "有一定长期观察价值，但优势不够充分，适合小仓位观察或继续等待基本面/趋势确认。"
    if advice == "长期暂不关注":
        return "长期吸引力不足，暂时不适合作为核心配置对象。"
    if advice == "长期回避":
        return "长期风险或弱势特征较明显，不建议作为中长期配置对象。"
    return "长期结论偏中性，需要继续结合财务、估值和行业趋势观察。"


def _short_action_sentence(short_view) -> str:
    advice = short_view.advice
    if advice == "短期买入":
        return "当前已有量价买点，若价格仍在计划区间内，可按止损约束考虑执行。"
    if advice == "短期加仓":
        return "当前处于较强量价买点或趋势延续状态，已有仓位可关注加仓机会，但不能放松止损。"
    if advice == "短期持有":
        return "短期结构尚可，但新买点不够明确；已有仓位可持有观察，新仓不宜追高。"
    if advice == "短期等待":
        return "当前不具备足够清晰的短线买点，建议等待回踩缩量、突破确认或风险释放。"
    if advice == "短期减仓":
        return "短期供应压力开始上升，已有仓位应降低风险敞口。"
    if advice == "短期卖出":
        return "短期出现明确卖出风险，优先保护本金和已有利润。"
    if advice == "短期回避":
        return "短期结构或风险信号偏弱，不适合开新仓。"
    return "短期结论偏中性，等待更明确的量价信号。"


def _long_factor_snapshot(factors: dict) -> str:
    items = []
    mapping = [
        ("return_60d", "60日收益", "{:.1%}"),
        ("return_120d", "120日收益", "{:.1%}"),
        ("return_250d", "250日收益", "{:.1%}"),
        ("roe", "ROE", "{:.1%}"),
        ("gross_margin", "毛利率", "{:.1%}"),
        ("net_margin", "净利率", "{:.1%}"),
        ("pe", "PE", "{:.1f}"),
        ("pb", "PB", "{:.1f}"),
        ("debt_to_assets", "资产负债率", "{:.1%}"),
    ]
    for key, label, fmt in mapping:
        value = factors.get(key)
        try:
            if value is not None:
                items.append(f"{label}{fmt.format(float(value))}")
        except (TypeError, ValueError):
            pass
        if len(items) >= 5:
            break
    return "，".join(items)


def _join_limited(items: list[str], limit: int) -> str:
    return "；".join(str(item) for item in items[:limit] if item)


def scan_candidate_score(result: AnalysisResult) -> float:
    scores = getattr(result, "factor_scores", None)
    if scores is not None:
        return float(scores.overall_score)
    decision = getattr(result, "decision", None)
    if decision is not None:
        return float(decision.candidate_score)
    return float(result.score.total)


def scan_action_code(result: AnalysisResult) -> str:
    classification = getattr(result, "classification", None)
    if classification is not None:
        return str(classification.action).lower()
    decision = getattr(result, "decision", None)
    if decision is not None:
        return str(decision.action_code)
    return ""


def classification_action_label(action: str) -> str:
    return {
        "BUY": "\u4e70\u5165",
        "ADD": "\u52a0\u4ed3",
        "WAIT": "\u7b49\u5f85",
        "WATCH": "\u89c2\u5bdf",
        "REDUCE": "\u51cf\u4ed3",
        "SELL": "\u5356\u51fa",
        "AVOID": "\u56de\u907f",
    }.get(str(action).upper(), str(action))


def result_to_scan_row(result: AnalysisResult) -> dict:
    sides = {"buy": "\u4e70", "sell": "\u5356", "watch": "\u89c2\u5bdf", "avoid": "\u56de\u907f"}
    long_view = getattr(result, "long_term_view", None)
    short_view = getattr(result, "short_term_view", None)
    row = {
        "\u4ee3\u7801": result.symbol,
        "\u65e5\u671f": result.as_of.date(),
        "\u6536\u76d8": round(result.close, 2),
        "\u7ed3\u6784": stage_label(result.structure.stage),
        "\u91cf\u4ef7\u5206": result.score.total,
        "\u91cf\u4ef7\u4fe1\u53f7": "\u3001".join(signal.name for signal in result.signals),
        "\u4fe1\u53f7\u65b9\u5411": "\u3001".join(sides.get(signal.side.value, signal.side.value) for signal in result.signals),
    }
    if long_view is not None:
        row.update(
            {
                "\u957f\u671f\u6295\u8d44\u4ef7\u503c": long_view.advice,
                "\u957f\u671f\u8bc4\u7ea7": long_view.rating,
                "\u957f\u671f\u5206": long_view.score,
                "\u957f\u671f\u98ce\u9669": "\u3001".join(long_view.risk_warnings),
            }
        )
    if short_view is not None:
        row.update(
            {
                "\u77ed\u671f\u4ea4\u6613\u673a\u4f1a": short_view.advice,
                "\u91cf\u4ef7\u4fe1\u53f7\u7c7b\u578b": short_view.signal_type,
                "\u4fe1\u53f7\u5f3a\u5ea6": short_view.signal_strength,
                "\u4fe1\u53f7\u65b9\u5411": short_view.signal_direction,
                "\u77ed\u671f\u98ce\u9669": "\u3001".join(short_view.risk_warnings),
            }
        )
    classification = getattr(result, "classification", None)
    scores = getattr(result, "factor_scores", None)
    if scores is not None:
        execution_score, execution_flags = execution_window_from_scores(scores)
        row.update(
            {
                "\u5019\u9009\u5206": scan_candidate_score(result),
                "\u7efc\u5408\u5206": scores.overall_score,
                "\u91cf\u4ef7\u5206": scores.liangjia_score,
                "\u77ed\u671f\u5206": scores.short_term_score,
                "\u957f\u671f\u5206": scores.long_term_score,
                "\u6267\u884c\u7a97\u53e3\u5206": execution_score,
                "\u6267\u884c\u7a97\u53e3": "\u3001".join(execution_flags[:3]),
            }
        )
    if classification is not None:
        row.update(
            {
                "\u4e70\u70b9\u7c7b\u578b": classification.buy_point_type,
                "\u98ce\u9669\u6807\u8bb0": "\u3001".join(classification.risk_flags),
            }
        )
    radar = getattr(result, "radar", None)
    if radar is not None:
        row.update(
            {
                "\u5e02\u573a\u72b6\u6001": MARKET_TEXT.get(radar.market_state, radar.market_state),
                "\u677f\u5757\u540d\u79f0": radar.sector_name,
                "\u677f\u5757\u5f3a\u5ea6\u5206\u4f4d": format_pct(radar.sector_rs_rank),
                "\u4e2a\u80a1\u5f3a\u4e8e\u677f\u5757": format_pct(radar.stock_vs_sector_rs),
                "\u5f3a\u80a1\u8bbe\u7f6e\u5206": radar.setup_score,
                "\u4e70\u70b9\u8d28\u91cf\u5206": radar.entry_quality_score,
                "\u5356\u51fa\u98ce\u9669\u5206": radar.exit_risk_score,
                "\u96f7\u8fbe\u5206\u7c7b": ACTION_TEXT.get(radar.action_code, radar.expected_action),
                "\u96f7\u8fbe\u6392\u9664\u539f\u56e0": translate_reject_reason(radar.reject_reason),
            }
        )
    row["\u5019\u9009\u5206"] = scan_candidate_score(result)
    return row


def scan_candidate_score(result: AnalysisResult) -> float:
    long_view = getattr(result, "long_term_view", None)
    short_view = getattr(result, "short_term_view", None)
    candidates: list[float] = []
    scores = getattr(result, "factor_scores", None)
    if scores is not None:
        execution_score, _ = execution_window_from_scores(scores)
        balanced_short = min(float(scores.short_term_score), float(execution_score) + 8.0)
        balanced_liangjia = min(float(short_view.liangjia_score if short_view is not None else scores.liangjia_score), float(execution_score) + 12.0)
        tradable_score = (
            float(execution_score) * 0.52
            + balanced_liangjia * 0.26
            + balanced_short * 0.17
            + float(scores.long_term_score) * 0.05
        )
        candidates.extend([float(scores.overall_score) * 0.85, tradable_score, float(scores.long_term_score) * 0.80, float(execution_score)])
    else:
        if long_view is not None:
            candidates.append(float(long_view.score))
        if short_view is not None:
            candidates.append(float(short_view.liangjia_score))
    if candidates:
        return max(candidates)
    decision = getattr(result, "decision", None)
    if decision is not None:
        return float(decision.candidate_score)
    return float(result.score.total)


def execution_window_from_scores(scores) -> tuple[float, list[str]]:
    components = getattr(scores, "components", {}) or {}
    item = components.get("execution_window", {}) or {}
    if isinstance(item, dict):
        score = _score_value(item.get("score"), _score_value(components.get("execution_score"), 70.0))
        flags = item.get("flags", []) or []
        return score, [str(flag) for flag in flags]
    return _score_value(item, _score_value(components.get("execution_score"), 70.0)), []


def execution_window_label(score: float | None) -> str:
    if score is None:
        return "暂无"
    if score >= 75:
        return "可执行区"
    if score >= 65:
        return "观察执行"
    if score >= 55:
        return "等待确认"
    return "不追/防守"


def execution_window_tooltip(score: float | None, flags: list[str] | None = None) -> str:
    current = "暂无执行窗口分。" if score is None else f"当前执行窗口分：{score:.0f}，属于「{execution_window_label(score)}」。"
    flag_text = "；".join((flags or [])[:4]) or "暂无明显执行窗口备注。"
    return _score_tooltip_lines(
        "买点执行窗口分：专门平衡量价分和短期分，回答“现在能不能下手”，不是替代量价主线。",
        "简易公式：执行窗口分 = 量价结构 + 日K买点 + 短期分是否刚转强 + MA20乖离 + 相对20日高点回撤 + 回踩量能。",
        current,
        "操作区间：75分以上：若有日K买点且风险可控，可按计划执行；65-74分：只做观察执行，等回踩不破支撑/VWAP；55-64分：等待确认，不急买；55分以下：不追高或优先防守。",
        f"当前备注：{flag_text}",
    )


def scan_action_code(result: AnalysisResult) -> str:
    short_view = getattr(result, "short_term_view", None)
    long_view = getattr(result, "long_term_view", None)
    if short_view is not None:
        code = str(short_view.action_code)
        if code in {"buy", "add", "wait"}:
            return code
        if code == "hold":
            return "watch"
        if long_view is not None and long_view.advice in {"\u957f\u671f\u53ef\u5173\u6ce8", "\u957f\u671f\u8c28\u614e\u5173\u6ce8"} and code not in {"avoid", "sell"}:
            return "watch"
        return code
    classification = getattr(result, "classification", None)
    if classification is not None:
        return str(classification.action).lower()
    decision = getattr(result, "decision", None)
    if decision is not None:
        return str(decision.action_code)
    return ""


def render_scan_output(table: pd.DataFrame, scan_results: list[AnalysisResult], min_score: int = 0) -> None:
    if table.empty:
        st.warning("\u6ca1\u6709\u547d\u4e2d\u5019\u9009\u3002\u53ef\u4ee5\u964d\u4f4e\u6700\u4f4e\u5165\u9009\u5206\uff0c\u6216\u6362\u4e00\u4e2a\u80a1\u7968\u6c60\u3002")
        return
    display = localize_scan_table(table.copy())
    display = filter_scan_display_by_min_score(display, min_score)
    if display.empty:
        st.warning("\u6ca1\u6709\u547d\u4e2d\u5019\u9009\u3002\u53ef\u4ee5\u964d\u4f4e\u6700\u4f4e\u5165\u9009\u5206\uff0c\u6216\u6362\u4e00\u4e2a\u80a1\u7968\u6c60\u3002")
        return
    if "\u5019\u9009\u5206" in display.columns:
        display = display.sort_values("\u5019\u9009\u5206", ascending=False)
    elif "candidate_score" in display.columns:
        display = display.sort_values("candidate_score", ascending=False)
    compact = compact_scan_display(display)
    st.caption(f"\u5f53\u524d\u53ea\u663e\u793a\u5019\u9009\u5206\u4e0d\u4f4e\u4e8e {min_score} \u5206\u7684\u80a1\u7968\uff1b\u4e0b\u8f7d CSV \u4f1a\u4fdd\u7559\u5b8c\u6574\u8bca\u65ad\u5b57\u6bb5\u3002")
    st.dataframe(compact, use_container_width=True, hide_index=True, column_config=scan_column_config(compact))
    st.download_button(
        "\u4e0b\u8f7d\u626b\u63cf\u7ed3\u679cCSV",
        data=display.to_csv(index=False, encoding="utf-8-sig"),
        file_name="candidates.csv",
        mime="text/csv",
        use_container_width=True,
    )
    render_scan_save_buttons(filter_scan_results_by_display(scan_results, display))


def render_scan_save_buttons(results: list[AnalysisResult]) -> None:
    if not results:
        return
    st.markdown("**加入备选池**")
    saved_symbols = {item.get("symbol", "") for item in load_watchlist_records()}
    ordered = sorted(results, key=scan_save_sort_score, reverse=True)
    for result in ordered:
        long_view = getattr(result, "long_term_view", None)
        short_view = getattr(result, "short_term_view", None)
        with st.container(border=True):
            cols = st.columns([0.12, 0.28, 0.22, 0.22, 0.16], gap="medium")
            cols[0].markdown(f"**{result.symbol}**")
            cols[1].markdown(factor_score_badges(result), unsafe_allow_html=True)
            cols[2].write(long_view.advice if long_view is not None else "长期-")
            cols[3].write(short_view.advice if short_view is not None else "短期-")
            already_saved = result.symbol in saved_symbols
            label = "更新备选池" if already_saved else "加入备选池"
            if cols[4].button(label, key=f"save_scan_{result.symbol}_{result.as_of.date()}", use_container_width=True):
                save_watchlist_result(result, source="股票池扫描")
                st.success(f"{result.symbol} 已保存到我的备选池。")


def watchlist_sort_score(record: dict) -> float:
    factors = record.get("factor_scores") or {}
    long_view = record.get("long_term_view") or {}
    short_view = record.get("short_term_view") or {}
    score = record.get("score") or {}
    values = [
        factors.get("overall_score"),
        factors.get("long_term_score"),
        factors.get("short_term_score"),
        short_view.get("liangjia_score"),
        long_view.get("score"),
        score.get("total"),
    ]
    numeric = []
    for value in values:
        try:
            if value not in [None, ""]:
                numeric.append(float(value))
        except (TypeError, ValueError):
            pass
    return max(numeric) if numeric else 0.0


def sort_watchlist_records(records: list[dict]) -> list[dict]:
    return sorted(
        records,
        key=lambda item: (
            watchlist_sort_score(item),
            item.get("saved_at", ""),
            item.get("symbol", ""),
        ),
        reverse=True,
    )


def watchlist_summary_row(record: dict) -> dict:
    factors = record.get("factor_scores") or {}
    long_view = record.get("long_term_view") or {}
    short_view = record.get("short_term_view") or {}
    structure = record.get("structure", {}) or {}
    signals = record.get("signals", [])
    status = watchlist_display_status(record)
    execution = watchlist_execution_status(record)
    return {
        "代码": record.get("symbol", ""),
        "候选分": round(watchlist_sort_score(record)),
        "长期投资价值": long_view.get("advice", ""),
        "长期分": round(float(long_view.get("score") or factors.get("long_term_score") or 0)),
        "短期交易机会": short_view.get("advice", record.get("suggestion", "")),
        "量价分": round(float(short_view.get("liangjia_score") or record.get("score", {}).get("total", 0))),
        "信号方向": short_view.get("signal_direction", ""),
        "信号强度": short_view.get("signal_strength", ""),
        "综合分": round(float(factors.get("overall_score") or 0)),
        "操作状态": execution_status_label(execution),
        "风控状态": operation_status_label(status),
        "实时/对照价": price_text(record.get("manual_price")),
        "买入区间": watchlist_entry_zone_text(signals),
        "止损参考": watchlist_stop_text(signals),
        "信号日期": record.get("as_of", ""),
        "收盘": round(float(record.get("close", 0)), 2),
        "结构": structure.get("stage", ""),
        "来源": record.get("source", ""),
    }


def style_watchlist_summary(summary: pd.DataFrame):
    def style_row(row) -> list[str]:
        status = row.get("操作状态", "")
        short_action = row.get("短期交易机会", "")
        if status in {"信号失效", "已失效"} or short_action in {"短期卖出", "短期回避"}:
            return ["background-color: #fee2e2; color: #7f1d1d"] * len(row)
        if short_action in {"短期买入", "短期加仓", "短期持有"}:
            return ["background-color: #dcfce7; color: #14532d"] * len(row)
        if short_action == "短期等待":
            return ["background-color: #fef9c3; color: #713f12"] * len(row)
        return ["background-color: #f8fafc; color: #334155"] * len(row)

    return summary.style.apply(style_row, axis=1)


def watchlist_summary_row(record: dict) -> dict:
    factors = record.get("factor_scores") or {}
    long_view = record.get("long_term_view") or {}
    short_view = record.get("short_term_view") or {}
    structure = record.get("structure", {}) or {}
    signals = record.get("signals", [])
    status = watchlist_display_status(record)
    execution = watchlist_execution_status(record)
    return {
        "\u4ee3\u7801": record.get("symbol", ""),
        "交易分层": watchlist_trade_tier(record),
        "\u5019\u9009\u5206": round(watchlist_sort_score(record)),
        "\u957f\u671f\u6295\u8d44\u4ef7\u503c": long_view.get("advice", ""),
        "\u957f\u671f\u5206": round(float(long_view.get("score") or factors.get("long_term_score") or 0)),
        "\u77ed\u671f\u4ea4\u6613\u673a\u4f1a": short_view.get("advice", record.get("suggestion", "")),
        "\u77ed\u671f\u5206": round(float(short_view.get("short_term_score") or factors.get("short_term_score") or 0)),
        "\u91cf\u4ef7\u5206": round(float(short_view.get("liangjia_score") or record.get("score", {}).get("total", 0))),
        "\u4fe1\u53f7\u65b9\u5411": short_view.get("signal_direction", ""),
        "\u4fe1\u53f7\u5f3a\u5ea6": short_view.get("signal_strength", ""),
        "\u7efc\u5408\u5206": round(float(factors.get("overall_score") or 0)),
        "\u64cd\u4f5c\u72b6\u6001": execution_status_label(execution),
        "\u98ce\u63a7\u72b6\u6001": operation_status_label(status),
        "\u5b9e\u65f6/\u5bf9\u7167\u4ef7": price_text(record.get("manual_price")),
        "\u4e70\u5165\u533a\u95f4": watchlist_entry_zone_text(signals),
        "\u6b62\u635f\u53c2\u8003": watchlist_stop_text(signals),
        "盘前动作": watchlist_preopen_action(record),
        "盘中确认": watchlist_intraday_rule(record),
        "风险处理": watchlist_risk_action(record),
        "\u4fe1\u53f7\u65e5\u671f": record.get("as_of", ""),
        "最近刷新": record.get("last_refreshed_at", ""),
        "刷新错误": record.get("refresh_error", ""),
        "\u6536\u76d8": round(float(record.get("close", 0)), 2),
        "\u7ed3\u6784": structure.get("stage", ""),
        "\u6765\u6e90": record.get("source", ""),
    }


def render_watchlist_trade_plan(records: list[dict]) -> None:
    plan_rows = [watchlist_plan_row(record) for record in records]
    if not plan_rows:
        return
    plan = pd.DataFrame(plan_rows)
    counts = plan["交易分层"].value_counts()
    cols = st.columns(4, gap="medium")
    cols[0].markdown(metric_card("A类可执行", str(int(counts.get("A-明日可执行", 0))), "盘中确认后执行"), unsafe_allow_html=True)
    cols[1].markdown(metric_card("B类观察", str(int(counts.get("B-等待观察", 0))), "只等买点触发"), unsafe_allow_html=True)
    cols[2].markdown(metric_card("C类剔除/防守", str(int(counts.get("C-先剔除", 0))), "风险优先处理"), unsafe_allow_html=True)
    cols[3].markdown(metric_card("备选总数", str(len(records)), "先计划，后执行"), unsafe_allow_html=True)

    st.markdown("**明日交易计划**")
    st.caption("A类是明天重点盯盘对象；B类只观察不追；C类优先剔除或防守。盘中买入仍必须服从日K量价买点、买入区间和分时确认。")
    tabs = st.tabs(["A类可执行", "B类观察", "C类剔除/防守", "全部计划"])
    tab_filters = [
        plan[plan["交易分层"] == "A-明日可执行"],
        plan[plan["交易分层"] == "B-等待观察"],
        plan[plan["交易分层"] == "C-先剔除"],
        plan,
    ]
    for index, (tab, frame) in enumerate(zip(tabs, tab_filters)):
        with tab:
            if frame.empty:
                st.info("当前没有这一类标的。")
            else:
                st.dataframe(frame, use_container_width=True, hide_index=True)
                st.download_button(
                    "下载当前计划CSV",
                    data=frame.to_csv(index=False, encoding="utf-8-sig"),
                    file_name="watchlist_trade_plan.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"download_plan_{index}",
                )


def render_watchlist_bulk_price_update(records: list[dict]) -> None:
    with st.expander("盘中批量更新对照价", expanded=False):
        st.caption("从行情软件或表格粘贴“代码 价格”即可，每行一只；支持空格、逗号、制表符分隔。保存后系统会重新判断未到买点、可执行、偏离买点或已失效。")
        sample = "\n".join(f"{record.get('symbol', '')} {price_text(record.get('manual_price'), price_text(record.get('close')))}" for record in records[:5])
        text = st.text_area("批量价格", value="", placeholder=sample, height=150)
        cols = st.columns([0.22, 0.78], gap="medium")
        if cols[0].button("保存批量价格", type="primary", use_container_width=True):
            updates, errors = parse_watchlist_price_updates(text)
            if not updates and not errors:
                st.warning("没有识别到可更新的价格。")
                return
            updated = update_watchlist_manual_prices(updates)
            if updated:
                st.success(f"已更新 {updated} 只股票的对照价。")
            if errors:
                st.warning("未识别的行：" + "；".join(errors[:5]))
            st.rerun()
        cols[1].caption("建议开盘后、10点后、午后各更新一次价格，用同一套买入区间和失效价判断，不临时追涨。")


def parse_watchlist_price_updates(text: str) -> tuple[dict[str, float], list[str]]:
    updates: dict[str, float] = {}
    errors: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.replace(",", " ").replace("，", " ").replace("\t", " ").split() if part.strip()]
        if len(parts) < 2:
            errors.append(line)
            continue
        symbol = parts[0]
        try:
            price = float(parts[1])
        except ValueError:
            errors.append(line)
            continue
        if price <= 0:
            errors.append(line)
            continue
        updates[symbol] = price
    return updates, errors


def update_watchlist_manual_prices(updates: dict[str, float]) -> int:
    if not updates:
        return 0
    records = load_watchlist_records()
    updated = 0
    now_text = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    for record in records:
        symbol = str(record.get("symbol", "")).strip()
        if symbol in updates:
            record["manual_price"] = float(updates[symbol])
            record["status_updated_at"] = now_text
            updated += 1
    if updated:
        save_watchlist_records(records)
    return updated


def watchlist_plan_row(record: dict) -> dict:
    signals = record.get("signals", [])
    short_view = record.get("short_term_view") or {}
    radar = record.get("radar") or {}
    return {
        "代码": record.get("symbol", ""),
        "交易分层": watchlist_trade_tier(record),
        "候选分": round(watchlist_sort_score(record)),
        "短期机会": short_view.get("advice", record.get("suggestion", "")),
        "操作状态": execution_status_label(watchlist_execution_status(record)),
        "买入区间": watchlist_entry_zone_text(signals),
        "止损参考": watchlist_stop_text(signals),
        "卖出风险": price_text(radar.get("卖出风险分")),
        "盘前动作": watchlist_preopen_action(record),
        "盘中确认": watchlist_intraday_rule(record),
        "风险处理": watchlist_risk_action(record),
    }


def watchlist_trade_tier(record: dict) -> str:
    short_view = record.get("short_term_view") or {}
    radar = record.get("radar") or {}
    stage = str((record.get("structure") or {}).get("stage", ""))
    short_action = str(short_view.get("advice", record.get("suggestion", "")))
    execution = watchlist_execution_status(record)
    exit_risk = _score_value(radar.get("卖出风险分"), 0.0)
    entry_quality = _score_value(radar.get("买点质量分"), 0.0)
    has_buy_zone = watchlist_best_entry_zone(record.get("signals", [])) is not None

    if (
        execution == "invalid"
        or short_action in {"短期卖出", "短期回避"}
        or exit_risk >= 70
        or any(word in stage for word in ["主跌", "筑顶", "高位"])
    ):
        return "C-先剔除"
    if (
        short_action in {"短期买入", "短期加仓"}
        and has_buy_zone
        and exit_risk < 55
        and (entry_quality == 0 or entry_quality >= 65)
        and execution not in {"extended", "invalid"}
    ):
        return "A-明日可执行"
    return "B-等待观察"


def watchlist_preopen_action(record: dict) -> str:
    tier = watchlist_trade_tier(record)
    if tier == "A-明日可执行":
        return "开盘前保留重点盯盘，不集合竞价追入"
    if tier == "B-等待观察":
        return "保留观察，只等回踩/突破确认"
    return "从买入清单移出，已有仓位先看风险"


def watchlist_intraday_rule(record: dict) -> str:
    tier = watchlist_trade_tier(record)
    execution = watchlist_execution_status(record)
    if tier == "C-先剔除":
        return "分时不能单独改成买入"
    if execution == "actionable":
        return "价格在买入区间，等VWAP上方承接或回踩不破"
    if execution == "waiting":
        return "未到买点，等价格进入区间再看分时"
    if execution == "extended":
        return "偏离买点，不追高，等回落确认"
    return "先填实时价，再判断区间和分时"


def watchlist_risk_action(record: dict) -> str:
    radar = record.get("radar") or {}
    execution = watchlist_execution_status(record)
    exit_risk = _score_value(radar.get("卖出风险分"), 0.0)
    if execution == "invalid":
        return "跌破失效价，放弃买入或减仓处理"
    if exit_risk >= 70:
        return "卖出风险高，优先保护本金和利润"
    stop_text = watchlist_stop_text(record.get("signals", []))
    if stop_text:
        return f"执行前确认止损：{stop_text}"
    return "没有明确止损时不执行"


def watchlist_detail_plan_text(record: dict) -> str:
    return (
        f"交易分层：{watchlist_trade_tier(record)}。"
        f"盘前动作：{watchlist_preopen_action(record)}。"
        f"盘中确认：{watchlist_intraday_rule(record)}。"
        f"风险处理：{watchlist_risk_action(record)}。"
    )


def style_watchlist_summary(summary: pd.DataFrame):
    def style_row(row) -> list[str]:
        status = row.get("\u64cd\u4f5c\u72b6\u6001", "")
        short_action = row.get("\u77ed\u671f\u4ea4\u6613\u673a\u4f1a", "")
        tier = row.get("交易分层", "")
        if tier == "C-先剔除" or status in {"\u4fe1\u53f7\u5931\u6548", "\u5df2\u5931\u6548"} or short_action in {"\u77ed\u671f\u5356\u51fa", "\u77ed\u671f\u56de\u907f"}:
            return ["background-color: #fee2e2; color: #7f1d1d"] * len(row)
        if tier == "A-明日可执行" or short_action in {"\u77ed\u671f\u4e70\u5165", "\u77ed\u671f\u52a0\u4ed3", "\u77ed\u671f\u6301\u6709"}:
            return ["background-color: #dcfce7; color: #14532d"] * len(row)
        if tier == "B-等待观察" or short_action == "\u77ed\u671f\u7b49\u5f85":
            return ["background-color: #fef9c3; color: #713f12"] * len(row)
        return ["background-color: #f8fafc; color: #334155"] * len(row)

    return summary.style.apply(style_row, axis=1)


def _score_component(scores, group: str, key: str, default: float = 0.0) -> float:
    try:
        components = getattr(scores, "components", {}) or {}
        if not isinstance(components, dict):
            return default
        item = components.get(group, default)
        if isinstance(item, dict):
            return float(item.get(key, default))
        if key in {"", None}:
            return float(item)
        return default
    except (TypeError, ValueError, AttributeError):
        return default


def price_text(value, empty: str = "") -> str:
    if value in [None, ""]:
        return empty
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _score_value(value, default: float = 0.0) -> float:
    try:
        if value in [None, ""]:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_tooltip_lines(*parts: str) -> str:
    return "\n".join(part for part in parts if part)


def score_badge_tooltip(result: AnalysisResult, label: str) -> str:
    scores = getattr(result, "factor_scores", None)
    if scores is None:
        score = getattr(result, "score", None)
        return _score_tooltip_lines(
            "量价分：来自原始量价供需评分。",
            "简易公式：结构分 + 位置分 + 量价信号分 + 风险分 + 相对强度分。",
            "" if score is None else f"当前拆分：结构{score.structure}、位置{score.position}、信号{score.signal}、风险{score.risk}、相对强度{score.relative_strength}。",
        )

    if label == "\u7efc\u5408":
        risk_penalty = _score_value((getattr(scores, "components", {}) or {}).get("risk_penalty"))
        return _score_tooltip_lines(
            "综合分：用于排序参考，不是单一最终买卖结论。",
            "简易公式：综合分 = 量价分*45% + 短期分*25% + 长期分*25% - 风险惩罚*5%。",
            f"当前代入：{scores.liangjia_score:.0f}*45% + {scores.short_term_score:.0f}*25% + {scores.long_term_score:.0f}*25% - {risk_penalty:.0f}*5% = {scores.overall_score:.0f}。",
            "注意：短期买入仍必须由量价买点触发，长期分高不会自动变成短期买入。",
        )

    if label == "\u91cf\u4ef7":
        score = getattr(result, "score", None)
        base = _score_component(scores, "liangjia", "base_score", scores.liangjia_score)
        market = _score_component(scores, "liangjia", "market_modifier")
        exit_risk = _score_component(scores, "liangjia", "exit_risk_modifier")
        breakdown = ""
        if score is not None:
            breakdown = (
                f"基础量价分拆分：结构{getattr(score, 'structure', 0)} + "
                f"位置{getattr(score, 'position', 0)} + "
                f"信号{getattr(score, 'signal', 0)} + "
                f"风险{getattr(score, 'risk', 0)} + "
                f"相对强度{getattr(score, 'relative_strength', 0)} = {getattr(score, 'total', base)}。"
            )
        return _score_tooltip_lines(
            "量价分：衡量价格 + 成交量 + 所处位置形成的供需结构质量。",
            "简易公式：量价分 = 基础量价分 + 市场环境修正 + 卖出风险修正。",
            breakdown,
            f"当前代入：基础{base:.0f} + 市场修正{market:+.0f} + 卖出风险修正{exit_risk:+.0f} = {scores.liangjia_score:.0f}。",
            "用到指标：四阶段结构、关键支撑/压力距离、突破/回踩/反转/滞涨信号、止损距离、相对强度、大盘环境、卖出风险。",
        )

    if label == "\u77ed\u671f":
        return_score = _score_component(scores, "short_term", "return_score")
        ma_score = _score_component(scores, "short_term", "ma_score")
        volume_score = _score_component(scores, "short_term", "volume_score")
        momentum_score = _score_component(scores, "short_term", "momentum_score")
        breakout_score = _score_component(scores, "short_term", "breakout_score")
        risk_score = _score_component(scores, "short_term", "risk_score")
        return _score_tooltip_lines(
            "短期分：衡量近期走势是否支持短期交易机会。",
            "简易公式：短期分 = (动量*30 + 均线*12 + 量能*12 + 技术动量*12 + 新高*8 + 风险*6) / 80。",
            f"当前子分：动量{return_score:.0f}、均线{ma_score:.0f}、量能{volume_score:.0f}、技术动量{momentum_score:.0f}、新高{breakout_score:.0f}、风险{risk_score:.0f}，合成短期分 {scores.short_term_score:.0f}。",
            "用到指标：1/3/5/10/20日涨跌幅、MA5/MA10/MA20、成交量放大倍数/量比、RSI、MACD柱、相对大盘强弱、20/60日新高、短期回撤、ATR。",
        )

    if label == "\u957f\u671f":
        trend_score = _score_component(scores, "long_term", "trend_score")
        return_score = _score_component(scores, "long_term", "return_score")
        quality_score = _score_component(scores, "long_term", "quality_score")
        growth_score = _score_component(scores, "long_term", "growth_score")
        valuation_score = _score_component(scores, "long_term", "valuation_score")
        risk_score = _score_component(scores, "long_term", "risk_score")
        return _score_tooltip_lines(
            "长期分：衡量股票是否值得中长期关注，不直接触发短期买入。",
            "简易公式：长期分 = 趋势*35% + 长期收益*20% + 基本面质量*20% + 成长*10% + 估值*10% + 财务风险*5%。",
            f"当前子分：趋势{trend_score:.0f}、长期收益{return_score:.0f}、质量{quality_score:.0f}、成长{growth_score:.0f}、估值{valuation_score:.0f}、风险{risk_score:.0f}，合成长期分 {scores.long_term_score:.0f}。",
            "用到指标：MA60/MA120/MA250、60/120/250日收益、ROE/ROA/毛利率/净利率/现金流质量、营收和利润增速、PE/PB/PEG及估值分位、资产负债率、有息负债率、长期回撤。",
        )

    return "该分数用于辅助排序和解释；长期价值与短期交易机会分开判断。"


def metric_score_tooltip(result: AnalysisResult, label: str) -> str:
    mapping = {
        "\u91cf\u4ef7\u5206": "\u91cf\u4ef7",
        "\u7efc\u5408\u5206": "\u7efc\u5408",
        "\u77ed\u671f\u5206": "\u77ed\u671f",
        "\u957f\u671f\u5206": "\u957f\u671f",
    }
    if label in mapping:
        return score_badge_tooltip(result, mapping[label])
    if label == "\u5019\u9009\u5206":
        scores = getattr(result, "factor_scores", None)
        if scores is not None:
            execution_score, _ = execution_window_from_scores(scores)
            balanced_short = min(float(scores.short_term_score), float(execution_score) + 8.0)
            balanced_liangjia = min(float(scores.liangjia_score), float(execution_score) + 12.0)
            tradable_score = float(execution_score) * 0.52 + balanced_liangjia * 0.26 + balanced_short * 0.17 + float(scores.long_term_score) * 0.05
            return _score_tooltip_lines(
                "候选分：用于扫描入选和备选池排序，不是最终买卖结论。",
                "交易候选分公式：执行窗口分*52% + 平衡后量价分*26% + 平衡后短期分*17% + 长期分*5%。长期分只保留很小辅助权重。",
                f"当前交易候选代入：{execution_score:.0f}*52% + {balanced_liangjia:.0f}*26% + {balanced_short:.0f}*17% + {scores.long_term_score:.0f}*5% = {tradable_score:.0f}。",
                f"最终候选分还会参考折扣后的综合分和执行窗口分，当前显示为 {scan_candidate_score(result):.0f}。",
            )
        return "候选分：用于扫描入选和排序；当前缺少多因子分时，退回量价分。"
    if label == "\u5f3a\u80a1\u8bbe\u7f6e":
        return _score_tooltip_lines(
            "强股设置分：强股雷达的辅助评分。",
            "简易组合：市场状态 + 板块强度 + 个股相对强度 + 结构改善 + 是否过度乖离。",
            "它只决定关注优先级，不单独触发短期买入。",
        )
    if label == "\u4e70\u70b9\u8d28\u91cf":
        return _score_tooltip_lines(
            "买点质量分：衡量当前量价买点是否值得执行。",
            "简易组合：支撑来源 + 回踩缩量 + K线收窄 + 未跌破支撑 + 止损距离 + 盈亏比。",
            "只有原量价买点存在时，这个分数才用于提高执行优先级。",
        )
    if label == "\u5356\u51fa\u98ce\u9669":
        return _score_tooltip_lines(
            "卖出风险分：衡量供应是否开始压过需求。",
            "简易组合：高位放量滞涨 + 向下反转/跌破关键位 + 需求衰竭 + 相对强度转弱 + 板块走弱。",
            "风险信号优先级高于机会信号。",
        )
    return "该分数用于辅助判断；长期和短期分开解释，短期动作仍遵守量价信号主流程。"


def score_badge_tooltip(result: AnalysisResult, label: str) -> str:
    scores = getattr(result, "factor_scores", None)
    if scores is None:
        score = getattr(result, "score", None)
        return _score_tooltip_lines(
            "量价分：来自原始量价供需评分。",
            "简易公式：结构分 + 位置分 + 量价信号分 + 风险分 + 相对强度分。",
            "" if score is None else (
                f"当前拆分：结构{getattr(score, 'structure', 0)}、位置{getattr(score, 'position', 0)}、"
                f"信号{getattr(score, 'signal', 0)}、风险{getattr(score, 'risk', 0)}、"
                f"相对强度{getattr(score, 'relative_strength', 0)}。"
            ),
        )

    if label == "\u7efc\u5408":
        risk_penalty = _score_value((getattr(scores, "components", {}) or {}).get("risk_penalty"))
        return _score_tooltip_lines(
            "综合分：用于排序参考，不是单一最终买卖结论。",
            "简易公式：Overall Score = Liangjia Score*45% + Short-term Score*25% + Long-term Score*25% - Risk Penalty*5%。",
            f"当前代入：{scores.liangjia_score:.0f}*45% + {scores.short_term_score:.0f}*25% + {scores.long_term_score:.0f}*25% - {risk_penalty:.0f}*5% = {scores.overall_score:.0f}。",
            "注意：短期买入仍必须由量价买点触发，长期分高不会自动变成短期买入。",
        )

    if label == "\u91cf\u4ef7":
        score = getattr(result, "score", None)
        base = _score_component(scores, "liangjia", "base_score", scores.liangjia_score)
        market = _score_component(scores, "liangjia", "market_modifier")
        exit_risk = _score_component(scores, "liangjia", "exit_risk_modifier")
        breakdown = ""
        if score is not None:
            breakdown = (
                f"基础量价分拆分：结构{getattr(score, 'structure', 0)} + "
                f"位置{getattr(score, 'position', 0)} + "
                f"信号{getattr(score, 'signal', 0)} + "
                f"风险{getattr(score, 'risk', 0)} + "
                f"相对强度{getattr(score, 'relative_strength', 0)} = {getattr(score, 'total', base)}。"
            )
        return _score_tooltip_lines(
            "量价分：衡量价格 Price + 成交量 Volume + 位置 Position 形成的供需结构质量。",
            "简易公式：Liangjia Score = Base Volume-Price Score + Market Modifier + Exit-risk Modifier。",
            breakdown,
            f"当前代入：基础{base:.0f} + 市场修正{market:+.0f} + 卖出风险修正{exit_risk:+.0f} = {scores.liangjia_score:.0f}。",
            "指标对照：Stage/Trend Structure 四阶段结构、Support/Resistance 支撑压力、Breakout 突破、Pullback 回踩、Reversal 反转、Volume Stalling 放量滞涨、Stop Distance 止损距离、Relative Strength 相对强度。",
        )

    if label == "\u77ed\u671f":
        return_score = _score_component(scores, "short_term", "return_score")
        ma_score = _score_component(scores, "short_term", "ma_score")
        volume_score = _score_component(scores, "short_term", "volume_score")
        momentum_score = _score_component(scores, "short_term", "momentum_score")
        breakout_score = _score_component(scores, "short_term", "breakout_score")
        risk_score = _score_component(scores, "short_term", "risk_score")
        return _score_tooltip_lines(
            "短期分 Short-term Score：衡量近期走势是否支持短期交易机会。",
            "简易公式：Short-term Score = (Return Momentum*30 + MA Trend*12 + Volume*12 + Technical Momentum*12 + New High*8 + Short Risk*6) / 80。",
            f"当前子分：Return Momentum动量{return_score:.0f}、MA Trend均线{ma_score:.0f}、Volume量能{volume_score:.0f}、Technical Momentum技术动量{momentum_score:.0f}、New High新高{breakout_score:.0f}、Short Risk风险{risk_score:.0f}，合成短期分 {scores.short_term_score:.0f}。",
            "英文指标对照：Return_1D/3D/5D/10D/20D 涨跌幅，MA5/MA10/MA20 短均线，Volume Ratio/Volume Expansion 量比/放量倍数，RSI，MACD Histogram，KDJ，ATR，20D High/60D High 阶段新高，Short Max Drawdown 短期最大回撤，Relative Strength vs Benchmark/Industry 相对大盘/行业强弱。",
        )

    if label == "\u957f\u671f":
        trend_score = _score_component(scores, "long_term", "trend_score")
        return_score = _score_component(scores, "long_term", "return_score")
        quality_score = _score_component(scores, "long_term", "quality_score")
        growth_score = _score_component(scores, "long_term", "growth_score")
        valuation_score = _score_component(scores, "long_term", "valuation_score")
        risk_score = _score_component(scores, "long_term", "risk_score")
        return _score_tooltip_lines(
            "长期分 Long-term Score：衡量股票是否值得中长期关注，不直接触发短期买入。",
            "简易公式：Long-term Score = Trend*35% + Long Return*20% + Quality*20% + Growth*10% + Valuation*10% + Financial Risk*5%。",
            f"当前子分：Trend趋势{trend_score:.0f}、Long Return长期收益{return_score:.0f}、Quality质量{quality_score:.0f}、Growth成长{growth_score:.0f}、Valuation估值{valuation_score:.0f}、Financial Risk财务风险{risk_score:.0f}，合成长期分 {scores.long_term_score:.0f}。",
            "英文指标对照：MA60/MA120/MA250 长期均线，Return_60D/120D/250D 长期收益，ROE，ROA，Gross Margin 毛利率，Net Margin 净利率，Revenue Growth 营收增速，Net Profit Growth 净利润增速，Deducted Profit Growth 扣非净利润增速，PE/PB/PS/PEG，PE Percentile/PB Percentile 估值分位，Operating Cash Flow to Net Profit 经营现金流/净利润，Debt to Assets 资产负债率，Interest-bearing Debt Ratio 有息负债率，Long Max Drawdown 长期最大回撤，Long Volatility 长期波动率。",
        )

    return "该分数用于辅助排序和解释；长期价值与短期交易机会分开判断。"


# Final tooltip override after the enhanced short-term technical scoring.
def metric_score_tooltip(result: AnalysisResult, label: str) -> str:
    return score_badge_tooltip(result, label)


def score_badge_tooltip(result: AnalysisResult, label: str) -> str:
    scores = getattr(result, "factor_scores", None)
    if scores is None:
        return "暂无多因子评分明细。"

    risk_penalty = _score_component(scores, "risk_penalty", "", 0.0)
    if label in {"\u5019\u9009", "\u5019\u9009\u5206"}:
        execution_score, _ = execution_window_from_scores(scores)
        balanced_short = min(float(scores.short_term_score), float(execution_score) + 8.0)
        balanced_liangjia = min(float(scores.liangjia_score), float(execution_score) + 12.0)
        tradable_score = float(execution_score) * 0.52 + balanced_liangjia * 0.26 + balanced_short * 0.17 + float(scores.long_term_score) * 0.05
        return _score_tooltip_lines(
            "候选分 Candidate Score：用于扫描表格和下方加入备选池列表的统一排序。",
            "交易候选分公式：执行窗口分*52% + 平衡后量价分*26% + 平衡后短期分*17% + 长期分*5%。",
            f"当前交易候选代入：{execution_score:.0f}*52% + {balanced_liangjia:.0f}*26% + {balanced_short:.0f}*17% + {scores.long_term_score:.0f}*5% = {tradable_score:.0f}。",
            f"最终候选分还会参考折扣后的综合分、长期分和执行窗口分，当前显示为 {scan_candidate_score(result):.0f}。",
        )

    if label in {"综合", "综合分"}:
        return _score_tooltip_lines(
            "综合分 Overall Score：用于排序和候选池展示，不是单一买卖结论。",
            "简易公式：Overall = 量价分*45% + 短期分*25% + 长期分*25% - 风险惩罚*5%。",
            f"当前代入：{scores.liangjia_score:.0f}*45% + {scores.short_term_score:.0f}*25% + {scores.long_term_score:.0f}*25% - {risk_penalty:.0f}*5% = {scores.overall_score:.0f}。",
        )

    if label in {"量价", "量价分"}:
        item = (getattr(scores, "components", {}) or {}).get("liangjia", {}) or {}
        factor_modifier = item.get("factor_modifier", 0)
        volume_position = item.get("volume_position_modifier", 0)
        money_flow = item.get("money_flow_modifier", 0)
        trend_participation = item.get("trend_participation_modifier", 0)
        position_compression = item.get("position_compression_modifier", 0)
        volatility_modifier = item.get("volatility_modifier", 0)
        return _score_tooltip_lines(
            "量价分 Liangjia Score：衡量价格、成交量、位置和供需结构，是短期买卖动作的主线。",
            "简易公式：量价分 = 原量价结构分 + 市场修正 + 卖出风险修正 + 供需确认修正。供需确认只使用与成交量、收盘位置、资金流、趋势参与和波动收缩相关的指标。",
            f"当前主项：基础{item.get('base_score', scores.liangjia_score):.0f}、结构{item.get('structure', 0):.0f}、位置{item.get('position', 0):.0f}、信号{item.get('signal', 0):.0f}、风险{item.get('risk', 0):.0f}、市场修正{item.get('market_modifier', 0):.0f}、卖出风险修正{item.get('exit_risk_modifier', 0):.0f}、供需确认修正{factor_modifier:.0f}，合成量价分 {scores.liangjia_score:.0f}。",
            f"供需确认明细：量能与收盘位置{volume_position:.0f}、OBV/MFI资金流{money_flow:.0f}、ADX/DMI趋势参与{trend_participation:.0f}、BOLL/波动收缩位置{position_compression:.0f}、ATR/回撤风险{volatility_modifier:.0f}。",
            "英文指标对照：Volume Ratio，Close Position，Price Position 20D，OBV/OBV MA20，MFI14，ADX14/DI+/DI-，BOLL Percent B/Bandwidth，Volatility Ratio，ATR%，Short Max Drawdown。",
        )

    if label in {"短期", "短期分"}:
        return_score = _score_component(scores, "short_term", "return_score")
        ma_score = _score_component(scores, "short_term", "ma_score")
        volume_score = _score_component(scores, "short_term", "volume_score")
        momentum_score = _score_component(scores, "short_term", "momentum_score")
        rsi_score = _score_component(scores, "short_term", "rsi_score")
        macd_score = _score_component(scores, "short_term", "macd_score")
        kdj_score = _score_component(scores, "short_term", "kdj_score")
        rs_score = _score_component(scores, "short_term", "relative_strength_score")
        breakout_score = _score_component(scores, "short_term", "breakout_score")
        risk_score = _score_component(scores, "short_term", "risk_score")
        return _score_tooltip_lines(
            "短期分 Short-term Score：用于确认近期走势优势和交易机会，重点看短线动量、短均线、量能和技术指标共振。",
            "简易公式：Short-term = Return Momentum*30% + MA Trend*12% + Volume*12% + Technical Momentum*12% + New High*8% + Short Risk*6%。Technical Momentum 内部由 RSI*28% + MACD*32% + KDJ*25% + Relative Strength*15% 合成。",
            f"当前一级子分：Return Momentum动量{return_score:.0f}、MA Trend均线{ma_score:.0f}、Volume量能{volume_score:.0f}、Technical Momentum技术动量{momentum_score:.0f}、New High新高{breakout_score:.0f}、Short Risk风险{risk_score:.0f}，合成短期分 {scores.short_term_score:.0f}。",
            f"技术动量明细：RSI强弱{rsi_score:.0f}、MACD趋势/金叉/柱线{macd_score:.0f}、KDJ位置/交叉/J值变化{kdj_score:.0f}、Relative Strength相对强弱{rs_score:.0f}。",
            "英文指标对照：RSI6/RSI14，MACD DIF/DEA/Histogram/Golden Cross，KDJ K/D/J/Golden Cross，Return_1D/3D/5D/10D/20D，MA5/MA10/MA20，Volume Ratio，20D High/60D High，ATR，Short Max Drawdown。",
        )

    if label in {"长期", "长期分"}:
        trend_score = _score_component(scores, "long_term", "trend_score")
        return_score = _score_component(scores, "long_term", "return_score")
        quality_score = _score_component(scores, "long_term", "quality_score")
        growth_score = _score_component(scores, "long_term", "growth_score")
        valuation_score = _score_component(scores, "long_term", "valuation_score")
        risk_score = _score_component(scores, "long_term", "risk_score")
        return _score_tooltip_lines(
            "长期分 Long-term Score：衡量股票是否值得中长期关注，不直接触发短期买入。",
            "简易公式：Long-term = Trend*35% + Long Return*20% + Quality*20% + Growth*10% + Valuation*10% + Financial Risk*5%。",
            f"当前子分：Trend趋势{trend_score:.0f}、Long Return长期收益{return_score:.0f}、Quality质量{quality_score:.0f}、Growth成长{growth_score:.0f}、Valuation估值{valuation_score:.0f}、Financial Risk财务风险{risk_score:.0f}，合成长期分 {scores.long_term_score:.0f}。",
        )

    return "该分数用于辅助排序和解释；长期价值与短期交易机会分开判断。"


# Final tooltip override after the extended short-term indicator set.
def metric_score_tooltip(result: AnalysisResult, label: str) -> str:
    return score_badge_tooltip(result, label)


def score_badge_tooltip(result: AnalysisResult, label: str) -> str:
    scores = getattr(result, "factor_scores", None)
    if scores is None:
        return "暂无多因子评分明细。"

    risk_penalty = _score_component(scores, "risk_penalty", "", 0.0)
    if label in {"\u5019\u9009", "\u5019\u9009\u5206"}:
        execution_score, _ = execution_window_from_scores(scores)
        balanced_short = min(float(scores.short_term_score), float(execution_score) + 8.0)
        balanced_liangjia = min(float(scores.liangjia_score), float(execution_score) + 12.0)
        tradable_score = float(execution_score) * 0.52 + balanced_liangjia * 0.26 + balanced_short * 0.17 + float(scores.long_term_score) * 0.05
        return _score_tooltip_lines(
            "候选分 Candidate Score：用于扫描表格和下方加入备选池列表的统一排序。",
            "交易候选分公式：执行窗口分*52% + 平衡后量价分*26% + 平衡后短期分*17% + 长期分*5%。",
            f"当前交易候选代入：{execution_score:.0f}*52% + {balanced_liangjia:.0f}*26% + {balanced_short:.0f}*17% + {scores.long_term_score:.0f}*5% = {tradable_score:.0f}。",
            f"最终候选分还会参考折扣后的综合分、长期分和执行窗口分，当前显示为 {scan_candidate_score(result):.0f}。",
        )

    if label in {"综合", "综合分"}:
        return _score_tooltip_lines(
            "综合分 Overall Score：用于排序和候选池展示，不是单一买卖结论。",
            "简易公式：Overall = 量价分*45% + 短期分*25% + 长期分*25% - 风险惩罚*5%。",
            f"当前代入：{scores.liangjia_score:.0f}*45% + {scores.short_term_score:.0f}*25% + {scores.long_term_score:.0f}*25% - {risk_penalty:.0f}*5% = {scores.overall_score:.0f}。",
        )

    if label in {"量价", "量价分"}:
        item = (getattr(scores, "components", {}) or {}).get("liangjia", {}) or {}
        return _score_tooltip_lines(
            "量价分 Liangjia Score：衡量价格、成交量、位置和供需结构，是短期买卖动作的主线。",
            "简易公式：先取原量价结构分，再根据市场状态和卖出风险修正。",
            f"当前子分：基础{item.get('base_score', scores.liangjia_score):.0f}、结构{item.get('structure', 0):.0f}、位置{item.get('position', 0):.0f}、信号{item.get('signal', 0):.0f}、风险{item.get('risk', 0):.0f}、市场修正{item.get('market_modifier', 0):.0f}、卖出风险修正{item.get('exit_risk_modifier', 0):.0f}，合成量价分 {scores.liangjia_score:.0f}。",
        )

    if label in {"短期", "短期分"}:
        return_score = _score_component(scores, "short_term", "return_score")
        ma_score = _score_component(scores, "short_term", "ma_score")
        volume_score = _score_component(scores, "short_term", "volume_score")
        momentum_score = _score_component(scores, "short_term", "momentum_score")
        rsi_score = _score_component(scores, "short_term", "rsi_score")
        macd_score = _score_component(scores, "short_term", "macd_score")
        kdj_score = _score_component(scores, "short_term", "kdj_score")
        oscillator_score = _score_component(scores, "short_term", "oscillator_score")
        bollinger_score = _score_component(scores, "short_term", "bollinger_score")
        obv_score = _score_component(scores, "short_term", "obv_score")
        trend_strength_score = _score_component(scores, "short_term", "trend_strength_score")
        rs_score = _score_component(scores, "short_term", "relative_strength_score")
        volatility_score = _score_component(scores, "short_term", "volatility_score")
        breakout_score = _score_component(scores, "short_term", "breakout_score")
        risk_score = _score_component(scores, "short_term", "risk_score")
        return _score_tooltip_lines(
            "短期分 Short-term Score：用于确认近期走势优势和交易机会，重点看短线动量、短均线、量能、趋势强度、资金流和波动质量。",
            "简易公式：Short-term = Return Momentum*30% + MA Trend*12% + Volume*12% + Technical Momentum*12% + New High*8% + Short Risk*6%。",
            "Technical Momentum = RSI*14% + MACD*20% + KDJ*14% + CCI/Williams/MFI*10% + BOLL*10% + OBV*10% + ADX/DMI*8% + Relative Strength*10% + Volatility*4%。",
            f"当前一级子分：动量{return_score:.0f}、均线{ma_score:.0f}、量能{volume_score:.0f}、技术动量{momentum_score:.0f}、新高{breakout_score:.0f}、风险{risk_score:.0f}，合成短期分 {scores.short_term_score:.0f}。",
            f"技术动量明细：RSI{rsi_score:.0f}、MACD{macd_score:.0f}、KDJ{kdj_score:.0f}、CCI/Williams/MFI{oscillator_score:.0f}、BOLL{bollinger_score:.0f}、OBV{obv_score:.0f}、ADX/DMI{trend_strength_score:.0f}、相对强度{rs_score:.0f}、波动质量{volatility_score:.0f}。",
            "英文指标对照：RSI6/RSI14，MACD DIF/DEA/Histogram/Golden Cross，KDJ K/D/J，CCI14，Williams %R14，MFI14，BOLL Percent B/Bandwidth，OBV/OBV MA20，ADX14/DI+/DI-，ATR%，Volatility Ratio，Return_1D/3D/5D/10D/20D，MA5/MA10/MA20。",
        )

    if label in {"长期", "长期分"}:
        trend_score = _score_component(scores, "long_term", "trend_score")
        return_score = _score_component(scores, "long_term", "return_score")
        quality_score = _score_component(scores, "long_term", "quality_score")
        growth_score = _score_component(scores, "long_term", "growth_score")
        valuation_score = _score_component(scores, "long_term", "valuation_score")
        risk_score = _score_component(scores, "long_term", "risk_score")
        return _score_tooltip_lines(
            "长期分 Long-term Score：衡量股票是否值得中长期关注，不直接触发短期买入。",
            "简易公式：Long-term = Trend*35% + Long Return*20% + Quality*20% + Growth*10% + Valuation*10% + Financial Risk*5%。",
            f"当前子分：趋势{trend_score:.0f}、长期收益{return_score:.0f}、质量{quality_score:.0f}、成长{growth_score:.0f}、估值{valuation_score:.0f}、财务风险{risk_score:.0f}，合成长期分 {scores.long_term_score:.0f}。",
        )

    return "该分数用于辅助排序和解释；长期价值与短期交易机会分开判断。"


# Final single-stock conclusion override with clearer short-term reasoning.
def _short_conclusion_lines(result: AnalysisResult, short_view) -> list[str]:
    return [
        f"- 总体结论：{short_view.advice}。{_short_action_sentence(short_view)}",
        f"- 短期分角度：短期分 {short_view.short_term_score:.0f}，主要衡量近期走势优势。{_short_score_summary(result)}",
        f"- 量价分角度：量价分 {short_view.liangjia_score:.0f}，主要衡量价格、成交量、位置背后的供需关系。{_liangjia_score_summary(result)}",
        f"- 信号方向：当前量价信号方向为「{short_view.signal_direction}」，信号类型为「{short_view.signal_type or '暂无明确量价信号'}」，强度 {short_view.signal_strength}。",
        _short_trade_boundary_line(short_view),
        _short_basis_line(short_view),
        _short_risk_line(short_view),
        "- 执行原则：短期分高说明短线状态较好，但短期买入或加仓必须有量价买点配合；量价分高但短期分一般，通常说明结构较好但动能尚未完全确认。",
    ]


def _short_score_summary(result: AnalysisResult) -> str:
    scores = getattr(result, "factor_scores", None)
    if scores is None:
        return "暂无短期分拆解。"
    components = (getattr(scores, "components", {}) or {}).get("short_term", {}) or {}
    return_score = float(components.get("return_score", 0) or 0)
    ma_score = float(components.get("ma_score", 0) or 0)
    volume_score = float(components.get("volume_score", 0) or 0)
    momentum_score = float(components.get("momentum_score", 0) or 0)
    breakout_score = float(components.get("breakout_score", 0) or 0)
    risk_score = float(components.get("risk_score", 0) or 0)
    strongest = _top_score_labels(
        [
            ("涨跌动量", return_score),
            ("短均线趋势", ma_score),
            ("量能", volume_score),
            ("技术动量", momentum_score),
            ("阶段新高", breakout_score),
            ("短期风险控制", risk_score),
        ],
        reverse=True,
    )
    weakest = _top_score_labels(
        [
            ("涨跌动量", return_score),
            ("短均线趋势", ma_score),
            ("量能", volume_score),
            ("技术动量", momentum_score),
            ("阶段新高", breakout_score),
            ("短期风险控制", risk_score),
        ],
        reverse=False,
    )
    detail = (
        f"子项为涨跌动量{return_score:.0f}、短均线{ma_score:.0f}、量能{volume_score:.0f}、"
        f"技术动量{momentum_score:.0f}、阶段新高{breakout_score:.0f}、风险控制{risk_score:.0f}。"
    )
    if strongest and weakest:
        return f"{detail} 主要加分来自{strongest}；主要拖累来自{weakest}。"
    return detail


def _liangjia_score_summary(result: AnalysisResult) -> str:
    scores = getattr(result, "factor_scores", None)
    if scores is None:
        return "暂无量价分拆解。"
    components = (getattr(scores, "components", {}) or {}).get("liangjia", {}) or {}
    base = float(components.get("base_score", scores.liangjia_score) or scores.liangjia_score)
    market_modifier = float(components.get("market_modifier", 0) or 0)
    exit_modifier = float(components.get("exit_risk_modifier", 0) or 0)
    factor_modifier = float(components.get("factor_modifier", 0) or 0)
    volume_position = float(components.get("volume_position_modifier", 0) or 0)
    money_flow = float(components.get("money_flow_modifier", 0) or 0)
    trend_participation = float(components.get("trend_participation_modifier", 0) or 0)
    position_compression = float(components.get("position_compression_modifier", 0) or 0)
    volatility_modifier = float(components.get("volatility_modifier", 0) or 0)
    supply_items = _modifier_text(
        [
            ("量能与收盘位置", volume_position),
            ("资金流", money_flow),
            ("趋势参与", trend_participation),
            ("位置/波动收缩", position_compression),
            ("波动与回撤风险", volatility_modifier),
        ]
    )
    return (
        f"原量价结构分 {base:.0f}，市场修正 {market_modifier:+.0f}，卖出风险修正 {exit_modifier:+.0f}，"
        f"供需确认修正 {factor_modifier:+.0f}。供需确认里，{supply_items}。"
    )


def _short_trade_boundary_line(short_view) -> str:
    if not (short_view.entry_zone or short_view.stop_loss or short_view.risk_pct is not None):
        return "- 交易边界：当前没有明确买入区间或止损位，说明更适合观察而不是直接执行。"
    stop_text = "-" if short_view.stop_loss is None else f"{short_view.stop_loss:.2f}"
    risk_text = "-" if short_view.risk_pct is None else f"{short_view.risk_pct:.1%}"
    return f"- 交易边界：买入区间 {_entry_zone_text(short_view.entry_zone)}，止损位 {stop_text}，风险比例 {risk_text}。"


def _short_basis_line(short_view) -> str:
    key_text = _join_limited(short_view.key_factors, 3)
    if key_text:
        return f"- 主要依据：{key_text}。"
    return "- 主要依据：当前没有特别突出的单项依据，需等待更清晰的量价确认。"


def _short_risk_line(short_view) -> str:
    if short_view.risk_warnings:
        return f"- 风险提示：{_join_limited(short_view.risk_warnings, 3)}。"
    return "- 风险提示：当前短期模块没有识别到强卖出或回避风险，但若跌破关键位、放量收低或趋势转弱，应重新评估。"


def _top_score_labels(items: list[tuple[str, float]], reverse: bool) -> str:
    if not items:
        return ""
    ordered = sorted(items, key=lambda item: item[1], reverse=reverse)
    selected = [item for item in ordered if (item[1] >= 65 if reverse else item[1] <= 55)]
    if not selected:
        selected = ordered[:2]
    return "、".join(f"{name}{value:.0f}" for name, value in selected[:2])


def _modifier_text(items: list[tuple[str, float]]) -> str:
    parts = []
    for name, value in items:
        if value > 0:
            parts.append(f"{name}加{value:.0f}分")
        elif value < 0:
            parts.append(f"{name}扣{abs(value):.0f}分")
        else:
            parts.append(f"{name}中性")
    return "、".join(parts)


if __name__ == "__main__":
    main()
