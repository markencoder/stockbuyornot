from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from stockbuyornot.analysis import analyze_ohlcv
from stockbuyornot.backtest import backtest_single
from stockbuyornot.data.providers import AkshareProvider, CsvProvider
from stockbuyornot.portfolio import PortfolioBacktestConfig, portfolio_backtest
from stockbuyornot.radar import market_state_from_benchmark
from stockbuyornot.view_engine import stage_label


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="stockbuyornot", description="A-share volume-price signal interpreter.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze one stock.")
    _add_data_args(analyze_parser)
    analyze_parser.add_argument("--json", action="store_true", help="Print full JSON result.")

    scan_parser = subparsers.add_parser("scan", help="Scan a stock universe.")
    scan_parser.add_argument("--symbols", nargs="+", default=[], help="Explicit stock codes, e.g. 000001 600519.")
    scan_parser.add_argument("--all", action="store_true", help="Shortcut for --pool all.")
    scan_parser.add_argument(
        "--pool",
        default="",
        help=(
            "Built-in pool: all, sse50, csi300, csi500, sse, szse, chinext, star. "
            "Chinese aliases such as 上证50, 中证500, 深证, 创业板 also work."
        ),
    )
    scan_parser.add_argument("--industry", default="", help="Eastmoney industry board name, e.g. 银行.")
    scan_parser.add_argument("--concept", default="", help="Eastmoney concept board name, e.g. 人工智能.")
    scan_parser.add_argument("--csv-dir", default="", help="Scan local CSV files in a directory. File stem is used as symbol.")
    scan_parser.add_argument("--limit", type=int, default=0, help="Limit scan count, useful for smoke tests.")
    scan_parser.add_argument("--start", default="20230101")
    scan_parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    scan_parser.add_argument("--adjust", default="qfq")
    scan_parser.add_argument("--min-score", type=int, default=60)
    scan_parser.add_argument("--output", default="", help="Optional CSV output path.")
    scan_parser.add_argument("--save-every", type=int, default=50, help="Write partial output every N scanned symbols when --output is set.")
    scan_parser.add_argument("--no-progress", action="store_true", help="Disable progress lines.")
    scan_parser.add_argument("--include-errors", action="store_true", help="Include failed symbols in the output table.")
    scan_parser.add_argument("--request-timeout", type=float, default=8.0, help="Per-request timeout in seconds.")
    scan_parser.add_argument("--request-retries", type=int, default=1, help="Retries per data endpoint.")

    backtest_parser = subparsers.add_parser("backtest", help="Backtest one stock signal stream.")
    _add_data_args(backtest_parser)
    backtest_parser.add_argument("--min-score", type=int, default=75)

    portfolio_parser = subparsers.add_parser("portfolio-backtest", help="Backtest the trend-pullback portfolio strategy.")
    portfolio_parser.add_argument("--symbols", nargs="+", default=[], help="Explicit stock codes, e.g. 000001 600519.")
    portfolio_parser.add_argument("--pool", default="", help="Built-in pool: all, sse50, csi300, csi500, sse, szse, chinext, star.")
    portfolio_parser.add_argument("--all", action="store_true", help="Shortcut for --pool all.")
    portfolio_parser.add_argument("--csv-dir", default="", help="Use local CSV files as the stock universe.")
    portfolio_parser.add_argument("--benchmark-csv", default="", help="Use local benchmark CSV instead of online HS300 data.")
    portfolio_parser.add_argument("--benchmark-symbol", default="", help="Benchmark index code. Defaults to the matching pool index, or HS300.")
    portfolio_parser.add_argument("--limit", type=int, default=0, help="Limit universe count for smoke tests.")
    portfolio_parser.add_argument("--start", default="20240101")
    portfolio_parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    portfolio_parser.add_argument("--adjust", default="qfq")
    portfolio_parser.add_argument("--max-positions", type=int, default=5)
    portfolio_parser.add_argument("--neutral-max-positions", type=int, default=2)
    portfolio_parser.add_argument("--initial-cash", type=float, default=100000.0)
    portfolio_parser.add_argument("--min-score", type=int, default=75)
    portfolio_parser.add_argument("--min-avg-amount", type=float, default=50_000_000.0)
    portfolio_parser.add_argument("--market-mode", choices=["strict", "balanced", "aggressive"], default="balanced")
    portfolio_parser.add_argument("--max-stop-distance", type=float, default=0.07)
    portfolio_parser.add_argument("--min-relative-strength", type=float, default=0.03)
    portfolio_parser.add_argument("--min-reward-risk", type=float, default=1.8)
    portfolio_parser.add_argument("--breakeven-r", type=float, default=1.0)
    portfolio_parser.add_argument("--trail-start-r", type=float, default=2.0)
    portfolio_parser.add_argument("--trail-pct", type=float, default=0.10)
    portfolio_parser.add_argument("--trailing-stop", type=float, default=None, help="Deprecated alias for --trail-pct.")
    portfolio_parser.add_argument("--stale-days", type=int, default=12)
    portfolio_parser.add_argument("--max-holding-days", type=int, default=45)
    portfolio_parser.add_argument("--output", default="", help="Output path prefix. Multiple CSV/JSON files will be written.")
    portfolio_parser.add_argument("--request-timeout", type=float, default=8.0)
    portfolio_parser.add_argument("--request-retries", type=int, default=1)
    portfolio_parser.add_argument("--no-progress", action="store_true")

    args = parser.parse_args()
    if args.command == "analyze":
        try:
            df, symbol = _load_data(args)
            result = analyze_ohlcv(df, symbol=symbol)
        except Exception as exc:
            raise SystemExit(f"分析失败：{_short_error(exc)}")
        print(_to_json(result) if args.json else format_analysis(result))
    elif args.command == "scan":
        _run_scan(args)
    elif args.command == "backtest":
        try:
            df, symbol = _load_data(args)
            result = backtest_single(df, symbol=symbol, min_score=args.min_score)
        except Exception as exc:
            raise SystemExit(f"回测失败：{_short_error(exc)}")
        print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
        if not result.trades.empty:
            print(result.trades.to_string(index=False))
    elif args.command == "portfolio-backtest":
        _run_portfolio_backtest(args)


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="", help="A-share code, e.g. 000001 or 600519.")
    parser.add_argument("--start", default="20230101")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y%m%d"))
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--csv", default="", help="Use local CSV instead of online data.")


def _load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.csv:
        path = Path(args.csv)
        provider = CsvProvider(path)
        symbol = args.symbol or path.stem
        return provider.daily(symbol, args.start, args.end, args.adjust), symbol
    if not args.symbol:
        raise SystemExit("--symbol is required when --csv is not provided.")
    provider = AkshareProvider()
    return provider.daily(args.symbol, args.start, args.end, args.adjust), args.symbol


def _run_scan(args: argparse.Namespace) -> None:
    provider = AkshareProvider(request_timeout=args.request_timeout, request_retries=args.request_retries)
    if not args.no_progress:
        print("正在获取股票池...", flush=True)
    sources = list(_scan_sources(args, provider))
    total = len(sources)
    if total == 0:
        raise SystemExit("scan found no symbols.")

    rows = []
    candidates = 0
    failures = 0
    started = time.time()
    progress = not args.no_progress
    benchmark = _load_scan_benchmark(provider, args.start, args.end)
    sector_name = args.industry or args.concept or ""
    sector_benchmark = _load_scan_sector_benchmark(provider, sector_name, args.start, args.end)
    sector_rs_rank = _sector_rank_hint(sector_benchmark, benchmark)

    if progress:
        print(f"开始扫描：共 {total} 只，时间区间 {args.start}-{args.end}，最低分 {args.min_score}")

    for index, (symbol, source) in enumerate(sources, start=1):
        item_started = time.time()
        try:
            if isinstance(source, Path):
                df = CsvProvider(source).daily(symbol=symbol, start=args.start, end=args.end, adjust=args.adjust)
            else:
                df = provider.daily(symbol, args.start, args.end, args.adjust)
            result = analyze_ohlcv(
                df,
                symbol=symbol,
                benchmark=benchmark,
                sector=sector_benchmark,
                sector_name=sector_name,
                sector_rs_rank=sector_rs_rank,
            )
        except Exception as exc:
            failures += 1
            error = _short_error(exc)
            if args.include_errors:
                rows.append({"symbol": symbol, "error": error})
            if progress:
                print(_progress_line(index, total, symbol, started, item_started, f"失败：{error}"), flush=True)
            _maybe_save(args, rows, index)
            continue

        candidate_score = _scan_candidate_score(result)
        matched = candidate_score >= args.min_score and _scan_action_code(result) in {"buy", "add", "wait", "watch"}
        if matched:
            candidates += 1
            rows.append(
                {
                    "symbol": symbol,
                    "date": str(result.as_of.date()),
                    "close": result.close,
                    "stage": stage_label(result.structure.stage),
                    "score": result.score.total,
                    "candidate_score": candidate_score,
                    "signals": ",".join(signal.name for signal in result.signals),
                    **_view_row_fields(result),
                    **_factor_row_fields(result),
                    **_radar_row_fields(result),
                }
            )

        if progress:
            status = f"命中 score={candidate_score:.0f}" if matched else f"跳过 score={candidate_score:.0f}"
            print(_progress_line(index, total, symbol, started, item_started, status), flush=True)
        _maybe_save(args, rows, index)

    output = pd.DataFrame(rows)
    if args.output and not output.empty:
        output.to_csv(args.output, index=False, encoding="utf-8-sig")

    elapsed = time.time() - started
    if progress:
        print(f"扫描完成：总数 {total}，命中 {candidates}，失败 {failures}，耗时 {_fmt_seconds(elapsed)}")
    print(output.to_string(index=False) if not output.empty else "No candidates.")


def _run_portfolio_backtest(args: argparse.Namespace) -> None:
    provider = AkshareProvider(request_timeout=args.request_timeout, request_retries=args.request_retries)
    fetch_start = _warmup_start(args.start)
    try:
        if not args.no_progress:
            print("正在加载股票池...", flush=True)
        data_by_symbol = _load_portfolio_data(args, provider, fetch_start)
        if not data_by_symbol:
            raise ValueError("没有可用于回测的股票数据。")

        benchmark_symbol = _benchmark_symbol_for_args(args)
        if args.benchmark_csv:
            benchmark = CsvProvider(args.benchmark_csv).daily(symbol=benchmark_symbol, start=fetch_start, end=args.end)
        else:
            benchmark = provider.index_daily(benchmark_symbol, fetch_start, args.end)

        config = PortfolioBacktestConfig(
            max_positions=args.max_positions,
            neutral_max_positions=args.neutral_max_positions,
            initial_cash=args.initial_cash,
            min_avg_amount_20=args.min_avg_amount,
            max_stop_distance_pct=args.max_stop_distance,
            min_relative_strength=args.min_relative_strength,
            min_reward_risk=args.min_reward_risk,
            trailing_stop_pct=args.trail_pct if args.trailing_stop is None else args.trailing_stop,
            stale_days=args.stale_days,
            max_holding_days=args.max_holding_days,
            breakeven_r=args.breakeven_r,
            trail_start_r=args.trail_start_r,
            min_score=args.min_score,
            market_mode=args.market_mode,
        )
        progress_callback = None if args.no_progress else lambda message: print(message, flush=True)
        result = portfolio_backtest(
            data_by_symbol,
            benchmark,
            args.start,
            args.end,
            portfolio_config=config,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        raise SystemExit(f"组合回测失败：{_short_error(exc)}")

    if args.output:
        result.write_outputs(args.output)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    if not result.trades.empty:
        print(result.trades.tail(20).to_string(index=False))


def _load_scan_benchmark(provider: AkshareProvider, start: str, end: str) -> pd.DataFrame | None:
    try:
        return provider.index_daily("000300", _warmup_start(start), end)
    except Exception:
        return None


def _load_scan_sector_benchmark(provider: AkshareProvider, sector_name: str, start: str, end: str) -> pd.DataFrame | None:
    if not sector_name:
        return None
    try:
        return provider.board_daily(sector_name, _warmup_start(start), end)
    except Exception:
        return None


def _sector_rank_hint(sector: pd.DataFrame | None, benchmark: pd.DataFrame | None) -> float | None:
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


def _radar_row_fields(result) -> dict:
    radar = getattr(result, "radar", None)
    if radar is None:
        return {}
    return {
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
        "stock_rs_20": radar.stock_rs_20,
        "stock_rs_60": radar.stock_rs_60,
        "reward_risk": radar.reward_risk,
        "stop_distance_pct": radar.stop_distance_pct,
    }


def _decision_row_fields(result) -> dict:
    decision = getattr(result, "decision", None)
    if decision is None:
        return {}
    return {
        "final_action": decision.action_code,
        "final_action_label": decision.action_label,
        "candidate_score": decision.candidate_score,
        "decision_confidence": decision.confidence,
        "primary_basis": decision.primary_basis,
        "decision_conflict": decision.conflict,
        "decision_reason": decision.reason,
    }


def _view_row_fields(result) -> dict:
    long_view = getattr(result, "long_term_view", None)
    short_view = getattr(result, "short_term_view", None)
    row = {}
    if long_view is not None:
        row.update(
            {
                "long_term_advice": long_view.advice,
                "long_term_rating": long_view.rating,
                "long_term_score_view": long_view.score,
                "long_term_risks": ";".join(long_view.risk_warnings),
            }
        )
    if short_view is not None:
        row.update(
            {
                "short_term_advice": short_view.advice,
                "short_term_score_view": short_view.short_term_score,
                "liangjia_signal_type": short_view.signal_type,
                "liangjia_signal_strength": short_view.signal_strength,
                "liangjia_signal_direction": short_view.signal_direction,
                "short_term_action_code": short_view.action_code,
                "short_term_risks": ";".join(short_view.risk_warnings),
            }
        )
    return row


def _factor_row_fields(result) -> dict:
    scores = getattr(result, "factor_scores", None)
    classification = getattr(result, "classification", None)
    if scores is None or classification is None:
        return {}
    return {
        "classification": classification.category,
        "action": classification.action,
        "liangjia_score": scores.liangjia_score,
        "short_term_score": scores.short_term_score,
        "long_term_score": scores.long_term_score,
        "overall_score": scores.overall_score,
        "main_signal": classification.main_signal,
        "buy_point_type": classification.buy_point_type,
        "risk_pct": classification.risk_pct,
        "risk_flags": ";".join(classification.risk_flags),
    }


def _scan_candidate_score(result) -> float:
    long_view = getattr(result, "long_term_view", None)
    short_view = getattr(result, "short_term_view", None)
    candidates: list[float] = []
    scores = getattr(result, "factor_scores", None)
    if scores is not None:
        execution_score = _execution_score(scores)
        balanced_short = min(float(scores.short_term_score), execution_score + 8.0)
        balanced_liangjia = min(float(short_view.liangjia_score if short_view is not None else scores.liangjia_score), execution_score + 12.0)
        tradable_score = execution_score * 0.52 + balanced_liangjia * 0.26 + balanced_short * 0.17 + float(scores.long_term_score) * 0.05
        candidates.extend([float(scores.overall_score) * 0.85, tradable_score, float(scores.long_term_score) * 0.80, execution_score])
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


def _execution_score(scores) -> float:
    components = getattr(scores, "components", {}) or {}
    item = components.get("execution_window", {}) or {}
    try:
        if isinstance(item, dict):
            return float(item.get("score", components.get("execution_score", 70.0)))
        return float(item)
    except (TypeError, ValueError):
        return 70.0


def _scan_action_code(result) -> str:
    short_view = getattr(result, "short_term_view", None)
    long_view = getattr(result, "long_term_view", None)
    if short_view is not None:
        code = str(short_view.action_code)
        if code in {"buy", "add", "wait"}:
            return code
        if code == "hold":
            return "watch"
        if long_view is not None and long_view.advice in {"长期可关注", "长期谨慎关注"} and code not in {"avoid", "sell"}:
            return "watch"
        return code
    classification = getattr(result, "classification", None)
    if classification is not None:
        return str(classification.action).lower()
    decision = getattr(result, "decision", None)
    if decision is not None:
        return str(decision.action_code)
    return ""


def _load_portfolio_data(args: argparse.Namespace, provider: AkshareProvider, fetch_start: str) -> dict[str, pd.DataFrame]:
    if args.csv_dir:
        paths = sorted(Path(args.csv_dir).glob("*.csv"))
        if args.benchmark_csv:
            benchmark_path = Path(args.benchmark_csv).resolve()
            paths = [path for path in paths if path.resolve() != benchmark_path]
        if args.limit and args.limit > 0:
            paths = paths[: args.limit]
        data = {}
        for path in paths:
            data[path.stem] = CsvProvider(path).daily(symbol=path.stem, start=fetch_start, end=args.end, adjust=args.adjust)
        return data

    symbols = _portfolio_symbols(args, provider)
    if args.limit and args.limit > 0:
        symbols = symbols[: args.limit]
    data_by_symbol = {}
    total = len(symbols)
    for index, symbol in enumerate(symbols, start=1):
        try:
            data_by_symbol[symbol] = provider.daily(symbol, fetch_start, args.end, args.adjust)
            if not args.no_progress:
                print(f"[{index}/{total}] {symbol} 数据已加载", flush=True)
        except Exception as exc:
            if not args.no_progress:
                print(f"[{index}/{total}] {symbol} 跳过：{_short_error(exc)}", flush=True)
    return data_by_symbol


def _portfolio_symbols(args: argparse.Namespace, provider: AkshareProvider) -> list[str]:
    if args.all:
        return provider.symbols_for_pool("all")
    if args.pool:
        return provider.symbols_for_pool(args.pool)
    if args.symbols:
        return args.symbols
    raise SystemExit("portfolio-backtest requires --symbols, --pool, --all, or --csv-dir.")


def _benchmark_symbol_for_args(args: argparse.Namespace) -> str:
    if args.benchmark_symbol:
        return args.benchmark_symbol
    pool = (args.pool or "").lower().strip()
    if pool in {"sse50", "sh50", "上证50"}:
        return "000016"
    if pool in {"csi500", "zz500", "中证500"}:
        return "000905"
    if pool in {"chinext", "cyb", "创业板"}:
        return "399006"
    return "000300"


def _scan_sources(args: argparse.Namespace, provider: AkshareProvider) -> Iterable[tuple[str, str | Path]]:
    if args.csv_dir:
        paths = sorted(Path(args.csv_dir).glob("*.csv"))
        if args.limit and args.limit > 0:
            paths = paths[: args.limit]
        if not paths:
            raise SystemExit(f"No CSV files found in {args.csv_dir}.")
        for path in paths:
            yield path.stem, path
        return

    if args.industry:
        symbols = provider.industry_symbols(args.industry)
    elif args.concept:
        symbols = provider.concept_symbols(args.concept)
    elif args.all:
        symbols = provider.symbols_for_pool("all")
    elif args.pool:
        symbols = provider.symbols_for_pool(args.pool)
    else:
        symbols = args.symbols

    if not symbols:
        raise SystemExit("scan requires --symbols, --all, --pool, --industry, --concept, or --csv-dir.")
    if args.limit and args.limit > 0:
        symbols = symbols[: args.limit]
    for symbol in symbols:
        yield symbol, "online"


def _progress_line(index: int, total: int, symbol: str, started: float, item_started: float, status: str) -> str:
    elapsed = time.time() - started
    item_elapsed = time.time() - item_started
    avg = elapsed / max(index, 1)
    remaining = avg * max(total - index, 0)
    percent = index / total * 100
    return (
        f"[{index}/{total} {percent:5.1f}%] {symbol} {status} "
        f"本只 {_fmt_seconds(item_elapsed)} 已用 {_fmt_seconds(elapsed)} 预计剩余 {_fmt_seconds(remaining)}"
    )


def _maybe_save(args: argparse.Namespace, rows: list[dict], index: int) -> None:
    if not args.output or not rows or args.save_every <= 0:
        return
    if index % args.save_every == 0:
        pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")


def format_analysis(result) -> str:
    lines = [
        f"股票：{result.symbol}",
        f"日期：{result.as_of.date()}",
        f"收盘：{result.close:.2f}",
        f"当前结构：{result.structure.stage.value}",
        f"结构解释：{result.structure.description}",
        f"操作建议：{result.suggestion}",
        f"信号评分：{result.score.total}",
        "",
        "关键支撑：",
        *[f"- {level.name}: {level.price:.2f}" for level in result.support_levels[:5]],
        "关键压力：",
        *[f"- {level.name}: {level.price:.2f}" for level in result.resistance_levels[:5]],
        "",
        "量价信号：",
    ]
    if result.signals:
        for signal in result.signals:
            lines.extend(
                [
                    f"- {signal.name} [{signal.side.value}] 强度 {signal.strength}",
                    f"  逻辑：{signal.logic}",
                    f"  止损：{signal.stop_loss if signal.stop_loss else '无'}",
                    f"  失效：{signal.invalidation if signal.invalidation else '无'}",
                    "  证据：" + "；".join(signal.evidence),
                ]
            )
    else:
        lines.append("- 暂无明确关键信号")
    lines.extend(["", "评分解释:", *[f"- {item}" for item in result.score.explanation]])
    return "\n".join(lines)


def _to_json(obj) -> str:
    if hasattr(obj, "output") and getattr(obj, "output"):
        return json.dumps(getattr(obj, "output"), ensure_ascii=False, indent=2)

    def default(value):
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if hasattr(value, "value"):
            return value.value
        return str(value)

    return json.dumps(obj, ensure_ascii=False, indent=2, default=default)


def _short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ").strip()
    if "Unable to fetch Eastmoney daily data" in message:
        return "无法获取东方财富日线数据，请稍后重试，或改用 --csv / --csv-dir 本地数据。"
    if "At least 60 trading days" in message:
        return "数据不足，至少需要 60 个交易日，推荐 120 个交易日以上。"
    if "ProxyError" in message or "Connection" in message or "timeout" in message.lower():
        return "网络连接失败，请稍后重试，或减少股票池数量后再扫。"
    return message[:500]


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _warmup_start(start: str) -> str:
    ts = pd.to_datetime(start) - pd.Timedelta(days=300)
    return ts.strftime("%Y%m%d")


def format_analysis(result) -> str:
    long_view = getattr(result, "long_term_view", None)
    short_view = getattr(result, "short_term_view", None)
    lines = [
        f"股票：{result.symbol}",
        f"日期：{result.as_of.date()}",
        f"收盘：{result.close:.2f}",
        f"当前结构：{stage_label(result.structure.stage)}",
        f"结构解释：{result.structure.description}",
        f"长期投资价值：{long_view.advice if long_view else '-'}",
        f"短期交易机会：{short_view.advice if short_view else result.suggestion}",
        f"量价分：{result.score.total}",
        "",
        "关键支撑：",
        *[f"- {level.name}: {level.price:.2f}" for level in result.support_levels[:5]],
        "关键压力：",
        *[f"- {level.name}: {level.price:.2f}" for level in result.resistance_levels[:5]],
        "",
        "量价信号：",
    ]
    if result.signals:
        for signal in result.signals:
            lines.extend(
                [
                    f"- {signal.name} [{signal.side.value}] 强度 {signal.strength}",
                    f"  逻辑：{signal.logic}",
                    f"  止损：{signal.stop_loss if signal.stop_loss else '无'}",
                    f"  失效：{signal.invalidation if signal.invalidation else '无'}",
                    "  证据：" + "；".join(signal.evidence),
                ]
            )
    else:
        lines.append("- 暂无明确关键量价信号")
    if long_view is not None:
        lines.extend(["", "长期解释：", f"- {long_view.explanation}", *[f"- {item}" for item in long_view.key_factors]])
    if short_view is not None:
        lines.extend(["", "短期解释：", f"- {short_view.explanation}", *[f"- {item}" for item in short_view.key_factors]])
    lines.extend(["", "量价评分解释:", *[f"- {item}" for item in result.score.explanation]])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
