from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
from pathlib import Path
import re
import time
from typing import Any, Protocol

import pandas as pd
import requests

from stockbuyornot.data.schema import normalize_ohlcv


class DataProvider(Protocol):
    def daily(self, symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        """Return normalized daily OHLCV data."""

    def intraday(self, symbol: str, period: str = "5", start: str = "", end: str = "", adjust: str = "") -> pd.DataFrame:
        """Return normalized intraday OHLCV data."""


@dataclass
class CsvProvider:
    path: str | Path

    def daily(self, symbol: str = "", start: str = "", end: str = "", adjust: str = "qfq") -> pd.DataFrame:
        data = normalize_ohlcv(pd.read_csv(self.path), symbol=symbol or None)
        if start:
            data = data[data["date"] >= pd.to_datetime(start)]
        if end:
            data = data[data["date"] <= pd.to_datetime(end)]
        return data.reset_index(drop=True)


class AkshareProvider:
    """A-share data adapter.

    The daily data path prefers a direct Eastmoney request through curl_cffi
    because the current machine has intermittent proxy/TLS issues with
    requests + AKShare's default endpoint. AKShare is still used for universe
    and index/board membership queries.
    """

    def __init__(self, request_timeout: float = 8.0, request_retries: int = 1) -> None:
        self.request_timeout = request_timeout
        self.request_retries = max(1, request_retries)

    def daily(self, symbol: str, start: str, end: str, adjust: str = "qfq") -> pd.DataFrame:
        symbol = _normalize_symbol(symbol)
        try:
            return self._tencent_daily(symbol, start, end, adjust)
        except Exception:
            pass

        try:
            return self._eastmoney_daily(symbol, start, end, adjust)
        except Exception:
            pass

        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AKShare is not installed. Run: pip install akshare") from exc

        raw = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=adjust,
        )
        return normalize_ohlcv(raw, symbol=symbol)

    def intraday(self, symbol: str, period: str = "5", start: str = "", end: str = "", adjust: str = "") -> pd.DataFrame:
        symbol = _normalize_symbol(symbol)
        period = str(period)
        if period not in {"1", "5", "15", "30", "60"}:
            raise ValueError("分时周期只支持 1、5、15、30、60 分钟。")

        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AKShare is not installed. Run: pip install akshare") from exc

        errors: list[str] = []
        start_time, end_time = _intraday_time_range(start, end)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                raw = ak.stock_zh_a_hist_min_em(
                    symbol=str(symbol).zfill(6),
                    start_date=start_time,
                    end_date=end_time,
                    period=period,
                    adjust=adjust or "",
                )
            return _normalize_intraday(raw, symbol=symbol)
        except Exception as exc:
            errors.append(f"hist_min_em: {exc}")

        if period != "1":
            try:
                return self._eastmoney_intraday(symbol, period, start_time, end_time, adjust or "")
            except Exception as exc:
                errors.append(f"eastmoney_direct_minute: {exc}")

        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                raw = ak.stock_zh_a_minute(symbol=_tencent_symbol(symbol), period=period, adjust=adjust or "")
            return _normalize_intraday(raw, symbol=symbol)
        except Exception as exc:
            errors.append(f"stock_zh_a_minute: {exc}")

        raise RuntimeError(
            f"未读取到 {str(symbol).zfill(6)} 的 {period}分钟分时数据。"
            "可能原因：代码不正确、股票停牌/退市、接口暂时无数据，或当前回看区间没有分钟线。"
            "可尝试扩大回看天数、切换5分钟/15分钟周期，或稍后重试。"
            f" 原始错误：{' | '.join(errors[-2:])}"
        )

    def index_daily(self, symbol: str = "000300", start: str = "20200101", end: str = "20500101") -> pd.DataFrame:
        try:
            return self._eastmoney_index_daily(symbol, start, end)
        except Exception as eastmoney_error:
            try:
                return self._tencent_index_daily(symbol, start, end)
            except Exception:
                pass
            proxy_symbol = _index_proxy_symbol(symbol)
            if proxy_symbol:
                try:
                    return self.daily(proxy_symbol, start, end, "qfq")
                except Exception:
                    pass

        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AKShare is not installed. Run: pip install akshare") from exc

        raw = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end)
        return normalize_ohlcv(raw, symbol=symbol)

    def _tencent_index_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        market_symbol = _tencent_index_symbol(symbol)
        start_fmt = _date_with_dash(start)
        end_fmt = _date_with_dash(end)
        count = _calendar_day_count(start, end)
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {"param": f"{market_symbol},day,{start_fmt},{end_fmt},{count},"}

        response = self._request_tencent(url, params)
        payload = response.json()
        index_data = (payload.get("data") or {}).get(market_symbol) or {}
        klines = index_data.get("day") or []
        rows = []
        for item in klines:
            if len(item) < 6:
                continue
            rows.append(
                {
                    "date": item[0],
                    "open": item[1],
                    "close": item[2],
                    "high": item[3],
                    "low": item[4],
                    "volume": item[5],
                    "symbol": symbol,
                }
            )
        return normalize_ohlcv(pd.DataFrame(rows), symbol=symbol)

    def _tencent_daily(self, symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
        market_symbol = _tencent_symbol(symbol)
        adjust_key = {"qfq": "qfqday", "hfq": "hfqday", "": "day"}.get(adjust, "qfqday")
        start_fmt = _date_with_dash(start)
        end_fmt = _date_with_dash(end)
        count = _calendar_day_count(start, end)
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {"param": f"{market_symbol},day,{start_fmt},{end_fmt},{count},{adjust or 'qfq'}"}

        response = self._request_tencent(url, params)
        payload = response.json()
        stock_data = (payload.get("data") or {}).get(market_symbol) or {}
        klines = stock_data.get(adjust_key) or stock_data.get("day") or []
        rows = []
        for item in klines:
            if len(item) < 6:
                continue
            rows.append(
                {
                    "date": item[0],
                    "open": item[1],
                    "close": item[2],
                    "high": item[3],
                    "low": item[4],
                    "volume": item[5],
                    "symbol": symbol,
                }
            )
        return normalize_ohlcv(pd.DataFrame(rows), symbol=symbol)

    def _request_tencent(self, url: str, params: dict[str, str]) -> Any:
        errors: list[str] = []
        try:
            from curl_cffi import requests as curl_requests
        except ImportError:
            curl_requests = None

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://gu.qq.com/",
        }
        if curl_requests is not None:
            for attempt in range(self.request_retries):
                try:
                    response = curl_requests.get(url, params=params, headers=headers, impersonate="chrome120", timeout=self.request_timeout)
                    response.raise_for_status()
                    return response
                except Exception as exc:
                    errors.append(f"curl_cffi tencent, attempt={attempt + 1}: {exc}")
                    time.sleep(0.2)

        for trust_env in [False, True]:
            for attempt in range(self.request_retries):
                session = requests.Session()
                session.trust_env = trust_env
                try:
                    response = session.get(url, params=params, headers=headers, timeout=self.request_timeout)
                    response.raise_for_status()
                    return response
                except requests.exceptions.RequestException as exc:
                    errors.append(f"requests tencent, trust_env={trust_env}, attempt={attempt + 1}: {exc}")
                    time.sleep(0.2)
        raise RuntimeError("Unable to fetch Tencent daily data. Last errors: " + " | ".join(errors[-3:]))

    def stock_list(self) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AKShare is not installed. Run: pip install akshare") from exc

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            raw = ak.stock_info_a_code_name()
        data = raw.rename(columns={"code": "symbol", "name": "name", "代码": "symbol", "名称": "name"}).copy()
        data["symbol"] = data["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
        data["name"] = data.get("name", "")
        data = data.dropna(subset=["symbol"])
        return data[["symbol", "name"]].drop_duplicates("symbol").reset_index(drop=True)

    def symbols_for_pool(self, pool: str) -> list[str]:
        pool = pool.lower().strip()
        if pool in {"all", "a", "ashare", "a-share", "全a", "全a股"}:
            return self.stock_list()["symbol"].tolist()
        if pool in {"sse50", "sh50", "上证50"}:
            return self.index_constituents("000016")
        if pool in {"csi300", "hs300", "沪深300"}:
            return self.index_constituents("000300")
        if pool in {"csi500", "zz500", "中证500"}:
            return self.index_constituents("000905")

        all_symbols = self.stock_list()["symbol"].tolist()
        if pool in {"sse", "sh", "上证", "沪市"}:
            return [symbol for symbol in all_symbols if symbol.startswith("6")]
        if pool in {"szse", "sz", "深证", "深市"}:
            return [symbol for symbol in all_symbols if symbol.startswith(("0", "3"))]
        if pool in {"chinext", "cyb", "创业板"}:
            return [symbol for symbol in all_symbols if symbol.startswith("3")]
        if pool in {"star", "科创板"}:
            return [symbol for symbol in all_symbols if symbol.startswith("688")]

        raise ValueError(f"Unknown pool: {pool}")

    def index_constituents(self, index_symbol: str) -> list[str]:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AKShare is not installed. Run: pip install akshare") from exc

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            raw = ak.index_stock_cons_csindex(symbol=index_symbol)
        return _extract_symbols(raw)

    def industry_symbols(self, name: str) -> list[str]:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AKShare is not installed. Run: pip install akshare") from exc

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            raw = ak.stock_board_industry_cons_em(symbol=name)
        return _extract_symbols(raw)

    def concept_symbols(self, name: str) -> list[str]:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AKShare is not installed. Run: pip install akshare") from exc

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            raw = ak.stock_board_concept_cons_em(symbol=name)
        return _extract_symbols(raw)

    def board_daily(self, name: str, start: str, end: str) -> pd.DataFrame:
        try:
            import akshare as ak
        except ImportError as exc:
            raise RuntimeError("AKShare is not installed. Run: pip install akshare") from exc

        errors: list[str] = []
        for func_name in ["stock_board_industry_hist_em", "stock_board_concept_hist_em"]:
            func = getattr(ak, func_name, None)
            if func is None:
                continue
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    raw = func(symbol=name, start_date=start, end_date=end, period="日k", adjust="")
                return _normalize_board_daily(raw, name)
            except TypeError:
                try:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        raw = func(symbol=name, start_date=start, end_date=end)
                    return _normalize_board_daily(raw, name)
                except Exception as exc:
                    errors.append(f"{func_name}: {exc}")
            except Exception as exc:
                errors.append(f"{func_name}: {exc}")
        raise RuntimeError("Unable to fetch board daily data. Last errors: " + " | ".join(errors[-3:]))

    def _eastmoney_daily(self, symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
        symbol = _normalize_symbol(symbol)
        fqt = {"": "0", "qfq": "1", "hfq": "2"}.get(adjust, "1")
        market = "1" if symbol.startswith(("6", "9")) else "0"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "klt": "101",
            "fqt": fqt,
            "secid": f"{market}.{symbol}",
            "beg": start,
            "end": end,
        }
        response = self._request_eastmoney(params)
        return _parse_eastmoney_klines(response, symbol)

    def _eastmoney_intraday(self, symbol: str, period: str, start_time: str, end_time: str, adjust: str) -> pd.DataFrame:
        symbol = _normalize_symbol(symbol)
        fqt = {"": "0", "qfq": "1", "hfq": "2"}.get(adjust, "0")
        market = "1" if symbol.startswith(("6", "9")) else "0"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "klt": str(period),
            "fqt": fqt,
            "secid": f"{market}.{symbol}",
            "beg": "0",
            "end": "20500000",
        }
        response = self._request_eastmoney(params)
        data = _parse_eastmoney_intraday_klines(response, symbol)
        start_ts = pd.to_datetime(start_time)
        end_ts = pd.to_datetime(end_time)
        data = data[(data["date"] >= start_ts) & (data["date"] <= end_ts)]
        return data.reset_index(drop=True)

    def _eastmoney_index_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        market = "0" if symbol.startswith("399") else "1"
        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
            "ut": "7eea3edcaed734bea9cbfc24409ed989",
            "klt": "101",
            "fqt": "0",
            "secid": f"{market}.{symbol}",
            "beg": start,
            "end": end,
        }
        response = self._request_eastmoney(params)
        return _parse_eastmoney_klines(response, symbol)

    def _request_eastmoney(self, params: dict[str, str]) -> Any:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://quote.eastmoney.com/",
        }
        hosts = [
            "82.push2his.eastmoney.com",
            "push2his.eastmoney.com",
        ]
        errors: list[str] = []

        try:
            from curl_cffi import requests as curl_requests
        except ImportError:
            curl_requests = None

        if curl_requests is not None:
            for host in hosts:
                url = f"http://{host}/api/qt/stock/kline/get"
                for attempt in range(self.request_retries):
                    try:
                        response = curl_requests.get(
                            url,
                            params=params,
                            headers=headers,
                            impersonate="chrome120",
                            timeout=self.request_timeout,
                        )
                        response.raise_for_status()
                        return response
                    except Exception as exc:
                        errors.append(f"curl_cffi {host} http, attempt={attempt + 1}: {exc}")
                        time.sleep(0.2)

        for host in hosts:
            for scheme in ["http", "https"]:
                url = f"{scheme}://{host}/api/qt/stock/kline/get"
                for trust_env in [False, True]:
                    for attempt in range(self.request_retries):
                        session = requests.Session()
                        session.trust_env = trust_env
                        try:
                            response = session.get(url, params=params, headers=headers, timeout=self.request_timeout)
                            response.raise_for_status()
                            return response
                        except requests.exceptions.RequestException as exc:
                            errors.append(f"{host} {scheme}, trust_env={trust_env}, attempt={attempt + 1}: {exc}")
                            time.sleep(0.3)

        raise RuntimeError("Unable to fetch Eastmoney daily data. Last errors: " + " | ".join(errors[-3:]))


def _parse_eastmoney_klines(response: Any, symbol: str) -> pd.DataFrame:
    payload = response.json()
    klines = (payload.get("data") or {}).get("klines") or []
    rows = []
    for item in klines:
        fields = item.split(",")
        if len(fields) < 7:
            continue
        rows.append(
            {
                "date": fields[0],
                "open": fields[1],
                "close": fields[2],
                "high": fields[3],
                "low": fields[4],
                "volume": fields[5],
                "amount": fields[6],
                "symbol": symbol,
            }
        )
    return normalize_ohlcv(pd.DataFrame(rows), symbol=symbol)


def _parse_eastmoney_intraday_klines(response: Any, symbol: str) -> pd.DataFrame:
    payload = response.json()
    klines = (payload.get("data") or {}).get("klines") or []
    rows = []
    for item in klines:
        fields = item.split(",")
        if len(fields) < 7:
            continue
        rows.append(
            {
                "date": fields[0],
                "open": fields[1],
                "close": fields[2],
                "high": fields[3],
                "low": fields[4],
                "volume": fields[5],
                "amount": fields[6],
                "symbol": symbol,
            }
        )
    return normalize_ohlcv(pd.DataFrame(rows), symbol=symbol)


def _normalize_intraday(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Intraday data is empty.")

    data = raw.copy()
    aliases = {
        "时间": "date",
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }
    data = data.rename(columns={key: value for key, value in aliases.items() if key in data.columns})
    lowered = {str(column).strip().lower(): column for column in data.columns}
    required = {"date", "open", "close", "high", "low", "volume"}
    if required.issubset(lowered):
        return normalize_ohlcv(data.rename(columns={lowered[key]: key for key in lowered}), symbol=symbol)

    if data.shape[1] < 6:
        return normalize_ohlcv(data, symbol=symbol)

    ordered = data.iloc[:, :9].copy()
    rename = {
        ordered.columns[0]: "date",
        ordered.columns[1]: "open",
        ordered.columns[2]: "close",
        ordered.columns[3]: "high",
        ordered.columns[4]: "low",
        ordered.columns[7] if ordered.shape[1] >= 8 else ordered.columns[5]: "volume",
    }
    if ordered.shape[1] >= 9:
        rename[ordered.columns[8]] = "amount"
    return normalize_ohlcv(ordered.rename(columns=rename), symbol=symbol)


def _normalize_board_daily(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    data = raw.copy()
    aliases = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    data = data.rename(columns={key: value for key, value in aliases.items() if key in data.columns})
    lowered = {str(column).strip().lower(): column for column in data.columns}
    if {"date", "open", "high", "low", "close", "volume"}.issubset(lowered):
        return normalize_ohlcv(data.rename(columns={lowered[key]: key for key in lowered}), symbol=symbol)

    if data.shape[1] < 6:
        return normalize_ohlcv(data, symbol=symbol)
    ordered = data.iloc[:, :7].copy()
    rename = {
        ordered.columns[0]: "date",
        ordered.columns[1]: "open",
        ordered.columns[2]: "close",
        ordered.columns[3]: "high",
        ordered.columns[4]: "low",
        ordered.columns[5]: "volume",
    }
    if ordered.shape[1] >= 7:
        rename[ordered.columns[6]] = "amount"
    return normalize_ohlcv(ordered.rename(columns=rename), symbol=symbol)


def _extract_symbols(df: pd.DataFrame) -> list[str]:
    symbols: list[str] = []
    preferred_columns = ["symbol", "code", "代码", "成分券代码", "品种代码", "证券代码"]
    columns = [column for column in preferred_columns if column in df.columns]
    columns.extend([column for column in df.columns if column not in columns])

    for column in columns:
        values = df[column].astype(str)
        found = values.str.extract(r"(\d{6})", expand=False).dropna().tolist()
        symbols.extend(found)
        if symbols:
            break

    deduped = []
    seen = set()
    for symbol in symbols:
        symbol = symbol.zfill(6)
        if re.fullmatch(r"\d{6}", symbol) and symbol not in seen:
            seen.add(symbol)
            deduped.append(symbol)
    return deduped


def _normalize_symbol(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    match = re.search(r"(\d{6})", text)
    if match:
        return match.group(1)
    digits = re.sub(r"\D", "", text)
    if 1 <= len(digits) <= 6:
        return digits.zfill(6)
    return text


def _tencent_symbol(symbol: str) -> str:
    symbol = _normalize_symbol(symbol)
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
    return f"{prefix}{symbol}"


def _tencent_index_symbol(symbol: str) -> str:
    symbol = _normalize_symbol(symbol)
    if symbol.startswith("399"):
        return f"sz{symbol}"
    return f"sh{symbol}"


def _index_proxy_symbol(symbol: str) -> str:
    return {
        "000300": "510300",
        "000016": "510050",
        "000905": "510500",
        "399006": "159915",
    }.get(_normalize_symbol(symbol), "")


def _date_with_dash(value: str) -> str:
    value = str(value)
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def _intraday_time_range(start: str, end: str) -> tuple[str, str]:
    end_ts = pd.Timestamp.today() if not end else pd.to_datetime(end)
    start_ts = end_ts - pd.Timedelta(days=15) if not start else pd.to_datetime(start)
    if str(start).isdigit() and len(str(start)) == 8:
        start_ts = pd.to_datetime(str(start), format="%Y%m%d")
    if str(end).isdigit() and len(str(end)) == 8:
        end_ts = pd.to_datetime(str(end), format="%Y%m%d")
    return f"{start_ts:%Y-%m-%d} 09:30:00", f"{end_ts:%Y-%m-%d} 15:00:00"


def _calendar_day_count(start: str, end: str) -> int:
    try:
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        return max(300, min(8000, int((end_ts - start_ts).days * 1.5) + 60))
    except Exception:
        return 3000
