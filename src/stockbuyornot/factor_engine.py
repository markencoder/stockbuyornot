from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stockbuyornot.config import FactorConfig
from stockbuyornot.models import FactorSnapshot


FUNDAMENTAL_ALIASES = {
    "roe": ["roe", "ROE", "净资产收益率"],
    "roa": ["roa", "ROA", "总资产收益率"],
    "gross_margin": ["gross_margin", "毛利率"],
    "net_margin": ["net_margin", "净利率"],
    "revenue_growth": ["revenue_growth", "营收增速", "营业收入增长率"],
    "net_profit_growth": ["net_profit_growth", "净利润增速"],
    "deducted_profit_growth": ["deducted_profit_growth", "扣非净利润增速"],
    "pe": ["pe", "PE", "市盈率"],
    "pb": ["pb", "PB", "市净率"],
    "ps": ["ps", "PS", "市销率"],
    "peg": ["peg", "PEG"],
    "pe_percentile": ["pe_percentile", "PE历史分位数", "市盈率分位数"],
    "pb_percentile": ["pb_percentile", "PB历史分位数", "市净率分位数"],
    "ocf_to_profit": ["ocf_to_profit", "经营现金流净利润比", "经营现金流/净利润"],
    "debt_to_assets": ["debt_to_assets", "资产负债率"],
    "interest_debt_ratio": ["interest_debt_ratio", "有息负债率"],
}


def compute_factors(
    df: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    sector: pd.DataFrame | None = None,
    fundamentals: dict[str, Any] | pd.DataFrame | None = None,
    config: FactorConfig = FactorConfig(),
) -> FactorSnapshot:
    data = _prepare(df)
    short_term = _short_term_factors(data, benchmark, sector)
    long_term = _long_term_factors(data, fundamentals)
    data_quality = _data_quality(data, benchmark, sector, long_term)
    return FactorSnapshot(short_term=short_term, long_term=long_term, data_quality=data_quality)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["date", "open", "high", "low", "close", "volume"]).reset_index(drop=True)


def _short_term_factors(data: pd.DataFrame, benchmark: pd.DataFrame | None, sector: pd.DataFrame | None) -> dict[str, Any]:
    close = data["close"]
    high = data["high"]
    low = data["low"]
    volume = data["volume"]
    factors: dict[str, Any] = {}
    factors["close"] = _last(close)
    for days in [1, 3, 5, 10, 20]:
        factors[f"return_{days}d"] = _return(close, days)
    for window in [5, 10, 20]:
        factors[f"ma{window}"] = _last(close.rolling(window, min_periods=max(3, window // 2)).mean())
    latest_close = factors.get("close")
    ma20 = factors.get("ma20")
    factors["ma20_gap_pct"] = _safe_div(None if latest_close is None or ma20 is None else latest_close - ma20, ma20)
    recent_high_20 = _last(high.rolling(20, min_periods=10).max())
    factors["pullback_from_20d_high"] = _safe_div(None if latest_close is None or recent_high_20 is None else latest_close - recent_high_20, recent_high_20)
    vol_ma5 = volume.rolling(5, min_periods=3).mean()
    vol_ma20 = volume.rolling(20, min_periods=5).mean()
    factors["volume_expand_ratio"] = _safe_div(_last(volume), _last(vol_ma5))
    factors["volume_ratio"] = _safe_div(_last(volume), _last(vol_ma20))
    factors["turnover"] = _last(data["turnover"]) if "turnover" in data.columns else None
    factors["turnover_ma5"] = _last(data["turnover"].rolling(5, min_periods=3).mean()) if "turnover" in data.columns else None
    factors["close_position_day"] = _safe_div(_last(close - low), _last(high - low))
    rolling_low_20 = low.rolling(20, min_periods=10).min()
    rolling_high_20 = high.rolling(20, min_periods=10).max()
    factors["price_position_20d"] = _safe_div(_last(close - rolling_low_20), _last(rolling_high_20 - rolling_low_20))
    rsi14 = _rsi(close)
    factors["rsi"] = _last(rsi14)
    factors["rsi14"] = factors["rsi"]
    factors["rsi6"] = _last(_rsi(close, 6))
    factors["cci14"] = _last(_cci(close, high, low))
    factors["williams_r14"] = _last(_williams_r(close, high, low))
    factors["mfi14"] = _last(_mfi(data))
    macd = _macd(close)
    factors.update({key: _last(value) for key, value in macd.items()})
    factors["macd_hist_prev"] = _lag(macd["macd_hist"], 1)
    factors["macd_hist_delta_3d"] = _series_delta(macd["macd_hist"], 3)
    factors["macd_golden_cross"] = bool(
        _lag(macd["macd_dif"], 1) is not None
        and _lag(macd["macd_dea"], 1) is not None
        and _last(macd["macd_dif"]) is not None
        and _last(macd["macd_dea"]) is not None
        and _lag(macd["macd_dif"], 1) <= _lag(macd["macd_dea"], 1)
        and _last(macd["macd_dif"]) > _last(macd["macd_dea"])
    )
    factors["macd_above_zero"] = bool((_last(macd["macd_dif"]) or 0) > 0 and (_last(macd["macd_dea"]) or 0) > 0)
    kdj = _kdj(close, high, low)
    factors.update({key: _last(value) for key, value in kdj.items()})
    k_now, d_now = _last(kdj["kdj_k"]), _last(kdj["kdj_d"])
    k_prev, d_prev = _lag(kdj["kdj_k"], 1), _lag(kdj["kdj_d"], 1)
    factors["kdj_golden_cross"] = bool(k_prev is not None and d_prev is not None and k_now is not None and d_now is not None and k_prev <= d_prev and k_now > d_now)
    factors["kdj_dead_cross"] = bool(k_prev is not None and d_prev is not None and k_now is not None and d_now is not None and k_prev >= d_prev and k_now < d_now)
    factors["kdj_j_delta_3d"] = _series_delta(kdj["kdj_j"], 3)
    boll = _bollinger(close)
    factors.update({key: _last(value) for key, value in boll.items()})
    factors["boll_bandwidth_ratio"] = _safe_div(factors.get("boll_bandwidth"), _last(boll["boll_bandwidth"].rolling(60, min_periods=20).mean()))
    obv = _obv(close, volume)
    factors["obv"] = _last(obv)
    factors["obv_ma20"] = _last(obv.rolling(20, min_periods=5).mean())
    factors["obv_above_ma20"] = bool(factors["obv"] is not None and factors["obv_ma20"] is not None and factors["obv"] > factors["obv_ma20"])
    factors["obv_change_5d"] = _safe_div(_series_delta(obv, 5), _last(vol_ma20))
    adx = _adx(data)
    factors.update({key: _last(value) for key, value in adx.items()})
    atr = _atr(data)
    factors["atr"] = _last(atr)
    factors["atr_pct"] = _safe_div(factors.get("atr"), _last(close))
    returns = close.pct_change()
    factors["volatility_20d"] = _last(returns.rolling(20, min_periods=10).std())
    factors["volatility_60d"] = _last(returns.rolling(60, min_periods=20).std())
    factors["volatility_ratio"] = _safe_div(factors.get("volatility_20d"), factors.get("volatility_60d"))
    factors["new_high_20"] = bool(len(data) >= 20 and close.iloc[-1] >= high.rolling(20, min_periods=20).max().iloc[-1])
    factors["new_high_60"] = bool(len(data) >= 60 and close.iloc[-1] >= high.rolling(60, min_periods=60).max().iloc[-1])
    factors["short_max_drawdown"] = _max_drawdown(close.tail(min(len(close), 20)))
    factors["stock_vs_benchmark_20d"] = _relative_strength(data, benchmark, 20)
    factors["stock_vs_sector_20d"] = _relative_strength(data, sector, 20)
    return _clean(factors)


def _long_term_factors(data: pd.DataFrame, fundamentals: dict[str, Any] | pd.DataFrame | None) -> dict[str, Any]:
    close = data["close"]
    factors: dict[str, Any] = {}
    for window in [60, 120, 250]:
        factors[f"ma{window}"] = _last(close.rolling(window, min_periods=max(20, window // 2)).mean())
        factors[f"return_{window}d"] = _return(close, window)
    factors["long_max_drawdown"] = _max_drawdown(close.tail(min(len(close), 250)))
    factors["long_volatility"] = _last(close.pct_change().rolling(120, min_periods=40).std()) or _last(close.pct_change().rolling(60, min_periods=20).std())
    extracted = _extract_fundamentals(data, fundamentals)
    factors.update(extracted)
    return _clean(factors)


def _extract_fundamentals(data: pd.DataFrame, fundamentals: dict[str, Any] | pd.DataFrame | None) -> dict[str, Any]:
    source: dict[str, Any] = {}
    if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty:
        row = fundamentals.iloc[-1].to_dict()
        source.update(row)
    elif isinstance(fundamentals, dict):
        source.update(fundamentals)
    if not data.empty:
        source.update({column: data[column].iloc[-1] for column in data.columns if column not in source})

    result: dict[str, Any] = {}
    for canonical, aliases in FUNDAMENTAL_ALIASES.items():
        result[canonical] = None
        for alias in aliases:
            if alias in source:
                result[canonical] = _to_float(source.get(alias))
                break
    return result


def _data_quality(data: pd.DataFrame, benchmark: pd.DataFrame | None, sector: pd.DataFrame | None, long_term: dict[str, Any]) -> dict[str, Any]:
    fundamental_keys = ["roe", "roa", "gross_margin", "net_margin", "revenue_growth", "net_profit_growth"]
    valuation_keys = ["pe", "pb", "ps", "peg", "pe_percentile", "pb_percentile"]
    risk_keys = ["ocf_to_profit", "debt_to_assets", "interest_debt_ratio"]
    return {
        "price_rows": int(len(data)),
        "has_250d_history": len(data) >= 250,
        "has_benchmark": benchmark is not None and not benchmark.empty,
        "has_sector": sector is not None and not sector.empty,
        "fundamental_fields_present": [key for key in fundamental_keys if long_term.get(key) is not None],
        "fundamental_fields_missing": [key for key in fundamental_keys if long_term.get(key) is None],
        "valuation_fields_present": [key for key in valuation_keys if long_term.get(key) is not None],
        "risk_fields_present": [key for key in risk_keys if long_term.get(key) is not None],
    }


def _return(close: pd.Series, days: int) -> float | None:
    if len(close) <= days:
        return None
    base = close.iloc[-days - 1]
    if pd.isna(base) or base == 0:
        return None
    return float(close.iloc[-1] / base - 1)


def _relative_strength(data: pd.DataFrame, ref: pd.DataFrame | None, days: int) -> float | None:
    stock_ret = _return(data["close"], days)
    if stock_ret is None or ref is None or ref.empty:
        return None
    benchmark = ref.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    aligned = benchmark[benchmark["date"].between(data["date"].iloc[-days - 1], data["date"].iloc[-1])]
    ref_ret = _return(aligned["close"].reset_index(drop=True), min(days, len(aligned) - 1)) if len(aligned) > 1 else None
    return None if ref_ret is None else float(stock_ret - ref_ret)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-diff.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _macd(close: pd.Series) -> dict[str, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2
    return {"macd_dif": dif, "macd_dea": dea, "macd_hist": hist}


def _kdj(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 9) -> dict[str, pd.Series]:
    lowest = low.rolling(window, min_periods=window).min()
    highest = high.rolling(window, min_periods=window).max()
    rsv = ((close - lowest) / (highest - lowest).replace(0, np.nan) * 100).fillna(50)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d
    return {"kdj_k": k, "kdj_d": d, "kdj_j": j}


def _cci(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    typical = (high + low + close) / 3
    ma = typical.rolling(window, min_periods=window).mean()
    mean_dev = (typical - ma).abs().rolling(window, min_periods=window).mean()
    return ((typical - ma) / (0.015 * mean_dev.replace(0, np.nan))).fillna(0)


def _williams_r(close: pd.Series, high: pd.Series, low: pd.Series, window: int = 14) -> pd.Series:
    highest = high.rolling(window, min_periods=window).max()
    lowest = low.rolling(window, min_periods=window).min()
    return (-100 * (highest - close) / (highest - lowest).replace(0, np.nan)).fillna(-50)


def _mfi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    typical = (data["high"] + data["low"] + data["close"]) / 3
    raw_money = typical * data["volume"]
    direction = typical.diff()
    positive = raw_money.where(direction > 0, 0.0).rolling(window, min_periods=window).sum()
    negative = raw_money.where(direction < 0, 0.0).rolling(window, min_periods=window).sum()
    ratio = positive / negative.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50)


def _bollinger(close: pd.Series, window: int = 20, width: float = 2.0) -> dict[str, pd.Series]:
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    upper = mid + width * std
    lower = mid - width * std
    band_width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return {
        "boll_mid": mid,
        "boll_upper": upper,
        "boll_lower": lower,
        "boll_percent_b": pct_b,
        "boll_bandwidth": band_width,
    }


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _adx(data: pd.DataFrame, window: int = 14) -> dict[str, pd.Series]:
    high = data["high"]
    low = data["low"]
    close = data["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=data.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=data.index)
    tr = _true_range(data)
    atr = tr.rolling(window, min_periods=window).sum()
    plus_di = 100 * plus_dm.rolling(window, min_periods=window).sum() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window, min_periods=window).sum() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window, min_periods=window).mean()
    return {"adx14": adx, "plus_di14": plus_di, "minus_di14": minus_di}


def _atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    return _true_range(data).rolling(window, min_periods=window).mean()


def _true_range(data: pd.DataFrame) -> pd.Series:
    prev_close = data["close"].shift(1)
    return pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _max_drawdown(close: pd.Series) -> float | None:
    clean = close.dropna()
    if clean.empty:
        return None
    drawdown = clean / clean.cummax() - 1
    return float(drawdown.min())


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    num = _to_float(numerator)
    den = _to_float(denominator)
    if num is None or den in [None, 0]:
        return None
    return float(num / den)


def _last(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return _to_float(series.iloc[-1])


def _lag(series: pd.Series, periods: int = 1) -> float | None:
    if len(series) <= periods:
        return None
    return _to_float(series.iloc[-1 - periods])


def _series_delta(series: pd.Series, periods: int = 1) -> float | None:
    latest = _last(series)
    previous = _lag(series, periods)
    if latest is None or previous is None:
        return None
    return float(latest - previous)


def _to_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(values: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in values.items():
        if isinstance(value, (np.bool_, bool)):
            cleaned[key] = bool(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            cleaned[key] = None if pd.isna(value) or np.isinf(value) else float(value)
        else:
            cleaned[key] = value
    return cleaned
