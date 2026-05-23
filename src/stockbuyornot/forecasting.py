from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastConfig:
    horizon: int = 10
    lookback_days: int = 750
    min_rows: int = 250
    model: str = "light_var"
    interval_z: float = 1.2815515655446004
    ridge_alpha: float = 3.0


@dataclass(frozen=True)
class ForecastPoint:
    step: int
    date: pd.Timestamp
    predicted_close: float
    predicted_return: float
    lower: float
    upper: float


@dataclass(frozen=True)
class ForecastResult:
    symbol: str
    as_of: pd.Timestamp
    expected_return_10d: float
    up_probability: float
    confidence: float
    conclusion: str
    points: list[ForecastPoint]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def forecast_symbol(
    provider,
    symbol: str,
    settings: dict,
    benchmark: pd.DataFrame | None = None,
    sector: pd.DataFrame | None = None,
    volume_price_action: str = "",
    stop_loss: float | None = None,
    config: ForecastConfig = ForecastConfig(),
) -> ForecastResult:
    end = str(settings.get("end") or pd.Timestamp.today().strftime("%Y%m%d"))
    end_ts = pd.to_datetime(end)
    start_ts = end_ts - pd.Timedelta(days=max(1500, config.lookback_days * 2))
    start = start_ts.strftime("%Y%m%d")
    adjust = str(settings.get("adjust", "qfq"))

    df = provider.daily(symbol, start, end, adjust)
    if benchmark is None:
        try:
            benchmark = provider.index_daily("000300", start, end)
        except Exception:
            benchmark = None

    return forecast_ohlcv(
        df,
        benchmark=benchmark,
        sector=sector,
        symbol=symbol,
        volume_price_action=volume_price_action,
        stop_loss=stop_loss,
        config=config,
    )


def forecast_ohlcv(
    df: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    sector: pd.DataFrame | None = None,
    symbol: str = "",
    volume_price_action: str = "",
    stop_loss: float | None = None,
    as_of: pd.Timestamp | str | None = None,
    config: ForecastConfig = ForecastConfig(),
) -> ForecastResult:
    stock = _prepare_ohlcv(df, as_of=as_of)
    if len(stock) < config.min_rows:
        raise ValueError(f"样本不足，至少需要 {config.min_rows} 个交易日才能进行预测。")

    frame = _build_feature_frame(stock, benchmark=benchmark, sector=sector, as_of=as_of)
    frame = frame.tail(config.lookback_days).reset_index(drop=True)
    if len(frame) < config.min_rows:
        raise ValueError(f"有效样本不足，至少需要 {config.min_rows} 个交易日。")

    ridge_prediction, residual_std, train_rows = _direct_ridge_forecast(frame, config)
    model_used = "Direct Ridge"
    model_error = ""
    predicted_returns = ridge_prediction

    if config.model == "light_var":
        try:
            predicted_returns = _var_forecast(frame, config)
            model_used = "VAR"
        except Exception as exc:
            model_error = str(exc)

    close = float(frame["close"].iloc[-1])
    as_of_date = pd.to_datetime(frame["date"].iloc[-1])
    points = _build_points(close, as_of_date, predicted_returns, residual_std, config)
    final_return = float(predicted_returns[min(config.horizon, len(predicted_returns)) - 1])
    final_std = float(residual_std[min(config.horizon, len(residual_std)) - 1])
    up_probability = _normal_cdf(final_return / max(final_std, 1e-6))
    confidence = _confidence(train_rows, final_return, final_std, model_used)
    lower_10d_return = points[-1].lower / close - 1 if points else 0.0
    conclusion = _forecast_conclusion(
        expected_return=final_return,
        up_probability=up_probability,
        lower_return=lower_10d_return,
        close=close,
        volume_price_action=volume_price_action,
        stop_loss=stop_loss,
    )

    diagnostics = {
        "model_used": model_used,
        "fallback_error": model_error,
        "train_rows": int(train_rows),
        "last_close": close,
        "recent_dates": [pd.to_datetime(item).strftime("%Y-%m-%d") for item in frame["date"].tail(90)],
        "recent_close": [float(item) for item in frame["close"].tail(90)],
        "disclaimer": "预测是统计辅助，不构成投资建议，不能替代止损和量价确认。",
    }
    return ForecastResult(
        symbol=symbol or str(stock.get("symbol", pd.Series([""])).iloc[-1]),
        as_of=as_of_date,
        expected_return_10d=final_return,
        up_probability=float(up_probability),
        confidence=float(confidence),
        conclusion=conclusion,
        points=points,
        diagnostics=diagnostics,
    )


def forecast_points_frame(result: ForecastResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "T+N": f"T+{point.step}",
                "预测日期": point.date.strftime("%Y-%m-%d"),
                "预测收盘价": round(point.predicted_close, 3),
                "预测涨跌幅": f"{point.predicted_return:.2%}",
                "下沿": round(point.lower, 3),
                "上沿": round(point.upper, 3),
            }
            for point in result.points
        ]
    )


def forecast_chart_frame(result: ForecastResult) -> pd.DataFrame:
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(result.diagnostics.get("recent_dates", [])),
            "value": result.diagnostics.get("recent_close", []),
            "lower": np.nan,
            "upper": np.nan,
            "type": "历史收盘",
        }
    )
    forecast = pd.DataFrame(
        [
            {
                "date": point.date,
                "value": point.predicted_close,
                "lower": point.lower,
                "upper": point.upper,
                "type": "预测中位",
            }
            for point in result.points
        ]
    )
    return pd.concat([history, forecast], ignore_index=True)


def _prepare_ohlcv(df: pd.DataFrame, as_of: pd.Timestamp | str | None = None) -> pd.DataFrame:
    if df.empty:
        raise ValueError("OHLCV data is empty.")
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    if as_of is not None:
        data = data[data["date"] <= pd.to_datetime(as_of)]
    data = data.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    if data.empty:
        raise ValueError("OHLCV data is empty after normalization.")
    return data


def _build_feature_frame(
    stock: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    sector: pd.DataFrame | None = None,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    data = stock[["date", "open", "high", "low", "close", "volume"]].copy()
    data["stock_ret"] = data["close"].pct_change().clip(-0.2, 0.2)
    data["volume_chg"] = np.log(data["volume"].replace(0, np.nan)).diff().replace([np.inf, -np.inf], np.nan).clip(-3, 3)
    data["volume_ratio"] = data["volume"] / data["volume"].rolling(20, min_periods=10).mean() - 1
    data["amplitude"] = (data["high"] - data["low"]) / data["close"].shift(1)
    data["volatility_20"] = data["stock_ret"].rolling(20, min_periods=10).std()
    data["ma20"] = data["close"].rolling(20, min_periods=10).mean()
    data["ma60"] = data["close"].rolling(60, min_periods=30).mean()
    data["ma20_gap"] = data["close"] / data["ma20"] - 1
    data["ma60_gap"] = data["close"] / data["ma60"] - 1
    data["ma20_slope"] = data["ma20"].pct_change(5)
    data["ma60_slope"] = data["ma60"].pct_change(10)

    data = _merge_return(data, benchmark, "bench_ret", as_of=as_of)
    data = _merge_return(data, sector, "sector_ret", as_of=as_of)
    data["relative_ret"] = data["stock_ret"] - data["bench_ret"]
    return data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _merge_return(data: pd.DataFrame, other: pd.DataFrame | None, column: str, as_of: pd.Timestamp | str | None = None) -> pd.DataFrame:
    if other is None or other.empty or "close" not in other.columns or "date" not in other.columns:
        data[column] = 0.0
        return data
    ref = other[["date", "close"]].copy()
    ref["date"] = pd.to_datetime(ref["date"])
    if as_of is not None:
        ref = ref[ref["date"] <= pd.to_datetime(as_of)]
    ref = ref.sort_values("date").drop_duplicates("date", keep="last")
    ref[column] = ref["close"].pct_change().clip(-0.2, 0.2)
    merged = data.merge(ref[["date", column]], on="date", how="left")
    merged[column] = merged[column].ffill().fillna(0.0)
    return merged


def _direct_ridge_forecast(frame: pd.DataFrame, config: ForecastConfig) -> tuple[np.ndarray, np.ndarray, int]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x_all, y_all, latest_x = _make_supervised_dataset(frame, config)
    if len(x_all) < 80:
        raise ValueError("可用于训练预测模型的样本不足。")
    model = make_pipeline(StandardScaler(), Ridge(alpha=config.ridge_alpha))
    model.fit(x_all, y_all)
    prediction = np.asarray(model.predict(latest_x), dtype=float).reshape(-1)
    fitted = np.asarray(model.predict(x_all), dtype=float)
    residual_std = np.nanstd(y_all - fitted, axis=0, ddof=1)
    historical_std = _historical_future_return_std(frame, config)
    residual_std = np.maximum(np.nan_to_num(residual_std, nan=0.0), historical_std * 0.35)
    residual_std = np.maximum(residual_std, 0.005)
    return _sanitize_returns(prediction), residual_std, len(x_all)


def _make_supervised_dataset(frame: pd.DataFrame, config: ForecastConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_columns = [
        "stock_ret",
        "bench_ret",
        "sector_ret",
        "relative_ret",
        "volume_chg",
        "volume_ratio",
        "amplitude",
        "volatility_20",
        "ma20_gap",
        "ma60_gap",
        "ma20_slope",
        "ma60_slope",
    ]
    features = pd.DataFrame(index=frame.index)
    for column in base_columns:
        for lag in [0, 1, 2, 3, 5, 10]:
            features[f"{column}_lag{lag}"] = frame[column].shift(lag)
    features["ret_mean_5"] = frame["stock_ret"].rolling(5, min_periods=3).mean()
    features["ret_mean_20"] = frame["stock_ret"].rolling(20, min_periods=10).mean()
    features["ret_std_20"] = frame["stock_ret"].rolling(20, min_periods=10).std()
    features["volume_mean_5"] = frame["volume_chg"].rolling(5, min_periods=3).mean()
    features["trend_quality"] = frame["ma20_gap"] + frame["ma20_slope"] * 2

    targets = pd.DataFrame(index=frame.index)
    for step in range(1, config.horizon + 1):
        targets[f"target_{step}"] = frame["close"].shift(-step) / frame["close"] - 1

    dataset = pd.concat([features, targets], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    feature_columns = list(features.columns)
    target_columns = list(targets.columns)
    latest_x = features.iloc[[-1]][feature_columns].replace([np.inf, -np.inf], np.nan)
    if latest_x.isna().any(axis=None):
        latest_x = features[feature_columns].dropna().iloc[[-1]]
    return dataset[feature_columns], dataset[target_columns], latest_x


def _var_forecast(frame: pd.DataFrame, config: ForecastConfig) -> np.ndarray:
    from statsmodels.tsa.api import VAR

    columns = ["stock_ret", "bench_ret", "sector_ret", "volume_chg", "volatility_20", "ma20_slope"]
    var_data = frame[columns].replace([np.inf, -np.inf], np.nan).dropna().tail(config.lookback_days)
    if len(var_data) < config.min_rows:
        raise ValueError("VAR effective sample is too small.")
    model = VAR(var_data)
    maxlags = max(1, min(5, len(var_data) // 80))
    fitted = model.fit(maxlags=maxlags, ic=None, trend="c")
    forecast_values = fitted.forecast(var_data.values[-fitted.k_ar :], steps=config.horizon)
    daily_stock_returns = np.asarray(forecast_values[:, 0], dtype=float)
    cumulative = np.cumprod(1 + np.clip(daily_stock_returns, -0.12, 0.12)) - 1
    return _sanitize_returns(cumulative)


def _historical_future_return_std(frame: pd.DataFrame, config: ForecastConfig) -> np.ndarray:
    values = []
    for step in range(1, config.horizon + 1):
        future_returns = frame["close"].shift(-step) / frame["close"] - 1
        std = float(future_returns.dropna().std(ddof=1) or 0.0)
        values.append(max(std, 0.005))
    return np.asarray(values, dtype=float)


def _sanitize_returns(values: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.25, neginf=-0.25)
    return np.clip(clean, -0.35, 0.35)


def _build_points(
    close: float,
    as_of: pd.Timestamp,
    predicted_returns: np.ndarray,
    residual_std: np.ndarray,
    config: ForecastConfig,
) -> list[ForecastPoint]:
    points = []
    for index in range(config.horizon):
        step = index + 1
        predicted_return = float(predicted_returns[index])
        std = float(residual_std[index])
        lower_return = predicted_return - config.interval_z * std
        upper_return = predicted_return + config.interval_z * std
        points.append(
            ForecastPoint(
                step=step,
                date=as_of + pd.offsets.BDay(step),
                predicted_close=max(0.01, close * (1 + predicted_return)),
                predicted_return=predicted_return,
                lower=max(0.01, close * (1 + lower_return)),
                upper=max(0.01, close * (1 + upper_return)),
            )
        )
    return points


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1 + erf(value / sqrt(2))))


def _confidence(train_rows: int, expected_return: float, final_std: float, model_used: str) -> float:
    sample_score = min(1.0, max(0.0, (train_rows - 80) / 500))
    signal_to_noise = min(2.0, abs(expected_return) / max(final_std, 1e-6)) / 2
    model_bonus = 5.0 if model_used == "VAR" else 0.0
    return max(20.0, min(90.0, 35.0 + sample_score * 30.0 + signal_to_noise * 20.0 + model_bonus))


def _forecast_conclusion(
    expected_return: float,
    up_probability: float,
    lower_return: float,
    close: float,
    volume_price_action: str,
    stop_loss: float | None,
) -> str:
    action = _normalize_action(volume_price_action)
    lower_price = close * (1 + lower_return)
    breaks_stop = stop_loss is not None and stop_loss > 0 and lower_price < float(stop_loss)

    if action == "buy":
        if expected_return > 0.03 and up_probability >= 0.60 and not breaks_stop:
            return "预测支持当前买点"
        if breaks_stop:
            return "量价有信号，但预测下沿跌破止损，暂不支持追入"
        if expected_return < 0 or up_probability < 0.50:
            return "量价有信号，但预测不支持追入"
        return "量价有信号，但预测优势一般，继续观察确认"

    if action in {"sell", "avoid"}:
        if expected_return > 0.03 and up_probability >= 0.60:
            return "预测偏强，但量价未确认"
        return "预测不支持买入，风险偏高"

    if expected_return > 0.03 and up_probability >= 0.60:
        return "预测偏强，等待量价买点确认"
    if expected_return < -0.02 or up_probability < 0.45:
        return "风险偏高"
    return "中性观察"


def _normalize_action(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"wait", "watch", "trade", "观察", "等待买点"} or "等待" in text or "观察" in text:
        return "watch"
    if text in {"sell", "减仓/卖出", "卖"} or "卖" in text or "减仓" in text:
        return "sell"
    if text in {"avoid", "回避"} or "回避" in text:
        return "avoid"
    if text in {"buy", "可买入", "买"} or "可买入" in text or "买入" in text:
        return "buy"
    return "watch"
