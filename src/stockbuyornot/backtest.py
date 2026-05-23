from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockbuyornot.analysis import analyze_ohlcv
from stockbuyornot.config import AppConfig
from stockbuyornot.features import add_features
from stockbuyornot.models import SignalSide


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    metrics: dict[str, float]


def backtest_single(
    df: pd.DataFrame,
    symbol: str = "",
    config: AppConfig = AppConfig(),
    min_score: int = 75,
    initial_cash: float = 100000.0,
    risk_per_trade: float = 0.01,
    fee_rate: float = 0.0003,
    stamp_tax: float = 0.0005,
    slippage: float = 0.001,
) -> BacktestResult:
    data = add_features(df, config.features).reset_index(drop=True)
    trades: list[dict] = []
    cash = initial_cash
    position = 0
    entry_price = 0.0
    stop_loss = 0.0

    for idx in range(120, len(data) - 1):
        window = data.iloc[: idx + 1].copy()
        try:
            result = analyze_ohlcv(window, symbol=symbol, config=config)
        except ValueError:
            continue

        next_open = float(data.iloc[idx + 1]["open"])
        date = data.iloc[idx + 1]["date"]
        sell_signal = any(signal.side == SignalSide.SELL for signal in result.signals)
        buy_signal = any(signal.side == SignalSide.BUY for signal in result.signals)

        if position > 0:
            stop_hit = float(data.iloc[idx + 1]["low"]) <= stop_loss
            if sell_signal or stop_hit:
                exit_price = (stop_loss if stop_hit else next_open) * (1 - slippage)
                cash += position * exit_price * (1 - fee_rate - stamp_tax)
                trades.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "side": "sell",
                        "price": exit_price,
                        "shares": position,
                        "reason": "stop_loss" if stop_hit else "sell_signal",
                        "cash": cash,
                    }
                )
                position = 0
                entry_price = 0.0
                stop_loss = 0.0
            continue

        if buy_signal and result.score.total >= min_score:
            best_stop = max((signal.stop_loss for signal in result.signals if signal.stop_loss), default=None)
            if not best_stop or best_stop >= next_open:
                continue
            entry = next_open * (1 + slippage)
            per_share_risk = entry - best_stop
            risk_budget = cash * risk_per_trade
            shares = int(min(cash / entry, risk_budget / per_share_risk) // 100 * 100)
            if shares <= 0:
                continue
            cash -= shares * entry * (1 + fee_rate)
            position = shares
            entry_price = entry
            stop_loss = best_stop
            trades.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "side": "buy",
                    "price": entry,
                    "shares": shares,
                    "reason": result.signals[0].name,
                    "cash": cash,
                }
            )

    if position > 0:
        final_price = float(data.iloc[-1]["close"]) * (1 - slippage)
        cash += position * final_price * (1 - fee_rate - stamp_tax)
        trades.append(
            {
                "date": data.iloc[-1]["date"],
                "symbol": symbol,
                "side": "sell",
                "price": final_price,
                "shares": position,
                "reason": "end_of_data",
                "cash": cash,
            }
        )

    trade_df = pd.DataFrame(trades)
    metrics = _metrics(trade_df, initial_cash, cash)
    return BacktestResult(trade_df, metrics)


def _metrics(trades: pd.DataFrame, initial_cash: float, final_cash: float) -> dict[str, float]:
    if trades.empty:
        return {
            "initial_cash": initial_cash,
            "final_cash": final_cash,
            "total_return": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
        }

    pairs = []
    open_trade = None
    for _, trade in trades.iterrows():
        if trade["side"] == "buy":
            open_trade = trade
        elif trade["side"] == "sell" and open_trade is not None:
            pnl = (trade["price"] - open_trade["price"]) / open_trade["price"]
            pairs.append(pnl)
            open_trade = None
    wins = [pnl for pnl in pairs if pnl > 0]
    return {
        "initial_cash": initial_cash,
        "final_cash": final_cash,
        "total_return": final_cash / initial_cash - 1,
        "trade_count": len(pairs),
        "win_rate": len(wins) / len(pairs) if pairs else 0.0,
        "avg_trade_return": sum(pairs) / len(pairs) if pairs else 0.0,
    }
