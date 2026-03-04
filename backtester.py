"""
Vectorised backtesting engine.

Simulates trade execution from a signal DataFrame produced by a strategy
function.  Tracks positions, equity, and per-trade records.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class TradeRecord:
    """Single completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int          # 1 long, -1 short
    entry_price: float
    exit_price: float
    size: float             # number of contracts / units
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trades: list = field(default_factory=list)
    signals_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def run(
    signals: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_pct: float = 0.5,
    point_value: float = 20.0,
    commission_per_contract: float = 4.0,
) -> BacktestResult:
    """Execute a backtest over a signal DataFrame.

    Parameters
    ----------
    signals : pd.DataFrame
        Must contain columns: open, high, low, close, signal, stop_price.
        ``signal`` values: 1 (long entry), -1 (short entry), 0 (no action).
        ``stop_price`` is the initial stop for the trade (NaN when no entry).
    initial_capital : float
        Starting account equity in USD.
    risk_pct : float
        Percentage of equity risked per trade (e.g. 0.5 = 0.5%).
    point_value : float
        Dollar value per point of price movement per contract (NQ = $20).
    commission_per_contract : float
        Round-trip commission per contract.

    Returns
    -------
    BacktestResult
        Equity curve, drawdown series, and list of TradeRecord objects.
    """
    closes = signals["close"].values
    highs = signals["high"].values
    lows = signals["low"].values
    opens = signals["open"].values
    sigs = signals["signal"].values.astype(int)
    stops = signals["stop_price"].values
    timestamps = signals.index

    n = len(closes)
    equity = np.empty(n, dtype=np.float64)
    equity[0] = initial_capital

    # Position state
    pos_dir = 0          # 1 long, -1 short, 0 flat
    pos_size = 0.0
    entry_price = 0.0
    stop_price = np.nan
    entry_idx = 0
    trades: list[TradeRecord] = []

    for i in range(1, n):
        bar_pnl = 0.0

        # --- Check stop-loss on open positions ---
        if pos_dir != 0:
            stopped = False
            if pos_dir == 1 and lows[i] <= stop_price:
                # Long stopped out — assume fill at stop price
                exit_price = stop_price
                stopped = True
            elif pos_dir == -1 and highs[i] >= stop_price:
                exit_price = stop_price
                stopped = True

            if stopped:
                bar_pnl = pos_dir * (exit_price - entry_price) * pos_size * point_value
                bar_pnl -= commission_per_contract * pos_size
                trades.append(
                    TradeRecord(
                        entry_time=timestamps[entry_idx],
                        exit_time=timestamps[i],
                        direction=pos_dir,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=pos_size,
                        pnl=bar_pnl,
                        return_pct=bar_pnl / equity[i - 1] * 100,
                    )
                )
                pos_dir = 0
                pos_size = 0.0

        # --- Check for new entry signal (only if flat) ---
        if pos_dir == 0 and sigs[i] != 0:
            sig = sigs[i]
            stop_val = stops[i]
            if not np.isnan(stop_val):
                # Size based on risk
                risk_amount = equity[i - 1] * (risk_pct / 100.0)
                stop_dist = abs(closes[i] - stop_val)
                if stop_dist > 0:
                    raw_size = risk_amount / (stop_dist * point_value)
                    contracts = max(int(raw_size), 1)

                    # Entry at next-bar open approximation: use this bar's close
                    entry_price = closes[i]
                    stop_price = stop_val
                    pos_dir = sig
                    pos_size = contracts
                    entry_idx = i
                    bar_pnl -= commission_per_contract * contracts  # entry commission

        # --- Mark-to-market for open position ---
        mtm = 0.0
        if pos_dir != 0:
            mtm = pos_dir * (closes[i] - entry_price) * pos_size * point_value

        equity[i] = equity[i - 1] + bar_pnl + (mtm - (
            pos_dir * (closes[i - 1] - entry_price) * pos_size * point_value
            if pos_dir != 0 and i > entry_idx else 0.0
        ))

    # --- Close any remaining position at last close ---
    if pos_dir != 0:
        exit_price = closes[-1]
        final_pnl = pos_dir * (exit_price - entry_price) * pos_size * point_value
        final_pnl -= commission_per_contract * pos_size
        trades.append(
            TradeRecord(
                entry_time=timestamps[entry_idx],
                exit_time=timestamps[-1],
                direction=pos_dir,
                entry_price=entry_price,
                exit_price=exit_price,
                size=pos_size,
                pnl=final_pnl,
                return_pct=final_pnl / equity[-2] * 100 if equity[-2] != 0 else 0.0,
            )
        )

    # Build output series
    equity_series = pd.Series(equity, index=timestamps, name="equity")

    # Drawdown
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max * 100
    drawdown.name = "drawdown_pct"

    # Rebuild equity from trade PnL for accuracy
    # (the bar-level equity above is mark-to-market; recalculate closed-trade equity)
    closed_equity = np.full(n, initial_capital, dtype=np.float64)
    trade_pnl_by_exit = {}
    for t in trades:
        idx = signals.index.get_loc(t.exit_time)
        trade_pnl_by_exit[idx] = trade_pnl_by_exit.get(idx, 0.0) + t.pnl
    cumulative = initial_capital
    for i in range(n):
        if i in trade_pnl_by_exit:
            cumulative += trade_pnl_by_exit[i]
        closed_equity[i] = cumulative

    equity_closed = pd.Series(closed_equity, index=timestamps, name="equity_closed")

    return BacktestResult(
        equity_curve=equity_series,
        drawdown_series=drawdown,
        trades=trades,
        signals_df=signals,
    )
