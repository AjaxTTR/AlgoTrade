"""
Backtesting engine for futures strategies.

Simulates trade execution from a signal DataFrame produced by a strategy
function.  Models next-bar-open fills, gap-aware stop-loss and take-profit,
per-side commissions, slippage, and proper position sizing.
"""

import math
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
    size: int               # number of contracts
    pnl: float
    return_pct: float
    exit_reason: str        # "stop", "target", "final_bar"


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    equity_mtm: pd.Series           # mark-to-market equity (bar-level)
    equity_closed: pd.Series        # closed-trade equity (steps on exits)
    drawdown_series: pd.Series      # drawdown % from MTM equity peak
    trades: list = field(default_factory=list)
    signals_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def run(
    signals: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_per_trade: float = 0.005,
    point_value: float = 20.0,
    commission_per_side: float = 2.0,
    slippage_points: float = 0.0,
) -> BacktestResult:
    """Execute a backtest over a signal DataFrame.

    Execution model
    ---------------
    - Signals fire on bar *i*; entry fills at bar *i+1* open.
    - Stop-loss is checked first each bar (priority over take-profit).
    - Take-profit is checked only if the stop was not hit.
    - Both stop and TP fill at the worse of target price or bar open (gap model).
    - Commission is charged per side (entry and exit separately).
    - Slippage is applied against the trade on both entry and exit.

    Parameters
    ----------
    signals : pd.DataFrame
        Must contain columns: open, high, low, close, signal, stop_price.
        Optionally contains: tp_price (take-profit level; NaN to disable).
        ``signal``: 1 (long), -1 (short), 0 (flat).
        ``stop_price``: initial stop for the trade (NaN when no signal).
        ``tp_price``: take-profit target (NaN to skip TP logic).
    initial_capital : float
        Starting account equity in USD.
    risk_per_trade : float
        Fraction of equity risked per trade in decimal form
        (e.g. 0.005 = 0.5%).
    point_value : float
        Dollar value per point per contract (NQ = $20).
    commission_per_side : float
        Commission per contract per side (entry or exit).
    slippage_points : float
        Points of slippage applied adversely on each fill.

    Returns
    -------
    BacktestResult
        Mark-to-market equity, closed-trade equity, drawdown series,
        trade records, and the signal DataFrame.
    """
    closes = signals["close"].values
    highs = signals["high"].values
    lows = signals["low"].values
    opens = signals["open"].values
    sigs = signals["signal"].values.astype(np.int8)
    stops = signals["stop_price"].values

    # tp_price is optional — fill with NaN if column is absent
    has_tp = "tp_price" in signals.columns
    tps = signals["tp_price"].values if has_tp else np.full(len(closes), np.nan)

    timestamps = signals.index

    n = len(closes)
    equity_mtm = np.empty(n, dtype=np.float64)
    equity_closed = np.empty(n, dtype=np.float64)
    equity_mtm[0] = initial_capital
    equity_closed[0] = initial_capital

    # Position state
    pos_dir = 0             # 1 long, -1 short, 0 flat
    pos_size = 0            # contracts (int)
    entry_price = 0.0
    stop_price = np.nan
    tp_price = np.nan
    entry_idx = 0

    # Pending entry from previous bar's signal
    pending_signal = 0      # direction of pending entry
    pending_stop = np.nan   # stop price for pending entry
    pending_tp = np.nan     # take-profit price for pending entry

    closed_equity = initial_capital  # running closed-trade equity
    prev_mtm = 0.0                  # previous bar's open-position MTM value

    trades: list[TradeRecord] = []

    for i in range(1, n):
        bar_cash_flow = 0.0   # realised cash changes this bar

        # ── 1. Check exits on open position ──
        if pos_dir != 0:
            exited = False
            fill_price = np.nan
            exit_reason = ""

            # Stop-loss checked FIRST (priority over take-profit)
            if pos_dir == 1 and lows[i] <= stop_price:
                # Long stop hit — fill at worse of stop or open (gap down)
                fill_price = min(stop_price, opens[i])
                fill_price -= slippage_points
                exited = True
                exit_reason = "stop"
            elif pos_dir == -1 and highs[i] >= stop_price:
                # Short stop hit — fill at worse of stop or open (gap up)
                fill_price = max(stop_price, opens[i])
                fill_price += slippage_points
                exited = True
                exit_reason = "stop"

            # Take-profit checked only if stop was not hit
            if not exited and not np.isnan(tp_price):
                if pos_dir == 1 and highs[i] >= tp_price:
                    # Long TP hit — fill at better of TP or open (gap up)
                    fill_price = max(tp_price, opens[i])
                    fill_price -= slippage_points
                    exited = True
                    exit_reason = "target"
                elif pos_dir == -1 and lows[i] <= tp_price:
                    # Short TP hit — fill at better of TP or open (gap down)
                    fill_price = min(tp_price, opens[i])
                    fill_price += slippage_points
                    exited = True
                    exit_reason = "target"

            if exited:
                gross_pnl = pos_dir * (fill_price - entry_price) * pos_size * point_value
                exit_commission = commission_per_side * pos_size
                net_pnl = gross_pnl - exit_commission

                bar_cash_flow += net_pnl + prev_mtm  # unwind MTM, book realised
                closed_equity += net_pnl
                prev_mtm = 0.0

                trades.append(TradeRecord(
                    entry_time=timestamps[entry_idx],
                    exit_time=timestamps[i],
                    direction=pos_dir,
                    entry_price=entry_price,
                    exit_price=fill_price,
                    size=pos_size,
                    pnl=net_pnl,
                    return_pct=net_pnl / equity_mtm[i - 1] * 100 if equity_mtm[i - 1] != 0 else 0.0,
                    exit_reason=exit_reason,
                ))
                pos_dir = 0
                pos_size = 0

        # ── 2. Fill pending entry at this bar's open ──
        if pos_dir == 0 and pending_signal != 0:
            sig = pending_signal
            stop_val = pending_stop
            tp_val = pending_tp

            # Apply slippage adversely to entry
            fill_price = opens[i] + slippage_points if sig == 1 else opens[i] - slippage_points

            # Size: risk_amount / (stop_distance * point_value), floored
            stop_dist = abs(fill_price - stop_val)
            if stop_dist > 0:
                risk_amount = equity_mtm[i - 1] * risk_per_trade
                contracts = math.floor(risk_amount / (stop_dist * point_value))

                if contracts >= 1:
                    entry_commission = commission_per_side * contracts
                    bar_cash_flow -= entry_commission

                    entry_price = fill_price
                    stop_price = stop_val
                    tp_price = tp_val
                    pos_dir = sig
                    pos_size = contracts
                    entry_idx = i

            pending_signal = 0
            pending_stop = np.nan
            pending_tp = np.nan

        # ── 3. Register new signal for next-bar fill ──
        if pos_dir == 0 and sigs[i] != 0 and not np.isnan(stops[i]):
            pending_signal = sigs[i]
            pending_stop = stops[i]
            pending_tp = tps[i]

        # ── 4. Mark-to-market ──
        current_mtm = 0.0
        if pos_dir != 0:
            current_mtm = pos_dir * (closes[i] - entry_price) * pos_size * point_value

        # Equity = previous equity + realised cash flow + change in MTM
        equity_mtm[i] = equity_mtm[i - 1] + bar_cash_flow + (current_mtm - prev_mtm)
        equity_closed[i] = closed_equity
        prev_mtm = current_mtm

    # ── Close any remaining position at last bar's close ──
    if pos_dir != 0:
        exit_price = closes[-1]
        if pos_dir == 1:
            exit_price -= slippage_points
        else:
            exit_price += slippage_points

        gross_pnl = pos_dir * (exit_price - entry_price) * pos_size * point_value
        exit_commission = commission_per_side * pos_size
        net_pnl = gross_pnl - exit_commission
        closed_equity += net_pnl

        trades.append(TradeRecord(
            entry_time=timestamps[entry_idx],
            exit_time=timestamps[-1],
            direction=pos_dir,
            entry_price=entry_price,
            exit_price=exit_price,
            size=pos_size,
            pnl=net_pnl,
            return_pct=net_pnl / equity_mtm[-2] * 100 if equity_mtm[-2] != 0 else 0.0,
            exit_reason="final_bar",
        ))
        equity_closed[-1] = closed_equity

    # ── Build output series ──
    mtm_series = pd.Series(equity_mtm, index=timestamps, name="equity_mtm")
    closed_series = pd.Series(equity_closed, index=timestamps, name="equity_closed")

    running_max = mtm_series.cummax()
    drawdown = (mtm_series - running_max) / running_max * 100
    drawdown.name = "drawdown_pct"

    return BacktestResult(
        equity_mtm=mtm_series,
        equity_closed=closed_series,
        drawdown_series=drawdown,
        trades=trades,
        signals_df=signals,
    )
