"""
Backtesting engine for futures strategies.

Simulates trade execution from a signal DataFrame produced by a strategy
function.  Models next-bar-open fills, gap-aware stop-loss and take-profit,
per-side commissions, slippage, and proper position sizing.

Supports trailing stops, partial take-profit exits, and equity guards.
"""

import logging
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


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
    exit_reason: str        # "stop", "target", "partial_tp", "trailing_stop",
                            # "session_close", "final_bar"


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    equity_mtm: pd.Series           # mark-to-market equity (bar-level)
    equity_closed: pd.Series        # closed-trade equity (steps on exits)
    drawdown_series: pd.Series      # drawdown % from MTM equity peak
    trades: list = field(default_factory=list)
    signals_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    halted: bool = False
    halt_bar_index: int = -1


def run(
    signals: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_per_trade: float = 0.005,
    point_value: float = 20.0,
    commission_per_side: float = 2.0,
    slippage_points: float = 0.0,
    use_trailing_stop: bool = False,
    trail_atr_multiple: float = 2.0,
    partial_tp_pct: float = 0.5,
    move_stop_to_be: bool = True,
    margin_call_pct: float = 0.0,
    **kwargs,
) -> BacktestResult:
    """Execute a backtest over a signal DataFrame.

    Execution model
    ---------------
    - Signals fire on bar *i*; entry fills at bar *i+1* open.
    - Stop-loss is checked first each bar (priority over take-profit).
    - Take-profit is checked only if the stop was not hit.
    - Stop fills at the worse of target price or bar open (gap model).
    - TP fills at the better of target price or bar open (limit order, no slippage).
    - Commission is charged per side (entry and exit separately).
    - Slippage is applied against the trade on entries and stop/session exits.

    Parameters
    ----------
    signals : pd.DataFrame
        Must contain columns: open, high, low, close, signal, stop_price.
        Optionally contains: tp_price, session_close, atr.
    initial_capital : float
        Starting account equity in USD.
    risk_per_trade : float
        Fraction of equity risked per trade (e.g. 0.005 = 0.5%).
    point_value : float
        Dollar value per point per contract (NQ = $20).
    commission_per_side : float
        Commission per contract per side.
    slippage_points : float
        Points of slippage applied adversely on market order fills.
    use_trailing_stop : bool
        Enable partial TP at target + ATR trailing stop on remainder.
    trail_atr_multiple : float
        ATR multiplier for trailing stop distance (default 2.0).
    partial_tp_pct : float
        Fraction of position to close at TP (default 0.5 = 50%).
    move_stop_to_be : bool
        Move stop to breakeven after partial TP (default True).
    margin_call_pct : float
        Halt new entries if equity drops below this fraction of initial
        capital (0 = disabled).

    Returns
    -------
    BacktestResult
    """
    closes = signals["close"].values
    highs = signals["high"].values
    lows = signals["low"].values
    opens = signals["open"].values
    sigs = signals["signal"].values.astype(np.int8)
    stops = signals["stop_price"].values

    # Optional columns
    tps = signals["tp_price"].values if "tp_price" in signals.columns else np.full(len(closes), np.nan)
    session_closes = signals["session_close"].values if "session_close" in signals.columns else np.zeros(len(closes), dtype=bool)
    has_atr = "atr" in signals.columns
    atrs = signals["atr"].values if has_atr else np.full(len(closes), np.nan)

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

    # Trailing stop / partial TP state
    partial_filled = False
    trail_stop = np.nan
    original_size = 0

    # Pending entry from previous bar's signal
    pending_signal = 0
    pending_stop = np.nan
    pending_tp = np.nan

    closed_equity = initial_capital
    prev_mtm = 0.0

    # Equity guard state
    halted = False
    halt_bar_index = -1

    trades: list[TradeRecord] = []

    def _record_exit(exit_bar, fill_px, size, reason):
        """Helper to record a trade exit and update equity."""
        nonlocal bar_cash_flow, closed_equity, prev_mtm
        gross_pnl = pos_dir * (fill_px - entry_price) * size * point_value
        exit_comm = commission_per_side * size
        net_pnl = gross_pnl - exit_comm
        # Proportion of MTM attributable to this leg
        mtm_share = prev_mtm * (size / pos_size) if pos_size > 0 else 0.0
        bar_cash_flow += net_pnl + mtm_share
        closed_equity += net_pnl
        trades.append(TradeRecord(
            entry_time=timestamps[entry_idx],
            exit_time=timestamps[exit_bar],
            direction=pos_dir,
            entry_price=entry_price,
            exit_price=fill_px,
            size=size,
            pnl=net_pnl,
            return_pct=net_pnl / equity_mtm[exit_bar - 1] * 100 if equity_mtm[exit_bar - 1] != 0 else 0.0,
            exit_reason=reason,
        ))
        return net_pnl

    for i in range(1, n):
        bar_cash_flow = 0.0

        # ── Equity guard check ──
        if not halted and margin_call_pct > 0:
            if equity_mtm[i - 1] <= initial_capital * margin_call_pct:
                halted = True
                halt_bar_index = i
                log.warning(
                    "Equity guard triggered at bar %d: equity $%.2f below threshold $%.2f",
                    i, equity_mtm[i - 1], initial_capital * margin_call_pct,
                )

        # ── 1. Check exits on open position ──
        if pos_dir != 0:
            exited = False
            fill_price = np.nan
            exit_reason = ""

            # Stop-loss checked FIRST (priority)
            if pos_dir == 1 and lows[i] <= stop_price:
                fill_price = min(stop_price, opens[i])
                fill_price -= slippage_points
                exited = True
                exit_reason = "stop"
            elif pos_dir == -1 and highs[i] >= stop_price:
                fill_price = max(stop_price, opens[i])
                fill_price += slippage_points
                exited = True
                exit_reason = "stop"

            # Trailing stop checked second (only when trailing is active)
            if not exited and partial_filled and not np.isnan(trail_stop):
                if pos_dir == 1:
                    # Ratchet trail stop up
                    new_trail = highs[i] - atrs[i] * trail_atr_multiple
                    if new_trail > trail_stop:
                        trail_stop = new_trail
                    if lows[i] <= trail_stop:
                        fill_price = min(trail_stop, opens[i])
                        fill_price -= slippage_points
                        exited = True
                        exit_reason = "trailing_stop"
                elif pos_dir == -1:
                    # Ratchet trail stop down
                    new_trail = lows[i] + atrs[i] * trail_atr_multiple
                    if new_trail < trail_stop:
                        trail_stop = new_trail
                    if highs[i] >= trail_stop:
                        fill_price = max(trail_stop, opens[i])
                        fill_price += slippage_points
                        exited = True
                        exit_reason = "trailing_stop"

            # Take-profit / partial TP
            if not exited and not np.isnan(tp_price):
                if pos_dir == 1 and highs[i] >= tp_price:
                    fill_price = max(tp_price, opens[i])  # limit order, no slippage

                    if use_trailing_stop and not partial_filled and pos_size > 1:
                        # Partial exit: close partial_tp_pct of position
                        partial_size = math.floor(pos_size * partial_tp_pct)
                        if partial_size < 1:
                            partial_size = 1
                        _record_exit(i, fill_price, partial_size, "partial_tp")
                        # Reduce position, keep remainder
                        remaining = pos_size - partial_size
                        # Update MTM for reduced position
                        prev_mtm = pos_dir * (closes[i] - entry_price) * remaining * point_value
                        pos_size = remaining
                        partial_filled = True
                        # Move stop to breakeven
                        if move_stop_to_be:
                            stop_price = entry_price
                        # Initialize trailing stop
                        if has_atr and not np.isnan(atrs[i]):
                            trail_stop = highs[i] - atrs[i] * trail_atr_multiple
                        tp_price = np.nan  # disable further TP checks
                    else:
                        exited = True
                        exit_reason = "target"

                elif pos_dir == -1 and lows[i] <= tp_price:
                    fill_price = min(tp_price, opens[i])  # limit order, no slippage

                    if use_trailing_stop and not partial_filled and pos_size > 1:
                        partial_size = math.floor(pos_size * partial_tp_pct)
                        if partial_size < 1:
                            partial_size = 1
                        _record_exit(i, fill_price, partial_size, "partial_tp")
                        remaining = pos_size - partial_size
                        prev_mtm = pos_dir * (closes[i] - entry_price) * remaining * point_value
                        pos_size = remaining
                        partial_filled = True
                        if move_stop_to_be:
                            stop_price = entry_price
                        if has_atr and not np.isnan(atrs[i]):
                            trail_stop = lows[i] + atrs[i] * trail_atr_multiple
                        tp_price = np.nan
                    else:
                        exited = True
                        exit_reason = "target"

            # Session close exit — checked last
            if not exited and session_closes[i]:
                fill_price = closes[i]
                if pos_dir == 1:
                    fill_price -= slippage_points
                else:
                    fill_price += slippage_points
                exited = True
                exit_reason = "session_close"

            if exited:
                _record_exit(i, fill_price, pos_size, exit_reason)
                # Reset all position state
                prev_mtm = 0.0
                pos_dir = 0
                pos_size = 0
                partial_filled = False
                trail_stop = np.nan
                original_size = 0

        # ── 2. Fill pending entry at this bar's open ──
        if pos_dir == 0 and pending_signal != 0 and not halted:
            sig = pending_signal
            stop_val = pending_stop
            tp_val = pending_tp

            fill_price = opens[i] + slippage_points if sig == 1 else opens[i] - slippage_points

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
                    original_size = contracts
                    entry_idx = i
                    partial_filled = False
                    trail_stop = np.nan

            pending_signal = 0
            pending_stop = np.nan
            pending_tp = np.nan

        # ── 3. Register new signal for next-bar fill ──
        if pos_dir == 0 and sigs[i] != 0 and not np.isnan(stops[i]) and not halted:
            pending_signal = sigs[i]
            pending_stop = stops[i]
            pending_tp = tps[i]

        # ── 4. Mark-to-market ──
        current_mtm = 0.0
        if pos_dir != 0:
            current_mtm = pos_dir * (closes[i] - entry_price) * pos_size * point_value

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
        halted=halted,
        halt_bar_index=halt_bar_index,
    )
