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
    mae: float = 0.0       # max adverse excursion (points, always >= 0)
    mfe: float = 0.0       # max favourable excursion (points, always >= 0)
    mae_mfe_ratio: float = 0.0  # MAE / MFE (lower = cleaner trade)
    stop_distance: float = 0.0  # initial stop distance in points


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
    spread_points: float = 0.25,
    use_trailing_stop: bool = False,
    trail_atr_multiple: float = 2.0,
    partial_tp_pct: float = 0.5,
    move_stop_to_be: bool = True,
    daily_dd_limit: float = 0.05,
    max_dd_limit: float = 0.10,
    use_volatility_sizing: bool = True,
    execution_delay_bars: int = 1,
    max_loss_per_trade: float = 0.0,
    vol_stop_tighten: bool = False,
    vol_stop_tighten_threshold: float = 1.5,
    vol_stop_tighten_factor: float = 0.75,
    max_bars_in_trade: int = 0,
    **kwargs,
) -> BacktestResult:
    """Execute a backtest over a signal DataFrame.

    Execution model
    ---------------
    - Signals fire on bar *i*; entry fills at bar *i+execution_delay_bars* open.
    - Stop-loss is checked first each bar (priority over take-profit).
    - Take-profit is checked only if the stop was not hit.
    - Stop fills at the worse of target price or bar open (gap model).
    - TP fills at the better of target price or bar open (limit order, no slippage).
    - Commission is charged per side (entry and exit separately).
    - Bid-ask spread (half-spread) is applied adversely on every fill.
    - Market order fills (entries, stops, session exits) pay slippage + half-spread.
    - Limit order fills (TP) pay half-spread only (slightly optimistic).

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
    spread_points : float
        Bid-ask spread in points (default 0.25 for NQ). Half-spread is
        applied adversely on every fill. Combined with slippage on market
        orders; applied alone on limit-order TP fills.
    use_trailing_stop : bool
        Enable partial TP at target + ATR trailing stop on remainder.
    trail_atr_multiple : float
        ATR multiplier for trailing stop distance (default 2.0).
    partial_tp_pct : float
        Fraction of position to close at TP (default 0.5 = 50%).
    move_stop_to_be : bool
        Move stop to breakeven after partial TP (default True).
    daily_dd_limit : float
        Max daily loss as fraction of day's starting equity (default 0.05 = 5%).
        If daily P&L hits this limit, halt trading for the rest of that day.
        Set to 0 to disable.
    max_dd_limit : float
        Max total drawdown from equity peak (default 0.10 = 10%).
        If equity drops this much from peak, halt all trading permanently.
        Set to 0 to disable.
    use_volatility_sizing : bool
        If True and ATR is available, size positions using ATR-based
        volatility normalisation instead of stop distance. Falls back
        to stop-distance sizing when ATR is missing or zero.
    execution_delay_bars : int
        Number of bars between signal generation and order fill
        (default 1 = next-bar open, matching prior behaviour).
        A new signal while one is pending replaces the earlier one.
    max_loss_per_trade : float
        Hard USD cap on any single trade loss (default 0 = disabled).
        If unrealised loss exceeds this on any bar, force-exit at market.
        Prevents fat-tail outliers from extreme gap/momentum events.
    vol_stop_tighten : bool
        Enable volatility-adjusted stop tightening (default False).
        When current ATR exceeds its rolling median by vol_stop_tighten_threshold,
        the initial stop distance is multiplied by vol_stop_tighten_factor.
    vol_stop_tighten_threshold : float
        ATR ratio above which stops are tightened (default 1.5 = 50% above median).
    vol_stop_tighten_factor : float
        Multiplier applied to stop distance when vol is elevated (default 0.75).
    max_bars_in_trade : int
        Force-exit after this many bars in a trade (default 0 = disabled).
        Prevents capital lock-up in stale positions.

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
    size_factors = signals["size_factor"].values if "size_factor" in signals.columns else np.ones(len(closes), dtype=np.float64)

    timestamps = signals.index
    n = len(closes)

    # Precompute rolling ATR median for volatility-adjusted stop tightening
    if vol_stop_tighten and has_atr:
        atr_series = pd.Series(atrs)
        atr_rolling_median = atr_series.rolling(
            window=50, min_periods=20,
        ).median().values
    else:
        atr_rolling_median = None

    # Precompute execution costs: market orders pay slippage + half-spread,
    # limit orders (TP) pay half-spread only.
    half_spread = spread_points / 2.0
    market_cost = slippage_points + half_spread  # entries, stops, session exits
    limit_cost = half_spread                     # TP fills only

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

    # MAE/MFE tracking (points from entry, always >= 0)
    trade_mae = 0.0   # max adverse excursion
    trade_mfe = 0.0   # max favourable excursion
    trade_stop_dist = 0.0  # initial stop distance in points

    # Pending entry: signal queued with a target fill bar
    pending_signal = 0
    pending_stop = np.nan
    pending_tp = np.nan
    pending_fill_bar = -1  # bar index at which to fill
    pending_size_factor = 1.0

    closed_equity = initial_capital
    prev_mtm = 0.0

    # Prop firm risk control state
    halted = False              # permanent halt (max DD breached)
    halt_bar_index = -1
    daily_halted = False        # daily halt (resets each new day)
    equity_peak = initial_capital
    day_start_equity = initial_capital
    current_date = timestamps[0].date()

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
        mae_mfe_ratio = trade_mae / trade_mfe if trade_mfe > 0 else 0.0
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
            mae=trade_mae,
            mfe=trade_mfe,
            mae_mfe_ratio=round(mae_mfe_ratio, 4),
            stop_distance=trade_stop_dist,
        ))
        return net_pnl

    for i in range(1, n):
        bar_cash_flow = 0.0

        # ── Prop firm risk controls ──
        bar_date = timestamps[i].date()
        if bar_date != current_date:
            # New trading day: reset daily halt, update day start equity
            current_date = bar_date
            daily_halted = False
            day_start_equity = equity_mtm[i - 1]

        # Track equity peak
        if equity_mtm[i - 1] > equity_peak:
            equity_peak = equity_mtm[i - 1]

        # Daily drawdown check
        if not daily_halted and daily_dd_limit > 0 and day_start_equity > 0:
            daily_loss = (day_start_equity - equity_mtm[i - 1]) / day_start_equity
            if daily_loss >= daily_dd_limit:
                daily_halted = True
                log.warning(
                    "Daily DD limit hit at bar %d: loss %.2f%% of day start $%.2f",
                    i, daily_loss * 100, day_start_equity,
                )

        # Max drawdown check (permanent halt)
        if not halted and max_dd_limit > 0 and equity_peak > 0:
            total_dd = (equity_peak - equity_mtm[i - 1]) / equity_peak
            if total_dd >= max_dd_limit:
                halted = True
                halt_bar_index = i
                log.warning(
                    "Max DD limit hit at bar %d: equity $%.2f, peak $%.2f, DD %.2f%%",
                    i, equity_mtm[i - 1], equity_peak, total_dd * 100,
                )

        # ── 1. Update MAE/MFE from this bar's extremes ──
        if pos_dir != 0:
            if pos_dir == 1:
                adverse = entry_price - lows[i]    # how far price moved against long
                favorable = highs[i] - entry_price  # how far price moved for long
            else:
                adverse = highs[i] - entry_price   # how far price moved against short
                favorable = entry_price - lows[i]   # how far price moved for short
            if adverse > trade_mae:
                trade_mae = adverse
            if favorable > trade_mfe:
                trade_mfe = favorable

        # ── Check exits on open position ──
        if pos_dir != 0:
            exited = False
            fill_price = np.nan
            exit_reason = ""

            # Max loss per trade cap — hard override, checked first
            if not exited and max_loss_per_trade > 0:
                unrealised = pos_dir * (closes[i] - entry_price) * pos_size * point_value
                if unrealised <= -max_loss_per_trade:
                    fill_price = closes[i]
                    if pos_dir == 1:
                        fill_price -= market_cost
                    else:
                        fill_price += market_cost
                    exited = True
                    exit_reason = "max_loss_cap"

            # Stop-loss checked FIRST (priority)
            if not exited and pos_dir == 1 and lows[i] <= stop_price:
                fill_price = min(stop_price, opens[i])
                fill_price -= market_cost
                exited = True
                exit_reason = "stop"
            elif not exited and pos_dir == -1 and highs[i] >= stop_price:
                fill_price = max(stop_price, opens[i])
                fill_price += market_cost
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
                        fill_price -= market_cost
                        exited = True
                        exit_reason = "trailing_stop"
                elif pos_dir == -1:
                    # Ratchet trail stop down
                    new_trail = lows[i] + atrs[i] * trail_atr_multiple
                    if new_trail < trail_stop:
                        trail_stop = new_trail
                    if highs[i] >= trail_stop:
                        fill_price = max(trail_stop, opens[i])
                        fill_price += market_cost
                        exited = True
                        exit_reason = "trailing_stop"

            # Take-profit / partial TP
            if not exited and not np.isnan(tp_price):
                if pos_dir == 1 and highs[i] >= tp_price:
                    fill_price = max(tp_price, opens[i]) - limit_cost

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
                    fill_price = min(tp_price, opens[i]) + limit_cost

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

            # Time-based exit — force close after max bars
            if not exited and max_bars_in_trade > 0:
                bars_held = i - entry_idx
                if bars_held >= max_bars_in_trade:
                    fill_price = closes[i]
                    if pos_dir == 1:
                        fill_price -= market_cost
                    else:
                        fill_price += market_cost
                    exited = True
                    exit_reason = "time_exit"

            # Session close exit — checked last
            if not exited and session_closes[i]:
                fill_price = closes[i]
                if pos_dir == 1:
                    fill_price -= market_cost
                else:
                    fill_price += market_cost
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
                trade_mae = 0.0
                trade_mfe = 0.0
                trade_stop_dist = 0.0

        # ── 2. Fill pending entry ──
        if pos_dir == 0 and pending_signal != 0 and i >= pending_fill_bar and not halted and not daily_halted:
            sig = pending_signal
            stop_val = pending_stop
            tp_val = pending_tp
            sfactor = pending_size_factor

            fill_price = opens[i] + market_cost if sig == 1 else opens[i] - market_cost

            risk_amount = equity_mtm[i - 1] * risk_per_trade * sfactor
            contracts = 0
            sizing_method = ""

            # Volatility-normalised sizing: ATR * point_value per contract
            if use_volatility_sizing and has_atr and not np.isnan(atrs[i]) and atrs[i] > 0:
                risk_per_contract = atrs[i] * point_value
                contracts = math.floor(risk_amount / risk_per_contract)
                sizing_method = "volatility"
            else:
                # Fallback: stop-distance sizing
                stop_dist = abs(fill_price - stop_val)
                if stop_dist > 0:
                    contracts = math.floor(risk_amount / (stop_dist * point_value))
                    sizing_method = "stop_distance"

            if contracts >= 1:
                if sizing_method == "volatility":
                    log.debug(
                        "Bar %d: vol sizing — ATR=%.2f, risk=$%.0f, contracts=%d",
                        i, atrs[i], risk_amount, contracts,
                    )
                else:
                    log.debug(
                        "Bar %d: stop-dist sizing — dist=%.2f, risk=$%.0f, contracts=%d",
                        i, abs(fill_price - stop_val), risk_amount, contracts,
                    )

                entry_commission = commission_per_side * contracts
                bar_cash_flow -= entry_commission

                entry_price = fill_price

                # Volatility-adjusted stop tightening
                if (vol_stop_tighten and atr_rolling_median is not None
                        and not np.isnan(atrs[i]) and atrs[i] > 0
                        and not np.isnan(atr_rolling_median[i])
                        and atr_rolling_median[i] > 0):
                    atr_ratio = atrs[i] / atr_rolling_median[i]
                    if atr_ratio >= vol_stop_tighten_threshold:
                        orig_dist = abs(fill_price - stop_val)
                        tight_dist = orig_dist * vol_stop_tighten_factor
                        if sig == 1:
                            stop_val = fill_price - tight_dist
                        else:
                            stop_val = fill_price + tight_dist
                        log.debug(
                            "Bar %d: vol stop tighten — ATR ratio=%.2f, "
                            "stop dist %.2f -> %.2f",
                            i, atr_ratio, orig_dist, tight_dist,
                        )

                stop_price = stop_val
                tp_price = tp_val
                pos_dir = sig
                pos_size = contracts
                original_size = contracts
                entry_idx = i
                partial_filled = False
                trail_stop = np.nan
                trade_mae = 0.0
                trade_mfe = 0.0
                trade_stop_dist = abs(fill_price - stop_val)

            pending_signal = 0
            pending_stop = np.nan
            pending_tp = np.nan
            pending_fill_bar = -1
            pending_size_factor = 1.0

        # ── 3. Register new signal ──
        if pos_dir == 0 and sigs[i] != 0 and not np.isnan(stops[i]) and not halted and not daily_halted:
            fill_bar = i + execution_delay_bars
            if fill_bar < n:  # enough bars remaining to fill
                pending_signal = sigs[i]
                pending_stop = stops[i]
                pending_tp = tps[i]
                pending_fill_bar = fill_bar
                pending_size_factor = size_factors[i]

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
            exit_price -= market_cost
        else:
            exit_price += market_cost

        gross_pnl = pos_dir * (exit_price - entry_price) * pos_size * point_value
        exit_commission = commission_per_side * pos_size
        net_pnl = gross_pnl - exit_commission
        closed_equity += net_pnl

        mae_mfe_ratio = trade_mae / trade_mfe if trade_mfe > 0 else 0.0
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
            mae=trade_mae,
            mfe=trade_mfe,
            mae_mfe_ratio=round(mae_mfe_ratio, 4),
            stop_distance=trade_stop_dist,
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
