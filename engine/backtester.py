"""
Backtesting engine for futures strategies.

Simulates trade execution from a signal DataFrame produced by a strategy
function.  Models next-bar-open fills, gap-aware stop-loss and take-profit,
per-side commissions, slippage, and proper position sizing.

Supports trailing stops, partial take-profit exits, equity guards, and
multiple concurrent positions with entry spacing constraints.
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
                            # "session_close", "final_bar", "time_exit"
    mae: float = 0.0       # max adverse excursion (points, always >= 0)
    mfe: float = 0.0       # max favourable excursion (points, always >= 0)
    mae_mfe_ratio: float = 0.0  # MAE / MFE (lower = cleaner trade)
    stop_distance: float = 0.0  # initial stop distance in points
    strategy_name: str = ""     # source strategy (for portfolio tracking)


@dataclass
class _Position:
    """Internal state for an active position."""
    direction: int
    size: int
    entry_price: float
    stop_price: float
    tp_price: float
    entry_idx: int
    partial_filled: bool = False
    trail_stop: float = float('nan')
    original_size: int = 0
    mae: float = 0.0
    mfe: float = 0.0
    stop_distance: float = 0.0
    prev_mtm: float = 0.0
    strategy_name: str = ""


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
    daily_trade_counts: dict = field(default_factory=dict)
    max_concurrent_positions: int = 0
    risk_skipped_count: int = 0


def run(
    signals: pd.DataFrame,
    initial_capital: float = 100_000.0,
    risk_per_trade: float = 0.005,
    point_value: float = 20.0,
    commission_per_side: float = 2.0,
    slippage_points: float = 0.0,
    spread_points: float = 0.25,
    stop_atr_multiple: float = 1.0,
    tp_atr_multiple: float = 0.0,
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
    max_daily_risk: float = 0.0,
    max_concurrent_trades: int = 2,
    min_bars_between_entries: int = 2,
    consec_loss_threshold: int = 2,
    loss_scale_down: float = 0.5,
    max_risk_per_trade: float = 0.006,
    strategy_name: str = "",
    **kwargs,
) -> BacktestResult:
    """Execute a backtest over a signal DataFrame.

    Execution model
    ---------------
    - Signals fire on bar *i*; entry fills at bar *i+execution_delay_bars* open.
    - Stops and take-profits are computed by the backtester at fill time
      using ATR and configurable multiples (stop_atr_multiple, tp_atr_multiple).
    - Stop-loss is checked first each bar (priority over take-profit).
    - Take-profit is checked only if the stop was not hit.
    - Stop fills at the worse of target price or bar open (gap model).
    - TP fills at the better of target price or bar open (limit order, no slippage).
    - Commission is charged per side (entry and exit separately).
    - Bid-ask spread (half-spread) is applied adversely on every fill.
    - Market order fills (entries, stops, session exits) pay slippage + half-spread.
    - Limit order fills (TP) pay half-spread only (slightly optimistic).

    Multi-position constraints
    --------------------------
    - max_concurrent_trades: max open positions at once (default 2).
    - min_bars_between_entries: min bars between consecutive fills (default 2).
    - When a pending signal conflicts with an existing pending of lower
      priority (higher signal_tier), the higher-priority signal wins.

    Parameters
    ----------
    signals : pd.DataFrame
        Must contain columns: open, high, low, close, signal, atr.
        Optionally contains: session_close, signal_tier, size_factor.
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
        applied adversely on every fill.
    stop_atr_multiple : float
        Stop distance as multiple of ATR at fill time (default 1.0).
        For longs: stop = fill_price - ATR * multiple.
    tp_atr_multiple : float
        Take-profit distance as multiple of ATR at fill time (default 0.0).
        0.0 disables TP (exit via session close or time exit).
        For longs: tp = fill_price + ATR * multiple.
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
    max_dd_limit : float
        Max total drawdown from equity peak (default 0.10 = 10%).
    use_volatility_sizing : bool
        If True and ATR is available, size positions using ATR-based
        volatility normalisation.
    execution_delay_bars : int
        Bars between signal and fill (default 1 = next-bar open).
    max_loss_per_trade : float
        Hard USD cap on any single trade loss (default 0 = disabled).
    vol_stop_tighten : bool
        Enable volatility-adjusted stop tightening (default False).
    vol_stop_tighten_threshold : float
        ATR ratio above which stops are tightened (default 1.5).
    vol_stop_tighten_factor : float
        Multiplier for stop distance when vol is elevated (default 0.75).
    max_bars_in_trade : int
        Force-exit after this many bars in a trade (default 0 = disabled).
    max_daily_risk : float
        Max cumulative risk as fraction of day-start equity (default 0).
    max_concurrent_trades : int
        Maximum positions open simultaneously (default 2).
    min_bars_between_entries : int
        Minimum bars between consecutive entry fills (default 2).
    consec_loss_threshold : int
        After this many consecutive losses, scale position size down
        (default 2).
    loss_scale_down : float
        Multiplier applied to risk_per_trade after consec_loss_threshold
        consecutive losses (default 0.5 = halve size).  Resets to 1.0
        after the next winning trade.

    Returns
    -------
    BacktestResult
    """
    closes = signals["close"].values
    highs = signals["high"].values
    lows = signals["low"].values
    opens = signals["open"].values
    sigs = signals["signal"].values.astype(np.int8)

    # ATR is required for stop/TP computation
    if "atr" not in signals.columns:
        raise ValueError("signals DataFrame must contain an 'atr' column for stop/TP computation")
    atrs = signals["atr"].values
    has_atr = True

    # Optional columns
    session_closes = signals["session_close"].values if "session_close" in signals.columns else np.zeros(len(closes), dtype=bool)
    size_factors = signals["size_factor"].values if "size_factor" in signals.columns else np.ones(len(closes), dtype=np.float64)
    signal_tiers = signals["signal_tier"].values if "signal_tier" in signals.columns else np.zeros(len(closes), dtype=int)

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

    # Execution costs: market orders pay slippage + half-spread,
    # limit orders (TP) pay half-spread only.
    half_spread = spread_points / 2.0
    market_cost = slippage_points + half_spread
    limit_cost = half_spread

    equity_mtm = np.empty(n, dtype=np.float64)
    equity_closed = np.empty(n, dtype=np.float64)
    equity_mtm[0] = initial_capital
    equity_closed[0] = initial_capital

    # Multi-position state
    positions: list[_Position] = []
    last_entry_bar = -999

    # Pending entry: signal queued with a target fill bar
    pending_signal = 0
    pending_fill_bar = -1
    pending_size_factor = 1.0
    pending_tier = 999  # lower = higher priority

    closed_equity = initial_capital

    # Prop firm risk control state
    halted = False
    halt_bar_index = -1
    daily_halted = False
    equity_peak = initial_capital
    day_start_equity = initial_capital
    day_realized_loss = 0.0
    current_date = timestamps[0].date()

    # Dynamic sizing: consecutive loss tracking
    consec_losses = 0
    size_scale = 1.0  # multiplied into risk_per_trade

    # Tracking
    daily_trade_counts: dict = {}
    max_concurrent_seen = 0
    risk_skipped_count = 0

    trades: list[TradeRecord] = []

    def _clear_pending():
        nonlocal pending_signal
        nonlocal pending_fill_bar, pending_size_factor, pending_tier
        pending_signal = 0
        pending_fill_bar = -1
        pending_size_factor = 1.0
        pending_tier = 999

    def _record_exit(pos, exit_bar, fill_px, size, reason):
        """Record a trade exit and update equity."""
        nonlocal bar_cash_flow, closed_equity, day_realized_loss
        nonlocal consec_losses, size_scale
        gross_pnl = pos.direction * (fill_px - pos.entry_price) * size * point_value
        exit_comm = commission_per_side * size
        net_pnl = gross_pnl - exit_comm
        # When a position closes we must REMOVE the previously-marked
        # unrealized P&L (mtm_share) from equity_mtm and ADD the now-realized
        # net_pnl. The previous code added mtm_share with the wrong sign,
        # which inflated equity_mtm by 2 * prev_mtm on every exit.
        mtm_share = pos.prev_mtm * (size / pos.size) if pos.size > 0 else 0.0
        bar_cash_flow += net_pnl - mtm_share
        closed_equity += net_pnl
        if net_pnl < 0:
            day_realized_loss += abs(net_pnl)

        # Dynamic sizing: track consecutive losses
        is_full_exit = (reason != "partial_tp")
        if is_full_exit:
            if net_pnl < 0:
                consec_losses += 1
                if consec_losses >= consec_loss_threshold:
                    size_scale = loss_scale_down
            else:
                consec_losses = 0
                size_scale = 1.0

        mae_mfe_ratio = pos.mae / pos.mfe if pos.mfe > 0 else 0.0
        trades.append(TradeRecord(
            entry_time=timestamps[pos.entry_idx],
            exit_time=timestamps[exit_bar],
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=fill_px,
            size=size,
            pnl=net_pnl,
            return_pct=net_pnl / equity_mtm[exit_bar - 1] * 100 if equity_mtm[exit_bar - 1] != 0 else 0.0,
            exit_reason=reason,
            mae=pos.mae,
            mfe=pos.mfe,
            mae_mfe_ratio=round(mae_mfe_ratio, 4),
            stop_distance=pos.stop_distance,
            strategy_name=pos.strategy_name,
        ))
        return net_pnl

    for i in range(1, n):
        bar_cash_flow = 0.0

        # ── Prop firm risk controls ──
        bar_date = timestamps[i].date()
        if bar_date != current_date:
            current_date = bar_date
            daily_halted = False
            day_start_equity = equity_mtm[i - 1]
            day_realized_loss = 0.0

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

        # ── 1. Update MAE/MFE for all open positions ──
        for pos in positions:
            if pos.direction == 1:
                adverse = pos.entry_price - lows[i]
                favorable = highs[i] - pos.entry_price
            else:
                adverse = highs[i] - pos.entry_price
                favorable = pos.entry_price - lows[i]
            if adverse > pos.mae:
                pos.mae = adverse
            if favorable > pos.mfe:
                pos.mfe = favorable

        # ── 2. Check exits for all open positions ──
        to_remove = []
        for pidx, pos in enumerate(positions):
            exited = False
            fill_price = np.nan
            exit_reason = ""

            # Max loss per trade cap
            if not exited and max_loss_per_trade > 0:
                unrealised = pos.direction * (closes[i] - pos.entry_price) * pos.size * point_value
                if unrealised <= -max_loss_per_trade:
                    fill_price = closes[i]
                    if pos.direction == 1:
                        fill_price -= market_cost
                    else:
                        fill_price += market_cost
                    exited = True
                    exit_reason = "max_loss_cap"

            # Stop-loss (priority)
            if not exited and pos.direction == 1 and lows[i] <= pos.stop_price:
                fill_price = min(pos.stop_price, opens[i]) - market_cost
                exited = True
                exit_reason = "stop"
            elif not exited and pos.direction == -1 and highs[i] >= pos.stop_price:
                fill_price = max(pos.stop_price, opens[i]) + market_cost
                exited = True
                exit_reason = "stop"

            # Trailing stop (when partial TP has fired)
            if not exited and pos.partial_filled and not np.isnan(pos.trail_stop):
                if pos.direction == 1:
                    new_trail = highs[i] - atrs[i] * trail_atr_multiple
                    if new_trail > pos.trail_stop:
                        pos.trail_stop = new_trail
                    if lows[i] <= pos.trail_stop:
                        fill_price = min(pos.trail_stop, opens[i]) - market_cost
                        exited = True
                        exit_reason = "trailing_stop"
                elif pos.direction == -1:
                    new_trail = lows[i] + atrs[i] * trail_atr_multiple
                    if new_trail < pos.trail_stop:
                        pos.trail_stop = new_trail
                    if highs[i] >= pos.trail_stop:
                        fill_price = max(pos.trail_stop, opens[i]) + market_cost
                        exited = True
                        exit_reason = "trailing_stop"

            # Take-profit / partial TP
            if not exited and not np.isnan(pos.tp_price):
                if pos.direction == 1 and highs[i] >= pos.tp_price:
                    fill_price = max(pos.tp_price, opens[i]) - limit_cost

                    if use_trailing_stop and not pos.partial_filled and pos.size > 1:
                        partial_size = math.floor(pos.size * partial_tp_pct)
                        if partial_size < 1:
                            partial_size = 1
                        _record_exit(pos, i, fill_price, partial_size, "partial_tp")
                        remaining = pos.size - partial_size
                        pos.prev_mtm = pos.direction * (closes[i] - pos.entry_price) * remaining * point_value
                        pos.size = remaining
                        pos.partial_filled = True
                        if move_stop_to_be:
                            pos.stop_price = pos.entry_price
                        if has_atr and not np.isnan(atrs[i]):
                            pos.trail_stop = highs[i] - atrs[i] * trail_atr_multiple
                        pos.tp_price = np.nan
                    else:
                        exited = True
                        exit_reason = "target"

                elif pos.direction == -1 and lows[i] <= pos.tp_price:
                    fill_price = min(pos.tp_price, opens[i]) + limit_cost

                    if use_trailing_stop and not pos.partial_filled and pos.size > 1:
                        partial_size = math.floor(pos.size * partial_tp_pct)
                        if partial_size < 1:
                            partial_size = 1
                        _record_exit(pos, i, fill_price, partial_size, "partial_tp")
                        remaining = pos.size - partial_size
                        pos.prev_mtm = pos.direction * (closes[i] - pos.entry_price) * remaining * point_value
                        pos.size = remaining
                        pos.partial_filled = True
                        if move_stop_to_be:
                            pos.stop_price = pos.entry_price
                        if has_atr and not np.isnan(atrs[i]):
                            pos.trail_stop = lows[i] + atrs[i] * trail_atr_multiple
                        pos.tp_price = np.nan
                    else:
                        exited = True
                        exit_reason = "target"

            # Time-based exit (holding period)
            if not exited and max_bars_in_trade > 0:
                bars_held = i - pos.entry_idx
                if bars_held >= max_bars_in_trade:
                    fill_price = closes[i]
                    if pos.direction == 1:
                        fill_price -= market_cost
                    else:
                        fill_price += market_cost
                    exited = True
                    exit_reason = "time_exit"

            # Session close exit
            if not exited and session_closes[i]:
                fill_price = closes[i]
                if pos.direction == 1:
                    fill_price -= market_cost
                else:
                    fill_price += market_cost
                exited = True
                exit_reason = "session_close"

            if exited:
                _record_exit(pos, i, fill_price, pos.size, exit_reason)
                to_remove.append(pidx)

        # Remove exited positions (reverse to preserve indices)
        for pidx in reversed(to_remove):
            positions.pop(pidx)

        # ── 3. Fill pending entry ──
        if (pending_signal != 0 and i >= pending_fill_bar
                and not halted and not daily_halted):

            # Concurrent capacity check
            if len(positions) >= max_concurrent_trades:
                _clear_pending()

            # Entry spacing check
            elif i - last_entry_bar < min_bars_between_entries:
                _clear_pending()

            else:
                sig = pending_signal
                sfactor = pending_size_factor

                fill_price = opens[i] + market_cost if sig == 1 else opens[i] - market_cost

                # Compute stop and TP from ATR at fill time.
                # Use atrs[i-1] (prior bar's ATR) — atrs[i] would include
                # bar i's high/low/close, which a real-time trader does not
                # know at the open of bar i. Using atrs[i] is lookahead.
                atr_at_fill = atrs[i - 1]
                if not np.isnan(atr_at_fill) and atr_at_fill > 0 and stop_atr_multiple > 0:
                    if sig == 1:
                        stop_val = fill_price - atr_at_fill * stop_atr_multiple
                    else:
                        stop_val = fill_price + atr_at_fill * stop_atr_multiple
                else:
                    stop_val = np.nan

                if tp_atr_multiple > 0 and not np.isnan(atr_at_fill) and atr_at_fill > 0:
                    if sig == 1:
                        tp_val = fill_price + atr_at_fill * tp_atr_multiple
                    else:
                        tp_val = fill_price - atr_at_fill * tp_atr_multiple
                else:
                    tp_val = np.nan

                # Max daily risk gate
                skip_entry = False
                if max_daily_risk > 0 and day_start_equity > 0:
                    trade_worst_case = equity_mtm[i - 1] * risk_per_trade * sfactor
                    total_if_loss = day_realized_loss + trade_worst_case
                    daily_risk_cap = day_start_equity * max_daily_risk
                    if total_if_loss > daily_risk_cap:
                        log.debug(
                            "Bar %d: daily risk gate — realized=$%.0f + new=$%.0f "
                            "> cap=$%.0f, skipping entry",
                            i, day_realized_loss, trade_worst_case, daily_risk_cap,
                        )
                        skip_entry = True

                if not skip_entry:
                    risk_amount = equity_mtm[i - 1] * risk_per_trade * sfactor * size_scale
                    contracts = 0

                    # Volatility-normalised sizing.
                    # Use atr_at_fill (= atrs[i-1]) — same lookahead reason
                    # as the stop/TP computation above.
                    if use_volatility_sizing and has_atr and not np.isnan(atr_at_fill) and atr_at_fill > 0:
                        risk_per_contract = atr_at_fill * point_value
                        contracts = max(1, math.floor(risk_amount / risk_per_contract))
                    else:
                        # Fallback: stop-distance sizing
                        stop_dist = abs(fill_price - stop_val)
                        if stop_dist > 0:
                            contracts = max(1, math.floor(risk_amount / (stop_dist * point_value)))

                    # Contract-level risk constraint: skip if 1-contract
                    # minimum would exceed max allowed risk per trade.
                    if contracts >= 1 and max_risk_per_trade > 0:
                        stop_dist_actual = abs(fill_price - stop_val)
                        actual_risk = stop_dist_actual * contracts * point_value
                        max_allowed = equity_mtm[i - 1] * max_risk_per_trade
                        if actual_risk > max_allowed:
                            log.debug(
                                "Bar %d: risk gate — actual=$%.0f > max=$%.0f "
                                "(stop_dist=%.1f, ATR=%.1f), skipping",
                                i, actual_risk, max_allowed,
                                stop_dist_actual, atr_at_fill if has_atr else 0,
                            )
                            contracts = 0
                            risk_skipped_count += 1

                    if contracts >= 1:
                        entry_commission = commission_per_side * contracts
                        bar_cash_flow -= entry_commission

                        entry_px = fill_price

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

                        pos = _Position(
                            direction=sig,
                            size=contracts,
                            entry_price=entry_px,
                            stop_price=stop_val,
                            tp_price=tp_val,
                            entry_idx=i,
                            original_size=contracts,
                            stop_distance=abs(entry_px - stop_val),
                            strategy_name=strategy_name,
                        )
                        positions.append(pos)
                        last_entry_bar = i
                        daily_trade_counts[bar_date] = daily_trade_counts.get(bar_date, 0) + 1

                _clear_pending()

        # ── 4. Register new signal (tier priority) ──
        if (sigs[i] != 0 and not halted and not daily_halted):
            effective_count = len(positions) + (1 if pending_signal != 0 else 0)
            if effective_count < max_concurrent_trades:
                fill_bar = i + execution_delay_bars
                if fill_bar < n:
                    new_tier = int(signal_tiers[i]) if signal_tiers[i] > 0 else 999
                    if pending_signal != 0:
                        # Replace only if new signal has higher priority (lower tier)
                        if new_tier < pending_tier:
                            pending_signal = sigs[i]
                            pending_fill_bar = fill_bar
                            pending_size_factor = size_factors[i]
                            pending_tier = new_tier
                    else:
                        pending_signal = sigs[i]
                        pending_fill_bar = fill_bar
                        pending_size_factor = size_factors[i]
                        pending_tier = new_tier

        # ── 5. Mark-to-market ──
        current_total_mtm = 0.0
        prev_total_mtm = 0.0
        for pos in positions:
            current_total_mtm += pos.direction * (closes[i] - pos.entry_price) * pos.size * point_value
            prev_total_mtm += pos.prev_mtm

        equity_mtm[i] = equity_mtm[i - 1] + bar_cash_flow + (current_total_mtm - prev_total_mtm)
        equity_closed[i] = closed_equity

        # Update per-position prev_mtm
        for pos in positions:
            pos.prev_mtm = pos.direction * (closes[i] - pos.entry_price) * pos.size * point_value

        # Track peak concurrent positions
        if len(positions) > max_concurrent_seen:
            max_concurrent_seen = len(positions)

    # ── Close any remaining positions at last bar's close ──
    for pos in positions:
        exit_price = closes[-1]
        if pos.direction == 1:
            exit_price -= market_cost
        else:
            exit_price += market_cost

        gross_pnl = pos.direction * (exit_price - pos.entry_price) * pos.size * point_value
        exit_comm = commission_per_side * pos.size
        net_pnl = gross_pnl - exit_comm
        closed_equity += net_pnl

        mae_mfe_ratio = pos.mae / pos.mfe if pos.mfe > 0 else 0.0
        trades.append(TradeRecord(
            entry_time=timestamps[pos.entry_idx],
            exit_time=timestamps[-1],
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            pnl=net_pnl,
            return_pct=net_pnl / equity_mtm[-2] * 100 if equity_mtm[-2] != 0 else 0.0,
            exit_reason="final_bar",
            mae=pos.mae,
            mfe=pos.mfe,
            mae_mfe_ratio=round(mae_mfe_ratio, 4),
            stop_distance=pos.stop_distance,
            strategy_name=pos.strategy_name,
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
        daily_trade_counts=daily_trade_counts,
        max_concurrent_positions=max_concurrent_seen,
        risk_skipped_count=risk_skipped_count,
    )
