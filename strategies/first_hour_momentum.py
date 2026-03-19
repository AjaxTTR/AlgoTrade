"""
First-Hour Momentum strategy.

Edge: when the first hour (09:30-10:30) shows a strong positive move,
the rest of the day tends to continue upward.

Source: ranked_edges.csv -- FH_Up -> Rest_Up (t=30.38, stability=0.946).
Long only -- the short side showed weaker edge quality.

Trade logic:
  1. Bias filter: only trade on days where first-hour return is in the
     top 20% historically (80th percentile, expanding window).
  2. Initial entry: first bar at/after 10:30 on bias days.
  3. Pullback re-entries: after the initial trade exits (holding period),
     re-enter long on pullback-and-recovery bars.  Pullback = price dips
     from running session high by at least pullback_atr_frac * ATR, then
     the bar closes in its upper half (buying pressure returning).
  4. Max trades per day caps total entries.

Every strategy module in /strategies must expose:

    def generate_signals(df: pd.DataFrame, **params) -> pd.DataFrame

The returned DataFrame must include a ``signal`` column
(1 = long, -1 = short, 0 = flat) and a ``stop_price`` column.
"""

import numpy as np
import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    fh_end: str = "10:30",
    entry_cutoff: str = "15:45",
    fh_threshold: float = 0.0,
    fh_percentile: float = 80.0,
    atr_period: int = 14,
    stop_atr_multiple: float = 1.5,
    tp_atr_multiple: float = 2.0,
    holding_bars: int = 8,
    max_trades_per_day: int = 3,
    pullback_atr_frac: float = 0.5,
    **kwargs,
) -> pd.DataFrame:
    """First-Hour Momentum with pullback re-entries.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with a DatetimeIndex (15-min bars expected).
    session_start / session_end / fh_end / entry_cutoff : str
        Time boundaries for the session and first-hour window.
    fh_threshold : float
        Hard minimum |fh_return| floor (default 0.0, disabled).
    fh_percentile : float
        Percentile of |fh_return| that today's move must exceed
        (default 80.0 = top 20%).  Expanding window, no lookahead.
    atr_period : int
        Lookback for Average True Range (default 14).
    stop_atr_multiple : float
        Stop distance as ATR multiple (default 1.5).
    tp_atr_multiple : float
        Take-profit distance as ATR multiple (default 2.0).
    holding_bars : int
        Bars to hold each trade (default 8 = 2 hours on 15-min bars).
    max_trades_per_day : int
        Maximum entries per bias day (default 3).  The first is the
        initial breakout; additional entries are pullback re-entries.
    pullback_atr_frac : float
        Minimum pullback depth from running session high as a fraction
        of ATR (default 0.5).  The bar must also close in its upper
        half to confirm buying pressure is returning.

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns: signal, stop_price, tp_price,
        session_close, atr, fh_return, size_factor, signal_tier.
    """
    out = df.copy()
    n = len(out)

    # ------------------------------------------------------------------
    # ATR (for stop/TP sizing)
    # ------------------------------------------------------------------
    prev_close = out["close"].shift(1)
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum(
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ),
    )
    out["atr"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

    # ------------------------------------------------------------------
    # Time masks
    # ------------------------------------------------------------------
    time = out.index.time
    t_session_start = pd.Timestamp(session_start).time()
    t_fh_end = pd.Timestamp(fh_end).time()
    t_session_end = pd.Timestamp(session_end).time()
    t_entry_cutoff = pd.Timestamp(entry_cutoff).time()

    out["date"] = out.index.date

    # ------------------------------------------------------------------
    # Session close flag (15:45 EOD backstop)
    # Holding-period exits are added per-signal below.
    # ------------------------------------------------------------------
    past_cutoff = time >= t_entry_cutoff
    new_day = out["date"] != pd.Series(out["date"]).shift(1).values
    prev_past = pd.Series(past_cutoff).shift(1, fill_value=False).values
    out["session_close"] = past_cutoff & (~prev_past | new_day)

    # ------------------------------------------------------------------
    # First-hour return per day (no lookahead)
    # ------------------------------------------------------------------
    in_fh = (time >= t_session_start) & (time < t_fh_end)
    fh_bars = out[in_fh]
    fh_open = fh_bars.groupby("date")["open"].first()
    fh_close = fh_bars.groupby("date")["close"].last()
    fh_return = (fh_close - fh_open) / fh_open
    fh_return.name = "fh_return"
    out["fh_return"] = out["date"].map(fh_return)

    # ------------------------------------------------------------------
    # Adaptive threshold: expanding percentile of |fh_return|
    # ------------------------------------------------------------------
    MIN_HISTORY = 20
    fh_abs = fh_return.abs()
    daily_dates = fh_abs.index

    if fh_percentile > 0 and len(daily_dates) > MIN_HISTORY:
        expanding_thresh = fh_abs.expanding(min_periods=MIN_HISTORY).quantile(
            fh_percentile / 100.0
        ).shift(1)
    else:
        expanding_thresh = pd.Series(np.nan, index=daily_dates)

    out["fh_pctile_thresh"] = out["date"].map(expanding_thresh)
    out["fh_eff_thresh"] = out["fh_pctile_thresh"].fillna(fh_threshold).clip(lower=fh_threshold)

    # ------------------------------------------------------------------
    # Running session high (for pullback detection)
    # Reset at session start each day, track cumulative high.
    # ------------------------------------------------------------------
    session_high = np.full(n, np.nan)
    current_high = np.nan
    current_date = None
    for i in range(n):
        d = out["date"].iloc[i]
        t = time[i]
        if d != current_date:
            current_date = d
            current_high = np.nan
        if t >= t_session_start and t < t_session_end:
            h = out["high"].iloc[i]
            if np.isnan(current_high) or h > current_high:
                current_high = h
            session_high[i] = current_high
    out["session_high"] = session_high

    # ------------------------------------------------------------------
    # Identify bias days: fh_return > threshold (top 20%)
    # ------------------------------------------------------------------
    bias_days = set()
    fh_thresh_by_date = {}
    for d in daily_dates:
        fhr = fh_return.get(d, np.nan)
        thresh_val = expanding_thresh.get(d, np.nan)
        eff = max(thresh_val if not pd.isna(thresh_val) else fh_threshold, fh_threshold)
        fh_thresh_by_date[d] = eff
        if not pd.isna(fhr) and fhr > eff:
            bias_days.add(d)

    # ------------------------------------------------------------------
    # Signal generation: initial entry + pullback re-entries
    #
    # Pass 1: Place initial entry on first bar at/after fh_end on bias days.
    # Pass 2: On bias days, after each trade's expected exit, scan for
    #          pullback re-entry bars until max_trades_per_day is reached.
    #
    # signal_tier: 1 = initial entry, 2 = pullback re-entry
    # ------------------------------------------------------------------
    out["signal"] = 0
    out["stop_price"] = np.nan
    out["tp_price"] = np.nan
    out["size_factor"] = 1.0
    out["signal_tier"] = 0

    at_or_after_fh = time >= t_fh_end
    before_cutoff = time < t_entry_cutoff
    in_session = (time >= t_session_start) & (time < t_session_end)

    # Precompute column indices for fast iloc assignment
    sig_col = out.columns.get_loc("signal")
    stop_col = out.columns.get_loc("stop_price")
    tp_col = out.columns.get_loc("tp_price")
    sf_col = out.columns.get_loc("size_factor")
    tier_col = out.columns.get_loc("signal_tier")
    sc_col = out.columns.get_loc("session_close")

    closes = out["close"].values
    highs = out["high"].values
    lows = out["low"].values
    atrs = out["atr"].values
    dates = out["date"].values
    sess_highs = out["session_high"].values

    def _place_signal(idx, tier):
        """Place a long signal at bar idx, return True if successful."""
        atr_val = atrs[idx]
        if pd.isna(atr_val) or atr_val <= 0:
            return False
        entry_px = closes[idx]
        out.iloc[idx, sig_col] = 1
        out.iloc[idx, stop_col] = entry_px - (atr_val * stop_atr_multiple)
        out.iloc[idx, tp_col] = entry_px + (atr_val * tp_atr_multiple)
        out.iloc[idx, sf_col] = 1.0
        out.iloc[idx, tier_col] = tier
        # Place holding-period exit
        if holding_bars > 0:
            exit_idx = idx + 1 + holding_bars
            if exit_idx < n:
                out.iloc[exit_idx, sc_col] = True
        return True

    # Group bars by date for efficient iteration
    date_bar_ranges = {}
    current_d = None
    start_i = 0
    for i in range(n):
        d = dates[i]
        if d != current_d:
            if current_d is not None:
                date_bar_ranges[current_d] = (start_i, i)
            current_d = d
            start_i = i
    if current_d is not None:
        date_bar_ranges[current_d] = (start_i, n)

    for d in sorted(bias_days):
        if d not in date_bar_ranges:
            continue
        d_start, d_end = date_bar_ranges[d]
        trades_today = 0

        # --- Pass 1: Initial entry at first bar at/after fh_end ---
        initial_signal_idx = None
        for i in range(d_start, d_end):
            t = time[i]
            if t >= t_fh_end and t < t_entry_cutoff and t >= t_session_start and t < t_session_end:
                if _place_signal(i, tier=1):
                    initial_signal_idx = i
                    trades_today = 1
                break  # only first qualifying bar

        if initial_signal_idx is None:
            continue

        # --- Pass 2: Pullback re-entries after each trade's exit ---
        # Each trade: signal at bar S, fills at S+1, exits at S+1+holding_bars.
        # Next signal can be placed at exit_bar or later (backtester is flat).
        next_eligible = initial_signal_idx + 1 + holding_bars

        while trades_today < max_trades_per_day:
            found = False
            for i in range(max(next_eligible, d_start), d_end):
                t = time[i]
                if t >= t_entry_cutoff or t >= t_session_end:
                    break  # past cutoff, done for the day

                atr_val = atrs[i]
                sh = sess_highs[i]
                if pd.isna(atr_val) or atr_val <= 0 or pd.isna(sh):
                    continue

                # Pullback condition: bar dipped from session high by threshold
                pullback_depth = sh - lows[i]
                min_pullback = pullback_atr_frac * atr_val
                if pullback_depth < min_pullback:
                    continue

                # Recovery condition: bar closes in upper half of its range
                bar_range = highs[i] - lows[i]
                if bar_range <= 0:
                    continue
                if closes[i] < (highs[i] + lows[i]) / 2.0:
                    continue

                # Place pullback re-entry
                if _place_signal(i, tier=2):
                    trades_today += 1
                    next_eligible = i + 1 + holding_bars
                    found = True
                    break  # scan for next re-entry from new eligible bar

            if not found:
                break  # no more pullbacks today

    # Clean up working columns
    out.drop(columns=["date", "fh_pctile_thresh", "fh_eff_thresh", "session_high"], inplace=True)

    return out
