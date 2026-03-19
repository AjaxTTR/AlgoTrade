"""
First-Hour Momentum strategy.

Edge: when the first hour (09:30-10:30) shows a strong positive move,
the rest of the day tends to continue upward.

Source: ranked_edges.csv -- FH_Up -> Rest_Up (t=30.38, stability=0.946).
Long only -- the short side showed weaker edge quality.

Trade logic:
  1. Bias filter: only trade on days where first-hour return is in the
     top 20% historically (80th percentile, expanding window).
  2. Early entry: if the first 30 minutes (09:30-10:00) already exceed
     the threshold, enter at 10:00 instead of waiting until 10:30.
  3. Standard entry: otherwise, enter at 10:30 after full first hour.
  4. Pullback re-entries: after each trade exits (holding period),
     re-enter long on pullback-and-recovery bars.
  5. Midday continuation: after 11:00, enter on consolidation breakout
     or higher-high after pullback, if aligned with first-hour bias.
     Max 1 additional trade per day.
  6. Max trades per day caps total entries.

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
    early_end: str = "10:00",
    entry_cutoff: str = "15:45",
    fh_threshold: float = 0.0,
    fh_percentile: float = 80.0,
    atr_period: int = 14,
    stop_atr_multiple: float = 1.5,
    tp_atr_multiple: float = 2.0,
    holding_bars: int = 8,
    max_trades_per_day: int = 3,
    pullback_atr_frac: float = 1.0,
    enable_early_entry: bool = True,
    midday_start: str = "11:00",
    midday_consol_bars: int = 4,
    midday_consol_atr_frac: float = 1.0,
    midday_pullback_atr_frac: float = 0.5,
    max_midday_trades: int = 1,
    min_bars_between_entries: int = 2,
    **kwargs,
) -> pd.DataFrame:
    """First-Hour Momentum with early entry, pullback re-entries, and midday continuation.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with a DatetimeIndex (15-min bars expected).
    session_start / session_end / fh_end / entry_cutoff : str
        Time boundaries for the session and first-hour window.
    early_end : str
        End of the early observation window (default "10:00").
        If the return from session_start to early_end already exceeds the
        threshold, enter at early_end instead of waiting for fh_end.
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
        Maximum entries per bias day (default 3).
    pullback_atr_frac : float
        Minimum pullback depth from running session high as a fraction
        of ATR (default 0.5).
    enable_early_entry : bool
        If True, check 30-min return for early entry at early_end
        (default True).  If False, always wait for fh_end.
    midday_start : str
        Earliest time for midday continuation entries (default "11:00").
    midday_consol_bars : int
        Number of prior bars to check for consolidation (default 4).
    midday_consol_atr_frac : float
        Max range of consolidation window as ATR fraction (default 1.0).
        If the high-low range of the prior N bars is below this * ATR,
        the market is consolidating.
    midday_pullback_atr_frac : float
        Min pullback from session high for higher-high trigger (default 0.5).
    max_midday_trades : int
        Max midday continuation entries per day (default 1).
    min_bars_between_entries : int
        Minimum bars between consecutive signal placements (default 2).
        Controls signal spacing so the backtester can overlap positions.

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns: signal, stop_price, tp_price,
        session_close, atr, fh_return, size_factor, signal_tier.
        signal_tier: 1 = early entry, 2 = standard entry, 3 = pullback,
                     4 = midday continuation.
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
    t_early_end = pd.Timestamp(early_end).time()
    t_fh_end = pd.Timestamp(fh_end).time()
    t_session_end = pd.Timestamp(session_end).time()
    t_entry_cutoff = pd.Timestamp(entry_cutoff).time()
    t_midday_start = pd.Timestamp(midday_start).time()

    out["date"] = out.index.date

    # ------------------------------------------------------------------
    # Session close flag (15:45 EOD backstop)
    # ------------------------------------------------------------------
    past_cutoff = time >= t_entry_cutoff
    new_day = out["date"] != pd.Series(out["date"]).shift(1).values
    prev_past = pd.Series(past_cutoff).shift(1, fill_value=False).values
    out["session_close"] = past_cutoff & (~prev_past | new_day)

    # ------------------------------------------------------------------
    # First-hour return per day (full 60 min, no lookahead)
    # Used for: threshold computation (expanding percentile) and
    #           standard 10:30 entry decision.
    # ------------------------------------------------------------------
    in_fh = (time >= t_session_start) & (time < t_fh_end)
    fh_bars = out[in_fh]
    fh_open = fh_bars.groupby("date")["open"].first()
    fh_close = fh_bars.groupby("date")["close"].last()
    fh_return = (fh_close - fh_open) / fh_open
    fh_return.name = "fh_return"
    out["fh_return"] = out["date"].map(fh_return)

    # ------------------------------------------------------------------
    # Early return per day (first 30 min: 09:30-10:00)
    # Used for: early entry decision at 10:00.
    # ------------------------------------------------------------------
    in_early = (time >= t_session_start) & (time < t_early_end)
    early_bars = out[in_early]
    early_open = early_bars.groupby("date")["open"].first()
    early_close = early_bars.groupby("date")["close"].last()
    early_return = (early_close - early_open) / early_open
    early_return.name = "early_return"
    out["early_return"] = out["date"].map(early_return)

    # ------------------------------------------------------------------
    # Adaptive threshold: expanding percentile of |fh_return|
    # Built from FULL first-hour returns (60 min) of prior days.
    # The same threshold is used for both early and standard entries.
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
    # Track whether a pullback from session high has occurred (per bar)
    # A pullback means price dipped >= midday_pullback_atr_frac * ATR
    # from the running session high at some point during the day.
    # ------------------------------------------------------------------
    atr_vals = out["atr"].values
    low_vals = out["low"].values
    had_pullback = np.zeros(n, dtype=bool)
    pullback_active = False
    current_date_pb = None
    for i in range(n):
        d = out["date"].iloc[i]
        t = time[i]
        if d != current_date_pb:
            current_date_pb = d
            pullback_active = False
        if t >= t_session_start and t < t_session_end:
            sh = session_high[i]
            atr_val = atr_vals[i]
            if not np.isnan(sh) and not np.isnan(atr_val) and atr_val > 0:
                dip = sh - low_vals[i]
                if dip >= midday_pullback_atr_frac * atr_val:
                    pullback_active = True
        had_pullback[i] = pullback_active
    out["had_pullback"] = had_pullback

    # ------------------------------------------------------------------
    # Identify bias days and early-eligible days
    #
    # bias_days: fh_return (full 60 min) > threshold
    # early_days: early_return (first 30 min) > threshold
    #   (subset of days where the move is so strong that 30 min
    #    already exceeds the typical full-hour threshold)
    # ------------------------------------------------------------------
    bias_days = set()
    early_days = set()
    for d in daily_dates:
        thresh_val = expanding_thresh.get(d, np.nan)
        eff = max(thresh_val if not pd.isna(thresh_val) else fh_threshold, fh_threshold)

        fhr = fh_return.get(d, np.nan)
        if not pd.isna(fhr) and fhr > eff:
            bias_days.add(d)

        if enable_early_entry:
            er = early_return.get(d, np.nan)
            if not pd.isna(er) and er > eff:
                early_days.add(d)

    # ------------------------------------------------------------------
    # Signal generation
    #
    # For each day:
    #   1. If early_days: enter at first bar at/after early_end (tier=1)
    #   2. Elif bias_days: enter at first bar at/after fh_end (tier=2)
    #   3. Pullback re-entries after each trade exits (tier=3)
    # ------------------------------------------------------------------
    out["signal"] = 0
    out["stop_price"] = np.nan
    out["tp_price"] = np.nan
    out["size_factor"] = 1.0
    out["signal_tier"] = 0

    sig_col = out.columns.get_loc("signal")
    stop_col = out.columns.get_loc("stop_price")
    tp_col = out.columns.get_loc("tp_price")
    sf_col = out.columns.get_loc("size_factor")
    tier_col = out.columns.get_loc("signal_tier")
    sc_col = out.columns.get_loc("session_close")

    opens = out["open"].values
    closes = out["close"].values
    highs = out["high"].values
    lows = out["low"].values
    atrs = out["atr"].values
    dates = out["date"].values
    sess_highs = out["session_high"].values
    had_pb = out["had_pullback"].values

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
        return True

    # Group bars by date
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

    # All tradeable days = union of early_days and bias_days
    all_trade_days = bias_days | early_days

    for d in sorted(all_trade_days):
        if d not in date_bar_ranges:
            continue
        d_start, d_end = date_bar_ranges[d]
        trades_today = 0
        initial_signal_idx = None
        initial_tier = None

        # --- Try early entry (10:00) ---
        if d in early_days:
            for i in range(d_start, d_end):
                t = time[i]
                if t >= t_early_end and t < t_entry_cutoff and t >= t_session_start and t < t_session_end:
                    if _place_signal(i, tier=1):
                        initial_signal_idx = i
                        initial_tier = 1
                        trades_today = 1
                    break

        # --- Fallback: standard entry (10:30) ---
        if initial_signal_idx is None and d in bias_days:
            for i in range(d_start, d_end):
                t = time[i]
                if t >= t_fh_end and t < t_entry_cutoff and t >= t_session_start and t < t_session_end:
                    if _place_signal(i, tier=2):
                        initial_signal_idx = i
                        initial_tier = 2
                        trades_today = 1
                    break

        if initial_signal_idx is None:
            continue

        next_eligible = initial_signal_idx + min_bars_between_entries

        # --- Pullback re-entry (tier 3): only after confirmed Tier 2 ---
        #
        # Two-stage trigger to avoid shallow / choppy re-entries:
        #   Stage 1 (pullback confirmation): bar dips >= 1 ATR from session
        #            high AND closes in top 25% of its range (rejection).
        #   Stage 2 (entry): a subsequent strong bullish bar — close > open,
        #            close in top 25% of range, range > 0.25 ATR (anti-chop).
        # Max 1 pullback trade per day.
        if initial_tier == 2 and trades_today < max_trades_per_day:
            pullback_confirmed = False

            for i in range(max(next_eligible, d_start), d_end):
                t = time[i]
                if t >= t_entry_cutoff or t >= t_session_end:
                    break

                atr_val = atrs[i]
                if pd.isna(atr_val) or atr_val <= 0:
                    continue

                sh = sess_highs[i]
                if pd.isna(sh):
                    continue

                # Stage 1: detect pullback with rejection
                if not pullback_confirmed:
                    pullback_depth = sh - lows[i]
                    if pullback_depth >= pullback_atr_frac * atr_val:
                        bar_range = highs[i] - lows[i]
                        if bar_range > 0 and closes[i] >= lows[i] + 0.75 * bar_range:
                            pullback_confirmed = True
                    continue  # never enter on the pullback bar itself

                # Stage 2: strong bullish bar after confirmed pullback
                bar_range = highs[i] - lows[i]
                if bar_range <= 0 or bar_range < 0.25 * atr_val:
                    continue  # reject narrow / choppy bars
                if closes[i] <= opens[i]:
                    continue  # must be bullish (close > open)
                if closes[i] < lows[i] + 0.75 * bar_range:
                    continue  # close must be in top 25% of bar range

                if _place_signal(i, tier=3):
                    trades_today += 1
                    next_eligible = i + min_bars_between_entries
                    break  # max 1 pullback per day

        # --- Midday continuation (tier 4) ---
        midday_trades = 0
        while trades_today < max_trades_per_day and midday_trades < max_midday_trades:
            found = False
            for i in range(max(next_eligible, d_start), d_end):
                t = time[i]
                if t >= t_entry_cutoff or t >= t_session_end:
                    break
                if t < t_midday_start:
                    continue

                atr_val = atrs[i]
                if pd.isna(atr_val) or atr_val <= 0:
                    continue

                midday_triggered = False

                # (A) Consolidation breakout
                if i >= midday_consol_bars:
                    consol_high = np.max(highs[i - midday_consol_bars:i])
                    consol_low = np.min(lows[i - midday_consol_bars:i])
                    consol_range = consol_high - consol_low
                    if consol_range < midday_consol_atr_frac * atr_val:
                        if closes[i] > consol_high:
                            midday_triggered = True

                # (B) Higher high after pullback
                if not midday_triggered and had_pb[i]:
                    if i >= 1 and highs[i] > highs[i - 1]:
                        bar_range = highs[i] - lows[i]
                        if bar_range > 0 and closes[i] >= (highs[i] + lows[i]) / 2.0:
                            midday_triggered = True

                if midday_triggered and _place_signal(i, tier=4):
                    midday_trades += 1
                    trades_today += 1
                    next_eligible = i + min_bars_between_entries
                    found = True
                    break

            if not found:
                break

    # Clean up working columns
    out.drop(columns=["date", "fh_pctile_thresh", "fh_eff_thresh",
                       "session_high", "early_return", "had_pullback"],
             inplace=True)

    return out
