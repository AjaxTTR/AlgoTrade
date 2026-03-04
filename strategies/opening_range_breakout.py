"""
Opening Range Breakout (ORB) strategy.

Every strategy module in /strategies must expose:

    def generate_signals(df: pd.DataFrame, **params) -> pd.DataFrame

The returned DataFrame must include a ``signal`` column
(1 = long, -1 = short, 0 = flat) and a ``stop_price`` column.
"""

import pandas as pd
import numpy as np


def generate_signals(
    df: pd.DataFrame,
    orb_minutes: int = 30,
    session_start: str = "09:30",
    session_end: str = "16:00",
    entry_cutoff: str = "14:30",
    atr_period: int = 14,
    tp_atr_multiple: float = 2.0,
    **kwargs,
) -> pd.DataFrame:
    """Opening Range Breakout strategy anchored to session open.

    Computes the high and low of the first *orb_minutes* after
    *session_start* each trading day, then generates a long signal on
    the first bar that closes above the range high and a short signal
    on the first bar that closes below the range low.  Only one signal
    per direction per day.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with a DatetimeIndex (intraday frequency).
    orb_minutes : int
        Length of the opening range window in minutes (default 30).
    session_start : str
        Session open time in "HH:MM" format (default "09:30").
    session_end : str
        Session close time in "HH:MM" format (default "16:00").
        Breakouts are only allowed before this time.
    entry_cutoff : str
        Latest time for new entries in "HH:MM" format (default "14:30").
        No new breakout signals after this time.
    atr_period : int
        Lookback for ATR calculation (default 14).
    tp_atr_multiple : float
        ATR multiplier for take-profit distance (default 2.0).

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns:
        - signal: 1 (long), -1 (short), 0 (flat)
        - stop_price: opposite boundary of the opening range
        - tp_price: ATR-based take-profit target
        - range_high: opening range upper boundary
        - range_low: opening range lower boundary
        - session_close: True on the first bar at or after session_end each day
        - atr: Average True Range
    """
    out = df.copy()
    n = len(out)

    # ── ATR (Wilder smoothing via EMA with alpha = 1/period) ──
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift(1)).abs(),
            (out["low"] - out["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

    # Parse session start, end, and entry cutoff into hour and minute
    ss_hour, ss_min = (int(x) for x in session_start.split(":"))
    se_hour, se_min = (int(x) for x in session_end.split(":"))
    ec_hour, ec_min = (int(x) for x in entry_cutoff.split(":"))

    # Bar time components
    bar_times = out.index
    bar_hours = bar_times.hour
    bar_minutes = bar_times.minute
    dates = bar_times.date

    # Minutes since midnight for each bar, session boundaries, and ORB end
    bar_mins_since_midnight = bar_hours * 60 + bar_minutes
    session_start_mins = ss_hour * 60 + ss_min
    session_end_mins = se_hour * 60 + se_min
    entry_cutoff_mins = ec_hour * 60 + ec_min
    orb_end_mins = session_start_mins + orb_minutes

    # ORB window: bars at or after session_start and before ORB end
    is_orb_window = (
        (bar_mins_since_midnight >= session_start_mins)
        & (bar_mins_since_midnight < orb_end_mins)
    )

    # Tradeable window: after ORB closes, before entry cutoff, and before session end
    after_orb = (
        (bar_mins_since_midnight >= orb_end_mins)
        & (bar_mins_since_midnight < entry_cutoff_mins)
        & (bar_mins_since_midnight < session_end_mins)
    )

    # Compute opening range high/low per day using only ORB window bars
    orb_high = out["high"].where(is_orb_window)
    orb_low = out["low"].where(is_orb_window)

    range_high_daily = orb_high.groupby(dates).transform("max")
    range_low_daily = orb_low.groupby(dates).transform("min")

    out["range_high"] = range_high_daily
    out["range_low"] = range_low_daily

    # Breakout conditions (only after ORB window closes)
    long_break = after_orb & (out["close"] > out["range_high"])
    short_break = after_orb & (out["close"] < out["range_low"])

    # Build raw signal
    raw_signal = np.zeros(n, dtype=np.int8)
    raw_signal[long_break.values] = 1
    raw_signal[short_break.values] = -1

    # Allow only one signal per direction per day
    signal = np.zeros(n, dtype=np.int8)
    prev_date = None
    long_fired = False
    short_fired = False

    dates_arr = np.array(dates)
    for i in range(n):
        if dates_arr[i] != prev_date:
            prev_date = dates_arr[i]
            long_fired = False
            short_fired = False

        if raw_signal[i] == 1 and not long_fired:
            signal[i] = 1
            long_fired = True
        elif raw_signal[i] == -1 and not short_fired:
            signal[i] = -1
            short_fired = True

    out["signal"] = signal

    # Stop price: opposite side of the opening range
    stop_price = np.full(n, np.nan)
    long_mask = signal == 1
    short_mask = signal == -1
    stop_price[long_mask] = out["range_low"].values[long_mask]
    stop_price[short_mask] = out["range_high"].values[short_mask]
    out["stop_price"] = stop_price

    # Take-profit: ATR-based target on signal bars only
    tp_price = np.full(n, np.nan)
    atr_vals = out["atr"].values
    close_vals = out["close"].values
    tp_price[long_mask] = close_vals[long_mask] + (atr_vals[long_mask] * tp_atr_multiple)
    tp_price[short_mask] = close_vals[short_mask] - (atr_vals[short_mask] * tp_atr_multiple)
    out["tp_price"] = tp_price

    # Session close: first bar at or after session_end each day
    past_session_end = pd.Series(
        bar_mins_since_midnight >= session_end_mins, index=out.index
    )
    # Mark only the first such bar per day using shift: True now but False previous bar (or new day)
    prev_past = past_session_end.shift(1, fill_value=False)
    prev_date_shifted = pd.Series(dates_arr, index=out.index).shift(1)
    new_day = pd.Series(dates_arr, index=out.index) != prev_date_shifted

    out["session_close"] = past_session_end & (~prev_past | new_day)

    return out
