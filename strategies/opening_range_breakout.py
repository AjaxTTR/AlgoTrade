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
    **kwargs,
) -> pd.DataFrame:
    """Opening Range Breakout strategy.

    Computes the high and low of the first *orb_minutes* of each trading
    day, then generates a long signal on the first bar that closes above
    the range high and a short signal on the first bar that closes below
    the range low.  Only one signal per direction per day.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with a DatetimeIndex (intraday frequency).
    orb_minutes : int
        Number of minutes from the start of each day's session used to
        define the opening range (default 30).

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns:
        - signal: 1 (long), -1 (short), 0 (flat)
        - stop_price: opposite boundary of the opening range
        - range_high: opening range upper boundary
        - range_low: opening range lower boundary
    """
    out = df.copy()
    n = len(out)

    # Trading date for each bar
    dates = out.index.date

    # First bar timestamp of each day
    date_series = pd.Series(dates, index=out.index)
    first_bar_time = date_series.groupby(dates).transform("idxmin")

    # Minutes elapsed since each day's first bar
    minutes_elapsed = (out.index - first_bar_time).total_seconds() / 60.0

    # Bars within the opening range window
    is_orb_window = minutes_elapsed < orb_minutes

    # Compute opening range high/low per day using only ORB bars
    orb_high = out["high"].where(is_orb_window)
    orb_low = out["low"].where(is_orb_window)

    range_high_daily = orb_high.groupby(dates).transform("max")
    range_low_daily = orb_low.groupby(dates).transform("min")

    out["range_high"] = range_high_daily
    out["range_low"] = range_low_daily

    # After the ORB window, check for breakouts
    after_orb = ~is_orb_window

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

    return out
