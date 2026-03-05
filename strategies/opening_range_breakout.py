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
    min_orb_atr_ratio: float = 0.3,
    use_vwap_filter: bool = True,
    min_atr_percent: float = 0.003,
    min_gap_atr_ratio: float = 0.25,
    trade_window_start: str = "09:30",
    trade_window_end: str = "11:00",
    max_trades_per_day: int = 1,
    atr_ma_period: int = 50,
    use_volume_filter: bool = True,
    volume_ma_period: int = 20,
    use_htf_filter: bool = True,
    htf_ema_period: int = 200,
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
    min_orb_atr_ratio : float
        Minimum ORB width as a fraction of ATR (default 0.3).
        Days where orb_width < ATR * ratio produce no signals.
    use_vwap_filter : bool
        When True, long breakouts require close > VWAP and short
        breakouts require close < VWAP (default True).
    min_atr_percent : float
        Minimum ATR as a percentage of close price (default 0.003).
        Filters out low-volatility regimes where breakouts are unreliable.
    min_gap_atr_ratio : float
        Minimum opening gap size as a fraction of ATR (default 0.25).
        Days with a smaller gap produce no signals.
    trade_window_start : str
        Earliest time for breakout entries in "HH:MM" format (default "09:30").
    trade_window_end : str
        Latest time for breakout entries in "HH:MM" format (default "11:00").
        Bars outside this window produce no signals.
    max_trades_per_day : int
        Maximum total trades (long + short) allowed per day (default 1).
    atr_ma_period : int
        Rolling window for ATR moving average (default 50).
        Trades are only allowed when ATR > its rolling mean.
    use_volume_filter : bool
        When True, require breakout bar volume > its rolling mean (default True).
    volume_ma_period : int
        Rolling window for volume moving average (default 20).
    use_htf_filter : bool
        When True, apply daily EMA trend filter: long only above EMA,
        short only below (default True).
    htf_ema_period : int
        Period for the daily EMA trend filter (default 200).

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
        - vwap: Volume-Weighted Average Price (reset daily)
        - atr_pct: ATR as a fraction of close price
        - gap_atr_ratio: opening gap size normalized by ATR
        - daily_ema: daily EMA trend line (ffilled to intraday, shifted 1 day)
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

    # ── ATR percentage (volatility regime) ──
    out["atr_pct"] = out["atr"] / out["close"]

    # ── ATR moving average (expanding volatility filter) ──
    out["atr_ma"] = out["atr"].rolling(atr_ma_period, min_periods=atr_ma_period).mean()

    # ── VWAP (reset daily) ──
    out["tp"] = (out["high"] + out["low"] + out["close"]) / 3
    dates_for_vwap = out.index.date
    out["cum_pv"] = (out["tp"] * out["volume"]).groupby(dates_for_vwap).cumsum()
    out["cum_vol"] = out["volume"].groupby(dates_for_vwap).cumsum()
    out["vwap"] = out["cum_pv"] / out["cum_vol"]
    out.drop(columns=["tp", "cum_pv", "cum_vol"], inplace=True)

    # Parse session start, end, and entry cutoff into hour and minute
    ss_hour, ss_min = (int(x) for x in session_start.split(":"))
    se_hour, se_min = (int(x) for x in session_end.split(":"))
    ec_hour, ec_min = (int(x) for x in entry_cutoff.split(":"))
    tw_start_hour, tw_start_min = (int(x) for x in trade_window_start.split(":"))
    tw_end_hour, tw_end_min = (int(x) for x in trade_window_end.split(":"))

    # Bar time components
    bar_times = out.index
    bar_hours = bar_times.hour
    bar_minutes = bar_times.minute
    dates = bar_times.date
    dates_arr = np.array(dates)

    # Minutes since midnight for each bar, session boundaries, and ORB end
    bar_mins_since_midnight = bar_hours * 60 + bar_minutes
    session_start_mins = ss_hour * 60 + ss_min
    session_end_mins = se_hour * 60 + se_min
    entry_cutoff_mins = ec_hour * 60 + ec_min
    trade_window_start_mins = tw_start_hour * 60 + tw_start_min
    trade_window_end_mins = tw_end_hour * 60 + tw_end_min
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

    # ── Opening gap filter ──
    daily_close = out.groupby(dates)["close"].transform("last")
    prev_day_close = daily_close.groupby(dates).transform("first").shift(1)
    # First bar at or after session start each day (handles missing exact-match bars)
    in_session = bar_mins_since_midnight >= session_start_mins
    prev_in_session = pd.Series(in_session, index=out.index).shift(1, fill_value=False)
    prev_date_gap = pd.Series(dates_arr, index=out.index).shift(1)
    new_day_gap = pd.Series(dates_arr, index=out.index) != prev_date_gap
    first_session_bar = in_session & (~prev_in_session | new_day_gap)
    session_open_price = out["open"].where(first_session_bar)
    session_open_price = session_open_price.groupby(dates).transform("first")
    gap_size = (session_open_price - prev_day_close).abs()
    out["gap_atr_ratio"] = gap_size / out["atr"]
    gap_ok = out["gap_atr_ratio"] >= min_gap_atr_ratio

    # Compute opening range high/low per day using only ORB window bars
    orb_high = out["high"].where(is_orb_window)
    orb_low = out["low"].where(is_orb_window)

    range_high_daily = orb_high.groupby(dates).transform("max")
    range_low_daily = orb_low.groupby(dates).transform("min")

    out["range_high"] = range_high_daily
    out["range_low"] = range_low_daily

    # ORB width filter: skip days where range is too narrow relative to ATR
    orb_width = out["range_high"] - out["range_low"]
    valid_range = orb_width >= (out["atr"] * min_orb_atr_ratio)

    # Volatility regime filters
    vol_ok = out["atr_pct"] >= min_atr_percent
    atr_expanding = out["atr"] > out["atr_ma"]

    # Trade window filter
    in_trade_window = (
        (bar_mins_since_midnight >= trade_window_start_mins)
        & (bar_mins_since_midnight < trade_window_end_mins)
    )

    # Volume confirmation filter
    if use_volume_filter:
        vol_ma = out["volume"].rolling(volume_ma_period, min_periods=volume_ma_period).mean()
        volume_ok = out["volume"] > vol_ma
    else:
        volume_ok = pd.Series(True, index=out.index)

    # Higher-timeframe trend filter (daily EMA, shifted 1 day to avoid look-ahead)
    if use_htf_filter:
        daily_close = out["close"].resample("D").last().dropna()
        daily_ema = daily_close.ewm(span=htf_ema_period, min_periods=htf_ema_period, adjust=False).mean()
        daily_ema_shifted = daily_ema.shift(1)  # use yesterday's EMA
        out["daily_ema"] = daily_ema_shifted.reindex(out.index, method="ffill")
        htf_long_ok = out["close"] > out["daily_ema"]
        htf_short_ok = out["close"] < out["daily_ema"]
    else:
        out["daily_ema"] = np.nan
        htf_long_ok = pd.Series(True, index=out.index)
        htf_short_ok = pd.Series(True, index=out.index)

    # Breakout conditions (all filters combined)
    base_filter = after_orb & in_trade_window & valid_range & vol_ok & atr_expanding & gap_ok & volume_ok
    long_break = base_filter & (out["close"] > out["range_high"]) & htf_long_ok
    short_break = base_filter & (out["close"] < out["range_low"]) & htf_short_ok

    # VWAP directional bias filter
    if use_vwap_filter:
        long_break = long_break & (out["close"] > out["vwap"])
        short_break = short_break & (out["close"] < out["vwap"])

    # Allow only one signal per direction per day (vectorized)
    long_first = long_break.groupby(dates).cumsum() == 1
    short_first = short_break.groupby(dates).cumsum() == 1

    raw_signal = np.zeros(n, dtype=np.int8)
    raw_signal[long_break.values & long_first.values] = 1
    raw_signal[short_break.values & short_first.values] = -1

    # Cap total trades per day
    any_signal = raw_signal != 0
    daily_cum = pd.Series(any_signal, index=out.index).groupby(dates).cumsum()
    signal = np.where(daily_cum.values <= max_trades_per_day, raw_signal, 0).astype(np.int8)

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
    entry_prices = out["open"].shift(-1).values
    tp_price[long_mask] = entry_prices[long_mask] + (atr_vals[long_mask] * tp_atr_multiple)
    tp_price[short_mask] = entry_prices[short_mask] - (atr_vals[short_mask] * tp_atr_multiple)
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
