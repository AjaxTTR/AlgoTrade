"""
Gap-Up First-Hour Continuation strategy (long only).

Validated edge: Gap Up + FH return >= 75th percentile → continuation.
Source: discover_conditional_edges.py → validate_gap_edge.py → extract_gap_edge.py → expand_gap_edge.py

Trade logic:
  1. Gap-up filter: session open > prior day close by >= gap_threshold_pct.
  2. First-hour return (09:30-10:30): compute expanding percentile from prior days.
  3. Signal fires when gap_up AND fh_return >= dynamic threshold.
  4. Enter long at 10:30 bar close.
  5. Stop: 1.0x ATR below entry. TP: 1.0x ATR above entry (1R).
  6. Max hold: 4 bars (1 hour on 15-min bars).
  7. At most 1 trade per day. Long only.

Every strategy module must expose:
    def generate_signals(df: pd.DataFrame, **params) -> pd.DataFrame
"""

import numpy as np
import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    fh_end: str = "10:30",
    entry_cutoff: str = "15:45",
    atr_period: int = 14,
    stop_atr_multiple: float = 1.0,
    tp_atr_multiple: float = 1.0,
    fh_percentile: float = 75.0,
    gap_threshold_pct: float = 0.10,
    **kwargs,
) -> pd.DataFrame:
    """Generate long-only gap-up first-hour continuation signals.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex (15-min bars expected).
    session_start / session_end : str
        RTH session boundaries.
    fh_end : str
        End of the first-hour window (default "10:30"). Also the entry time.
    entry_cutoff : str
        Latest time to enter a new trade (default "15:45").
    atr_period : int
        Lookback for Average True Range (default 14).
    stop_atr_multiple : float
        Stop distance as ATR multiple (default 1.0).
    tp_atr_multiple : float
        Take-profit distance as ATR multiple (default 1.0 = 1R).
    fh_percentile : float
        Percentile of fh_return distribution used as dynamic threshold
        (default 75.0). Computed from prior days only (shifted).
    gap_threshold_pct : float
        Minimum gap-up size in percent (default 0.10).

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns: signal, stop_price, tp_price,
        session_close, atr, fh_return, size_factor, signal_tier, entry_type.
    """
    out = df.copy()
    n = len(out)

    # -- ATR --
    prev_close = out["close"].shift(1)
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum(
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ),
    )
    out["atr"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

    # -- Time boundaries --
    time = out.index.time
    t_session_start = pd.Timestamp(session_start).time()
    t_fh_end = pd.Timestamp(fh_end).time()
    t_session_end = pd.Timestamp(session_end).time()
    t_entry_cutoff = pd.Timestamp(entry_cutoff).time()

    out["date"] = out.index.date

    # -- Session close flag --
    past_cutoff = time >= t_entry_cutoff
    new_day = out["date"] != pd.Series(out["date"]).shift(1).values
    prev_past = pd.Series(past_cutoff).shift(1, fill_value=False).values
    out["session_close"] = past_cutoff & (~prev_past | new_day)

    # -- First-hour return (full 60 min, 09:30-10:30) --
    in_fh = (time >= t_session_start) & (time < t_fh_end)
    fh_bars = out[in_fh]
    fh_open = fh_bars.groupby("date")["open"].first()
    fh_close = fh_bars.groupby("date")["close"].last()
    fh_return = (fh_close - fh_open) / fh_open
    fh_return.name = "fh_return"
    out["fh_return"] = out["date"].map(fh_return)

    # -- Daily session open and prior day close for gap detection --
    session_bars = out[(time >= t_session_start) & (time < t_session_end)]
    daily_open = session_bars.groupby("date")["open"].first()
    daily_close = session_bars.groupby("date")["close"].last()

    # -- Expanding percentile threshold on fh_return (shifted, no lookahead) --
    _MIN_HISTORY = 20
    daily_dates = fh_return.index

    dynamic_thresh = fh_return.expanding(min_periods=_MIN_HISTORY).quantile(
        fh_percentile / 100.0
    ).shift(1)

    # -- Classify qualifying days --
    signal_days = set()
    sorted_dates = sorted(daily_dates)

    for i, d in enumerate(sorted_dates):
        # Gap-up check: today's open vs yesterday's close
        if i == 0:
            continue
        prev_d = sorted_dates[i - 1]

        today_open = daily_open.get(d, np.nan)
        prev_day_close = daily_close.get(prev_d, np.nan)

        if pd.isna(today_open) or pd.isna(prev_day_close) or prev_day_close <= 0:
            continue

        gap_pct = (today_open - prev_day_close) / prev_day_close * 100
        if gap_pct <= gap_threshold_pct:
            continue

        # FH return threshold check
        thresh = dynamic_thresh.get(d, np.nan)
        if pd.isna(thresh):
            continue

        fh_ret = fh_return.get(d, np.nan)
        if pd.isna(fh_ret) or fh_ret < thresh:
            continue

        signal_days.add(d)

    # -- Signal output columns --
    out["signal"] = 0
    out["stop_price"] = np.nan
    out["tp_price"] = np.nan
    out["size_factor"] = 1.0
    out["signal_tier"] = 0
    out["entry_type"] = ""

    sig_col = out.columns.get_loc("signal")
    stop_col = out.columns.get_loc("stop_price")
    tp_col = out.columns.get_loc("tp_price")
    sf_col = out.columns.get_loc("size_factor")
    tier_col = out.columns.get_loc("signal_tier")
    et_col = out.columns.get_loc("entry_type")

    closes = out["close"].values
    atrs = out["atr"].values
    dates = out["date"].values

    def _place_long(idx):
        atr_val = atrs[idx]
        if pd.isna(atr_val) or atr_val <= 0:
            return False
        px = closes[idx]
        out.iloc[idx, sig_col] = 1
        out.iloc[idx, stop_col] = px - atr_val * stop_atr_multiple
        out.iloc[idx, tp_col] = px + atr_val * tp_atr_multiple
        out.iloc[idx, sf_col] = 1.0
        out.iloc[idx, tier_col] = 1
        out.iloc[idx, et_col] = "gap_fh_continuation"
        return True

    # -- Group bars by date --
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

    # -- Generate signals: enter at 10:30 bar --
    for d in sorted(signal_days):
        if d not in date_bar_ranges:
            continue
        d_start, d_end = date_bar_ranges[d]

        for i in range(d_start, d_end):
            t = time[i]
            if t >= t_fh_end and t < t_entry_cutoff and t < t_session_end:
                _place_long(i)
                break

    # Clean up
    out.drop(columns=["date"], inplace=True)

    # -- Validate output --
    signal_bars = out[out["signal"] != 0]

    invalid_vals = set(out["signal"].unique()) - {0, 1}
    if invalid_vals:
        raise ValueError(f"signal contains invalid values: {invalid_vals}.")

    if signal_bars["stop_price"].isna().any():
        bad = signal_bars[signal_bars["stop_price"].isna()].index.tolist()
        raise ValueError(f"stop_price is NaN on {len(bad)} signal bar(s): {bad[:5]}")

    signals_per_day = signal_bars.groupby(signal_bars.index.date).size()
    multi = signals_per_day[signals_per_day > 1]
    if len(multi) > 0:
        raise ValueError(f"More than 1 signal on {len(multi)} day(s): {multi.index.tolist()[:5]}")

    return out
