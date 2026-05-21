"""
Centralized feature engineering for strategy consumption.

Computes all shared features ONCE from OHLCV data.  Strategies import
``build_features`` and consume the resulting columns — they never
compute ATR, FH return, gap, etc. themselves.

Every feature uses only past/current data — no lookahead bias.

Usage:
    from engine.features import build_features
    out = build_features(df)
    # out now has: atr, atr_percentile, fh_return, early_return,
    #   gap_pct, session_close, vol_regime, hour_of_day, ...
"""

import numpy as np
import pandas as pd


_MIN_HISTORY = 20  # minimum days before expanding stats are valid


def build_features(
    df: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    fh_end: str = "10:30",
    early_end: str = "10:00",
    entry_cutoff: str = "15:45",
    atr_period: int = 14,
) -> pd.DataFrame:
    """Build all shared features from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex (15-min bars expected).
    session_start : str
        RTH session start (default "09:30").
    session_end : str
        RTH session end (default "16:00").
    fh_end : str
        End of full first-hour window (default "10:30").
    early_end : str
        End of early observation window (default "10:00").
    entry_cutoff : str
        Latest entry time / session close trigger (default "15:45").
    atr_period : int
        ATR averaging period (default 14).

    Returns
    -------
    pd.DataFrame
        Copy of input with feature columns appended.

    Feature columns added
    ---------------------
    **Volatility:**
        atr, atr_percentile, vol_regime

    **Session:**
        date, session_close, daily_open, daily_close, daily_prev_close

    **First hour:**
        fh_return, early_return

    **Gap:**
        gap_pct

    **Time:**
        hour_of_day, minute_of_day, day_of_week
    """
    out = df.copy()

    # -- Time boundaries --
    time_vals = out.index.time
    t_session_start = pd.Timestamp(session_start).time()
    t_early_end = pd.Timestamp(early_end).time()
    t_fh_end = pd.Timestamp(fh_end).time()
    t_session_end = pd.Timestamp(session_end).time()
    t_entry_cutoff = pd.Timestamp(entry_cutoff).time()

    out["date"] = out.index.date

    # =====================================================================
    # ATR (exponential weighted, matches strategy convention)
    # =====================================================================
    prev_close = out["close"].shift(1)
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum(
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ),
    )
    out["atr"] = tr.ewm(
        alpha=1.0 / atr_period, min_periods=atr_period, adjust=False,
    ).mean()

    # =====================================================================
    # ATR percentile (expanding rank, 0-100)
    # =====================================================================
    # Both atr[i] and the rank that includes atr[i] require bar i's H/L/C.
    # Shifting by 1 makes this column the percentile rank KNOWABLE at bar i's
    # open — i.e. rank of atr[i-1] within atr[0..i-1]. Any consumer reading
    # atr_percentile[i] at a bar-i entry is now lookahead-safe by definition.
    out["atr_percentile"] = (
        out["atr"].expanding(min_periods=atr_period).rank(pct=True) * 100.0
    ).shift(1)

    # =====================================================================
    # Volatility regime (ATR vs expanding median — simple placeholder)
    # =====================================================================
    # Same lookahead reasoning as atr_percentile — compare the PRIOR bar's
    # ATR against the expanding median up to the prior bar.
    prev_atr = out["atr"].shift(1)
    prev_atr_median = out["atr"].expanding(min_periods=_MIN_HISTORY).median().shift(1)
    out["vol_regime"] = np.where(prev_atr > prev_atr_median, "high", "low")

    # =====================================================================
    # Session close flag
    # =====================================================================
    past_cutoff = time_vals >= t_entry_cutoff
    new_day = out["date"] != pd.Series(out["date"]).shift(1).values
    prev_past = pd.Series(past_cutoff).shift(1, fill_value=False).values
    out["session_close"] = past_cutoff & (~prev_past | new_day)

    # =====================================================================
    # Daily open / close / prev close
    # =====================================================================
    sess_mask = (time_vals >= t_session_start) & (time_vals < t_session_end)
    session_bars = out[sess_mask]
    _daily_open = session_bars.groupby("date")["open"].first()
    _daily_close = session_bars.groupby("date")["close"].last()

    out["daily_open"] = out["date"].map(_daily_open)
    out["daily_close"] = out["date"].map(_daily_close)

    # Previous day's close (shifted by 1 trading day, no lookahead)
    sorted_dates = sorted(_daily_close.index)
    prev_close_map = {}
    for i, d in enumerate(sorted_dates):
        if i == 0:
            prev_close_map[d] = np.nan
        else:
            prev_close_map[d] = _daily_close[sorted_dates[i - 1]]
    out["daily_prev_close"] = out["date"].map(prev_close_map)

    # =====================================================================
    # Gap % (session open vs previous session close)
    # =====================================================================
    out["gap_pct"] = np.where(
        out["daily_prev_close"] > 0,
        (out["daily_open"] - out["daily_prev_close"]) / out["daily_prev_close"] * 100,
        np.nan,
    )

    # =====================================================================
    # First-hour return (09:30-10:30)
    # =====================================================================
    in_fh = (time_vals >= t_session_start) & (time_vals < t_fh_end)
    fh_bars = out[in_fh]
    fh_open = fh_bars.groupby("date")["open"].first()
    fh_close = fh_bars.groupby("date")["close"].last()
    _fh_return = (fh_close - fh_open) / fh_open
    out["fh_return"] = out["date"].map(_fh_return)

    # =====================================================================
    # Early return (09:30-10:00, first 30 min)
    # =====================================================================
    in_early = (time_vals >= t_session_start) & (time_vals < t_early_end)
    early_bars = out[in_early]
    early_open = early_bars.groupby("date")["open"].first()
    early_close = early_bars.groupby("date")["close"].last()
    _early_return = (early_close - early_open) / early_open
    out["early_return"] = out["date"].map(_early_return)

    # =====================================================================
    # Time-of-day features
    # =====================================================================
    out["hour_of_day"] = out.index.hour
    out["minute_of_day"] = out.index.hour * 60 + out.index.minute
    out["day_of_week"] = out.index.dayofweek

    return out


def build_atws_features(
    df: pd.DataFrame,
    anchor_window: int = 20,
) -> pd.DataFrame:
    """Compute ATWS (Asymmetric Trailing-Window Shock) hypothesis features.

    Adds:
      anchor_mean   = EMA(close, span=anchor_window)
      rolling_std   = close.rolling(anchor_window).std()
      z_score       = (close - anchor_mean) / rolling_std
      signal_zscore = z_score.shift(1)   — entry uses prior bar's z (no look-ahead)
      target_price  = anchor_mean.shift(1) — TP anchor frozen at prior bar
                       (prevents intra-bar look-ahead on the limit-touch exit)

    See research/hypotheses/atws_zscore_reversion.md for the locked spec.
    """
    out = df.copy()
    out["anchor_mean"] = out["close"].ewm(
        span=anchor_window, adjust=False, min_periods=anchor_window,
    ).mean()
    out["rolling_std"] = out["close"].rolling(anchor_window).std()
    out["z_score"] = (out["close"] - out["anchor_mean"]) / out["rolling_std"]
    out["signal_zscore"] = out["z_score"].shift(1)
    out["target_price"] = out["anchor_mean"].shift(1)
    return out


def build_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VWAP-reversion hypothesis features (daily reset at session open).

    Adds:
      typical_price   = (high + low + close) / 3
      vwap            = cumulative-since-RTH-open of (typical_price * volume)
                        divided by cumulative-since-RTH-open of volume.
                        Resets at each session boundary (groupby on date).
      vwap_dev        = close - vwap
      intraday_std    = expanding std of vwap_dev within the day (min_periods=2)
      vwap_z          = vwap_dev / intraday_std
      signal_zscore   = vwap_z.shift(1)         — entry uses prior bar's z
      target_price    = vwap.shift(1)           — TP anchor frozen at prior bar

    Notes:
      - Globex / overnight bars are NOT excluded here; caller is expected to
        feed RTH-filtered data (the project's load_csv default does this).
      - Shifts cross day boundaries (signal_zscore at 09:30 references the
        prior day's 16:00 bar). Strategy gates entries at ≥10:30, so this
        boundary noise is invisible to the locked signal rule. Documented
        for future readers.

    See research/hypotheses/vwap_reversion.md for the locked spec.
    """
    out = df.copy()
    out["typical_price"] = (out["high"] + out["low"] + out["close"]) / 3.0

    date_key = pd.Series(out.index.date, index=out.index)
    tpv = out["typical_price"] * out["volume"]
    cum_tpv = tpv.groupby(date_key).cumsum()
    cum_v = out["volume"].groupby(date_key).cumsum()
    out["vwap"] = cum_tpv / cum_v

    out["vwap_dev"] = out["close"] - out["vwap"]
    out["intraday_std"] = (
        out["vwap_dev"]
        .groupby(date_key)
        .expanding(min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )
    out["vwap_z"] = out["vwap_dev"] / out["intraday_std"]
    out["signal_zscore"] = out["vwap_z"].shift(1)
    out["target_price"] = out["vwap"].shift(1)
    return out


def compute_fh_threshold(
    fh_return_series: pd.Series,
    percentile: float = 75.0,
) -> pd.Series:
    """Compute expanding percentile threshold on a daily FH return series.

    Uses only prior days (shift 1) — no lookahead.

    Parameters
    ----------
    fh_return_series : pd.Series
        Daily FH return series (index = date).
    percentile : float
        Percentile cutoff (default 75.0).

    Returns
    -------
    pd.Series
        Dynamic threshold per day (same index as input).
    """
    return fh_return_series.expanding(min_periods=_MIN_HISTORY).quantile(
        percentile / 100.0
    ).shift(1)


def get_date_bar_ranges(dates: np.ndarray, n: int) -> dict:
    """Map each unique date to (start_idx, end_idx) in the bar array.

    Parameters
    ----------
    dates : np.ndarray
        Array of date values (one per bar).
    n : int
        Total number of bars.

    Returns
    -------
    dict
        {date: (start_idx, end_idx)} for each trading day.
    """
    ranges = {}
    current_d = None
    start_i = 0
    for i in range(n):
        d = dates[i]
        if d != current_d:
            if current_d is not None:
                ranges[current_d] = (start_i, i)
            current_d = d
            start_i = i
    if current_d is not None:
        ranges[current_d] = (start_i, n)
    return ranges
