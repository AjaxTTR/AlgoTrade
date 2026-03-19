"""
Feature engineering for intraday futures research.

Generates a comprehensive set of bar-level features from OHLCV data for
edge discovery and strategy research.  Every feature uses only past and
current bar data — no lookahead bias.

Usage:
    from engine.data_loader import load_csv
    from engine.feature_engineering import add_features

    df = load_csv("data/nq_15m_data.csv")
    df = add_features(df)
"""

import logging
from datetime import time as dt_time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Price structure
# ---------------------------------------------------------------------------

def _add_returns(df: pd.DataFrame) -> None:
    """Lagged log returns over multiple horizons."""
    c = df["close"]
    for n in (1, 3, 5):
        df[f"return_{n}"] = np.log(c / c.shift(n))


def _add_rolling_stats(
    df: pd.DataFrame,
    window: int = 20,
) -> None:
    """Rolling mean and std of close prices (window bars back, no lookahead)."""
    c = df["close"]
    df["rolling_mean"] = c.rolling(window, min_periods=window).mean()
    df["rolling_std"] = c.rolling(window, min_periods=window).std()


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def _add_atr(df: pd.DataFrame, period: int = 14) -> None:
    """Average True Range — uses prior bars only (shifted high/low/close)."""
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(period, min_periods=period).mean()


def _add_atr_percentile(
    df: pd.DataFrame,
    lookback: int = 100,
) -> None:
    """Rolling percentile rank of current ATR within its own history.

    Requires ``atr`` column (call ``_add_atr`` first).
    Result is 0-100 — e.g. 90 means current ATR is higher than 90% of
    the past *lookback* ATR readings.
    """
    atr = df["atr"]

    def _pctile(window):
        if np.isnan(window.iloc[-1]):
            return np.nan
        valid = window.dropna()
        if len(valid) < 2:
            return np.nan
        return (valid < valid.iloc[-1]).sum() / (len(valid) - 1) * 100

    df["atr_percentile"] = atr.rolling(
        lookback, min_periods=lookback // 2,
    ).apply(_pctile, raw=False)


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------

def _add_time_features(df: pd.DataFrame) -> None:
    """Calendar/clock features extracted from the index.

    - ``hour_of_day``: 0-23
    - ``minute_bucket``: 15-min bucket within the hour (0-3 for 15-min bars)
    - ``day_of_week``: 0=Monday … 4=Friday
    """
    df["hour_of_day"] = df.index.hour
    df["minute_bucket"] = df.index.minute // 15
    df["day_of_week"] = df.index.dayofweek


# ---------------------------------------------------------------------------
# Positional
# ---------------------------------------------------------------------------

def _add_vwap_distance(df: pd.DataFrame) -> None:
    """Intraday VWAP and signed distance from it.

    VWAP resets each calendar day and uses only data up to the current bar
    (cumulative sums, no lookahead).

    ``distance_from_vwap`` is expressed in points (close - vwap).
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)
    tp_vol = typical * vol

    dates = df.index.date
    day_change = np.empty(len(df), dtype=bool)
    day_change[0] = True
    day_change[1:] = dates[1:] != dates[:-1]

    # Cumulative sums that reset each day — forward pass
    cum_tp_vol = np.empty(len(df), dtype=np.float64)
    cum_vol = np.empty(len(df), dtype=np.float64)

    tp_vol_arr = tp_vol.values
    vol_arr = vol.values

    for i in range(len(df)):
        if day_change[i]:
            cum_tp_vol[i] = tp_vol_arr[i] if not np.isnan(tp_vol_arr[i]) else 0.0
            cum_vol[i] = vol_arr[i] if not np.isnan(vol_arr[i]) else 0.0
        else:
            prev_tp = cum_tp_vol[i - 1]
            prev_v = cum_vol[i - 1]
            cum_tp_vol[i] = prev_tp + (tp_vol_arr[i] if not np.isnan(tp_vol_arr[i]) else 0.0)
            cum_vol[i] = prev_v + (vol_arr[i] if not np.isnan(vol_arr[i]) else 0.0)

    vwap = cum_tp_vol / (cum_vol + _EPS)
    df["vwap"] = vwap
    df["distance_from_vwap"] = df["close"].values - vwap


def _add_position_in_day_range(df: pd.DataFrame) -> None:
    """Where the current close sits within today's running high-low range.

    0 = at the day low, 1 = at the day high.  Uses only data up to the
    current bar (expanding max/min per day, no lookahead).
    """
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    dates = df.index.date

    day_change = np.empty(len(df), dtype=bool)
    day_change[0] = True
    day_change[1:] = dates[1:] != dates[:-1]

    day_high = np.empty(len(df), dtype=np.float64)
    day_low = np.empty(len(df), dtype=np.float64)

    for i in range(len(df)):
        if day_change[i]:
            day_high[i] = highs[i]
            day_low[i] = lows[i]
        else:
            day_high[i] = max(day_high[i - 1], highs[i])
            day_low[i] = min(day_low[i - 1], lows[i])

    day_range = day_high - day_low
    df["position_in_day_range"] = np.where(
        day_range > _EPS,
        (closes - day_low) / day_range,
        0.5,
    )


# ---------------------------------------------------------------------------
# Session features
# ---------------------------------------------------------------------------

def _add_gap_from_prev_close(df: pd.DataFrame) -> None:
    """Gap between this session's first bar open and the previous bar's close.

    Computed at every bar but only meaningful on session-open bars.
    For intrabar use the value is forward-filled within each day so every
    bar in the day carries its session's opening gap.
    """
    opens = df["open"].values
    prev_close = df["close"].shift(1).values
    dates = df.index.date

    day_change = np.empty(len(df), dtype=bool)
    day_change[0] = True
    day_change[1:] = dates[1:] != dates[:-1]

    gap = np.full(len(df), np.nan)
    for i in range(len(df)):
        if day_change[i]:
            gap[i] = opens[i] - prev_close[i] if not np.isnan(prev_close[i]) else np.nan

    # Forward fill within each day
    df["gap_from_prev_close"] = pd.Series(gap, index=df.index).ffill()


def _add_session_range_and_trend(df: pd.DataFrame) -> None:
    """Running session range and trend using only past/current data.

    - ``session_range``: day_high - day_low up to the current bar (points).
    - ``session_trend``: (close - day_open) / session_range.
      Positive = bullish bias, negative = bearish.  Bounded roughly [-1, 1].
    """
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values
    dates = df.index.date

    day_change = np.empty(len(df), dtype=bool)
    day_change[0] = True
    day_change[1:] = dates[1:] != dates[:-1]

    day_high = np.empty(len(df), dtype=np.float64)
    day_low = np.empty(len(df), dtype=np.float64)
    day_open = np.empty(len(df), dtype=np.float64)

    for i in range(len(df)):
        if day_change[i]:
            day_high[i] = highs[i]
            day_low[i] = lows[i]
            day_open[i] = opens[i]
        else:
            day_high[i] = max(day_high[i - 1], highs[i])
            day_low[i] = min(day_low[i - 1], lows[i])
            day_open[i] = day_open[i - 1]

    sess_range = day_high - day_low
    df["session_range"] = sess_range
    df["session_trend"] = np.where(
        sess_range > _EPS,
        (closes - day_open) / sess_range,
        0.0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    atr_pctile_lookback: int = 100,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """Add all research features to an OHLCV DataFrame.

    Appends columns in-place and returns the same DataFrame.  No rows are
    dropped — early bars where lookback windows are insufficient will have
    NaN in the corresponding feature columns.

    **No lookahead bias**: every feature is computed from data at or before
    the current bar only.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex and columns: open, high, low, close, volume.
    atr_period : int
        ATR averaging period (default 14).
    atr_pctile_lookback : int
        Window for ATR percentile ranking (default 100).
    rolling_window : int
        Window for rolling mean/std (default 20).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with feature columns appended.

    Features added
    --------------
    **Price structure** (3):
        return_1, return_3, return_5, rolling_mean, rolling_std

    **Volatility** (2):
        atr, atr_percentile

    **Time** (3):
        hour_of_day, minute_bucket, day_of_week

    **Positional** (3):
        vwap, distance_from_vwap, position_in_day_range

    **Session** (3):
        gap_from_prev_close, session_range, session_trend
    """
    n_before = len(df.columns)

    # Price structure
    _add_returns(df)
    _add_rolling_stats(df, window=rolling_window)

    # Volatility
    _add_atr(df, period=atr_period)
    _add_atr_percentile(df, lookback=atr_pctile_lookback)

    # Time
    _add_time_features(df)

    # Positional
    _add_vwap_distance(df)
    _add_position_in_day_range(df)

    # Session
    _add_gap_from_prev_close(df)
    _add_session_range_and_trend(df)

    n_added = len(df.columns) - n_before
    n_rows = len(df)
    # Count rows where all new features are available (no NaN warm-up)
    feature_cols = [c for c in df.columns if c not in (
        "open", "high", "low", "close", "volume", "is_rth",
        "session_type", "session_id", "session_open", "session_high", "session_low",
    )]
    n_complete = int(df[feature_cols].dropna().shape[0])

    logger.info(
        "Feature engineering: %d features added, %d/%d rows complete (%.1f%%)",
        n_added, n_complete, n_rows, n_complete / n_rows * 100 if n_rows > 0 else 0,
    )

    return df
