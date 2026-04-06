"""
Regime-filtered Gap-Up + FH Continuation strategy (long only).

Wraps gap_fh_continuation and adds pre-trade regime filters:
  1. Dead zone:        skip FH returns between fh_dead_zone_lo and fh_dead_zone_hi
  2. Trend extension:  skip when 20-day trend > trend_20d_max above MA
  3. Low volatility:   skip when 5-day rolling vol is below expanding median

All filters use only data available at 10:30 -- no lookahead.
Core signal logic is unchanged.

Every strategy module must expose:
    def generate_signals(df: pd.DataFrame, **params) -> pd.DataFrame
"""

import pandas as pd

from strategies.gap_fh_continuation import generate_signals as _base_generate_signals


def generate_signals(
    df: pd.DataFrame,
    # Regime filter params
    fh_dead_zone_lo: float = 0.50,
    fh_dead_zone_hi: float = 0.75,
    trend_20d_max: float = 1.0,
    vol_filter: bool = True,
    # Pass-through to base strategy
    **kwargs,
) -> pd.DataFrame:
    """Generate regime-filtered gap-up + FH continuation signals.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex (15-min bars expected).
    fh_dead_zone_lo / fh_dead_zone_hi : float
        FH return percentage range to exclude (default 0.50-0.75%).
    trend_20d_max : float
        Max allowed 20-day trend (prev close vs SMA20, in %). Default 1.0.
    vol_filter : bool
        If True, skip trades when 5-day rolling vol is below expanding
        median (default True).
    **kwargs
        Passed through to gap_fh_continuation.generate_signals().
    """
    # Step 1: Get base signals (uses build_features internally)
    out = _base_generate_signals(df, **kwargs)

    # Step 2: Compute daily regime features using only pre-trade data
    time_vals = out.index.time
    t_start = pd.Timestamp("09:30").time()
    t_end = pd.Timestamp("16:00").time()

    out["_date"] = out.index.date

    # Daily closes from session bars
    sess_mask = (time_vals >= t_start) & (time_vals <= t_end)
    daily_close = out.loc[sess_mask].groupby(out.loc[sess_mask, "_date"])["close"].last()

    # --- 20-day trend (prev_close vs SMA20, shifted for no lookahead) ---
    prev_close = daily_close.shift(1)
    sma20 = daily_close.rolling(20, min_periods=10).mean().shift(1)
    trend_20d = (prev_close - sma20) / sma20 * 100
    out["regime_trend_20d"] = out["_date"].map(trend_20d)

    # --- 5-day rolling vol (shifted for no lookahead) ---
    daily_ret = daily_close.pct_change()
    vol_5d = daily_ret.rolling(5, min_periods=3).std() * 100
    vol_5d_prev = vol_5d.shift(1)
    vol_5d_median = vol_5d_prev.expanding(min_periods=20).median()
    above_median = vol_5d_prev > vol_5d_median

    out["regime_vol_5d"] = out["_date"].map(vol_5d_prev)
    out["regime_vol_above_median"] = out["_date"].map(above_median)

    # Step 3: Apply regime filters -- zero out signals that fail
    signal_idx = out.index[out["signal"] != 0]

    if len(signal_idx) > 0:
        # Filter 1: Dead zone
        fh_ret_pct = out.loc[signal_idx, "fh_return"] * 100
        dead_zone = (fh_ret_pct >= fh_dead_zone_lo) & (fh_ret_pct <= fh_dead_zone_hi)

        # Filter 2: Trend extension
        trend = out.loc[signal_idx, "regime_trend_20d"]
        trend_extended = trend > trend_20d_max

        # Filter 3: Low volatility
        if vol_filter:
            low_vol = out.loc[signal_idx, "regime_vol_above_median"] != True
        else:
            low_vol = pd.Series(False, index=signal_idx)

        # Combined: skip if ANY filter triggers
        skip = dead_zone | trend_extended | low_vol
        skip_idx = skip[skip].index

        out.loc[skip_idx, "signal"] = 0
        out.loc[skip_idx, "signal_tier"] = 0
        out.loc[skip_idx, "entry_type"] = ""

    out.drop(columns=["_date"], inplace=True)

    return out
