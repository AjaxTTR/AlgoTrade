"""
Regime-Filtered Momentum strategy (long only, gap-agnostic).

Edge: Strong first-hour return → continuation, filtered to favorable
      volatility/trend regime.
Source: ranked_edges.csv — FH_Up × Rest_Up (t=30.39, stability=0.946).
       Regime filters from gap_fh_regime_filtered research:
         - Dead zone (0.50-0.75% FH return) hurts profit factor
         - Low volatility periods produce weaker continuation
         - Extended trends (>1% above SMA20) reduce edge

Trade logic:
  1. First-hour return (09:30-10:30) >= expanding percentile threshold.
     No gap requirement — this strategy fires on ANY strong FH move.
  2. Dead zone filter: skip FH returns in the ambiguous range.
  3. Trend filter: skip when 20-day trend is overextended.
  4. Volatility filter: skip when 5-day vol is below expanding median.
  5. Enter long at 10:30.
  6. At most 1 trade per day. Long only.

All filters use only data available at 10:30 — no lookahead.

This strategy is independent — it does NOT suppress signals from other
strategies running on the same data.

Every strategy module must expose:
    def generate_signals(df: pd.DataFrame, **params) -> pd.DataFrame
"""

import numpy as np
import pandas as pd

from engine.features import build_features, compute_fh_threshold, get_date_bar_ranges


def generate_signals(
    df: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    fh_end: str = "10:30",
    entry_cutoff: str = "15:45",
    atr_period: int = 14,
    fh_percentile: float = 75.0,
    fh_dead_zone_lo: float = 0.50,
    fh_dead_zone_hi: float = 0.75,
    trend_20d_max: float = 1.0,
    vol_filter: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Generate regime-filtered momentum signals (gap-agnostic).

    Parameters
    ----------
    fh_percentile : float
        FH return must exceed this expanding percentile (default 75.0).
    fh_dead_zone_lo / fh_dead_zone_hi : float
        FH return percentage range to exclude (default 0.50-0.75%).
    trend_20d_max : float
        Max allowed 20-day trend (prev close vs SMA20, %). Default 1.0.
    vol_filter : bool
        If True, skip when 5-day rolling vol is below expanding median.
    """
    out = build_features(
        df, session_start=session_start, session_end=session_end,
        fh_end=fh_end, entry_cutoff=entry_cutoff, atr_period=atr_period,
    )
    n = len(out)
    time_vals = out.index.time
    t_start = pd.Timestamp(session_start).time()
    t_fh_end = pd.Timestamp(fh_end).time()
    t_session_end = pd.Timestamp(session_end).time()
    t_entry_cutoff = pd.Timestamp(entry_cutoff).time()

    # -- FH percentile threshold (no lookahead) --
    fh_daily = out.groupby("date")["fh_return"].first().dropna()
    dynamic_thresh = compute_fh_threshold(fh_daily, fh_percentile)

    # -- Regime features (all use prior-day data only) --
    sess_mask = (time_vals >= t_start) & (time_vals <= t_session_end)
    daily_close = out.loc[sess_mask].groupby(
        out.loc[sess_mask, "date"]
    )["close"].last()

    # 20-day trend: prev_close vs SMA20, shifted for no lookahead
    prev_close = daily_close.shift(1)
    sma20 = daily_close.rolling(20, min_periods=10).mean().shift(1)
    trend_20d = (prev_close - sma20) / sma20 * 100

    # 5-day rolling vol, shifted for no lookahead
    daily_ret = daily_close.pct_change()
    vol_5d = daily_ret.rolling(5, min_periods=3).std() * 100
    vol_5d_prev = vol_5d.shift(1)
    vol_5d_median = vol_5d_prev.expanding(min_periods=20).median()
    vol_above_median = vol_5d_prev > vol_5d_median

    # -- Classify qualifying days --
    signal_days = set()
    sorted_dates = sorted(fh_daily.index)

    for i, d in enumerate(sorted_dates):
        if i == 0:
            continue

        # FH return must exceed percentile threshold
        thresh = dynamic_thresh.get(d, np.nan)
        if pd.isna(thresh):
            continue
        fh_ret = fh_daily.get(d, np.nan)
        if pd.isna(fh_ret) or fh_ret < thresh:
            continue

        # Dead zone filter: skip ambiguous FH returns
        fh_ret_pct = fh_ret * 100
        if fh_dead_zone_lo <= fh_ret_pct <= fh_dead_zone_hi:
            continue

        # Trend filter: skip overextended trends
        t20 = trend_20d.get(d, np.nan)
        if not pd.isna(t20) and t20 > trend_20d_max:
            continue

        # Volatility filter: skip low-vol regimes
        if vol_filter:
            vol_ok = vol_above_median.get(d, np.nan)
            if pd.isna(vol_ok) or not vol_ok:
                continue

        signal_days.add(d)

    # -- Signal output columns --
    out["signal"] = 0
    out["signal_tier"] = 0
    out["entry_type"] = ""

    sig_col = out.columns.get_loc("signal")
    tier_col = out.columns.get_loc("signal_tier")
    et_col = out.columns.get_loc("entry_type")

    dates = out["date"].values
    date_bar_ranges = get_date_bar_ranges(dates, n)

    for d in sorted(signal_days):
        if d not in date_bar_ranges:
            continue
        d_start, d_end = date_bar_ranges[d]

        for i in range(d_start, d_end):
            t = time_vals[i]
            if t >= t_fh_end and t < t_entry_cutoff and t < t_session_end:
                out.iloc[i, sig_col] = 1
                out.iloc[i, tier_col] = 1
                out.iloc[i, et_col] = "regime_filtered_momentum"
                break

    out.drop(columns=["date"], inplace=True)

    # -- Validate --
    signal_bars = out[out["signal"] != 0]
    invalid_vals = set(out["signal"].unique()) - {0, 1}
    if invalid_vals:
        raise ValueError(f"signal contains invalid values: {invalid_vals}.")
    signals_per_day = signal_bars.groupby(signal_bars.index.date).size()
    multi = signals_per_day[signals_per_day > 1]
    if len(multi) > 0:
        raise ValueError(f"More than 1 signal on {len(multi)} day(s): {multi.index.tolist()[:5]}")

    return out
