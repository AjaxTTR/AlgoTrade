"""
First-Hour Mean Reversion strategy (long only).

Edge: Extreme first-hour selloff → rest-of-day reversal.
Source: ranked_edges.csv — FH_Down × Rest_Up (t=7.95, stability=0.44).
       refined_edge_results.csv — Large_Drop in high-ATR context had
       the highest edge score (0.78) among reversal sub-conditions.
       edge_analysis: extreme down 1st hour (bottom 10%) tested for reversal.

Trade logic:
  1. First-hour return (09:30-10:30) must be negative and below the
     expanding Nth percentile (bottom tail = extreme selloff).
  2. High-vol filter: ATR percentile >= threshold (reversal edge is
     strongest in elevated volatility — refined_edge Large_Drop × High_ATR).
  3. Enter long at 10:30 (betting on mean reversion after the selloff).
  4. At most 1 trade per day. Long only.

This is the weakest of the four decomposed edges (stability=0.44,
no sub-condition passed FDR after correction). Use with conservative
sizing in any portfolio allocation.

This strategy is independent — it does NOT suppress signals from other
strategies running on the same data.

Every strategy module must expose:
    def generate_signals(df: pd.DataFrame, **params) -> pd.DataFrame
"""

import numpy as np
import pandas as pd

from engine.features import build_features, get_date_bar_ranges


_MIN_HISTORY = 20


def generate_signals(
    df: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    fh_end: str = "10:30",
    entry_cutoff: str = "15:45",
    atr_period: int = 14,
    fh_percentile_lo: float = 20.0,
    atr_percentile_min: float = 50.0,
    **kwargs,
) -> pd.DataFrame:
    """Generate long-only first-hour mean reversion signals.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex (15-min bars expected).
    fh_percentile_lo : float
        FH return must be below this expanding percentile to qualify
        (default 20.0 = bottom 20% of FH returns). Lower = more selective.
    atr_percentile_min : float
        Minimum ATR percentile required (default 50.0). The reversal edge
        is strongest in higher-volatility regimes.
    """
    out = build_features(
        df, session_start=session_start, session_end=session_end,
        fh_end=fh_end, entry_cutoff=entry_cutoff, atr_period=atr_period,
    )
    n = len(out)
    time_vals = out.index.time
    t_session_start = pd.Timestamp(session_start).time()
    t_fh_end = pd.Timestamp(fh_end).time()
    t_session_end = pd.Timestamp(session_end).time()
    t_entry_cutoff = pd.Timestamp(entry_cutoff).time()

    # -- Daily FH return --
    fh_daily = out.groupby("date")["fh_return"].first().dropna()

    # -- Expanding low-percentile threshold (no lookahead) --
    # This gives us the Nth percentile of all prior FH returns.
    # A day qualifies if its FH return is BELOW this threshold (extreme down).
    dynamic_lo_thresh = fh_daily.expanding(min_periods=_MIN_HISTORY).quantile(
        fh_percentile_lo / 100.0
    ).shift(1)

    # -- Daily ATR percentile (use last bar before 10:30) --
    pre_entry = out[(time_vals >= t_session_start) & (time_vals < t_fh_end)]
    daily_atr_pctile = pre_entry.groupby("date")["atr_percentile"].last()

    # -- Classify qualifying days: extreme FH down + high vol --
    signal_days = set()
    sorted_dates = sorted(fh_daily.index)

    for i, d in enumerate(sorted_dates):
        if i == 0:
            continue

        # FH return must be negative
        fh_ret = fh_daily.get(d, np.nan)
        if pd.isna(fh_ret) or fh_ret >= 0:
            continue

        # Must be below the low-percentile threshold (extreme selloff)
        lo_thresh = dynamic_lo_thresh.get(d, np.nan)
        if pd.isna(lo_thresh):
            continue
        if fh_ret > lo_thresh:
            continue

        # Volatility filter: ATR percentile must be elevated
        atr_pct = daily_atr_pctile.get(d, np.nan)
        if pd.isna(atr_pct) or atr_pct < atr_percentile_min:
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
                out.iloc[i, et_col] = "fh_mean_reversion"
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
