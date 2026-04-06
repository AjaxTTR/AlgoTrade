"""
No-Gap Breakout strategy (long only).

Edge: No significant overnight gap + strong first-hour return → continuation.
Source: ranked_edges.csv — FH_Up × Rest_Up (t=30.39) filtered to no-gap context.

Trade logic:
  1. No-gap filter: |gap| <= gap_threshold_pct.
  2. First-hour return (09:30-10:30) >= expanding percentile threshold.
  3. Enter long at 10:30.
  4. At most 1 trade per day. Long only.

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
    gap_threshold_pct: float = 0.10,
    **kwargs,
) -> pd.DataFrame:
    """Generate long-only no-gap breakout signals."""
    out = build_features(
        df, session_start=session_start, session_end=session_end,
        fh_end=fh_end, entry_cutoff=entry_cutoff, atr_period=atr_period,
    )
    n = len(out)
    time_vals = out.index.time
    t_fh_end = pd.Timestamp(fh_end).time()
    t_session_end = pd.Timestamp(session_end).time()
    t_entry_cutoff = pd.Timestamp(entry_cutoff).time()

    # -- FH percentile threshold (no lookahead) --
    fh_daily = out.groupby("date")["fh_return"].first().dropna()
    dynamic_thresh = compute_fh_threshold(fh_daily, fh_percentile)

    # -- Classify qualifying days: no-gap + strong FH --
    signal_days = set()
    sorted_dates = sorted(fh_daily.index)

    for i, d in enumerate(sorted_dates):
        if i == 0:
            continue

        gap = out.loc[out["date"] == d, "gap_pct"].iloc[0]
        if pd.isna(gap) or abs(gap) > gap_threshold_pct:
            continue

        thresh = dynamic_thresh.get(d, np.nan)
        if pd.isna(thresh):
            continue

        fh_ret = fh_daily.get(d, np.nan)
        if pd.isna(fh_ret) or fh_ret < thresh:
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
                out.iloc[i, et_col] = "no_gap_breakout"
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
