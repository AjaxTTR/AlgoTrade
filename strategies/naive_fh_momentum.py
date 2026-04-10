"""
Naive first-hour momentum — maximally stripped-down baseline.

Single rule: if FH return (09:30-10:30) > 0, enter long at the first
bar at/after 10:30. Exit via the backtester's stop / TP / session-close.

NO filters: no percentile, no gap, no regime, no dead zone.

Used as a baseline to determine whether the filters in
gap_momentum / regime_filtered_momentum add real value, or whether
they are curve-fit selection on top of an already-existing edge.

If this naive version is profitable, the underlying edge is real and
the filters are refinements.  If it dies, the headline edge is being
manufactured by the filter combinations and we are curve-fitting.
"""

import pandas as pd

from engine.features import build_features, get_date_bar_ranges


def generate_signals(
    df: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    fh_end: str = "10:30",
    entry_cutoff: str = "15:45",
    atr_period: int = 14,
    **kwargs,
) -> pd.DataFrame:
    """Generate naive long-only FH-up signals (no filters)."""
    out = build_features(
        df, session_start=session_start, session_end=session_end,
        fh_end=fh_end, entry_cutoff=entry_cutoff, atr_period=atr_period,
    )
    n = len(out)
    time_vals = out.index.time
    t_fh_end = pd.Timestamp(fh_end).time()
    t_session_end = pd.Timestamp(session_end).time()
    t_entry_cutoff = pd.Timestamp(entry_cutoff).time()

    out["signal"] = 0
    out["signal_tier"] = 0
    out["entry_type"] = ""

    sig_col = out.columns.get_loc("signal")
    tier_col = out.columns.get_loc("signal_tier")
    et_col = out.columns.get_loc("entry_type")

    # Single rule: any day with positive FH return
    fh_daily = out.groupby("date")["fh_return"].first().dropna()
    signal_days = set(fh_daily[fh_daily > 0].index)

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
                out.iloc[i, et_col] = "naive_fh_momentum"
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
        raise ValueError(f"More than 1 signal on {len(multi)} day(s).")

    return out
