"""
Combined First-Hour system (long only).

Merges two validated edges into one strategy:
  A. Gap-Up Continuation -- gap > +0.10%, FH return >= percentile threshold
  B. No-Gap FH Breakout  -- |gap| <= 0.10%, FH return >= percentile threshold

Priority: gap-up checked first; no-gap only if gap-up doesn't fire.
Max 1 trade per day. EOD exit (or stop/TP).

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
    stop_atr_multiple: float = 1.0,
    tp_atr_multiple: float = 1.0,
    fh_percentile: float = 75.0,
    gap_threshold_pct: float = 0.10,
    max_bars_in_trade: int = 999,
    **kwargs,
) -> pd.DataFrame:
    """Generate signals from combined gap-up continuation + no-gap breakout."""
    out = build_features(
        df, session_start=session_start, session_end=session_end,
        fh_end=fh_end, entry_cutoff=entry_cutoff, atr_period=atr_period,
    )
    n = len(out)
    time_vals = out.index.time
    t_fh_end = pd.Timestamp(fh_end).time()
    t_session_end = pd.Timestamp(session_end).time()
    t_entry_cutoff = pd.Timestamp(entry_cutoff).time()

    # -- FH percentile threshold --
    fh_daily = out.groupby("date")["fh_return"].first().dropna()
    dynamic_thresh = compute_fh_threshold(fh_daily, fh_percentile)

    # -- Classify days: gap-up continuation OR no-gap breakout --
    signal_days = {}  # date -> entry_type
    sorted_dates = sorted(fh_daily.index)

    for i, d in enumerate(sorted_dates):
        if i == 0:
            continue

        # FH threshold check (same for both signals)
        thresh = dynamic_thresh.get(d, np.nan)
        if pd.isna(thresh):
            continue
        fh_ret = fh_daily.get(d, np.nan)
        if pd.isna(fh_ret) or fh_ret < thresh:
            continue

        gap = out.loc[out["date"] == d, "gap_pct"].iloc[0]
        if pd.isna(gap):
            continue

        # Priority A: gap-up continuation
        if gap > gap_threshold_pct:
            signal_days[d] = "gap_up_continuation"
        # Priority B: no-gap breakout
        elif abs(gap) <= gap_threshold_pct:
            signal_days[d] = "no_gap_breakout"

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

    def _place_long(idx, entry_type, tier):
        atr_val = atrs[idx]
        if pd.isna(atr_val) or atr_val <= 0:
            return False
        px = closes[idx]
        out.iloc[idx, sig_col] = 1
        out.iloc[idx, stop_col] = px - atr_val * stop_atr_multiple
        out.iloc[idx, tp_col] = px + atr_val * tp_atr_multiple
        out.iloc[idx, sf_col] = 1.0
        out.iloc[idx, tier_col] = tier
        out.iloc[idx, et_col] = entry_type
        return True

    date_bar_ranges = get_date_bar_ranges(dates, n)

    for d in sorted(signal_days.keys()):
        if d not in date_bar_ranges:
            continue
        d_start, d_end = date_bar_ranges[d]

        entry_type = signal_days[d]
        tier = 1 if entry_type == "gap_up_continuation" else 2

        for i in range(d_start, d_end):
            t = time_vals[i]
            if t >= t_fh_end and t < t_entry_cutoff and t < t_session_end:
                _place_long(i, entry_type, tier)
                break

    out.drop(columns=["date"], inplace=True)

    # -- Validate --
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
