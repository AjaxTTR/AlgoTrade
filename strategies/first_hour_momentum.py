"""
First-Hour Momentum strategy (long only, edge-filtered).

Edge: when the first 30 minutes (09:30-10:00) show a strong positive move,
the rest of the day tends to continue upward.

Source: ranked_edges.csv -- FH_Up -> Rest_Up (t=30.38, stability=0.946).

Trade logic:
  1. Compute expanding percentile of early_return (09:30-10:00)
     from prior days only -- positive returns only, no abs().
  2. Compute ATR percentile as volatility regime filter.
  3. A day qualifies only if early_return exceeds the dynamic threshold
     AND ATR percentile >= atr_percentile_min.
  4. Enter long at 10:00 (Tier 1) with full size. No fallback entry.
  5. At most 1 trade per day. Long only.

Every strategy module must expose:
    def generate_signals(df: pd.DataFrame, **params) -> pd.DataFrame
"""

import numpy as np
import pandas as pd

from engine.features import build_features, get_date_bar_ranges


def generate_signals(
    df: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "16:00",
    fh_end: str = "10:30",
    early_end: str = "10:00",
    entry_cutoff: str = "15:45",
    fh_threshold: float = 0.0,
    atr_period: int = 14,
    stop_atr_multiple: float = 1.5,
    tp_atr_multiple: float = 2.0,
    fh_percentile: float = 80.0,
    atr_percentile_min: float = 60.0,
    **kwargs,
) -> pd.DataFrame:
    """Generate long-only first-hour momentum signals.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex (15-min bars expected).
    session_start / session_end : str
        RTH session boundaries.
    fh_end : str
        End of the full first-hour window (default "10:30").
    early_end : str
        End of the early observation window (default "10:00").
    entry_cutoff : str
        Latest time to enter a new trade (default "15:45").
    fh_threshold : float
        Hard minimum floor for early_return (default 0.0).
    atr_period : int
        Lookback for Average True Range (default 14).
    stop_atr_multiple : float
        Stop distance as ATR multiple (default 1.5).
    tp_atr_multiple : float
        Take-profit distance as ATR multiple (default 2.0).
    fh_percentile : float
        Percentile of early_return distribution used as dynamic threshold
        (default 80.0 = only top 20% qualify). No lookahead.
    atr_percentile_min : float
        Minimum ATR percentile (0-100) required to trade (default 60.0).

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns: signal, stop_price, tp_price,
        session_close, atr, fh_return, size_factor, signal_tier,
        entry_type, early_return_threshold, is_top_percentile,
        fh_rank_percentile, atr_percentile, is_high_vol.
    """
    out = build_features(
        df, session_start=session_start, session_end=session_end,
        fh_end=fh_end, early_end=early_end, entry_cutoff=entry_cutoff,
        atr_period=atr_period,
    )
    n = len(out)
    time_vals = out.index.time
    t_session_start = pd.Timestamp(session_start).time()
    t_early_end = pd.Timestamp(early_end).time()
    t_session_end = pd.Timestamp(session_end).time()
    t_entry_cutoff = pd.Timestamp(entry_cutoff).time()

    _MIN_HISTORY = 20

    # -- is_high_vol from pre-computed atr_percentile --
    out["is_high_vol"] = out["atr_percentile"] >= atr_percentile_min

    # -- Daily ATR percentile: use the last bar before entry (just before 10:00) --
    pre_entry = out[(time_vals >= t_session_start) & (time_vals < t_early_end)]
    daily_atr_pctile = pre_entry.groupby("date")["atr_percentile"].last()

    # -- Early return expanding percentile threshold (no lookahead) --
    early_daily = out.groupby("date")["early_return"].first().dropna()
    dynamic_thresh = early_daily.expanding(min_periods=_MIN_HISTORY).quantile(
        fh_percentile / 100.0
    ).shift(1)

    out["early_return_threshold"] = out["date"].map(dynamic_thresh)

    # -- Debug: rank percentile of today's early_return vs prior days --
    rank_pctile = early_daily.expanding(min_periods=_MIN_HISTORY).rank(pct=True) * 100.0
    out["fh_rank_percentile"] = out["date"].map(rank_pctile)

    # -- Classify qualifying days --
    out["is_top_percentile"] = False
    early_days = set()

    for d in early_daily.index:
        thresh = dynamic_thresh.get(d, np.nan)
        if pd.isna(thresh):
            continue
        eff_thresh = max(thresh, fh_threshold)

        er = early_daily.get(d, np.nan)
        if pd.isna(er) or er < eff_thresh:
            continue

        atr_pct = daily_atr_pctile.get(d, np.nan)
        if pd.isna(atr_pct) or atr_pct < atr_percentile_min:
            continue

        early_days.add(d)

    is_top = out["date"].isin(early_days)
    out.loc[is_top, "is_top_percentile"] = True

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
        out.iloc[idx, et_col] = "early"
        return True

    date_bar_ranges = get_date_bar_ranges(dates, n)

    for d in sorted(early_days):
        if d not in date_bar_ranges:
            continue
        d_start, d_end = date_bar_ranges[d]

        for i in range(d_start, d_end):
            t = time_vals[i]
            if t >= t_early_end and t < t_entry_cutoff and t < t_session_end:
                _place_long(i)
                break

    out.drop(columns=["date"], inplace=True)

    # -- Validate --
    signal_bars = out[out["signal"] != 0]
    invalid_vals = set(out["signal"].unique()) - {0, 1}
    if invalid_vals:
        raise ValueError(f"signal contains invalid values: {invalid_vals}. Expected {{0, 1}}.")
    if signal_bars["signal"].isna().any():
        raise ValueError("signal column contains NaN on signal bars.")
    if signal_bars["stop_price"].isna().any():
        bad = signal_bars[signal_bars["stop_price"].isna()].index.tolist()
        raise ValueError(f"stop_price is NaN on {len(bad)} signal bar(s): {bad[:5]}")
    signals_per_day = signal_bars.groupby(signal_bars.index.date).size()
    multi = signals_per_day[signals_per_day > 1]
    if len(multi) > 0:
        raise ValueError(f"More than 1 signal on {len(multi)} day(s): {multi.index.tolist()[:5]}")

    return out
