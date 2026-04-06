"""
First-Hour Momentum strategy (long only, edge-filtered).

Edge: when the first 30 minutes (09:30-10:00) show a strong positive move,
the rest of the day tends to continue upward.

Source: ranked_edges.csv -- FH_Up -> Rest_Up (t=30.38, stability=0.946).

Trade logic:
  1. Compute expanding 80th percentile of early_return (09:30-10:00)
     from prior days only — positive returns only, no abs().
  2. Compute ATR percentile (expanding rank) as volatility regime filter.
  3. A day qualifies only if early_return exceeds the dynamic threshold
     AND ATR percentile >= 60 (sufficient volatility).
  4. Enter long at 10:00 (Tier 1) with full size. No fallback entry.
  5. At most 1 trade per day. Long only.

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
        (default 80.0 = only top 20% of early moves qualify).
        Computed from prior days only — no lookahead.
    atr_percentile_min : float
        Minimum ATR percentile (0-100) required to trade (default 60.0).
        Filters out low-volatility regimes where momentum follow-through
        is weaker.

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns: signal, stop_price, tp_price,
        session_close, atr, fh_return, size_factor, signal_tier,
        entry_type, early_return_threshold, is_top_percentile,
        fh_rank_percentile, atr_percentile, is_high_vol.
    """
    out = df.copy()
    n = len(out)

    # -- ATR (for stop/TP sizing) --
    prev_close = out["close"].shift(1)
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum(
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ),
    )
    out["atr"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

    # -- ATR percentile (volatility regime filter) --
    # Expanding rank of current ATR vs all prior bars, scaled to 0-100.
    out["atr_percentile"] = out["atr"].expanding(min_periods=atr_period).rank(pct=True) * 100.0
    out["is_high_vol"] = out["atr_percentile"] >= atr_percentile_min

    # -- Time boundaries --
    time = out.index.time
    t_session_start = pd.Timestamp(session_start).time()
    t_early_end = pd.Timestamp(early_end).time()
    t_session_end = pd.Timestamp(session_end).time()
    t_entry_cutoff = pd.Timestamp(entry_cutoff).time()

    out["date"] = out.index.date

    # -- Daily ATR percentile: use the last bar before entry (just before 10:00) --
    # This is the ATR percentile the strategy sees at decision time.
    pre_entry = out[(time >= t_session_start) & (time < t_early_end)]
    daily_atr_pctile = pre_entry.groupby("date")["atr_percentile"].last()

    # -- Session close flag (EOD backstop at entry_cutoff) --
    past_cutoff = time >= t_entry_cutoff
    new_day = out["date"] != pd.Series(out["date"]).shift(1).values
    prev_past = pd.Series(past_cutoff).shift(1, fill_value=False).values
    out["session_close"] = past_cutoff & (~prev_past | new_day)

    # -- First-hour return (full 60 min, 09:30-10:30) — kept as info column --
    in_fh = (time >= t_session_start) & (time < pd.Timestamp(fh_end).time())
    fh_bars = out[in_fh]
    fh_open = fh_bars.groupby("date")["open"].first()
    fh_close = fh_bars.groupby("date")["close"].last()
    fh_return = (fh_close - fh_open) / fh_open
    fh_return.name = "fh_return"
    out["fh_return"] = out["date"].map(fh_return)

    # -- Early return (first 30 min, 09:30-10:00) --
    # This is the SOLE basis for threshold and filtering.
    in_early = (time >= t_session_start) & (time < t_early_end)
    early_bars = out[in_early]
    early_open = early_bars.groupby("date")["open"].first()
    early_close = early_bars.groupby("date")["close"].last()
    early_return = (early_close - early_open) / early_open

    # -- Expanding percentile threshold on early_return (no lookahead) --
    # No abs() — only positive early returns matter for a long-only strategy.
    # shift(1) ensures today's return is excluded from its own threshold.
    _MIN_HISTORY = 20
    daily_dates = early_return.index

    dynamic_thresh = early_return.expanding(min_periods=_MIN_HISTORY).quantile(
        fh_percentile / 100.0
    ).shift(1)

    # Map daily threshold to every bar
    out["early_return_threshold"] = out["date"].map(dynamic_thresh)

    # -- Debug: rank percentile of today's early_return vs prior days --
    # Approximate — uses expanding rank which includes today, acceptable for debug.
    rank_pctile = early_return.expanding(min_periods=_MIN_HISTORY).rank(pct=True) * 100.0
    out["fh_rank_percentile"] = out["date"].map(rank_pctile)

    # -- Classify days --
    # A day qualifies only if:
    #   1. early_return exceeds the dynamic threshold AND hard floor
    #   2. ATR percentile at decision time >= atr_percentile_min
    out["is_top_percentile"] = False
    early_days = set()

    for d in daily_dates:
        thresh = dynamic_thresh.get(d, np.nan)
        if pd.isna(thresh):
            continue
        eff_thresh = max(thresh, fh_threshold)

        er = early_return.get(d, np.nan)
        if pd.isna(er) or er < eff_thresh:
            continue

        atr_pct = daily_atr_pctile.get(d, np.nan)
        if pd.isna(atr_pct) or atr_pct < atr_percentile_min:
            continue

        early_days.add(d)

    # Mark qualifying days on all their bars
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
        """Place a Tier 1 long signal at bar idx. Returns True if successful."""
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

    # -- Group bars by date for fast iteration --
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

    # -- Generate signals: Tier 1 only, 1 trade per day max --
    for d in sorted(early_days):
        if d not in date_bar_ranges:
            continue
        d_start, d_end = date_bar_ranges[d]

        # Enter at the first bar at or after 10:00
        for i in range(d_start, d_end):
            t = time[i]
            if t >= t_early_end and t < t_entry_cutoff and t < t_session_end:
                _place_long(i)
                break

    # Clean up working columns
    out.drop(columns=["date"], inplace=True)

    # -- Validate output --
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
