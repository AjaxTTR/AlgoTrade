"""
Data loader module for OHLCV CSV data.

Handles reading, validation, timezone conversion, preprocessing,
and session segmentation of futures market data for the backtesting engine.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import time as dt_time

logger = logging.getLogger(__name__)


_TIMESTAMP_ALIASES = ("timestamp", "time", "datetime", "ts_event", "date")


def load_csv(
    filepath: str,
    target_tz: str = "America/New_York",
    session_filter: bool = False,
    session_start: str = "09:30",
    session_end: str = "16:00",
) -> pd.DataFrame:
    """Load OHLCV data from CSV and return a clean, timezone-aware DataFrame.

    Processing pipeline:
        1. Read CSV and auto-detect the timestamp column
        2. Validate required columns exist
        3. Drop NaN OHLC rows
        4. Remove duplicate timestamps
        5. Localise to UTC, convert to target timezone
        6. Enforce OHLC integrity (high >= low, open/close within range)
        7. Sort ascending by timestamp
        8. Tag rows with ``is_rth`` (always added for intraday data)
        9. Optionally filter to RTH rows only

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    target_tz : str
        IANA timezone to convert timestamps into (default: America/New_York).
    session_filter : bool
        If True, keep only rows within the session window (default False).
    session_start : str
        Session start time as HH:MM (default "09:30").
    session_end : str
        Session end time as HH:MM (default "16:00").

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: open, high, low, close, volume, is_rth
        and a DatetimeIndex named ``timestamp`` in *target_tz*, sorted ascending.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing, no recognised timestamp column is
        found, or the file contains no valid rows.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Normalise column names to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # --- Auto-detect timestamp column ---
    ts_col = None
    for alias in _TIMESTAMP_ALIASES:
        if alias in df.columns:
            ts_col = alias
            break

    if ts_col is None:
        raise ValueError(
            f"No recognised timestamp column found. "
            f"Expected one of: {_TIMESTAMP_ALIASES}. "
            f"Got: {list(df.columns)}"
        )

    df[ts_col] = pd.to_datetime(df[ts_col])
    df.rename(columns={ts_col: "timestamp"}, inplace=True)
    df.set_index("timestamp", inplace=True)

    # Cast OHLCV to float (handles mixed types after generic read)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_rows = len(df)

    # --- Drop rows with NaN OHLC values ---
    nan_mask = df[["open", "high", "low", "close"]].isna().any(axis=1)
    nan_count = nan_mask.sum()
    if nan_count:
        logger.warning("Dropped %d rows with NaN OHLC values", nan_count)
        df = df.loc[~nan_mask]

    # --- Remove duplicate timestamps (keep first occurrence) ---
    dup_mask = df.index.duplicated(keep="first")
    dup_count = dup_mask.sum()
    if dup_count:
        logger.warning("Dropped %d duplicate timestamps", dup_count)
        df = df.loc[~dup_mask]

    # --- Timezone handling: assume UTC input, convert to target ---
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(target_tz)

    # --- OHLC integrity checks (vectorised) ---
    bad_hl = df["high"] < df["low"]
    bad_open = (df["open"] > df["high"]) | (df["open"] < df["low"])
    bad_close = (df["close"] > df["high"]) | (df["close"] < df["low"])
    integrity_mask = bad_hl | bad_open | bad_close

    integrity_count = integrity_mask.sum()
    if integrity_count:
        logger.warning(
            "Dropped %d rows failing OHLC integrity "
            "(high<low: %d, open out of range: %d, close out of range: %d)",
            integrity_count,
            bad_hl.sum(),
            bad_open.sum(),
            bad_close.sum(),
        )
        df = df.loc[~integrity_mask]

    # --- Final sort and naming ---
    df.sort_index(inplace=True)
    df.index.name = "timestamp"

    # --- Session tagging and filtering ---
    # Detect intraday data: multiple rows share the same calendar date
    is_intraday = df.index.date[0] == df.index.date[min(1, len(df) - 1)] if len(df) > 1 else False
    if is_intraday:
        t_start = pd.Timestamp(session_start).time()
        t_end = pd.Timestamp(session_end).time()
        bar_times = df.index.time
        df["is_rth"] = (bar_times >= t_start) & (bar_times <= t_end)

        if session_filter:
            pre_filter = len(df)
            df = df[df["is_rth"]].copy()
            removed = pre_filter - len(df)
            logger.info(
                "Session filter %s–%s: kept %d rows, removed %d",
                session_start, session_end, len(df), removed,
            )
    else:
        # Daily or non-intraday data: all rows are RTH by convention
        df["is_rth"] = True

    if df.empty:
        raise ValueError("No valid rows remain after cleaning")

    total_dropped = initial_rows - len(df)
    if total_dropped:
        logger.info(
            "Data cleaning complete: %d/%d rows kept (%d removed)",
            len(df),
            initial_rows,
            total_dropped,
        )

    return df


def add_session_segments(
    df: pd.DataFrame,
    rth_start: str = "09:30",
    rth_end: str = "16:00",
) -> pd.DataFrame:
    """Add session segmentation columns to an intraday DataFrame.

    Classifies every bar as RTH or OVERNIGHT without dropping any rows.
    Assigns a unique ``session_id`` per trading day and computes running
    session OHLC aggregates so downstream code can reference the session
    open, high, and low at any point within the session.

    Session boundaries
    ------------------
    - **RTH**: ``rth_start`` <= bar time <= ``rth_end`` (inclusive on both
      ends so the 16:00 close bar is tagged RTH).
    - **OVERNIGHT**: everything else.  An overnight session belongs to the
      *next* RTH date it precedes (e.g. bars from 18:00 Sun through 09:29
      Mon share the Monday session_id).

    Columns added
    -------------
    session_type : str
        ``"RTH"`` or ``"OVERNIGHT"``.
    session_id : str
        ``YYYY-MM-DD`` of the RTH date the bar belongs to.
    session_open : float
        Open price of the first bar in the current session segment.
    session_high : float
        Running high of the current session segment up to this bar.
    session_low : float
        Running low of the current session segment up to this bar.

    Parameters
    ----------
    df : pd.DataFrame
        Timezone-aware DataFrame from ``load_csv()`` with a DatetimeIndex
        and columns ``open``, ``high``, ``low``, ``close``.
    rth_start : str
        RTH session start as HH:MM (default "09:30").
    rth_end : str
        RTH session end as HH:MM (default "16:00").

    Returns
    -------
    pd.DataFrame
        Same DataFrame with five new columns appended.  Row count is
        unchanged — no data is dropped.
    """
    if df.empty:
        df["session_type"] = pd.Series(dtype=str)
        df["session_id"] = pd.Series(dtype=str)
        df["session_open"] = pd.Series(dtype=float)
        df["session_high"] = pd.Series(dtype=float)
        df["session_low"] = pd.Series(dtype=float)
        return df

    t_start = dt_time(*map(int, rth_start.split(":")))
    t_end = dt_time(*map(int, rth_end.split(":")))

    bar_times = df.index.time
    is_rth = (bar_times >= t_start) & (bar_times <= t_end)

    # --- session_type ---
    df["session_type"] = np.where(is_rth, "RTH", "OVERNIGHT")

    # --- session_id ---
    # RTH bars: session_id = that bar's calendar date.
    # Overnight bars: session_id = the *next* RTH date they precede.
    #
    # Strategy: find all unique RTH dates, then for each overnight bar
    # assign the earliest RTH date that is >= the bar's date.
    bar_dates = df.index.date
    rth_dates = np.unique(bar_dates[is_rth])

    session_ids = np.empty(len(df), dtype="U10")

    # RTH bars map directly
    session_ids[is_rth] = pd.DatetimeIndex(bar_dates[is_rth]).strftime("%Y-%m-%d")

    # Overnight bars: searchsorted into RTH dates to find next trading day
    ovn_mask = ~is_rth
    if ovn_mask.any():
        ovn_dates = bar_dates[ovn_mask]
        ovn_times = bar_times[ovn_mask]

        # Bars after RTH end on a given day belong to the *next* RTH day.
        # Bars before RTH start on a given day belong to *that* RTH day.
        # Determine the target RTH date for each overnight bar:
        ovn_target = np.empty(ovn_mask.sum(), dtype="datetime64[D]")
        for j, (d, t) in enumerate(zip(ovn_dates, ovn_times)):
            if t > t_end:
                # After RTH close — belongs to next RTH day
                idx = np.searchsorted(rth_dates, d, side="right")
            else:
                # Before RTH open — belongs to this RTH day
                idx = np.searchsorted(rth_dates, d, side="left")

            if idx < len(rth_dates):
                ovn_target[j] = rth_dates[idx]
            else:
                # Past last known RTH date — use the bar's own date
                ovn_target[j] = d

        session_ids[ovn_mask] = pd.DatetimeIndex(ovn_target).strftime("%Y-%m-%d")

    df["session_id"] = session_ids

    # --- session_open, session_high, session_low ---
    # Compute per (session_id, session_type) group using vectorised groupby.
    group_key = df["session_id"].values + "_" + df["session_type"].values

    # Detect group boundaries
    boundary = np.empty(len(group_key), dtype=bool)
    boundary[0] = True
    boundary[1:] = group_key[1:] != group_key[:-1]

    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values

    sess_open = np.empty(len(df), dtype=np.float64)
    sess_high = np.empty(len(df), dtype=np.float64)
    sess_low = np.empty(len(df), dtype=np.float64)

    # Single forward pass — O(n)
    for i in range(len(df)):
        if boundary[i]:
            sess_open[i] = opens[i]
            sess_high[i] = highs[i]
            sess_low[i] = lows[i]
        else:
            sess_open[i] = sess_open[i - 1]
            sess_high[i] = max(sess_high[i - 1], highs[i])
            sess_low[i] = min(sess_low[i - 1], lows[i])

    df["session_open"] = sess_open
    df["session_high"] = sess_high
    df["session_low"] = sess_low

    n_rth = int(is_rth.sum())
    n_ovn = len(df) - n_rth
    n_sessions = len(np.unique(session_ids))
    logger.info(
        "Session segmentation: %d RTH bars, %d overnight bars, %d unique sessions",
        n_rth, n_ovn, n_sessions,
    )

    return df
