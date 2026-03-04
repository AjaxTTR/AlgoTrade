"""
Data loader module for OHLCV CSV data.

Handles reading, validation, timezone conversion, and preprocessing
of futures market data for the backtesting engine.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


_TIMESTAMP_ALIASES = ("timestamp", "time", "datetime", "ts_event", "date")


def load_csv(
    filepath: str,
    target_tz: str = "America/New_York",
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

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    target_tz : str
        IANA timezone to convert timestamps into (default: America/New_York).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: open, high, low, close, volume
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
