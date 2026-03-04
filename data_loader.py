"""
Data loader module for OHLCV CSV data.

Handles reading, validation, and preprocessing of futures market data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_csv(filepath: str, datetime_col: str = "timestamp") -> pd.DataFrame:
    """Load OHLCV data from CSV and return a datetime-indexed DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    datetime_col : str
        Name of the column containing timestamps.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: open, high, low, close, volume
        and a DatetimeIndex sorted ascending.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(
        filepath,
        parse_dates=[datetime_col],
        index_col=datetime_col,
        dtype={
            "open": np.float64,
            "high": np.float64,
            "low": np.float64,
            "close": np.float64,
            "volume": np.float64,
        },
    )

    # Normalise column names to lowercase
    df.columns = df.columns.str.strip().str.lower()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df.sort_index(inplace=True)
    df.index.name = "timestamp"

    # Drop rows where any OHLC value is NaN
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    return df
