"""
Strategy module.

Strategies are pure functions that receive an OHLCV DataFrame and return
a signal Series: 1 (long), -1 (short), 0 (flat).
"""

import pandas as pd
import numpy as np


def compression_breakout(
    df: pd.DataFrame,
    atr_period: int = 14,
    compression_lookback: int = 12,
    compression_ratio: float = 0.75,
    atr_stop_mult: float = 1.5,
) -> pd.DataFrame:
    """Compression breakout strategy.

    Detects periods where the recent price range contracts below a fraction
    of ATR, then generates a long signal when price breaks above the
    compression high.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    atr_period : int
        Lookback for ATR calculation.
    compression_lookback : int
        Number of bars to measure the compression range.
    compression_ratio : float
        Compression threshold — range must be below ATR * ratio.
    atr_stop_mult : float
        ATR multiplier for the initial stop-loss distance.

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns:
        - signal: 1 (long), -1 (short), 0 (flat)
        - stop_price: ATR-based stop level (set on entry bar)
    """
    out = df.copy()

    # --- ATR (Wilder smoothing via EMA with alpha = 1/period) ---
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift(1)).abs(),
            (out["low"] - out["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

    # --- Compression range (rolling high - low over lookback) ---
    out["range_high"] = out["high"].rolling(compression_lookback).max()
    out["range_low"] = out["low"].rolling(compression_lookback).min()
    out["range_width"] = out["range_high"] - out["range_low"]

    # --- Compression condition ---
    out["compressed"] = out["range_width"] < (out["atr"] * compression_ratio)

    # --- Breakout signal: compressed on previous bar, close breaks above range high ---
    # Use shifted compression state so we enter on the bar AFTER compression is detected
    was_compressed = out["compressed"].shift(1).fillna(False)
    prev_range_high = out["range_high"].shift(1)
    prev_range_low = out["range_low"].shift(1)

    long_entry = was_compressed & (out["close"] > prev_range_high)
    short_entry = was_compressed & (out["close"] < prev_range_low)

    # --- Build signal series ---
    out["signal"] = 0
    out.loc[long_entry, "signal"] = 1
    out.loc[short_entry, "signal"] = -1

    # --- Stop price (set on entry bars only, used by backtester) ---
    out["stop_price"] = np.nan
    out.loc[long_entry, "stop_price"] = out.loc[long_entry, "close"] - (
        out.loc[long_entry, "atr"] * atr_stop_mult
    )
    out.loc[short_entry, "stop_price"] = out.loc[short_entry, "close"] + (
        out.loc[short_entry, "atr"] * atr_stop_mult
    )

    return out
