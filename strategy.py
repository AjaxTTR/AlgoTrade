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
    stop_atr_buffer: float = 0.5,
    require_candle_confirm: bool = True,
) -> pd.DataFrame:
    """Compression breakout strategy.

    Detects periods where volatility contracts (range < ATR * ratio AND ATR
    declining over 3 bars), then generates entry signals when price breaks
    out of the compression range with candle confirmation.

    Only one signal per breakout move — a new entry requires the previous
    bar's signal to have been flat (0).

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
    stop_atr_buffer : float
        ATR multiplier added beyond the compression range for stop placement.
        Long stop = range_low - ATR * buffer, Short stop = range_high + ATR * buffer.
    require_candle_confirm : bool
        If True, long entries require close > open (bullish candle) and
        short entries require close < open (bearish candle).

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns:
        - signal: 1 (long), -1 (short), 0 (flat)
        - stop_price: stop level anchored to compression range (set on entry bar)
    """
    out = df.copy()
    n = len(out)

    # ── ATR (Wilder smoothing: EMA with alpha = 1/period) ──
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - out["close"].shift(1)).abs(),
            (out["low"] - out["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

    # ── Compression range (rolling high/low over lookback) ──
    out["range_high"] = out["high"].rolling(compression_lookback).max()
    out["range_low"] = out["low"].rolling(compression_lookback).min()
    out["range_width"] = out["range_high"] - out["range_low"]

    # ── Compression condition: tight range AND declining ATR ──
    range_tight = out["range_width"] < (out["atr"] * compression_ratio)
    atr_declining = (out["atr"] < out["atr"].shift(1)) & (out["atr"].shift(1) < out["atr"].shift(2))
    out["compressed"] = range_tight & atr_declining

    # ── Breakout detection ──
    # Use previous bar's compression state and range levels so we enter
    # on the bar AFTER compression is confirmed.
    was_compressed = out["compressed"].shift(1).fillna(False)
    prev_range_high = out["range_high"].shift(1)
    prev_range_low = out["range_low"].shift(1)

    # Close must be outside the compression range
    long_break = was_compressed & (out["close"] > prev_range_high)
    short_break = was_compressed & (out["close"] < prev_range_low)

    # Optional candle confirmation: bullish candle for longs, bearish for shorts
    if require_candle_confirm:
        long_break = long_break & (out["close"] > out["open"])
        short_break = short_break & (out["close"] < out["open"])

    # ── Build raw signal then suppress consecutive duplicates ──
    raw_signal = np.zeros(n, dtype=np.int8)
    raw_signal[long_break.values] = 1
    raw_signal[short_break.values] = -1

    # Only allow entry when previous signal was flat (prevent stacking)
    signal = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        if raw_signal[i] != 0 and signal[i - 1] == 0:
            signal[i] = raw_signal[i]

    out["signal"] = signal

    # ── Stop prices anchored to compression range ──
    atr_vals = out["atr"].values
    range_low_vals = prev_range_low.values
    range_high_vals = prev_range_high.values

    stop_price = np.full(n, np.nan)

    long_mask = signal == 1
    stop_price[long_mask] = range_low_vals[long_mask] - (atr_vals[long_mask] * stop_atr_buffer)

    short_mask = signal == -1
    stop_price[short_mask] = range_high_vals[short_mask] + (atr_vals[short_mask] * stop_atr_buffer)

    out["stop_price"] = stop_price

    return out
