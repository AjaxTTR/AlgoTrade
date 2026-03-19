"""
Pure statistical edge scanner for NQ futures.

Detects raw return anomalies grouped by time-of-day, volatility regime,
gap behaviour, and momentum/reversal — without running any strategy or
backtester.  All tests use forward 1-bar returns so results are directly
tradeable.

Usage:
    python -m research.baseline_scan

Outputs to research/baseline_results.csv and research/baseline_hourly.csv.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_FILE = "data/nq_15m_data.csv"
OUTPUT_DIR = Path("research")
RESULTS_PATH = OUTPUT_DIR / "baseline_results.csv"
HOURLY_PATH = OUTPUT_DIR / "baseline_hourly.csv"

# Significance threshold (BH-FDR corrected downstream)
ALPHA = 0.05
MIN_OBS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    adjusted = np.minimum(p * n / ranks, 1.0)
    # Enforce monotonicity
    adj_sorted = adjusted[order].copy()
    for i in range(n - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adjusted[order] = adj_sorted
    return adjusted


def _group_stats(returns: pd.Series, label: str, group_name: str) -> dict:
    """Compute mean, t-stat, p-value for a return series."""
    clean = returns.dropna()
    n = len(clean)
    if n < MIN_OBS:
        return None
    mean_ret = float(clean.mean())
    std_ret = float(clean.std())
    if std_ret < 1e-12:
        return None
    t_stat, p_value = sp_stats.ttest_1samp(clean, 0)
    return {
        "category": label,
        "group": group_name,
        "n_obs": n,
        "mean_return": round(mean_ret * 1e4, 4),  # basis points
        "std_return": round(std_ret * 1e4, 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "win_rate": round(float((clean > 0).mean()) * 100, 2),
        "skew": round(float(clean.skew()), 4) if n >= 3 else 0.0,
    }


def _collect_group_stats(
    df: pd.DataFrame,
    group_col: str,
    return_col: str,
    category: str,
) -> list[dict]:
    """Run _group_stats for each unique value in group_col."""
    rows = []
    for val, grp in df.groupby(group_col):
        result = _group_stats(grp[return_col], category, str(val))
        if result is not None:
            rows.append(result)
    return rows


# ---------------------------------------------------------------------------
# Analysis modules
# ---------------------------------------------------------------------------

def analyze_time_of_day(df: pd.DataFrame) -> list[dict]:
    """Mean forward return by hour and by 30-min slot."""
    rows = []

    # By hour
    rows.extend(_collect_group_stats(df, "hour_of_day", "fwd_return", "time_hour"))

    # By 30-min slot (more granular)
    df = df.copy()
    df["slot_30m"] = (
        df.index.hour.astype(str).str.zfill(2) + ":"
        + (df.index.minute // 30 * 30).astype(str).str.zfill(2)
    )
    rows.extend(_collect_group_stats(df, "slot_30m", "fwd_return", "time_slot"))

    return rows


def analyze_volatility_buckets(df: pd.DataFrame) -> list[dict]:
    """Mean forward return by ATR regime and ATR percentile bucket."""
    rows = []

    # ATR terciles (low / medium / high volatility)
    valid = df.dropna(subset=["atr"]).copy()
    if len(valid) < MIN_OBS * 3:
        return rows
    valid["atr_tercile"] = pd.qcut(valid["atr"], 3, labels=["Low", "Medium", "High"])
    rows.extend(_collect_group_stats(valid, "atr_tercile", "fwd_return", "vol_atr_tercile"))

    # ATR percentile buckets (0-25, 25-50, 50-75, 75-100)
    valid_pct = df.dropna(subset=["atr_percentile"]).copy()
    if len(valid_pct) >= MIN_OBS * 4:
        valid_pct["atr_pct_bucket"] = pd.cut(
            valid_pct["atr_percentile"],
            bins=[0, 25, 50, 75, 100],
            labels=["0-25", "25-50", "50-75", "75-100"],
            include_lowest=True,
        )
        rows.extend(_collect_group_stats(
            valid_pct, "atr_pct_bucket", "fwd_return", "vol_atr_pctile",
        ))

    # Interaction: hour x vol regime
    if "atr_tercile" in valid.columns:
        valid["hour_vol"] = valid["hour_of_day"].astype(str) + "_" + valid["atr_tercile"].astype(str)
        rows.extend(_collect_group_stats(valid, "hour_vol", "fwd_return", "hour_x_vol"))

    return rows


def analyze_gap_behaviour(df: pd.DataFrame) -> list[dict]:
    """Mean forward return conditional on session gap size and direction."""
    rows = []

    valid = df.dropna(subset=["gap_from_prev_close"]).copy()
    if len(valid) < MIN_OBS:
        return rows

    # Gap direction
    valid["gap_dir"] = np.where(
        valid["gap_from_prev_close"] > 0, "Gap_Up",
        np.where(valid["gap_from_prev_close"] < 0, "Gap_Down", "Flat"),
    )
    rows.extend(_collect_group_stats(valid, "gap_dir", "fwd_return", "gap_direction"))

    # Gap size terciles (among non-zero gaps)
    non_flat = valid[valid["gap_from_prev_close"] != 0].copy()
    if len(non_flat) >= MIN_OBS * 3:
        non_flat["gap_abs"] = non_flat["gap_from_prev_close"].abs()
        non_flat["gap_size"] = pd.qcut(non_flat["gap_abs"], 3, labels=["Small", "Medium", "Large"])
        rows.extend(_collect_group_stats(non_flat, "gap_size", "fwd_return", "gap_size"))

    # Gap fill tendency: does the first-hour return reverse the gap?
    # Only look at RTH bars in the first hour (09:30-10:30)
    first_hour = valid[
        (valid["hour_of_day"] == 9) | (valid["hour_of_day"] == 10)
    ].copy()
    if len(first_hour) >= MIN_OBS:
        first_hour["gap_fill_signal"] = np.where(
            first_hour["gap_from_prev_close"] > 0, "GapUp_1stHr",
            np.where(first_hour["gap_from_prev_close"] < 0, "GapDown_1stHr", "Flat_1stHr"),
        )
        rows.extend(_collect_group_stats(
            first_hour, "gap_fill_signal", "fwd_return", "gap_fill",
        ))

    return rows


def analyze_momentum_reversal(df: pd.DataFrame) -> list[dict]:
    """Test momentum vs reversal across multiple lookback horizons."""
    rows = []

    for lookback, col in [(1, "return_1"), (3, "return_3"), (5, "return_5")]:
        valid = df.dropna(subset=[col, "fwd_return"]).copy()
        if len(valid) < MIN_OBS * 3:
            continue

        # Tercile split on past return
        valid["past_bucket"] = pd.qcut(
            valid[col], 3, labels=["Losers", "Neutral", "Winners"],
        )
        rows.extend(_collect_group_stats(
            valid, "past_bucket", "fwd_return", f"momentum_{lookback}bar",
        ))

        # Does past direction predict next bar?
        valid["past_dir"] = np.where(valid[col] > 0, "Up", "Down")
        rows.extend(_collect_group_stats(
            valid, "past_dir", "fwd_return", f"continuation_{lookback}bar",
        ))

    # Position in day range: is mean-reversion or trend-following stronger?
    valid_pos = df.dropna(subset=["position_in_day_range", "fwd_return"]).copy()
    if len(valid_pos) >= MIN_OBS * 3:
        valid_pos["pos_bucket"] = pd.cut(
            valid_pos["position_in_day_range"],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["Bottom_25%", "Lower_Mid", "Upper_Mid", "Top_25%"],
            include_lowest=True,
        )
        rows.extend(_collect_group_stats(
            valid_pos, "pos_bucket", "fwd_return", "day_range_position",
        ))

    # VWAP distance: above/below VWAP
    valid_vwap = df.dropna(subset=["distance_from_vwap", "fwd_return"]).copy()
    if len(valid_vwap) >= MIN_OBS:
        valid_vwap["vwap_side"] = np.where(
            valid_vwap["distance_from_vwap"] > 0, "Above_VWAP", "Below_VWAP",
        )
        rows.extend(_collect_group_stats(
            valid_vwap, "vwap_side", "fwd_return", "vwap_position",
        ))

    # Session trend: does intraday trend predict next bar?
    valid_trend = df.dropna(subset=["session_trend", "fwd_return"]).copy()
    if len(valid_trend) >= MIN_OBS * 3:
        valid_trend["trend_bucket"] = pd.qcut(
            valid_trend["session_trend"], 3, labels=["Bearish", "Neutral", "Bullish"],
        )
        rows.extend(_collect_group_stats(
            valid_trend, "trend_bucket", "fwd_return", "session_trend",
        ))

    return rows


def analyze_day_of_week(df: pd.DataFrame) -> list[dict]:
    """Mean forward return by day of week."""
    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    df = df.copy()
    df["dow_name"] = df["day_of_week"].map(dow_map)
    return _collect_group_stats(df, "dow_name", "fwd_return", "day_of_week")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def baseline_scan() -> pd.DataFrame:
    """Run pure statistical edge scan on raw NQ data."""
    t_start = time.perf_counter()

    from engine.data_loader import load_csv
    from engine.feature_engineering import add_features

    if not Path(DATA_FILE).exists():
        log.error("Data file not found: %s", DATA_FILE)
        sys.exit(1)

    # Load and enrich
    log.info("Loading data from %s", DATA_FILE)
    df = load_csv(DATA_FILE)
    log.info("Loaded %d bars: %s -> %s", len(df), df.index[0], df.index[-1])

    df = add_features(df)

    # Forward 1-bar return (what we're trying to predict — NOT a feature)
    df["fwd_return"] = df["close"].shift(-1) / df["close"] - 1

    # Drop last bar (no forward return) and warm-up NaNs for core features
    df = df.iloc[:-1]
    log.info("Analysis universe: %d bars after forward-return alignment", len(df))

    # Run all analyses
    all_rows: list[dict] = []

    log.info("Analysing time-of-day returns...")
    all_rows.extend(analyze_time_of_day(df))

    log.info("Analysing volatility buckets...")
    all_rows.extend(analyze_volatility_buckets(df))

    log.info("Analysing gap behaviour...")
    all_rows.extend(analyze_gap_behaviour(df))

    log.info("Analysing momentum vs reversal...")
    all_rows.extend(analyze_momentum_reversal(df))

    log.info("Analysing day-of-week effects...")
    all_rows.extend(analyze_day_of_week(df))

    if not all_rows:
        log.error("No edges found — check data quality")
        sys.exit(1)

    results_df = pd.DataFrame(all_rows)

    # Apply BH-FDR correction across all tests
    results_df["p_adj"] = _bh_fdr(results_df["p_value"].values)
    results_df["significant"] = results_df["p_adj"] < ALPHA

    # Sort by absolute t-stat (strongest signal first)
    results_df["abs_t"] = results_df["t_stat"].abs()
    results_df.sort_values("abs_t", ascending=False, inplace=True)
    results_df.drop(columns="abs_t", inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    log.info("Full results saved to %s (%d tests)", RESULTS_PATH, len(results_df))

    # Save hourly detail separately
    hourly = results_df[results_df["category"].isin(["time_hour", "time_slot"])].copy()
    if not hourly.empty:
        hourly.to_csv(HOURLY_PATH, index=False)
        log.info("Hourly detail saved to %s", HOURLY_PATH)

    elapsed = time.perf_counter() - t_start
    sig_df = results_df[results_df["significant"]]
    n_sig = len(sig_df)
    n_total = len(results_df)

    # ── Print results ──
    print("\n" + "=" * 100)
    print("  NQ FUTURES — PURE STATISTICAL EDGE SCAN")
    print("=" * 100)
    print(f"  Bars analysed: {len(df):,}  |  Tests run: {n_total}  |  "
          f"Significant (FDR<{ALPHA}): {n_sig}  |  Runtime: {elapsed:.1f}s")

    # Section 1: Significant edges
    print("\n" + "-" * 100)
    print(f"  SIGNIFICANT EDGES ({n_sig} found, ranked by |t-stat|)")
    print("-" * 100)
    if n_sig > 0:
        display_cols = [
            "category", "group", "n_obs", "mean_return",
            "t_stat", "p_adj", "win_rate", "skew",
        ]
        print(sig_df[display_cols].head(30).to_string(index=False))
    else:
        print("  No statistically significant edges found after FDR correction.")

    # Section 2: Category summary
    print("\n" + "-" * 100)
    print("  SUMMARY BY CATEGORY")
    print("-" * 100)
    cat_summary = results_df.groupby("category").agg(
        tests=("group", "count"),
        significant=("significant", "sum"),
        best_t=("t_stat", lambda x: x.iloc[x.abs().argmax()]),
        best_group=("group", lambda x: x.iloc[results_df.loc[x.index, "t_stat"].abs().argmax()]),
    ).sort_values("significant", ascending=False)
    print(cat_summary.to_string())

    # Section 3: Time-of-day detail
    time_sig = sig_df[sig_df["category"].isin(["time_hour", "time_slot"])]
    if not time_sig.empty:
        print("\n" + "-" * 100)
        print("  TIME-OF-DAY EDGES (significant only)")
        print("-" * 100)
        display_cols = ["category", "group", "n_obs", "mean_return", "t_stat", "p_adj", "win_rate"]
        print(time_sig[display_cols].to_string(index=False))

    # Section 4: Momentum/reversal summary
    mom_sig = sig_df[sig_df["category"].str.startswith("momentum") |
                      sig_df["category"].str.startswith("continuation")]
    if not mom_sig.empty:
        print("\n" + "-" * 100)
        print("  MOMENTUM / REVERSAL EDGES (significant only)")
        print("-" * 100)
        display_cols = ["category", "group", "n_obs", "mean_return", "t_stat", "p_adj", "win_rate"]
        print(mom_sig[display_cols].to_string(index=False))

    # Section 5: Strongest single edges (top 10 by |t|)
    print("\n" + "-" * 100)
    print("  TOP 10 STRONGEST EDGES (by |t-stat|, regardless of significance)")
    print("-" * 100)
    display_cols = [
        "category", "group", "n_obs", "mean_return",
        "t_stat", "p_value", "p_adj", "significant", "win_rate",
    ]
    print(results_df[display_cols].head(10).to_string(index=False))

    print("\n" + "=" * 100 + "\n")

    return results_df


if __name__ == "__main__":
    baseline_scan()
