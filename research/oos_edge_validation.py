"""
Out-of-sample edge validation using 1-hour Yahoo Finance data.

Tests whether the core edge (FH_Up -> Rest_Up) survived in 2025-2026.
Does NOT simulate execution — purely measures directional edge persistence.

Usage:
    python -m research.oos_edge_validation
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/nq_1h_data.csv")
# Overlap cutoff: existing 15-min data ends 2024-12-30
OOS_START = "2025-01-01"


def load_hourly_data():
    """Load 1h data, convert to ET, return full DataFrame."""
    df = pd.read_csv(DATA_PATH, parse_dates=["ts_event"], index_col="ts_event")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    df.index.name = "timestamp"
    return df


def compute_daily_edge(df):
    """
    For each trading day, compute:
    - first_hour_return: return of the 09:30 bar (open->close)
    - rest_of_day_return: return from first-hour close to session close (16:00)
    - session_return: full day open to close
    """
    df = df.copy()
    df["date"] = df.index.date
    df["time"] = df.index.time

    from datetime import time as dt_time
    # Yahoo 1h bars land on the hour: 09:00 bar covers 09:00-10:00
    # This captures the RTH open (09:30) through 10:00 — our first-hour window
    t_fh = dt_time(9, 0)
    t_rth_start = dt_time(9, 0)
    t_close = dt_time(15, 0)  # 15:00 bar closes at 16:00

    records = []
    for date, day_df in df.groupby("date"):
        times = day_df["time"].values

        # Find 09:00 bar (covers 09:00-10:00 = first hour)
        fh_mask = times == t_fh
        if not fh_mask.any():
            continue
        fh_bar = day_df[fh_mask].iloc[0]
        fh_open = fh_bar["open"]
        fh_close = fh_bar["close"]
        fh_high = fh_bar["high"]
        fh_low = fh_bar["low"]
        fh_return = (fh_close - fh_open) / fh_open * 100

        # Find session close (last RTH bar)
        rth = day_df[(times >= t_rth_start) & (times <= t_close)]
        if len(rth) < 2:
            continue
        session_close = rth.iloc[-1]["close"]
        rest_return = (session_close - fh_close) / fh_close * 100
        session_return = (session_close - fh_open) / fh_open * 100

        records.append({
            "date": date,
            "fh_open": fh_open,
            "fh_close": fh_close,
            "fh_return": fh_return,
            "rest_return": rest_return,
            "session_return": session_return,
            "fh_direction": "UP" if fh_return > 0 else "DOWN",
            "rest_direction": "UP" if rest_return > 0 else "DOWN",
            "continuation": (fh_return > 0 and rest_return > 0) or
                           (fh_return < 0 and rest_return < 0),
        })

    return pd.DataFrame(records)


def run_validation():
    print("=" * 65)
    print("OUT-OF-SAMPLE EDGE VALIDATION (1h bars, yfinance NQ=F)")
    print("=" * 65)

    df = load_hourly_data()
    daily = compute_daily_edge(df)
    daily["date"] = pd.to_datetime(daily["date"])

    # Split into in-sample (overlap with existing data) and OOS
    is_mask = daily["date"] < OOS_START
    oos_mask = daily["date"] >= OOS_START

    daily_is = daily[is_mask]
    daily_oos = daily[oos_mask]

    print(f"\nIn-sample period:  {daily_is['date'].min().date()} to "
          f"{daily_is['date'].max().date()} ({len(daily_is)} days)")
    print(f"Out-of-sample:     {daily_oos['date'].min().date()} to "
          f"{daily_oos['date'].max().date()} ({len(daily_oos)} days)")

    # --- Compute expanding percentile threshold (no lookahead) ---
    # Use only positive FH returns for threshold (matching strategy logic)
    all_fh = daily["fh_return"].copy()
    threshold_75 = all_fh.expanding(min_periods=20).quantile(0.75).shift(1)
    daily["fh_threshold"] = threshold_75.values
    daily["qualifies"] = daily["fh_return"] > daily["fh_threshold"]

    # Recompute OOS with threshold
    daily_oos = daily[oos_mask].copy()
    daily_is = daily[is_mask].copy()

    for label, subset in [("IN-SAMPLE (validation)", daily_is),
                          ("OUT-OF-SAMPLE (2025-2026)", daily_oos)]:
        print(f"\n{'=' * 65}")
        print(f"  {label}")
        print(f"{'=' * 65}")

        # All days
        n = len(subset)
        fh_up = subset[subset["fh_direction"] == "UP"]
        fh_down = subset[subset["fh_direction"] == "DOWN"]

        print(f"\n  All days (n={n}):")
        if len(fh_up) > 0:
            cont_up = fh_up["rest_direction"].eq("UP").mean() * 100
            avg_rest_up = fh_up["rest_return"].mean()
            print(f"    FH Up  -> Rest Up:   {cont_up:.1f}%  "
                  f"(n={len(fh_up)}, avg rest return: {avg_rest_up:+.3f}%)")
        if len(fh_down) > 0:
            cont_dn = fh_down["rest_direction"].eq("DOWN").mean() * 100
            avg_rest_dn = fh_down["rest_return"].mean()
            print(f"    FH Down-> Rest Down: {cont_dn:.1f}%  "
                  f"(n={len(fh_down)}, avg rest return: {avg_rest_dn:+.3f}%)")

        # Filtered days (top 25% FH return = our strategy)
        qual = subset[subset["qualifies"] == True]
        print(f"\n  Filtered days (FH return > 75th pctile, n={len(qual)}):")
        if len(qual) > 0:
            qual_up = qual[qual["fh_direction"] == "UP"]
            qual_dn = qual[qual["fh_direction"] == "DOWN"]

            if len(qual_up) > 0:
                cont = qual_up["rest_direction"].eq("UP").mean() * 100
                avg_r = qual_up["rest_return"].mean()
                med_r = qual_up["rest_return"].median()
                print(f"    FH Up  -> Rest Up:   {cont:.1f}%  "
                      f"(n={len(qual_up)}, avg: {avg_r:+.3f}%, "
                      f"med: {med_r:+.3f}%)")

            if len(qual_dn) > 0:
                cont = qual_dn["rest_direction"].eq("DOWN").mean() * 100
                avg_r = qual_dn["rest_return"].mean()
                print(f"    FH Down-> Rest Down: {cont:.1f}%  "
                      f"(n={len(qual_dn)}, avg: {avg_r:+.3f}%)")

            # Overall edge stats for qualifying long days
            if len(qual_up) > 0:
                print(f"\n    --- Long-only edge (qualifying FH Up days) ---")
                returns = qual_up["rest_return"]
                print(f"    Win rate:      {(returns > 0).mean() * 100:.1f}%")
                print(f"    Avg winner:    {returns[returns > 0].mean():+.3f}%")
                print(f"    Avg loser:     {returns[returns <= 0].mean():+.3f}%")
                winners = returns[returns > 0].sum()
                losers = abs(returns[returns <= 0].sum())
                pf = winners / losers if losers > 0 else float("inf")
                print(f"    Profit factor: {pf:.2f}")
                print(f"    Total return:  {returns.sum():+.3f}%")
        else:
            print("    No qualifying days found.")

    # --- Monthly breakdown for OOS ---
    print(f"\n{'=' * 65}")
    print(f"  OOS MONTHLY BREAKDOWN (qualifying FH Up days)")
    print(f"{'=' * 65}")

    oos_qual_up = daily_oos[
        (daily_oos["qualifies"] == True) & (daily_oos["fh_direction"] == "UP")
    ].copy()

    if len(oos_qual_up) > 0:
        oos_qual_up["month"] = pd.to_datetime(oos_qual_up["date"]).dt.to_period("M")
        monthly = oos_qual_up.groupby("month").agg(
            trades=("rest_return", "count"),
            win_rate=("rest_return", lambda x: (x > 0).mean() * 100),
            avg_return=("rest_return", "mean"),
            total_return=("rest_return", "sum"),
        )
        print(f"\n  {'Month':<10} {'Trades':>6} {'WR%':>7} {'Avg Ret':>10} {'Total':>10}")
        print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*10} {'-'*10}")
        for period, row in monthly.iterrows():
            print(f"  {str(period):<10} {int(row['trades']):>6} "
                  f"{row['win_rate']:>6.1f}% {row['avg_return']:>+9.3f}% "
                  f"{row['total_return']:>+9.3f}%")

    print()


if __name__ == "__main__":
    run_validation()
