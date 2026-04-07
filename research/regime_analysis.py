"""
Regime analysis for the prop-firm portfolio (gap_momentum + regime_filtered).

Identifies pre-trade market conditions that predict success or failure.
All features are computed BEFORE the entry decision -- no lookahead.

Runs the locked 2-strategy portfolio and enriches every trade with daily
regime features, then analyses terciles, correlations, loss clustering,
and candidate filters.

Usage:
    python -m research.regime_analysis
"""

import importlib
import math
import numpy as np
import pandas as pd
from dataclasses import asdict
from pathlib import Path

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest

# ---------------------------------------------------------------------------
# Locked portfolio config (matches run_portfolio.py / walk_forward_portfolio)
# ---------------------------------------------------------------------------
DATA_FILE = "data/nq_15m_data.csv"
OUTPUT_DIR = Path("output/portfolio")

TOTAL_CAPITAL = 100_000.0

STRATEGY_CATALOGUE = {
    "gap_momentum": {
        "module": "strategies.gap_momentum",
        "strategy_params": {
            "fh_percentile": 75.0,
            "gap_threshold_pct": 0.10,
        },
        "backtest_overrides": {
            "stop_atr_multiple": 1.0,
            "tp_atr_multiple": 2.0,
        },
    },
    "regime_filtered_momentum": {
        "module": "strategies.regime_filtered_momentum",
        "strategy_params": {
            "fh_percentile": 75.0,
            "fh_dead_zone_lo": 0.50,
            "fh_dead_zone_hi": 0.75,
            "trend_20d_max": 1.0,
            "vol_filter": True,
        },
        "backtest_overrides": {
            "stop_atr_multiple": 1.0,
            "tp_atr_multiple": 2.0,
        },
    },
}

WEIGHTS = {"gap_momentum": 0.60, "regime_filtered_momentum": 0.40}
RISK_PER_TRADE = 0.014

BASE_BACKTEST_CONFIG = {
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "use_trailing_stop": False,
    "daily_dd_limit": 0.02,
    "max_daily_risk": 0.02,
    "max_dd_limit": 0.0,
    "max_bars_in_trade": 0,
    "max_concurrent_trades": 1,
    "min_bars_between_entries": 2,
    "consec_loss_threshold": 2,
    "loss_scale_down": 0.5,
    "max_risk_per_trade": 0.0,
}

# Walk-forward bad periods (Sharpe < -0.5 from walk_forward_portfolio.py)
BAD_PERIODS = ["2019-Q1", "2019-Q2", "2022-Q3", "2022-Q4", "2023-Q4", "2024-Q1"]

# Features to analyse (all pre-trade, no lookahead)
FEATURE_LIST = [
    ("overnight_gap_pct", "Overnight Gap (%)"),
    ("fh_return_pct", "First-Hour Return (%)"),
    ("fh_return_abs", "FH Return Magnitude (%)"),
    ("fh_range_pct", "FH Volatility / Range (%)"),
    ("prev_day_return", "Previous Day Return (%)"),
    ("prev_day_range_pct", "Previous Day Range (%)"),
    ("rolling_vol_5d", "5-Day Rolling Vol (%)"),
    ("rolling_vol_10d", "10-Day Rolling Vol (%)"),
    ("trend_5d", "5-Day Trend (% from MA)"),
    ("trend_10d", "10-Day Trend (% from MA)"),
    ("trend_20d", "20-Day Trend (% from MA)"),
    ("fh_range_vs_avg", "FH Range vs Average"),
    ("momentum_2d", "2-Day Momentum (%)"),
]


# ---------------------------------------------------------------------------
# Run portfolio
# ---------------------------------------------------------------------------

def _run_portfolio(df: pd.DataFrame) -> list:
    """Run both strategies and return combined trade list."""
    all_trades = []

    for name, weight in WEIGHTS.items():
        capital = TOTAL_CAPITAL * weight
        cat = STRATEGY_CATALOGUE[name]

        mod = importlib.import_module(cat["module"])
        signals = mod.generate_signals(df.copy(), **cat["strategy_params"])

        bt_cfg = {
            **BASE_BACKTEST_CONFIG,
            **cat.get("backtest_overrides", {}),
            "risk_per_trade": RISK_PER_TRADE,
            "initial_capital": capital,
            "strategy_name": name,
        }
        result = run_backtest(signals, **bt_cfg)

        for t in result.trades:
            t_dict = asdict(t)
            t_dict["strategy"] = name
            all_trades.append(t_dict)

    all_trades.sort(key=lambda t: t["entry_time"])
    return all_trades


# ---------------------------------------------------------------------------
# Daily pre-trade features
# ---------------------------------------------------------------------------

def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily pre-trade features from 15-min bars.

    Every feature is available BEFORE the entry decision.
    """
    time = df.index.time
    t_start = pd.Timestamp("09:30").time()
    t_end = pd.Timestamp("16:00").time()
    t_fh = pd.Timestamp("10:30").time()

    sess = df[(time >= t_start) & (time <= t_end)].copy()
    sess["date"] = sess.index.date
    fh = df[(time >= t_start) & (time < t_fh)].copy()
    fh["date"] = fh.index.date

    daily = sess.groupby("date").agg(
        session_open=("open", "first"),
        session_high=("high", "max"),
        session_low=("low", "min"),
        session_close=("close", "last"),
    )
    fh_daily = fh.groupby("date").agg(
        fh_open=("open", "first"),
        fh_high=("high", "max"),
        fh_low=("low", "min"),
        fh_close=("close", "last"),
    )
    daily = daily.join(fh_daily)

    daily["prev_close"] = daily["session_close"].shift(1)
    daily["overnight_gap_pct"] = (
        (daily["session_open"] - daily["prev_close"]) / daily["prev_close"] * 100
    )
    daily["fh_return_pct"] = (
        (daily["fh_close"] - daily["fh_open"]) / daily["fh_open"] * 100
    )
    daily["fh_return_abs"] = daily["fh_return_pct"].abs()
    daily["fh_range"] = daily["fh_high"] - daily["fh_low"]
    daily["fh_range_pct"] = daily["fh_range"] / daily["fh_open"] * 100
    daily["prev_day_return"] = (
        (daily["session_close"].shift(1) - daily["session_open"].shift(1))
        / daily["session_open"].shift(1) * 100
    )
    daily["prev_day_range"] = daily["session_high"].shift(1) - daily["session_low"].shift(1)
    daily["prev_day_range_pct"] = daily["prev_day_range"] / daily["prev_close"] * 100

    daily_ret = daily["session_close"].pct_change()
    daily["rolling_vol_5d"] = daily_ret.rolling(5, min_periods=3).std() * 100
    daily["rolling_vol_10d"] = daily_ret.rolling(10, min_periods=5).std() * 100

    sma5 = daily["session_close"].rolling(5, min_periods=3).mean().shift(1)
    sma10 = daily["session_close"].rolling(10, min_periods=5).mean().shift(1)
    sma20 = daily["session_close"].rolling(20, min_periods=10).mean().shift(1)
    daily["trend_5d"] = (daily["prev_close"] - sma5) / sma5 * 100
    daily["trend_10d"] = (daily["prev_close"] - sma10) / sma10 * 100
    daily["trend_20d"] = (daily["prev_close"] - sma20) / sma20 * 100

    avg_fh = daily["fh_range"].expanding(min_periods=20).mean().shift(1)
    daily["fh_range_vs_avg"] = daily["fh_range"] / avg_fh
    daily["gap_aligned"] = (daily["overnight_gap_pct"] > 0).astype(int)
    daily["momentum_2d"] = daily["session_close"].pct_change(2).shift(1) * 100

    return daily


# ---------------------------------------------------------------------------
# Enrich trades with features
# ---------------------------------------------------------------------------

def enrich_trades(trades_list: list, daily: pd.DataFrame) -> pd.DataFrame:
    trades = pd.DataFrame(trades_list)
    trades["date"] = pd.to_datetime(trades["entry_time"]).dt.date
    enriched = trades.merge(daily, left_on="date", right_index=True, how="left")

    enriched["return_pct"] = enriched["pnl"] / TOTAL_CAPITAL * 100
    enriched["is_winner"] = enriched["pnl"] > 0
    win_med = enriched.loc[enriched["pnl"] > 0, "pnl"].median() if (enriched["pnl"] > 0).any() else 0
    loss_med = enriched.loc[enriched["pnl"] < 0, "pnl"].median() if (enriched["pnl"] < 0).any() else 0
    enriched["is_large_win"] = enriched["pnl"] > win_med
    enriched["is_large_loss"] = enriched["pnl"] < loss_med

    entry_dt = pd.to_datetime(enriched["entry_time"])
    enriched["year_quarter"] = (
        entry_dt.dt.year.astype(str) + "-Q" + entry_dt.dt.quarter.astype(str)
    )
    enriched["is_bad_period"] = enriched["year_quarter"].isin(BAD_PERIODS)

    return enriched


# ---------------------------------------------------------------------------
# Tercile analysis
# ---------------------------------------------------------------------------

def tercile_stats(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    valid = df.dropna(subset=[feature])
    if len(valid) < 9:
        return pd.DataFrame()
    try:
        cuts, bins = pd.qcut(valid[feature], 3, retbins=True, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    valid = valid.copy()
    valid["_bin"] = cuts
    labels = ["Bottom", "Middle", "Top"]

    rows = []
    for b in sorted(valid["_bin"].dropna().unique()):
        sub = valid[valid["_bin"] == b]
        n = len(sub)
        if n == 0:
            continue
        std = sub["return_pct"].std()
        rows.append({
            "tercile": labels[int(b)] if int(b) < len(labels) else f"Bin{int(b)}",
            "range_lo": bins[int(b)],
            "range_hi": bins[int(b) + 1],
            "n": n,
            "win_rate": sub["is_winner"].mean() * 100,
            "avg_pnl": sub["pnl"].mean(),
            "total_pnl": sub["pnl"].sum(),
            "large_loss_pct": sub["is_large_loss"].mean() * 100,
            "sharpe_est": (sub["return_pct"].mean() / std * math.sqrt(252 / max(1, n))
                          if std > 0 else 0.0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Filter impact test
# ---------------------------------------------------------------------------

def test_filter(enriched: pd.DataFrame, name: str, mask: pd.Series) -> dict:
    """Compute performance if we SKIP trades where mask is True."""
    kept = enriched[~mask]
    skipped = enriched[mask]
    n_kept = len(kept)
    if n_kept == 0:
        return {}
    winners = kept[kept["pnl"] > 0]
    losers = kept[kept["pnl"] < 0]
    pf = winners["pnl"].sum() / abs(losers["pnl"].sum()) if len(losers) > 0 and losers["pnl"].sum() != 0 else float("inf")
    return {
        "name": name,
        "trades_kept": n_kept,
        "trades_skipped": len(skipped),
        "skip_pct": len(skipped) / len(enriched) * 100,
        "total_pnl": kept["pnl"].sum(),
        "avg_pnl": kept["pnl"].mean(),
        "win_rate": kept["is_winner"].mean() * 100,
        "profit_factor": pf,
        "pnl_skipped": skipped["pnl"].sum(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 100

    print(f"\n{sep}")
    print("  REGIME ANALYSIS: Prop-Firm Portfolio (gap_momentum + regime_filtered)")
    print(f"  Config: 60/40 split | risk=1.4% | stop=1.0x ATR | TP=2.0x ATR")
    print(sep)

    df = load_csv(DATA_FILE)
    print(f"\n  Data: {len(df):,} bars | {df.index[0].date()} -> {df.index[-1].date()}")

    daily = compute_daily_features(df)
    print(f"  Daily features: {len(daily)} trading days")

    trades_list = _run_portfolio(df)
    n_trades = len(trades_list)
    print(f"  Total portfolio trades: {n_trades}")

    if n_trades == 0:
        print("\n  ERROR: No trades. Check config.")
        return

    enriched = enrich_trades(trades_list, daily)
    enriched.sort_values("entry_time", inplace=True)
    enriched.reset_index(drop=True, inplace=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(OUTPUT_DIR / "regime_analysis_trades.csv", index=False)

    # Per-strategy breakdown
    for strat in enriched["strategy"].unique():
        sub = enriched[enriched["strategy"] == strat]
        w = sub[sub["pnl"] > 0]
        print(f"    {strat}: {len(sub)} trades, WR={len(w)/len(sub)*100:.1f}%, "
              f"PnL=${sub['pnl'].sum():,.0f}")

    # =======================================================================
    # 1. TRADE SUMMARY
    # =======================================================================
    winners = enriched[enriched["is_winner"]]
    losers = enriched[~enriched["is_winner"]]

    print(f"\n{sep}")
    print("  1. TRADE SUMMARY")
    print(sep)
    print(f"  Total trades:  {n_trades}")
    print(f"  Winners:       {len(winners)} ({len(winners)/n_trades*100:.1f}%)")
    print(f"  Losers:        {len(losers)} ({len(losers)/n_trades*100:.1f}%)")
    print(f"  Total P&L:     ${enriched['pnl'].sum():>12,.0f}")
    if len(winners):
        print(f"  Avg winner:    ${winners['pnl'].mean():>12,.0f}")
    if len(losers):
        print(f"  Avg loser:     ${losers['pnl'].mean():>12,.0f}")
    if len(losers) and losers["pnl"].sum() != 0:
        print(f"  Profit factor: {winners['pnl'].sum() / abs(losers['pnl'].sum()):>12.2f}")

    print(f"\n  Exit reasons:")
    for reason, cnt in enriched["exit_reason"].value_counts().items():
        print(f"    {reason:<20s} {cnt:>4d}  ({cnt/n_trades*100:.1f}%)")

    # =======================================================================
    # 2. FEATURE ANALYSIS (tercile breakdown)
    # =======================================================================
    print(f"\n{sep}")
    print("  2. FEATURE ANALYSIS (Tercile Breakdown)")
    print(sep)

    feature_scores = []
    for col, name in FEATURE_LIST:
        if col not in enriched.columns:
            continue
        stats = tercile_stats(enriched, col)
        if stats.empty:
            continue

        print(f"\n  {name}")
        print(f"  {'Tercile':<8} {'Range':>24} {'N':>5} {'WR%':>6} {'AvgPnL':>10}"
              f" {'TotalPnL':>12} {'LgLoss%':>8}")
        print("  " + "-" * 78)
        for _, r in stats.iterrows():
            rng = f"[{r['range_lo']:.3f}, {r['range_hi']:.3f}]"
            print(f"  {r['tercile']:<8} {rng:>24} {int(r['n']):>5} {r['win_rate']:>5.1f}%"
                  f" ${r['avg_pnl']:>9,.0f} ${r['total_pnl']:>11,.0f} {r['large_loss_pct']:>7.1f}%")

        if len(stats) >= 2:
            best = stats.loc[stats["avg_pnl"].idxmax()]
            worst = stats.loc[stats["avg_pnl"].idxmin()]
            feature_scores.append({
                "col": col, "name": name,
                "pnl_spread": best["avg_pnl"] - worst["avg_pnl"],
                "wr_spread": best["win_rate"] - worst["win_rate"],
                "best_terc": best["tercile"],
                "best_lo": best["range_lo"], "best_hi": best["range_hi"],
                "best_wr": best["win_rate"], "best_pnl": best["avg_pnl"],
                "worst_terc": worst["tercile"],
                "worst_lo": worst["range_lo"], "worst_hi": worst["range_hi"],
                "worst_wr": worst["win_rate"], "worst_pnl": worst["avg_pnl"],
            })

    # =======================================================================
    # 3. WALK-FORWARD PERIOD ANALYSIS
    # =======================================================================
    print(f"\n{sep}")
    print("  3. QUARTERLY PERFORMANCE & BAD PERIOD ANALYSIS")
    print(sep)

    qtr = enriched.groupby("year_quarter").agg(
        n=("pnl", "count"),
        wr=("is_winner", lambda x: x.mean() * 100),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
    ).round(1)

    print(f"\n  {'Quarter':<10} {'N':>4} {'WR%':>6} {'TotalPnL':>12} {'AvgPnL':>10} {'Flag':>10}")
    print("  " + "-" * 56)
    for q, r in qtr.iterrows():
        flag = "*** BAD ***" if q in BAD_PERIODS else ""
        print(f"  {q:<10} {int(r['n']):>4} {r['wr']:>5.1f}% ${r['total_pnl']:>11,.0f}"
              f" ${r['avg_pnl']:>9,.0f} {flag}")

    # Feature profile: bad vs good periods
    feat_cols = [c for c, _ in FEATURE_LIST if c in enriched.columns]
    bad_mask = enriched["is_bad_period"]

    if bad_mask.any():
        print(f"\n  Feature profile: BAD periods ({bad_mask.sum()} trades) vs GOOD periods ({(~bad_mask).sum()} trades)")
        print(f"  {'Feature':<25} {'Bad mean':>12} {'Good mean':>12} {'Delta':>10}")
        print("  " + "-" * 62)
        for f in feat_cols:
            bm = enriched.loc[bad_mask, f].mean()
            gm = enriched.loc[~bad_mask, f].mean()
            d = bm - gm
            marker = " <--" if abs(d) > 0.5 * max(abs(bm), abs(gm), 0.01) else ""
            print(f"  {f:<25} {bm:>12.4f} {gm:>12.4f} {d:>10.4f}{marker}")

    # =======================================================================
    # 4. STRATEGY-LEVEL REGIME BREAKDOWN
    # =======================================================================
    print(f"\n{sep}")
    print("  4. STRATEGY-LEVEL REGIME BREAKDOWN")
    print(sep)

    for strat in enriched["strategy"].unique():
        sub = enriched[enriched["strategy"] == strat]
        bad_sub = sub[sub["is_bad_period"]]
        good_sub = sub[~sub["is_bad_period"]]

        bad_wr = bad_sub["is_winner"].mean() * 100 if len(bad_sub) > 0 else 0
        good_wr = good_sub["is_winner"].mean() * 100 if len(good_sub) > 0 else 0

        print(f"\n  {strat}:")
        print(f"    BAD periods:  {len(bad_sub)} trades, WR={bad_wr:.1f}%, "
              f"PnL=${bad_sub['pnl'].sum():,.0f}, Avg=${bad_sub['pnl'].mean():,.0f}" if len(bad_sub) else f"    BAD periods: 0 trades")
        print(f"    GOOD periods: {len(good_sub)} trades, WR={good_wr:.1f}%, "
              f"PnL=${good_sub['pnl'].sum():,.0f}, Avg=${good_sub['pnl'].mean():,.0f}" if len(good_sub) else f"    GOOD periods: 0 trades")

    # =======================================================================
    # 5. LOSS CLUSTERING
    # =======================================================================
    print(f"\n{sep}")
    print("  5. LOSS CLUSTERING")
    print(sep)

    streak = 0
    streaks = []
    for i in range(len(enriched)):
        if not enriched.iloc[i]["is_winner"]:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    enriched["loss_streak"] = streaks

    max_streak = enriched["loss_streak"].max()
    print(f"\n  Max consecutive losses: {max_streak}")

    bad_streak = enriched[enriched["loss_streak"] >= 3]
    if len(bad_streak) > 0:
        print(f"  Trades in 3+ loss streaks: {len(bad_streak)}")
        print(f"\n  Feature averages during 3+ loss streaks vs overall:")
        print(f"  {'Feature':<25} {'Streak':>12} {'Overall':>12}")
        print("  " + "-" * 52)
        for f in feat_cols:
            sm = bad_streak[f].mean()
            om = enriched[f].mean()
            print(f"  {f:<25} {sm:>12.4f} {om:>12.4f}")

    large_losses = enriched[enriched["is_large_loss"]]
    if len(large_losses) > 3:
        print(f"\n  Large loss trades ({len(large_losses)}) -- feature averages:")
        print(f"  {'Feature':<25} {'LgLoss':>12} {'Overall':>12}")
        print("  " + "-" * 52)
        for f in feat_cols:
            lm = large_losses[f].mean()
            om = enriched[f].mean()
            print(f"  {f:<25} {lm:>12.4f} {om:>12.4f}")

    # =======================================================================
    # 6. CORRELATIONS
    # =======================================================================
    print(f"\n{sep}")
    print("  6. FEATURE CORRELATIONS WITH OUTCOME")
    print(sep)

    print(f"\n  {'Feature':<25} {'Corr(PnL)':>10} {'Corr(Win)':>10}")
    print("  " + "-" * 48)
    for f in feat_cols:
        v = enriched.dropna(subset=[f])
        if len(v) < 10:
            continue
        cp = v[f].corr(v["pnl"])
        cw = v[f].corr(v["is_winner"].astype(float))
        marker = " <--" if abs(cp) > 0.10 else ""
        print(f"  {f:<25} {cp:>10.4f} {cw:>10.4f}{marker}")

    # =======================================================================
    # 7. TOP DISCRIMINATING FEATURES
    # =======================================================================
    print(f"\n{sep}")
    print("  7. TOP DISCRIMINATING FEATURES")
    print(sep)

    scores = []
    if feature_scores:
        scores = sorted(feature_scores, key=lambda x: x["pnl_spread"], reverse=True)
        for i, s in enumerate(scores[:6]):
            print(f"\n  #{i+1} {s['name']} (spread: ${s['pnl_spread']:,.0f})")
            print(f"     Best:  {s['best_terc']} [{s['best_lo']:.3f}, {s['best_hi']:.3f}]"
                  f" -> WR={s['best_wr']:.1f}%, AvgPnL=${s['best_pnl']:,.0f}")
            print(f"     Worst: {s['worst_terc']} [{s['worst_lo']:.3f}, {s['worst_hi']:.3f}]"
                  f" -> WR={s['worst_wr']:.1f}%, AvgPnL=${s['worst_pnl']:,.0f}")

    # =======================================================================
    # 8. CANDIDATE FILTERS
    # =======================================================================
    print(f"\n{sep}")
    print("  8. CANDIDATE FILTERS (skip worst tercile of top features)")
    print(sep)

    baseline = {
        "trades": n_trades,
        "total_pnl": enriched["pnl"].sum(),
        "win_rate": enriched["is_winner"].mean() * 100,
        "pf": (winners["pnl"].sum() / abs(losers["pnl"].sum())
               if len(losers) and losers["pnl"].sum() != 0 else float("inf")),
    }
    print(f"\n  Baseline: {baseline['trades']} trades, PnL=${baseline['total_pnl']:,.0f},"
          f" WR={baseline['win_rate']:.1f}%, PF={baseline['pf']:.2f}")

    filter_results = []
    if scores:
        for s in scores[:6]:
            col = s["col"]
            if col not in enriched.columns:
                continue
            if s["worst_pnl"] < s["best_pnl"]:
                lo, hi = s["worst_lo"], s["worst_hi"]
                mask = (enriched[col] >= lo) & (enriched[col] <= hi)
                label = f"Skip {s['worst_terc']} {col} [{lo:.3f},{hi:.3f}]"
            else:
                continue
            r = test_filter(enriched, label, mask)
            if r:
                filter_results.append(r)

    if len(scores) >= 2:
        masks_combined = pd.Series(False, index=enriched.index)
        combo_parts = []
        for s in scores[:3]:
            col = s["col"]
            if col not in enriched.columns:
                continue
            lo, hi = s["worst_lo"], s["worst_hi"]
            masks_combined |= ((enriched[col] >= lo) & (enriched[col] <= hi))
            combo_parts.append(col)
        if combo_parts:
            r = test_filter(enriched, f"Combined: skip worst of {'+'.join(combo_parts)}", masks_combined)
            if r:
                filter_results.append(r)

    if filter_results:
        print(f"\n  {'Filter':<55} {'Kept':>5} {'Skip':>5} {'PnL':>12}"
              f" {'WR%':>6} {'PF':>6} {'PnL Skip':>10}")
        print("  " + "-" * 102)
        for r in filter_results:
            print(f"  {r['name']:<55} {r['trades_kept']:>5} {r['trades_skipped']:>5}"
                  f" ${r['total_pnl']:>11,.0f} {r['win_rate']:>5.1f}%"
                  f" {r['profit_factor']:>5.2f} ${r['pnl_skipped']:>9,.0f}")

    # =======================================================================
    # 9. SUMMARY
    # =======================================================================
    print(f"\n{sep}")
    print("  9. SUMMARY")
    print(sep)

    if scores:
        print("\n  TOP 3 CONDITIONS TO AVOID (worst tercile, negative avg PnL):")
        avoid = [s for s in scores if s["worst_pnl"] < 0][:3]
        for i, s in enumerate(avoid):
            print(f"    {i+1}. {s['name']}: {s['worst_terc']} tercile"
                  f" [{s['worst_lo']:.3f}, {s['worst_hi']:.3f}]"
                  f" -> WR={s['worst_wr']:.1f}%, AvgPnL=${s['worst_pnl']:,.0f}")

        print("\n  TOP 3 CONDITIONS WHERE EDGE IS STRONGEST:")
        strong = sorted(scores, key=lambda x: x["best_pnl"], reverse=True)[:3]
        for i, s in enumerate(strong):
            print(f"    {i+1}. {s['name']}: {s['best_terc']} tercile"
                  f" [{s['best_lo']:.3f}, {s['best_hi']:.3f}]"
                  f" -> WR={s['best_wr']:.1f}%, AvgPnL=${s['best_pnl']:,.0f}")

    print(f"\n  Enriched trades saved to: {OUTPUT_DIR / 'regime_analysis_trades.csv'}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
