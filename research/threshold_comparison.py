"""
First-Hour Momentum: early entry comparison.

Tests early entry (10:00) vs standard (10:30) vs combined.

Run: python -m research.threshold_comparison
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
from engine.data_loader import load_csv
from engine.backtester import run as run_backtest
from engine.metrics import compute_metrics
from strategies.first_hour_momentum import generate_signals

OUTPUT_DIR = Path("output/edge_analysis")

BACKTEST_CONFIG = {
    "initial_capital": 100_000.0,
    "risk_per_trade": 0.01,
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "use_trailing_stop": False,
    "daily_dd_limit": 0.02,
    "max_daily_risk": 0.02,
    "max_dd_limit": 0.0,
}


def _tier_stats(trades, label):
    """Compute summary stats for a list of trades."""
    n = len(trades)
    if n < 2:
        return {"config": label, "trades": n}
    wins = sum(1 for t in trades if t.pnl > 0)
    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / n
    gross_win = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")
    return {
        "config": label,
        "trades": n,
        "trades_per_year": round(n / 7, 1),
        "win_rate": round(100 * wins / n, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "profit_factor": pf,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  FIRST-HOUR MOMENTUM: EARLY ENTRY COMPARISON")
    print("=" * 80)
    print("\nLoading data...")
    df = load_csv("data/nq_15m_data.csv")
    print(f"  {len(df):,} bars  |  {df.index[0].date()} to {df.index[-1].date()}")

    # ================================================================
    # Part 1: Configuration comparison
    # ================================================================
    configs = [
        ("Standard only (10:30)", {
            "enable_early_entry": False,
            "max_trades_per_day": 3,
        }),
        ("Early + standard (10:00/10:30)", {
            "enable_early_entry": True,
            "max_trades_per_day": 3,
        }),
        ("Early + standard, max 2", {
            "enable_early_entry": True,
            "max_trades_per_day": 2,
        }),
    ]

    results = []

    for name, strategy_params in configs:
        print(f"\n{'-' * 80}")
        print(f"  {name}")
        print(f"{'-' * 80}")

        signals = generate_signals(df, fh_percentile=80.0, pullback_atr_frac=0.5,
                                    **strategy_params)
        n_signals = int((signals["signal"] != 0).sum())

        t1 = int((signals["signal_tier"] == 1).sum())
        t2 = int((signals["signal_tier"] == 2).sum())
        t3 = int((signals["signal_tier"] == 3).sum())
        print(f"  Signals: {n_signals}  (early: {t1}, standard: {t2}, pullback: {t3})")

        result = run_backtest(signals, **BACKTEST_CONFIG)
        n_trades = len(result.trades)

        if n_trades < 2:
            print(f"  Only {n_trades} trades -- skipping.")
            results.append({"config": name, "trades": n_trades})
            continue

        metrics = compute_metrics(result, initial_capital=BACKTEST_CONFIG["initial_capital"])

        wins = sum(1 for t in result.trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in result.trades)
        avg_pnl = total_pnl / n_trades

        row = {
            "config": name,
            "trades": n_trades,
            "trades_per_year": round(n_trades / 7, 1),
            "early_sigs": t1,
            "standard_sigs": t2,
            "pullback_sigs": t3,
            "win_rate": round(100 * wins / n_trades, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "sharpe": metrics["mtm"]["sharpe"],
            "sortino": metrics["mtm"]["sortino"],
            "calmar": metrics["mtm"]["calmar"],
            "max_drawdown": metrics["mtm"]["max_drawdown"],
            "total_return": metrics["mtm"]["total_return"],
            "cagr": metrics["mtm"]["cagr"],
            "profit_factor": metrics["trades"].get("profit_factor", 0),
        }
        results.append(row)

        print(f"  Trades: {n_trades}  ({row['trades_per_year']}/yr)")
        print(f"  Win rate: {row['win_rate']}%")
        print(f"  Sharpe: {row['sharpe']}  Sortino: {row['sortino']}")
        print(f"  Max DD: {row['max_drawdown']}%")
        print(f"  Total return: {row['total_return']}%  CAGR: {row['cagr']}%")
        print(f"  Profit factor: {row['profit_factor']}")

    comp_df = pd.DataFrame(results)
    comp_df.to_csv(OUTPUT_DIR / "early_entry_comparison.csv", index=False)

    print(f"\n{'=' * 80}")
    print("  CONFIGURATION COMPARISON")
    print(f"{'=' * 80}\n")

    header_cols = ["config", "trades", "trades_per_year",
                   "win_rate", "sharpe", "max_drawdown", "total_return",
                   "profit_factor", "avg_pnl"]
    available = [c for c in header_cols if c in comp_df.columns]

    fmt = "  {:<30s}" + "  {:<14s}" * (len(available) - 1)
    print(fmt.format(*available))
    print("  " + "-" * (30 + 16 * (len(available) - 1)))

    for _, r in comp_df.iterrows():
        vals = []
        for c in available:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.2f}" if abs(v) < 10000 else f"{v:.0f}")
            else:
                vals.append(str(v))
        print(fmt.format(*vals))

    # ================================================================
    # Part 2: Per-tier breakdown
    # ================================================================
    print(f"\n{'=' * 80}")
    print("  PER-TIER BREAKDOWN (Early + standard, max 3)")
    print(f"{'=' * 80}\n")

    signals = generate_signals(df, fh_percentile=80.0, pullback_atr_frac=0.5,
                                enable_early_entry=True, max_trades_per_day=3)

    tier_signals = signals[signals["signal"] != 0][["signal_tier"]].copy()
    tier_map = {ts: int(row["signal_tier"]) for ts, row in tier_signals.iterrows()}
    signal_times = sorted(tier_map.keys())

    result = run_backtest(signals, **BACKTEST_CONFIG)

    early_trades = []
    standard_trades = []
    pullback_trades = []
    for t in result.trades:
        matched_tier = 0
        for st in signal_times:
            if st < t.entry_time:
                matched_tier = tier_map.get(st, 0)
            elif st >= t.entry_time:
                break
        if matched_tier == 1:
            early_trades.append(t)
        elif matched_tier == 2:
            standard_trades.append(t)
        else:
            pullback_trades.append(t)

    tier_rows = [
        _tier_stats(result.trades, "All trades"),
        _tier_stats(early_trades, "Early (10:00)"),
        _tier_stats(standard_trades, "Standard (10:30)"),
        _tier_stats(pullback_trades, "Pullback re-entries"),
    ]
    tier_df = pd.DataFrame(tier_rows)
    tier_df.to_csv(OUTPUT_DIR / "early_entry_tier_breakdown.csv", index=False)

    tier_cols = ["config", "trades", "trades_per_year", "win_rate",
                 "total_pnl", "avg_pnl", "profit_factor"]
    avail_tier = [c for c in tier_cols if c in tier_df.columns]

    fmt2 = "  {:<22s}" + "  {:<14s}" * (len(avail_tier) - 1)
    print(fmt2.format(*avail_tier))
    print("  " + "-" * (22 + 16 * (len(avail_tier) - 1)))

    for _, r in tier_df.iterrows():
        vals = []
        for c in avail_tier:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.2f}" if abs(v) < 10000 else f"{v:.0f}")
            else:
                vals.append(str(v))
        print(fmt2.format(*vals))

    # ================================================================
    # Part 3: Trades-per-day distribution
    # ================================================================
    print(f"\n{'=' * 80}")
    print("  TRADES PER DAY DISTRIBUTION (Early + standard, max 3)")
    print(f"{'=' * 80}\n")

    trade_dates = [t.entry_time.date() for t in result.trades]
    if trade_dates:
        trades_per_day = pd.Series(trade_dates).value_counts()
        dist = trades_per_day.value_counts().sort_index()
        for count, n_days in dist.items():
            print(f"  {count} trade(s)/day:  {n_days} days")
        print(f"\n  Average: {trades_per_day.mean():.2f} trades/day")
        print(f"  Total trading days: {len(trades_per_day)}")

    print(f"\n  Saved to: {OUTPUT_DIR.resolve()}/")
    print(f"    - early_entry_comparison.csv")
    print(f"    - early_entry_tier_breakdown.csv")


if __name__ == "__main__":
    main()
