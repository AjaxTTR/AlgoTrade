"""
Prop Firm Speed Optimization Grid Search.

Runs Variant B (gap_fh_regime_filtered, 75% threshold, dead zone, EOD exit)
across a 3x3 grid of risk_per_trade x TP settings.

Objective: find the fastest path to +10% / +15% while maintaining acceptable risk.

Usage:
    python -m research.prop_firm_speed_grid
"""

import itertools
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest
from engine.metrics import compute_metrics
from engine.prop_firm import simulate_prop_firm


# ---------------------------------------------------------------------------
# Variant B baseline config
# ---------------------------------------------------------------------------
STRATEGY_MODULE = "strategies.gap_fh_regime_filtered"

STRATEGY_PARAMS = {
    "session_start": "09:30",
    "session_end": "16:00",
    "fh_end": "10:30",
    "entry_cutoff": "15:45",
    "atr_period": 14,
    "fh_percentile": 75.0,          # 75th percentile (Variant B)
    "gap_threshold_pct": 0.10,
    "fh_dead_zone_lo": 0.50,        # dead zone ON
    "fh_dead_zone_hi": 0.75,
    "trend_20d_max": 999.0,         # trend filter disabled
    "vol_filter": False,            # vol filter disabled
}

BACKTEST_BASE = {
    "initial_capital": 100_000.0,
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "use_trailing_stop": False,
    "daily_dd_limit": 0.0,          # no daily DD limit in backtest
    "max_daily_risk": 0.0,
    "max_dd_limit": 0.0,
    "max_bars_in_trade": 0,         # EOD exit (session_close handles it)
    "max_concurrent_trades": 1,
    "min_bars_between_entries": 0,
    "consec_loss_threshold": 0,
    "loss_scale_down": 1.0,
    "max_risk_per_trade": 0.0,      # no hard cap — let risk_per_trade drive sizing
}

# Prop firm configs for +10% and +15% targets
PROP_CONFIGS = {
    "+10%": {
        "starting_balance": 100_000.0,
        "profit_target": 0.10,
        "daily_drawdown_limit": 0.05,
        "max_drawdown_limit": 0.10,
        "max_days": 0,              # no time limit for speed measurement
    },
    "+15%": {
        "starting_balance": 100_000.0,
        "profit_target": 0.15,
        "daily_drawdown_limit": 0.05,
        "max_drawdown_limit": 0.10,
        "max_days": 0,
    },
}

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
RISK_LEVELS = [0.005, 0.0075, 0.01]       # 0.5%, 0.75%, 1.0%
TP_LEVELS = [999.0, 1.5, 2.0]             # No TP, 1.5R, 2R (as ATR multiples)

TP_LABELS = {999.0: "No TP", 1.5: "1.5R TP", 2.0: "2R TP"}
RISK_LABELS = {0.005: "0.50%", 0.0075: "0.75%", 0.01: "1.00%"}

N_SIMS = 5000
SEED = 42


def run_experiment(signals_cache, risk, tp_mult):
    """Run one backtest + prop firm simulation for a (risk, tp) combo."""

    signals = signals_cache.copy()

    # Backtest config — stop/TP computed by backtester from ATR
    bt_cfg = {
        **BACKTEST_BASE,
        "risk_per_trade": risk,
        "stop_atr_multiple": 1.0,
        "tp_atr_multiple": tp_mult if tp_mult < 900 else 0.0,
    }
    result = run_backtest(signals, **bt_cfg)

    # Metrics
    metrics = compute_metrics(result, initial_capital=BACKTEST_BASE["initial_capital"])

    # Prop firm sims for both targets
    prop_results = {}
    for label, pcfg in PROP_CONFIGS.items():
        pf = simulate_prop_firm(result, config=pcfg, n_simulations=N_SIMS, seed=SEED)
        prop_results[label] = pf

    # Compute trades per year
    if len(result.trades) > 0:
        first_trade = result.trades[0].entry_time
        last_trade = result.trades[-1].exit_time
        span_days = (last_trade - first_trade).total_seconds() / 86400
        span_years = span_days / 365.25 if span_days > 0 else 1.0
        trades_per_year = len(result.trades) / span_years
    else:
        trades_per_year = 0.0
        span_years = 0.0

    # Extract key numbers
    row = {
        "risk": RISK_LABELS[risk],
        "tp": TP_LABELS[tp_mult],
        "trades": len(result.trades),
        "trades_yr": round(trades_per_year, 1),
        "pf": metrics["trades"].get("profit_factor", 0.0),
        "sharpe": metrics["mtm"].get("sharpe", 0.0),
        "max_dd": metrics["mtm"].get("max_drawdown", 0.0),
        "total_return": metrics["mtm"].get("total_return", 0.0),
    }

    # Prop firm results for each target
    for label in ["+10%", "+15%"]:
        pf = prop_results[label]
        ttp = pf.get("days_distribution", {})
        row[f"pass_rate_{label}"] = pf["pass_rate"]
        row[f"median_days_{label}"] = ttp.get("ttp_p50", 0.0)
        row[f"fail_rate_{label}"] = pf["fail_rate"]

        # Failure breakdown: DD failures specifically
        fail_reasons = pf.get("fail_reasons", {})
        dd_fails = fail_reasons.get("max_dd", 0) + fail_reasons.get("daily_dd", 0)
        row[f"dd_fail_pct_{label}"] = round(dd_fails / N_SIMS * 100, 1)

    return row


def main():
    t0 = time.perf_counter()

    print("\n" + "=" * 80)
    print("  PROP FIRM SPEED OPTIMIZATION — 3x3 GRID")
    print("  Variant B: 75% threshold + dead zone + EOD exit")
    print("=" * 80)

    # Load data
    print("\n  Loading data...")
    df = load_csv("data/nq_15m_data.csv")
    print(f"  Loaded {len(df):,} bars  |  {df.index[0]}  ->  {df.index[-1]}")

    # Generate signals ONCE (signal logic is identical across all configs)
    print("  Generating Variant B signals...")
    import importlib
    strategy = importlib.import_module(STRATEGY_MODULE)

    signals = strategy.generate_signals(df, **STRATEGY_PARAMS)
    n_signals = int((signals["signal"] != 0).sum())
    print(f"  Generated {n_signals} entry signals")

    # Run grid
    grid = list(itertools.product(RISK_LEVELS, TP_LEVELS))
    results = []

    print(f"\n  Running {len(grid)} experiments ({N_SIMS} MC sims each)...\n")

    for idx, (risk, tp) in enumerate(grid, 1):
        label = f"  [{idx}/{len(grid)}] Risk={RISK_LABELS[risk]}, TP={TP_LABELS[tp]}"
        print(f"{label} ...", end="", flush=True)
        row = run_experiment(signals, risk, tp)
        results.append(row)
        print(f"  done  (pass +10%: {row['pass_rate_+10%']:.1f}%, "
              f"median days: {row['median_days_+10%']:.0f})")

    # ---------------------------------------------------------------------------
    # Format comparison table
    # ---------------------------------------------------------------------------
    print("\n\n" + "=" * 120)
    print("  PROP FIRM SPEED COMPARISON TABLE")
    print("=" * 120)

    # Header
    header = (
        f"  {'Config':<22s} {'Trades':>6s} {'Tr/Yr':>6s} {'PF':>6s} "
        f"{'Sharpe':>7s} {'MaxDD':>7s} {'Return':>8s} "
        f"{'P10%':>6s} {'D10':>5s} {'F10%':>6s} "
        f"{'P15%':>6s} {'D15':>5s} {'F15%':>6s} "
        f"{'DDFail10':>8s} {'DDFail15':>8s}"
    )
    print(header)
    print("  " + "-" * 116)

    for r in results:
        config_label = f"{r['risk']} / {r['tp']}"
        line = (
            f"  {config_label:<22s} "
            f"{r['trades']:>6d} "
            f"{r['trades_yr']:>6.1f} "
            f"{r['pf']:>6.2f} "
            f"{r['sharpe']:>7.2f} "
            f"{r['max_dd']:>7.2f} "
            f"{r['total_return']:>8.1f} "
            f"{r['pass_rate_+10%']:>6.1f} "
            f"{r['median_days_+10%']:>5.0f} "
            f"{r['fail_rate_+10%']:>6.1f} "
            f"{r['pass_rate_+15%']:>6.1f} "
            f"{r['median_days_+15%']:>5.0f} "
            f"{r['fail_rate_+15%']:>6.1f} "
            f"{r['dd_fail_pct_+10%']:>8.1f} "
            f"{r['dd_fail_pct_+15%']:>8.1f}"
        )
        print(line)

    # ---------------------------------------------------------------------------
    # Identify fastest viable config
    # ---------------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("  SPEED vs RISK ANALYSIS")
    print("=" * 80)

    # Viable = pass rate >= 40% on +10% AND dd_fail < 30%
    viable = [r for r in results if r["pass_rate_+10%"] >= 40 and r["dd_fail_pct_+10%"] < 30]

    if viable:
        # Fastest = lowest median days to +10%
        fastest = min(viable, key=lambda r: r["median_days_+10%"] if r["median_days_+10%"] > 0 else 9999)
        print(f"\n  FASTEST VIABLE CONFIG: {fastest['risk']} risk / {fastest['tp']}")
        print(f"    Pass rate (+10%):    {fastest['pass_rate_+10%']:.1f}%")
        print(f"    Median days to +10%: {fastest['median_days_+10%']:.0f}")
        print(f"    Pass rate (+15%):    {fastest['pass_rate_+15%']:.1f}%")
        print(f"    Median days to +15%: {fastest['median_days_+15%']:.0f}")
        print(f"    Max DD:              {fastest['max_dd']:.2f}%")
        print(f"    DD failure rate:     {fastest['dd_fail_pct_+10%']:.1f}%")
    else:
        print("\n  WARNING: No config meets viability threshold (40% pass, <30% DD fail)")
        # Show best anyway
        best = max(results, key=lambda r: r["pass_rate_+10%"])
        print(f"  Best available: {best['risk']} risk / {best['tp']}")
        print(f"    Pass rate (+10%): {best['pass_rate_+10%']:.1f}%")

    # Speed vs risk tradeoff summary
    print("\n  RISK SCALING IMPACT (No TP configs):")
    no_tp = [r for r in results if r["tp"] == "No TP"]
    for r in no_tp:
        d10 = r['median_days_+10%']
        d10_str = f"{d10:.0f}" if d10 > 0 else "N/A"
        print(f"    {r['risk']:>6s} risk -> Pass: {r['pass_rate_+10%']:5.1f}%, "
              f"Median days: {d10_str:>4s}, MaxDD: {r['max_dd']:6.2f}%, "
              f"DD fail: {r['dd_fail_pct_+10%']:.1f}%")

    print("\n  TP IMPACT (0.75% risk configs):")
    mid_risk = [r for r in results if r["risk"] == "0.75%"]
    for r in mid_risk:
        d10 = r['median_days_+10%']
        d10_str = f"{d10:.0f}" if d10 > 0 else "N/A"
        print(f"    {r['tp']:>8s} -> Pass: {r['pass_rate_+10%']:5.1f}%, "
              f"Median days: {d10_str:>4s}, PF: {r['pf']:.2f}, "
              f"DD fail: {r['dd_fail_pct_+10%']:.1f}%")

    # ---------------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    df_results = pd.DataFrame(results)
    csv_path = out_dir / "prop_firm_speed_grid.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Results exported to {csv_path}")

    json_path = out_dir / "prop_firm_speed_grid.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"  JSON exported to {json_path}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
