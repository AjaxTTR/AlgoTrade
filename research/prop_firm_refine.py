"""
Refinement grid around 1.00% / 2R TP sweet spot.

TP:   1.5R, 1.75R, 2.0R
Risk: 1.0%, 1.1%, 1.25%

Two-phase prop firm simulation (Phase 1: +10%, Phase 2: +5%, reset between).

Usage:
    python -m research.prop_firm_refine
"""

import importlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest
from engine.metrics import compute_metrics
from engine.prop_firm import simulate_prop_firm_2phase

# ---------------------------------------------------------------------------
# Variant B baseline
# ---------------------------------------------------------------------------
STRATEGY_MODULE = "strategies.gap_fh_regime_filtered"

STRATEGY_PARAMS = {
    "session_start": "09:30",
    "session_end": "16:00",
    "fh_end": "10:30",
    "entry_cutoff": "15:45",
    "atr_period": 14,
    "stop_atr_multiple": 1.0,
    "tp_atr_multiple": 999.0,
    "fh_percentile": 75.0,
    "gap_threshold_pct": 0.10,
    "fh_dead_zone_lo": 0.50,
    "fh_dead_zone_hi": 0.75,
    "trend_20d_max": 999.0,
    "vol_filter": False,
}

BACKTEST_BASE = {
    "initial_capital": 100_000.0,
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "use_trailing_stop": False,
    "daily_dd_limit": 0.0,
    "max_daily_risk": 0.0,
    "max_dd_limit": 0.0,
    "max_bars_in_trade": 0,
    "max_concurrent_trades": 1,
    "min_bars_between_entries": 0,
    "consec_loss_threshold": 0,
    "loss_scale_down": 1.0,
    "max_risk_per_trade": 0.0,
}

PHASE1 = {
    "starting_balance": 100_000.0,
    "profit_target": 0.10,
    "daily_drawdown_limit": 0.05,
    "max_drawdown_limit": 0.10,
    "max_days": 0,
}

PHASE2 = {
    "starting_balance": 100_000.0,
    "profit_target": 0.05,
    "daily_drawdown_limit": 0.05,
    "max_drawdown_limit": 0.10,
    "max_days": 0,
}

# Grid
RISK_LEVELS = [0.01, 0.011, 0.0125]
TP_LEVELS = [1.5, 1.75, 2.0]

RISK_LABELS = {0.01: "1.00%", 0.011: "1.10%", 0.0125: "1.25%"}
TP_LABELS = {1.5: "1.50R", 1.75: "1.75R", 2.0: "2.00R"}

N_SIMS = 5000
SEED = 42


def run_one(signals_base, risk, tp_mult):
    signals = signals_base.copy()
    sig_mask = signals["signal"] != 0
    if sig_mask.any():
        signals.loc[sig_mask, "tp_price"] = (
            signals.loc[sig_mask, "close"] + signals.loc[sig_mask, "atr"] * tp_mult
        )

    result = run_backtest(signals, **{**BACKTEST_BASE, "risk_per_trade": risk})
    metrics = compute_metrics(result, initial_capital=BACKTEST_BASE["initial_capital"])

    tp = simulate_prop_firm_2phase(
        result, phase1_config=PHASE1, phase2_config=PHASE2,
        n_simulations=N_SIMS, seed=SEED,
    )

    td = tp.get("total_days_dist", {})
    p1d = tp.get("p1_days_dist", {})
    p2d = tp.get("p2_days_dist", {})

    return {
        "risk": RISK_LABELS[risk],
        "tp": TP_LABELS[tp_mult],
        "trades": len(result.trades),
        "pf": metrics["trades"].get("profit_factor", 0.0),
        "sharpe": metrics["mtm"].get("sharpe", 0.0),
        "max_dd": metrics["mtm"].get("max_drawdown", 0.0),
        "total_return": metrics["mtm"].get("total_return", 0.0),
        # 2-phase results
        "pass_rate": tp["overall_pass_rate"],
        "p1_pass": tp["phase1_pass_rate"],
        "p2_cond": tp["phase2_conditional_pass_rate"],
        "p1_fail": tp["fail_in_phase1_pct"],
        "p2_fail": tp["fail_in_phase2_pct"],
        "p1_dd_fail": tp["p1_dd_fail_pct"],
        "p2_dd_fail": tp["p2_dd_fail_pct"],
        # Timing
        "med_total": td.get("total_p50", 0),
        "p10_total": td.get("total_p10", 0),
        "p25_total": td.get("total_p25", 0),
        "p75_total": td.get("total_p75", 0),
        "p90_total": td.get("total_p90", 0),
        "mean_total": td.get("total_mean", 0),
        "med_p1": p1d.get("p1_p50", 0),
        "med_p2": p2d.get("p2_p50", 0),
        # Failure detail
        "fail_p1_reasons": tp.get("fail_p1_reasons", {}),
        "fail_p2_reasons": tp.get("fail_p2_reasons", {}),
    }


def main():
    t0 = time.perf_counter()

    print("\n" + "=" * 90)
    print("  REFINEMENT GRID: TP x RISK around 1%/2R sweet spot")
    print("  Two-phase: Phase 1 +10% | Phase 2 +5% (balance & DD reset)")
    print("=" * 90)

    df = load_csv("data/nq_15m_data.csv")
    strategy = importlib.import_module(STRATEGY_MODULE)
    signals = strategy.generate_signals(df, **STRATEGY_PARAMS)
    n_sig = int((signals["signal"] != 0).sum())
    print(f"  {len(df):,} bars, {n_sig} signals\n")

    grid = list(itertools.product(RISK_LEVELS, TP_LEVELS))
    results = []

    for idx, (risk, tp) in enumerate(grid, 1):
        label = f"{RISK_LABELS[risk]} / {TP_LABELS[tp]}"
        print(f"  [{idx}/{len(grid)}] {label} ...", end="", flush=True)
        row = run_one(signals, risk, tp)
        results.append(row)
        print(f"  pass={row['pass_rate']:.1f}%  med={row['med_total']:.0f}d  "
              f"p10={row['p10_total']:.0f}d")

    # =========================================================================
    # Main comparison table
    # =========================================================================
    print("\n\n" + "=" * 105)
    print("  REFINEMENT RESULTS")
    print("=" * 105)

    print(f"  {'Config':<16s} {'Pass%':>6s} {'MedD':>5s} {'P10':>5s} "
          f"{'P25':>5s} {'P75':>5s} {'P90':>5s} "
          f"{'P1Pass':>7s} {'P2Cond':>7s} "
          f"{'P1DDFl':>7s} {'P2DDFl':>7s} "
          f"{'MaxDD':>7s} {'PF':>6s} {'Sharpe':>7s}")
    print("  " + "-" * 101)

    for r in results:
        label = f"{r['risk']}/{r['tp']}"
        # Highlight if meets target: >=60% pass, <=90 med days
        marker = ""
        if r["pass_rate"] >= 60 and r["med_total"] <= 90:
            marker = " <<<"
        elif r["pass_rate"] >= 60 and r["med_total"] <= 135:
            marker = " <"

        print(
            f"  {label:<16s} "
            f"{r['pass_rate']:>6.1f} "
            f"{r['med_total']:>5.0f} "
            f"{r['p10_total']:>5.0f} "
            f"{r['p25_total']:>5.0f} "
            f"{r['p75_total']:>5.0f} "
            f"{r['p90_total']:>5.0f} "
            f"{r['p1_pass']:>7.1f} "
            f"{r['p2_cond']:>7.1f} "
            f"{r['p1_dd_fail']:>7.1f} "
            f"{r['p2_dd_fail']:>7.1f} "
            f"{r['max_dd']:>7.2f} "
            f"{r['pf']:>6.2f} "
            f"{r['sharpe']:>7.2f}"
            f"{marker}"
        )

    # =========================================================================
    # Heatmap-style: pass rate by TP x Risk
    # =========================================================================
    print("\n\n" + "=" * 50)
    print("  PASS RATE HEATMAP (TP rows x Risk cols)")
    print("=" * 50)

    print(f"  {'TP \\ Risk':<10s}", end="")
    for risk in RISK_LEVELS:
        print(f" {RISK_LABELS[risk]:>8s}", end="")
    print()
    print("  " + "-" * 36)

    for tp in TP_LEVELS:
        print(f"  {TP_LABELS[tp]:<10s}", end="")
        for risk in RISK_LEVELS:
            match = [r for r in results if r["risk"] == RISK_LABELS[risk]
                     and r["tp"] == TP_LABELS[tp]]
            if match:
                val = match[0]["pass_rate"]
                print(f" {val:>7.1f}%", end="")
            else:
                print(f" {'N/A':>8s}", end="")
        print()

    print("\n  MEDIAN DAYS HEATMAP (TP rows x Risk cols)")
    print("  " + "-" * 36)

    for tp in TP_LEVELS:
        print(f"  {TP_LABELS[tp]:<10s}", end="")
        for risk in RISK_LEVELS:
            match = [r for r in results if r["risk"] == RISK_LABELS[risk]
                     and r["tp"] == TP_LABELS[tp]]
            if match:
                val = match[0]["med_total"]
                print(f" {val:>7.0f}d", end="")
            else:
                print(f" {'N/A':>8s}", end="")
        print()

    # =========================================================================
    # Identify best configs
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  VIABLE CONFIGS (>=60% pass rate)")
    print("=" * 80)

    viable = sorted(
        [r for r in results if r["pass_rate"] >= 60],
        key=lambda r: r["med_total"],
    )

    if viable:
        for r in viable:
            tag = ""
            if r["med_total"] <= 90:
                tag = " *** TARGET ZONE ***"
            elif r["med_total"] <= 135:
                tag = " (close)"
            print(f"  {r['risk']}/{r['tp']:<6s}  "
                  f"Pass: {r['pass_rate']:5.1f}%  "
                  f"Med: {r['med_total']:4.0f}d  "
                  f"P10: {r['p10_total']:3.0f}d  "
                  f"DD fail: {r['p1_dd_fail'] + r['p2_dd_fail']:5.1f}%  "
                  f"MaxDD: {r['max_dd']:6.2f}%"
                  f"{tag}")

        fastest = viable[0]
        print(f"\n  FASTEST VIABLE: {fastest['risk']}/{fastest['tp']}")
        print(f"    Pass rate:      {fastest['pass_rate']:.1f}%")
        print(f"    Median days:    {fastest['med_total']:.0f}")
        print(f"    P10 (fast):     {fastest['p10_total']:.0f}")
        print(f"    P90 (slow):     {fastest['p90_total']:.0f}")
        print(f"    Phase 1 DD fail: {fastest['p1_dd_fail']:.1f}%")
        print(f"    Phase 2 DD fail: {fastest['p2_dd_fail']:.1f}%")
        print(f"    Max DD:         {fastest['max_dd']:.2f}%")
    else:
        print("  No config meets >=60% pass rate threshold.")
        best = max(results, key=lambda r: r["pass_rate"])
        print(f"  Best available: {best['risk']}/{best['tp']} at {best['pass_rate']:.1f}%")

    # Export
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "prop_firm_refine_grid.csv", index=False)
    (out_dir / "prop_firm_refine_grid.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    print(f"\n  Exported to output/prop_firm_refine_grid.csv")

    elapsed = time.perf_counter() - t0
    print(f"  Runtime: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
