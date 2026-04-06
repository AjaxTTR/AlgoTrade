"""
Two-Phase Prop Firm Simulation (realistic FTMO-style).

Phase 1: +10% target, 5% daily DD, 10% max DD
Phase 2: +5% target, same DD limits, balance & DD RESET

Compares to previous continuous +15% model.

Configs tested:
  1. 1.00% risk / 2R TP
  2. 0.75% risk / 2R TP
  3. 1.00% risk / No TP

Usage:
    python -m research.prop_firm_2phase
"""

import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest
from engine.metrics import compute_metrics
from engine.prop_firm import (
    simulate_prop_firm,
    simulate_prop_firm_2phase,
    print_prop_firm_2phase,
)

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

# Two-phase prop firm rules (FTMO-style)
PHASE1_CONFIG = {
    "starting_balance": 100_000.0,
    "profit_target": 0.10,          # +10%
    "daily_drawdown_limit": 0.05,
    "max_drawdown_limit": 0.10,
    "max_days": 0,                  # no time limit for measurement
}

PHASE2_CONFIG = {
    "starting_balance": 100_000.0,  # reset to same balance
    "profit_target": 0.05,          # +5%
    "daily_drawdown_limit": 0.05,
    "max_drawdown_limit": 0.10,
    "max_days": 0,
}

# Continuous +15% for comparison (old model)
CONTINUOUS_15_CONFIG = {
    "starting_balance": 100_000.0,
    "profit_target": 0.15,
    "daily_drawdown_limit": 0.05,
    "max_drawdown_limit": 0.10,
    "max_days": 0,
}

# Configs to test: (label, risk_per_trade, tp_atr_multiple)
EXPERIMENTS = [
    ("1.00% / 2R TP",  0.01,   2.0),
    ("0.75% / 2R TP",  0.0075, 2.0),
    ("1.00% / No TP",  0.01,   999.0),
]

N_SIMS = 5000
SEED = 42


def run_one(signals_base, risk, tp_mult):
    """Run backtest, then both 2-phase and continuous +15% sims."""
    signals = signals_base.copy()

    bt_cfg = {
        **BACKTEST_BASE,
        "risk_per_trade": risk,
        "stop_atr_multiple": 1.0,
        "tp_atr_multiple": tp_mult if tp_mult < 900 else 0.0,
    }
    result = run_backtest(signals, **bt_cfg)
    metrics = compute_metrics(result, initial_capital=BACKTEST_BASE["initial_capital"])

    # Two-phase simulation
    two_phase = simulate_prop_firm_2phase(
        result,
        phase1_config=PHASE1_CONFIG,
        phase2_config=PHASE2_CONFIG,
        n_simulations=N_SIMS,
        seed=SEED,
    )

    # Continuous +15% for comparison
    continuous = simulate_prop_firm(
        result,
        config=CONTINUOUS_15_CONFIG,
        n_simulations=N_SIMS,
        seed=SEED,
    )

    return result, metrics, two_phase, continuous


def main():
    t0 = time.perf_counter()

    print("\n" + "=" * 80)
    print("  TWO-PHASE PROP FIRM SIMULATION")
    print("  Phase 1: +10% | Phase 2: +5% (balance & DD reset)")
    print("  vs. Continuous +15% (old model)")
    print("=" * 80)

    # Load data & signals
    print("\n  Loading data...")
    df = load_csv("data/nq_15m_data.csv")
    print(f"  Loaded {len(df):,} bars")

    print("  Generating Variant B signals...")
    strategy = importlib.import_module(STRATEGY_MODULE)
    signals = strategy.generate_signals(df, **STRATEGY_PARAMS)
    n_sig = int((signals["signal"] != 0).sum())
    print(f"  {n_sig} entry signals\n")

    all_results = []

    for label, risk, tp in EXPERIMENTS:
        print(f"  Running: {label} ...", end="", flush=True)
        result, metrics, two_phase, continuous = run_one(signals, risk, tp)
        all_results.append((label, result, metrics, two_phase, continuous))
        print(f"  done")

        # Print detailed two-phase results
        print(f"\n  --- {label} ---")
        print_prop_firm_2phase(two_phase)

    # =========================================================================
    # COMPARISON TABLE
    # =========================================================================
    print("\n" + "=" * 100)
    print("  COMPARISON: TWO-PHASE vs CONTINUOUS +15%")
    print("=" * 100)

    header = (
        f"  {'Config':<18s} | "
        f"{'2-Phase':^42s} | "
        f"{'Continuous +15%':^30s}"
    )
    sub_header = (
        f"  {'':18s} | "
        f"{'Pass%':>6s} {'MedD':>5s} {'P1Fail':>7s} {'P2Fail':>7s} {'P1DD':>5s} {'P2DD':>5s} | "
        f"{'Pass%':>6s} {'MedD':>5s} {'DDFail':>7s}"
    )
    print(header)
    print(sub_header)
    print("  " + "-" * 96)

    for label, result, metrics, two_phase, continuous in all_results:
        tp = two_phase
        td = tp.get("total_days_dist", {})
        med_total = td.get("total_p50", 0)

        c_ttp = continuous.get("days_distribution", {})
        c_med = c_ttp.get("ttp_p50", 0)
        c_fr = continuous.get("fail_reasons", {})
        c_dd = c_fr.get("max_dd", 0) + c_fr.get("daily_dd", 0)
        c_dd_pct = c_dd / N_SIMS * 100

        line = (
            f"  {label:<18s} | "
            f"{tp['overall_pass_rate']:>6.1f} "
            f"{med_total:>5.0f} "
            f"{tp['fail_in_phase1_pct']:>7.1f} "
            f"{tp['fail_in_phase2_pct']:>7.1f} "
            f"{tp['p1_dd_fail_pct']:>5.1f} "
            f"{tp['p2_dd_fail_pct']:>5.1f} | "
            f"{continuous['pass_rate']:>6.1f} "
            f"{c_med:>5.0f} "
            f"{c_dd_pct:>7.1f}"
        )
        print(line)

    # =========================================================================
    # TIMING DETAIL
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  TIMING DETAIL (passing sims only, calendar days)")
    print("=" * 80)

    print(f"  {'Config':<18s} | {'Phase':>7s} | {'P10':>5s} {'P25':>5s} "
          f"{'Med':>5s} {'P75':>5s} {'P90':>5s} {'Mean':>5s}")
    print("  " + "-" * 76)

    for label, _, _, two_phase, _ in all_results:
        td = two_phase.get("total_days_dist", {})
        p1d = two_phase.get("p1_days_dist", {})
        p2d = two_phase.get("p2_days_dist", {})

        for phase_label, dist, prefix in [
            ("Phase 1", p1d, "p1"),
            ("Phase 2", p2d, "p2"),
            ("TOTAL", td, "total"),
        ]:
            print(
                f"  {label:<18s} | {phase_label:>7s} | "
                f"{dist.get(f'{prefix}_p10', 0):>5.0f} "
                f"{dist.get(f'{prefix}_p25', 0):>5.0f} "
                f"{dist.get(f'{prefix}_p50', 0):>5.0f} "
                f"{dist.get(f'{prefix}_p75', 0):>5.0f} "
                f"{dist.get(f'{prefix}_p90', 0):>5.0f} "
                f"{dist.get(f'{prefix}_mean', 0):>5.0f}"
            )
        print("  " + "-" * 76)

    # =========================================================================
    # KEY INSIGHT
    # =========================================================================
    print("\n" + "=" * 80)
    print("  KEY INSIGHT: 2-PHASE vs CONTINUOUS")
    print("=" * 80)

    for label, _, _, two_phase, continuous in all_results:
        tp_rate = two_phase["overall_pass_rate"]
        c_rate = continuous["pass_rate"]
        delta = tp_rate - c_rate
        direction = "HIGHER" if delta > 0 else "LOWER"
        print(f"  {label:<18s}:  2-phase {tp_rate:.1f}%  vs  continuous {c_rate:.1f}%  "
              f"({direction} by {abs(delta):.1f}pp)")

    # Export
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    export = {}
    for label, _, _, two_phase, continuous in all_results:
        export[label] = {
            "two_phase": {k: v for k, v in two_phase.items()
                         if k not in ("phase1_config", "phase2_config")},
            "continuous_15": {k: v for k, v in continuous.items()
                            if k != "config"},
        }

    json_path = out_dir / "prop_firm_2phase_results.json"
    json_path.write_text(json.dumps(export, indent=2, default=str))
    print(f"\n  Exported to {json_path}")

    elapsed = time.perf_counter() - t0
    print(f"  Runtime: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
