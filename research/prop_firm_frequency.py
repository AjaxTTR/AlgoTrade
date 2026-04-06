"""
Frequency optimization: increase trades/year toward 40-60 while keeping PF >= 1.8.

Levers tested:
  1. FH percentile: 75 (baseline), 70, 65, 60
  2. Dead zone: [0.50-0.75] (baseline), [0.55-0.70] (narrower), [0.60-0.65] (minimal), OFF
  3. Gap threshold: 0.10% (baseline), 0.05% (relaxed)

Locked: 1.00% risk, 2.0R TP, 1.0x ATR stop, EOD exit.
Two-phase prop sim on viable configs.

Usage:
    python -m research.prop_firm_frequency
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

STRATEGY_MODULE = "strategies.gap_fh_regime_filtered"

BACKTEST_BASE = {
    "initial_capital": 100_000.0,
    "risk_per_trade": 0.01,         # locked at 1%
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
    "stop_atr_multiple": 1.0,
    "tp_atr_multiple": 2.0,        # locked 2R TP
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

N_SIMS = 5000
SEED = 42

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
# Each experiment: (label, strategy_params_overrides)
# Base strategy params shared by all:
BASE_STRAT = {
    "session_start": "09:30",
    "session_end": "16:00",
    "fh_end": "10:30",
    "entry_cutoff": "15:45",
    "atr_period": 14,
    "trend_20d_max": 999.0,
    "vol_filter": False,
}

# Dead zone configs: (label, lo, hi) — use lo > hi to effectively disable
DZ_CONFIGS = [
    ("DZ 0.50-0.75", 0.50, 0.75),   # baseline
    ("DZ 0.55-0.70", 0.55, 0.70),   # narrower
    ("DZ 0.60-0.65", 0.60, 0.65),   # minimal
    ("DZ OFF",       999.0, -1.0),   # disabled (lo > hi = nothing in range)
]

FH_PCTILES = [75.0, 70.0, 65.0, 60.0]
GAP_THRESHOLDS = [0.10, 0.05]


def run_one(df, strat_params):
    """Generate signals, backtest, compute metrics + 2-phase sim."""
    strategy = importlib.import_module(STRATEGY_MODULE)
    signals = strategy.generate_signals(df, **strat_params)

    n_signals = int((signals["signal"] != 0).sum())
    result = run_backtest(signals, **BACKTEST_BASE)
    metrics = compute_metrics(result, initial_capital=BACKTEST_BASE["initial_capital"])

    # Trades per year
    if len(result.trades) >= 2:
        span = (result.trades[-1].exit_time - result.trades[0].entry_time).total_seconds() / 86400
        trades_yr = len(result.trades) / (span / 365.25) if span > 0 else 0
    else:
        trades_yr = 0

    pf = metrics["trades"].get("profit_factor", 0.0)
    sharpe = metrics["mtm"].get("sharpe", 0.0)
    max_dd = metrics["mtm"].get("max_drawdown", 0.0)
    win_rate = metrics["trades"].get("win_rate", 0.0)

    return {
        "signals": n_signals,
        "trades": len(result.trades),
        "trades_yr": round(trades_yr, 1),
        "pf": pf,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "result": result,
    }


def run_2phase(result):
    """Run 2-phase prop sim and extract key numbers."""
    tp = simulate_prop_firm_2phase(
        result, phase1_config=PHASE1, phase2_config=PHASE2,
        n_simulations=N_SIMS, seed=SEED,
    )
    td = tp.get("total_days_dist", {})
    return {
        "pass_rate": tp["overall_pass_rate"],
        "med_days": td.get("total_p50", 0),
        "p10_days": td.get("total_p10", 0),
        "p90_days": td.get("total_p90", 0),
        "p1_dd_fail": tp["p1_dd_fail_pct"],
        "p2_dd_fail": tp["p2_dd_fail_pct"],
    }


def main():
    t0 = time.perf_counter()

    print("\n" + "=" * 95)
    print("  FREQUENCY OPTIMIZATION")
    print("  Goal: 40-60 trades/yr with PF >= 1.8")
    print("  Locked: 1% risk, 2R TP, 1x ATR stop, EOD exit")
    print("=" * 95)

    df = load_csv("data/nq_15m_data.csv")
    print(f"  {len(df):,} bars loaded\n")

    # =========================================================================
    # PHASE 1: Broad sweep — backtest only (fast), find PF-viable configs
    # =========================================================================
    print("  PHASE 1: Signal count & edge quality scan")
    print("  " + "-" * 90)
    print(f"  {'Config':<42s} {'Sig':>5s} {'Tr':>5s} {'Tr/Yr':>6s} "
          f"{'PF':>6s} {'WR%':>6s} {'Sharpe':>7s} {'MaxDD':>7s} {'Viable':>7s}")
    print("  " + "-" * 90)

    all_rows = []

    for gap_th in GAP_THRESHOLDS:
        for fh_pct in FH_PCTILES:
            for dz_label, dz_lo, dz_hi in DZ_CONFIGS:
                strat_params = {
                    **BASE_STRAT,
                    "fh_percentile": fh_pct,
                    "gap_threshold_pct": gap_th,
                    "fh_dead_zone_lo": dz_lo,
                    "fh_dead_zone_hi": dz_hi,
                }

                label = f"Gap>{gap_th:.2f} FH{fh_pct:.0f}p {dz_label}"
                res = run_one(df, strat_params)

                viable = "YES" if res["pf"] >= 1.8 and res["trades_yr"] >= 25 else ""
                if res["pf"] >= 1.8 and res["trades_yr"] >= 35:
                    viable = "YES++"

                row = {
                    "label": label,
                    "gap_th": gap_th,
                    "fh_pct": fh_pct,
                    "dz_label": dz_label,
                    **{k: v for k, v in res.items() if k != "result"},
                    "viable": viable,
                    "_result": res["result"],
                }
                all_rows.append(row)

                print(
                    f"  {label:<42s} "
                    f"{res['signals']:>5d} "
                    f"{res['trades']:>5d} "
                    f"{res['trades_yr']:>6.1f} "
                    f"{res['pf']:>6.2f} "
                    f"{res['win_rate']:>6.1f} "
                    f"{res['sharpe']:>7.2f} "
                    f"{res['max_dd']:>7.2f} "
                    f"{viable:>7s}"
                )

    # =========================================================================
    # PHASE 2: Run 2-phase sim on viable configs only
    # =========================================================================
    viable_rows = [r for r in all_rows if r["pf"] >= 1.8 and r["trades_yr"] >= 25]

    print(f"\n\n  PHASE 2: Two-phase prop sim on {len(viable_rows)} viable configs")
    print("  " + "-" * 95)
    print(f"  {'Config':<42s} {'Tr/Yr':>6s} {'PF':>6s} {'Pass%':>6s} "
          f"{'MedD':>5s} {'P10':>5s} {'P90':>5s} {'DDFail':>7s} {'Sharpe':>7s}")
    print("  " + "-" * 95)

    final_rows = []
    for r in viable_rows:
        tp = run_2phase(r["_result"])
        combined = {k: v for k, v in r.items() if k != "_result"}
        combined.update(tp)
        final_rows.append(combined)

        tag = ""
        if tp["pass_rate"] >= 60 and tp["med_days"] <= 90:
            tag = " *** TARGET ***"
        elif tp["pass_rate"] >= 60 and tp["med_days"] <= 120:
            tag = " <<<"
        elif tp["pass_rate"] >= 60:
            tag = " <"

        print(
            f"  {r['label']:<42s} "
            f"{r['trades_yr']:>6.1f} "
            f"{r['pf']:>6.2f} "
            f"{tp['pass_rate']:>6.1f} "
            f"{tp['med_days']:>5.0f} "
            f"{tp['p10_days']:>5.0f} "
            f"{tp['p90_days']:>5.0f} "
            f"{tp['p1_dd_fail'] + tp['p2_dd_fail']:>7.1f} "
            f"{r['sharpe']:>7.2f}"
            f"{tag}"
        )

    # =========================================================================
    # Summary: best configs
    # =========================================================================
    print("\n\n" + "=" * 80)
    print("  TOP CONFIGS (sorted by median days, >=60% pass)")
    print("=" * 80)

    top = sorted(
        [r for r in final_rows if r["pass_rate"] >= 60],
        key=lambda r: r["med_days"],
    )

    if top:
        print(f"  {'Config':<42s} {'Tr/Yr':>6s} {'PF':>6s} {'Pass%':>6s} "
              f"{'MedD':>5s} {'P10':>5s}")
        print("  " + "-" * 72)
        for r in top[:10]:
            tag = " ***" if r["med_days"] <= 90 else ""
            print(
                f"  {r['label']:<42s} "
                f"{r['trades_yr']:>6.1f} "
                f"{r['pf']:>6.2f} "
                f"{r['pass_rate']:>6.1f} "
                f"{r['med_days']:>5.0f} "
                f"{r['p10_days']:>5.0f}"
                f"{tag}"
            )

        fastest = top[0]
        print(f"\n  RECOMMENDED: {fastest['label']}")
        print(f"    Trades/year:  {fastest['trades_yr']}")
        print(f"    PF:           {fastest['pf']:.2f}")
        print(f"    Pass rate:    {fastest['pass_rate']:.1f}%")
        print(f"    Median days:  {fastest['med_days']:.0f}")
        print(f"    P10 (fast):   {fastest['p10_days']:.0f}")
    else:
        print("  No config meets >=60% pass rate.")
        if final_rows:
            best = max(final_rows, key=lambda r: r["pass_rate"])
            print(f"  Best: {best['label']} — {best['pass_rate']:.1f}% pass, "
                  f"{best['med_days']:.0f}d, {best['trades_yr']} tr/yr, PF {best['pf']:.2f}")

    # Compare to baseline
    baseline = [r for r in final_rows
                if r["fh_pct"] == 75 and r["dz_label"] == "DZ 0.50-0.75"
                and r["gap_th"] == 0.10]
    if baseline and top:
        b = baseline[0]
        f = top[0]
        print(f"\n  vs BASELINE ({b['label']}):")
        print(f"    Trades/yr:  {b['trades_yr']} -> {f['trades_yr']}  "
              f"({f['trades_yr'] - b['trades_yr']:+.1f})")
        print(f"    PF:         {b['pf']:.2f} -> {f['pf']:.2f}  "
              f"({f['pf'] - b['pf']:+.2f})")
        print(f"    Pass rate:  {b.get('pass_rate', 0):.1f}% -> {f['pass_rate']:.1f}%  "
              f"({f['pass_rate'] - b.get('pass_rate', 0):+.1f}pp)")
        print(f"    Med days:   {b.get('med_days', 0):.0f} -> {f['med_days']:.0f}  "
              f"({f['med_days'] - b.get('med_days', 0):+.0f})")

    # Export
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    export_rows = [{k: v for k, v in r.items() if k != "_result"} for r in all_rows]
    pd.DataFrame(export_rows).to_csv(out_dir / "prop_firm_frequency_scan.csv", index=False)
    if final_rows:
        pd.DataFrame(final_rows).to_csv(out_dir / "prop_firm_frequency_viable.csv", index=False)
    print(f"\n  Exported to output/prop_firm_frequency_*.csv")

    elapsed = time.perf_counter() - t0
    print(f"  Runtime: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
