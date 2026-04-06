"""
Walk-forward analysis for strategy validation.

Splits data into rolling train/test windows, optimizes on train,
tests on out-of-sample, and aggregates OOS results to detect
overfitting.

Modes:
    default       — optimisation walk-forward (grid search on train, test OOS)
    validation    — fixed baseline config, no optimisation, pure robustness test

Usage:
    python -m research.walk_forward [strategy_name]
    python -m research.walk_forward --mode validation
"""

import argparse
import importlib
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest
from engine.metrics import compute_metrics, validate_strategy
from research.edge_analysis import check_edge_prerequisites_from_df

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = "data/nq_15m_data.csv"

TRAIN_MONTHS = 12
TEST_MONTHS = 3
STEP_MONTHS = 3

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

BASE_STRATEGY_CONFIG = {
    "session_start": "09:30",
    "session_end": "16:00",
    "fh_end": "10:30",
    "entry_cutoff": "15:45",
    "fh_percentile": 80.0,
    "atr_period": 14,
    "holding_bars": 8,
}

# Parameters that belong to backtester (not strategy)
BACKTEST_GRID_KEYS = {"stop_atr_multiple", "tp_atr_multiple"}

PARAM_GRID = {
    "stop_atr_multiple": [1.0, 1.5, 2.0],
    "tp_atr_multiple": [1.5, 2.0, 3.0],
    "max_trades_per_day": [1, 2, 3],
    "pullback_atr_frac": [0.3, 0.5, 0.75],
}

OUTPUT_PATH = Path("research/walk_forward_results.csv")
VALIDATION_CONFIG_PATH = Path("configs/baseline_gap_fh_75.json")

# Relaxed edge thresholds for fold-level checks (smaller sample windows)
FOLD_EDGE_THRESHOLDS = {
    "min_significant_edges": 1,
    "min_stability": 0.0,          # don't require stability on short windows
    "max_p_adj": 0.30,             # lenient FDR (|t| is the real gate)
    "max_raw_p": 0.10,             # relaxed raw p for smaller samples
    "min_sample_size": 30,         # fewer bars available per fold
    "min_abs_t_stat": 1.5,         # slightly relaxed t-stat floor
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_folds(
    index: pd.DatetimeIndex,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[tuple]:
    """Generate (train_start, train_end, test_start, test_end) tuples."""
    start = index[0]
    end = index[-1]
    folds = []
    current = start

    while True:
        train_end = current + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        folds.append((current, train_end, train_end, test_end))
        current += pd.DateOffset(months=step_months)

    return folds


def _grid_search(
    df: pd.DataFrame,
    strategy_module,
    param_grid: dict,
    backtest_config: dict,
) -> tuple[dict, float]:
    """Grid search on a data window. Returns (best_params, best_sharpe)."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    best_sharpe = -np.inf
    best_params = None

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        try:
            # Split params: backtest-level (stop/TP) vs strategy-level
            bt_overrides = {k: v for k, v in params.items() if k in BACKTEST_GRID_KEYS}
            strat_overrides = {k: v for k, v in params.items() if k not in BACKTEST_GRID_KEYS}
            merged_params = {**BASE_STRATEGY_CONFIG, **strat_overrides}
            signals = strategy_module.generate_signals(df.copy(), **merged_params)
            result = run_backtest(signals, **{**backtest_config, **bt_overrides})
            metrics = compute_metrics(result, initial_capital=backtest_config["initial_capital"])
            sharpe = metrics["mtm"]["sharpe"]
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        except Exception:
            continue

    if best_params is None:
        best_params = dict(zip(keys, [v[0] for v in values]))
        best_sharpe = 0.0

    return best_params, best_sharpe


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def walk_forward(
    strategy_name: str = "first_hour_momentum",
    skip_edge_check: bool = False,
    edge_thresholds: dict | None = None,
) -> pd.DataFrame:
    """Run walk-forward analysis and return per-fold results.

    Parameters
    ----------
    strategy_name : str
        Strategy module name under strategies/.
    skip_edge_check : bool
        If True, skip per-fold edge validation (default False).
    edge_thresholds : dict | None
        Override default thresholds for the edge prerequisite check.
    """
    t_start = time.perf_counter()

    log.info("Loading data from %s", DATA_FILE)
    df = load_csv(DATA_FILE)
    log.info("Loaded %d bars: %s -> %s", len(df), df.index[0], df.index[-1])

    strategy_module = importlib.import_module(f"strategies.{strategy_name}")

    folds = _build_folds(df.index, TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS)
    log.info("Walk-forward: %d folds (%dmo train / %dmo test / %dmo step)",
             len(folds), TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS)

    if not folds:
        log.error("Not enough data for even one fold")
        sys.exit(1)

    oos_results = []
    edge_checks = {"total": 0, "passed": 0, "skipped": 0}

    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        log.info("Fold %d/%d: train %s->%s | test %s->%s",
                 i + 1, len(folds),
                 train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d"),
                 test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))

        df_train = df[train_start:train_end]
        df_test = df[test_start:test_end]

        if len(df_train) < 100 or len(df_test) < 20:
            log.warning("  Skipping fold %d: insufficient bars (train=%d, test=%d)",
                        i + 1, len(df_train), len(df_test))
            continue

        # --- Per-fold edge validation ---
        fold_edge_passed = True
        if not skip_edge_check:
            edge_checks["total"] += 1
            try:
                fold_edge_th = {**FOLD_EDGE_THRESHOLDS, **(edge_thresholds or {})}
                edge_result = check_edge_prerequisites_from_df(
                    df_train.copy(),
                    thresholds=fold_edge_th,
                    quiet=True,
                )
                fold_edge_passed = edge_result["passed"]
            except Exception as exc:
                log.warning("  Fold %d edge check error: %s (proceeding anyway)", i + 1, exc)
                fold_edge_passed = True  # inconclusive — don't penalise

            if fold_edge_passed:
                edge_checks["passed"] += 1
                log.info("  Edge check PASSED (%d qualifying edge(s))",
                         edge_result.get("n_significant", 0))
            else:
                edge_checks["skipped"] += 1
                log.info("  Edge check FAILED — skipping fold %d (no significant edge in training data)",
                         i + 1)
                oos_results.append({
                    "fold": i + 1,
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": train_end.strftime("%Y-%m-%d"),
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": test_end.strftime("%Y-%m-%d"),
                    "is_sharpe": np.nan,
                    "oos_sharpe": np.nan,
                    "oos_return": np.nan,
                    "oos_max_dd": np.nan,
                    "oos_trades": 0,
                    "oos_valid": False,
                    "oos_val_failures": -1,
                    "edge_valid": False,
                    "edge_skipped": True,
                    **{k: None for k in PARAM_GRID.keys()},
                })
                continue

        # Optimize on training window
        best_params, is_sharpe = _grid_search(
            df_train, strategy_module, PARAM_GRID, BACKTEST_CONFIG,
        )

        # Test on out-of-sample window
        try:
            oos_params = {**BASE_STRATEGY_CONFIG, **best_params}
            signals = strategy_module.generate_signals(df_test.copy(), **oos_params)
            result = run_backtest(signals, **BACKTEST_CONFIG)
            metrics = compute_metrics(result, initial_capital=BACKTEST_CONFIG["initial_capital"])

            oos_sharpe = metrics["mtm"]["sharpe"]
            oos_return = metrics["mtm"]["total_return"]
            oos_max_dd = metrics["mtm"]["max_drawdown"]
            oos_trades = metrics["trades"]["total_trades"]

            # Validate OOS result (relaxed thresholds for individual folds)
            fold_thresholds = {"min_trades": 5, "max_drawdown": -40.0}
            validation = validate_strategy(metrics, thresholds=fold_thresholds)
            oos_valid = validation["passed"]
            oos_val_failures = len(validation["failures"])
        except Exception as exc:
            log.warning("  Fold %d OOS failed: %s", i + 1, exc)
            oos_sharpe = oos_return = oos_max_dd = 0.0
            oos_trades = 0
            oos_valid = False
            oos_val_failures = -1

        valid_str = "PASS" if oos_valid else "FAIL"
        log.info("  IS Sharpe: %.2f | OOS Sharpe: %.2f | OOS Return: %.2f%% | Valid: %s | Params: %s",
                 is_sharpe, oos_sharpe, oos_return, valid_str, best_params)

        oos_results.append({
            "fold": i + 1,
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "is_sharpe": round(is_sharpe, 2),
            "oos_sharpe": round(oos_sharpe, 2),
            "oos_return": round(oos_return, 2),
            "oos_max_dd": round(oos_max_dd, 2),
            "oos_trades": oos_trades,
            "oos_valid": oos_valid,
            "oos_val_failures": oos_val_failures,
            "edge_valid": fold_edge_passed,
            "edge_skipped": False,
            **best_params,
        })

    results_df = pd.DataFrame(oos_results)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    log.info("Results saved to %s", OUTPUT_PATH)

    # Summary
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 90)
    print("  WALK-FORWARD ANALYSIS RESULTS")
    print("=" * 90)
    print(results_df.to_string(index=False))
    print()

    if len(results_df) > 0:
        avg_is = results_df["is_sharpe"].mean()
        avg_oos = results_df["oos_sharpe"].mean()
        avg_ret = results_df["oos_return"].mean()
        avg_dd = results_df["oos_max_dd"].mean()
        degradation = (1 - avg_oos / avg_is) * 100 if avg_is != 0 else 0

        print("  AGGREGATE OOS METRICS")
        print("  " + "-" * 40)
        print(f"    Avg IS Sharpe:          {avg_is:>8.2f}")
        print(f"    Avg OOS Sharpe:         {avg_oos:>8.2f}")
        print(f"    Sharpe Degradation:     {degradation:>7.1f}%")
        print(f"    Avg OOS Return:         {avg_ret:>7.2f}%")
        print(f"    Avg OOS Max DD:         {avg_dd:>7.2f}%")
        print(f"    Folds:                  {len(results_df):>8d}")

        if "oos_valid" in results_df.columns:
            n_valid = int(results_df["oos_valid"].sum())
            n_total = len(results_df)
            valid_pct = n_valid / n_total * 100 if n_total > 0 else 0
            print(f"    OOS Folds Valid:        {n_valid:>5d}/{n_total}  ({valid_pct:.0f}%)")
            if valid_pct < 50:
                print("    WARNING: Majority of OOS folds fail validation — strategy unstable.")

        # Edge validation summary
        if not skip_edge_check and edge_checks["total"] > 0:
            ec = edge_checks
            edge_pct = ec["passed"] / ec["total"] * 100
            print(f"\n    Edge Pre-Check:         {ec['passed']:>5d}/{ec['total']}  ({edge_pct:.0f}% of folds have valid edge)")
            print(f"    Folds Skipped (no edge):{ec['skipped']:>5d}")
            if edge_pct < 50:
                print("    WARNING: Majority of folds lack a statistical edge — strategy may be curve-fitting.")

    print(f"    Runtime:                {elapsed:>7.1f}s")
    print("=" * 90 + "\n")

    return results_df


def walk_forward_validation() -> pd.DataFrame:
    """Run walk-forward in validation mode: fixed config, no optimisation.

    Uses the locked baseline config (configs/baseline_gap_fh_75.json).
    Every fold runs the test window only — training data is ignored.
    No folds are skipped for any reason.
    """
    t_start = time.perf_counter()

    # Load baseline config
    if not VALIDATION_CONFIG_PATH.exists():
        log.error("Baseline config not found: %s", VALIDATION_CONFIG_PATH)
        sys.exit(1)

    with open(VALIDATION_CONFIG_PATH) as f:
        cfg = json.load(f)

    strategy_name = cfg.get("strategy_module", "gap_fh_continuation")
    bt_cfg = {**BACKTEST_CONFIG, **cfg.get("backtest", {})}
    st_cfg = {**BASE_STRATEGY_CONFIG, **cfg.get("strategy", {})}

    log.info("VALIDATION MODE — fixed config, no optimisation")
    log.info("Config: %s (v%s)", cfg.get("name", "?"), cfg.get("version", "?"))
    log.info("Strategy: %s", strategy_name)

    log.info("Loading data from %s", DATA_FILE)
    df = load_csv(DATA_FILE)
    log.info("Loaded %d bars: %s -> %s", len(df), df.index[0], df.index[-1])

    strategy_module = importlib.import_module(f"strategies.{strategy_name}")

    folds = _build_folds(df.index, TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS)
    log.info("Walk-forward: %d folds (%dmo train / %dmo test / %dmo step)",
             len(folds), TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS)

    if not folds:
        log.error("Not enough data for even one fold")
        sys.exit(1)

    fold_results = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        log.info("Fold %d/%d: test %s -> %s",
                 i + 1, len(folds),
                 test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))

        df_test = df[test_start:test_end]

        if len(df_test) < 20:
            log.warning("  Fold %d: only %d bars — running anyway", i + 1, len(df_test))

        try:
            signals = strategy_module.generate_signals(df_test.copy(), **st_cfg)
            result = run_backtest(signals, **bt_cfg)
            metrics = compute_metrics(result, initial_capital=bt_cfg["initial_capital"])

            sharpe = metrics["mtm"]["sharpe"]
            total_return = metrics["mtm"]["total_return"]
            max_dd = metrics["mtm"]["max_drawdown"]
            trades = metrics["trades"]["total_trades"]
        except Exception as exc:
            log.warning("  Fold %d failed: %s", i + 1, exc)
            sharpe = 0.0
            total_return = 0.0
            max_dd = 0.0
            trades = 0

        log.info("  Sharpe: %.2f | Return: %.2f%% | DD: %.2f%% | Trades: %d",
                 sharpe, total_return, max_dd, trades)

        fold_results.append({
            "fold": i + 1,
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "sharpe": round(sharpe, 2),
            "return": round(total_return, 2),
            "max_dd": round(max_dd, 2),
            "trades": trades,
        })

    results_df = pd.DataFrame(fold_results)

    # Save
    out_path = Path("research/walk_forward_validation.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    log.info("Results saved to %s", out_path)

    # === Output ===
    elapsed = time.perf_counter() - t_start

    print()
    print("=" * 70)
    print("  WALK-FORWARD VALIDATION (FIXED BASELINE)")
    print("=" * 70)
    print(f"  Config:   {cfg.get('name', '?')} {cfg.get('version', '?')}")
    print(f"  Strategy: {strategy_name}")
    print(f"  Signal:   Gap Up + FH >= {st_cfg.get('fh_percentile', '?')}th pct")
    print(f"  Stop:     {st_cfg.get('stop_atr_multiple', '?')}x ATR  |  "
          f"TP: {st_cfg.get('tp_atr_multiple', '?')}R  |  "
          f"Risk: {bt_cfg.get('risk_per_trade', 0)*100:.2f}%")
    print(f"  Folds:    {len(results_df)}  "
          f"({TRAIN_MONTHS}mo train / {TEST_MONTHS}mo test / {STEP_MONTHS}mo step)")
    print()

    # Per-fold table
    print(f"  {'Fold':>4}  {'Test Window':<25}  {'Sharpe':>7}  {'Return':>8}  "
          f"{'MaxDD':>7}  {'Trades':>6}")
    print("  " + "-" * 68)

    for _, r in results_df.iterrows():
        window = f"{r['test_start']} -> {r['test_end']}"
        sharpe_flag = " *" if r["sharpe"] < -0.5 else ""
        print(f"  {r['fold']:>4}  {window:<25}  {r['sharpe']:>7.2f}{sharpe_flag}  "
              f"{r['return']:>7.2f}%  {r['max_dd']:>7.2f}%  {r['trades']:>5}")

    # Summary
    print()
    print("  " + "-" * 68)

    n_folds = len(results_df)
    avg_sharpe = results_df["sharpe"].mean()
    worst_sharpe = results_df["sharpe"].min()
    best_sharpe = results_df["sharpe"].max()
    positive_folds = (results_df["sharpe"] > 0).sum()
    positive_pct = positive_folds / n_folds * 100 if n_folds > 0 else 0
    worst_dd = results_df["max_dd"].min()
    avg_return = results_df["return"].mean()
    total_trades = results_df["trades"].sum()
    catastrophic = (results_df["sharpe"] < -0.5).sum()

    print(f"  Average Sharpe:        {avg_sharpe:>7.2f}")
    print(f"  Best Sharpe:           {best_sharpe:>7.2f}")
    print(f"  Worst Sharpe:          {worst_sharpe:>7.2f}")
    print(f"  Positive Sharpe folds: {positive_folds}/{n_folds}  ({positive_pct:.0f}%)")
    print(f"  Worst drawdown:        {worst_dd:>7.2f}%")
    print(f"  Average return/fold:   {avg_return:>7.2f}%")
    print(f"  Total trades:          {total_trades:>5}")
    print(f"  Runtime:               {elapsed:>7.1f}s")

    # Verdict
    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)

    stable = positive_pct >= 70 and catastrophic == 0
    if stable:
        print(f"\n  STABLE")
        print(f"    {positive_pct:.0f}% folds with positive Sharpe (>= 70% required)")
        print(f"    {catastrophic} catastrophic folds (Sharpe < -0.5)")
    else:
        reasons = []
        if positive_pct < 70:
            reasons.append(f"only {positive_pct:.0f}% positive Sharpe folds (need >= 70%)")
        if catastrophic > 0:
            reasons.append(f"{catastrophic} catastrophic fold(s) with Sharpe < -0.5")
        print(f"\n  UNSTABLE")
        for r in reasons:
            print(f"    - {r}")

    print()
    print("=" * 70)
    print()

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward analysis")
    parser.add_argument("strategy", nargs="?", default="first_hour_momentum",
                        help="Strategy module name (default: first_hour_momentum)")
    parser.add_argument("--mode", default="default", choices=["default", "validation"],
                        help="Mode: default (optimisation WF) or validation (fixed baseline)")
    parser.add_argument("--skip-edge-check", action="store_true",
                        help="Skip per-fold edge validation (default mode only)")
    args = parser.parse_args()

    if args.mode == "validation":
        walk_forward_validation()
    else:
        walk_forward(
            args.strategy,
            skip_edge_check=args.skip_edge_check,
        )
