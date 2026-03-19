"""
Parameter optimizer for strategy modules.

Runs a grid search over a parameter space, backtests each combination
in parallel using multiprocessing, and saves ranked results to
research/results.csv.

Usage:
    python -m research.optimizer [strategy_name]

If no strategy name is provided, defaults to "first_hour_momentum".
"""

import importlib
import itertools
import logging
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

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

BACKTEST_CONFIG = {
    "initial_capital": 100_000.0,
    "risk_per_trade": 0.005,
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "daily_dd_limit": 0.02,
    "max_daily_risk": 0.02,
    "max_bars_in_trade": 8,
    "max_concurrent_trades": 1,
    "min_bars_between_entries": 2,
    "consec_loss_threshold": 2,
    "loss_scale_down": 0.5,
}

BASE_STRATEGY_CONFIG = {
    "session_start": "09:30",
    "session_end": "16:00",
    "fh_end": "10:30",
    "entry_cutoff": "15:45",
    "fh_percentile": 80.0,
    "atr_period": 14,
    "holding_bars": 8,
    "pullback_atr_frac": 1.0,
    "min_bars_between_entries": 2,
    "enable_short": True,
}

PARAM_GRID = {
    "stop_atr_multiple": [1.0, 1.5, 2.0],
    "tp_atr_multiple": [1.5, 2.0, 3.0],
    "max_trades_per_day": [2, 3, 4],
}

# Multi-metric scoring configuration
SCORING = {
    "weights": {
        "sharpe": 0.5,
        "calmar": 0.3,
        "profit_factor": 0.2,
    },
    "filters": {
        "max_drawdown_threshold": -15.0,   # reject combos with DD worse than -15%
        "min_trades": 30,                   # reject combos with fewer trades
    },
}

OUTPUT_PATH = Path("research/results.csv")
TOP_N = 10


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_score(row: dict, scoring: dict) -> float:
    """Compute composite score for a parameter combination."""
    filters = scoring.get("filters", {})

    # Apply filters — return -inf if any filter fails
    max_dd = row.get("max_drawdown")
    if max_dd is not None and max_dd < filters.get("max_drawdown_threshold", -100):
        return -np.inf

    n_trades = row.get("trades", 0) or 0
    if n_trades < filters.get("min_trades", 0):
        return -np.inf

    # Weighted composite score
    weights = scoring["weights"]
    score = 0.0
    for metric, weight in weights.items():
        val = row.get(metric)
        if val is None:
            return -np.inf
        score += weight * val
    return score


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _build_combos(grid: dict) -> list[dict]:
    """Expand a parameter grid into a list of individual param dicts."""
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------------------
# Worker function (runs in child process)
# ---------------------------------------------------------------------------

def _evaluate_combo(args: tuple) -> dict:
    """Evaluate a single parameter combination.

    Each worker loads its own copy of data and strategy module to avoid
    shared-state issues across processes.
    """
    params, strategy_name, data_file = args

    from engine.data_loader import load_csv
    from engine.backtester import run
    from engine.metrics import compute_metrics, validate_strategy

    try:
        df = load_csv(data_file)
        strategy_module = importlib.import_module(f"strategies.{strategy_name}")
        merged_params = {**BASE_STRATEGY_CONFIG, **params}
        signals = strategy_module.generate_signals(df.copy(), **merged_params)
        bt_result = run(signals, **BACKTEST_CONFIG)
        metrics = compute_metrics(bt_result, initial_capital=BACKTEST_CONFIG["initial_capital"])

        mtm = metrics["mtm"]
        trades = metrics["trades"]
        validation = validate_strategy(metrics)

        return {
            "strategy_name": strategy_name,
            **params,
            "total_return": mtm["total_return"],
            "sharpe": mtm["sharpe"],
            "sortino": mtm["sortino"],
            "calmar": mtm["calmar"],
            "max_drawdown": mtm["max_drawdown"],
            "profit_factor": trades["profit_factor"],
            "trades": trades["total_trades"],
            "valid": validation["passed"],
            "validation_failures": len(validation["failures"]),
        }

    except Exception as exc:
        return {
            "strategy_name": strategy_name,
            **params,
            "total_return": None,
            "sharpe": None,
            "sortino": None,
            "calmar": None,
            "max_drawdown": None,
            "profit_factor": None,
            "trades": 0,
            "valid": False,
            "validation_failures": -1,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def optimize(
    strategy_name: str = "first_hour_momentum",
    skip_validation: bool = False,
    skip_edge_check: bool = False,
    validation_thresholds: dict | None = None,
    edge_thresholds: dict | None = None,
) -> pd.DataFrame:
    """Run grid search in parallel and return a DataFrame of results.

    Parameters
    ----------
    strategy_name : str
        Strategy module name under strategies/.
    skip_validation : bool
        If True, skip the baseline backtest validation gate (default False).
    skip_edge_check : bool
        If True, skip the edge prerequisite check (default False).
        When False, the optimizer runs a fast statistical scan of the data
        to confirm at least one significant, stable edge exists before
        committing to the grid search.
    validation_thresholds : dict | None
        Override default validation thresholds for the baseline check.
    edge_thresholds : dict | None
        Override default thresholds for the edge prerequisite check.
    """
    t_start = time.perf_counter()

    # Validate data file exists before spawning workers
    if not Path(DATA_FILE).exists():
        log.error("Data file not found: %s", DATA_FILE)
        sys.exit(1)

    # Validate strategy module exists
    try:
        mod = importlib.import_module(f"strategies.{strategy_name}")
    except ModuleNotFoundError:
        log.error("Strategy module 'strategies/%s.py' not found.", strategy_name)
        sys.exit(1)
    if not hasattr(mod, "generate_signals"):
        log.error("Strategy '%s' missing generate_signals().", strategy_name)
        sys.exit(1)

    # --- Edge prerequisite gate (hard stop) ---
    if not skip_edge_check:
        from research.edge_analysis import check_edge_prerequisites

        log.info("Running edge prerequisite check on raw data...")
        try:
            edge_result = check_edge_prerequisites(
                data_file=DATA_FILE,
                thresholds=edge_thresholds,
            )
            if not edge_result["passed"]:
                log.error(
                    "EDGE CHECK FAILED: no qualifying statistical edge found. "
                    "Optimisation aborted. Use skip_edge_check=True to override."
                )
                for reason in edge_result["failures"]:
                    log.error("  %s", reason)
                return pd.DataFrame()
            log.info(
                "Edge check PASSED: %d qualifying edge(s) confirmed.",
                edge_result["n_significant"],
            )
        except Exception as exc:
            log.warning("Edge prerequisite check could not run: %s", exc)
            log.warning("Proceeding with optimisation (edge check inconclusive).")

    # --- Baseline backtest validation gate (soft warning) ---
    if not skip_validation:
        from engine.data_loader import load_csv
        from engine.backtester import run as run_backtest
        from engine.metrics import compute_metrics, validate_strategy, print_validation

        log.info("Running baseline validation with default parameters...")
        df_val = load_csv(DATA_FILE)
        # Use the first set of values from the grid as baseline
        baseline_params = {k: v[0] for k, v in PARAM_GRID.items()}
        merged = {**BASE_STRATEGY_CONFIG, **baseline_params}
        try:
            signals = mod.generate_signals(df_val.copy(), **merged)
            result = run_backtest(signals, **BACKTEST_CONFIG)
            metrics = compute_metrics(result, initial_capital=BACKTEST_CONFIG["initial_capital"])
            validation = validate_strategy(metrics, thresholds=validation_thresholds)
            print_validation(validation)

            if not validation["passed"]:
                log.warning(
                    "Baseline validation FAILED (%d/%d checks). "
                    "Strategy may not be worth optimising. "
                    "Use skip_validation=True to override.",
                    validation["checks_passed"], validation["checks_run"],
                )
                # Still continue — the grid may find passing combos
                # but warn loudly so the user is aware
        except Exception as exc:
            log.warning("Baseline validation could not run: %s", exc)

    # Build parameter combinations
    combos = _build_combos(PARAM_GRID)
    total = len(combos)
    n_workers = cpu_count()
    log.info("Parameter combinations: %d  |  Workers: %d", total, n_workers)

    # Build args list for pool.map
    args_list = [(params, strategy_name, DATA_FILE) for params in combos]

    # Run in parallel
    with Pool(n_workers) as pool:
        results = pool.map(_evaluate_combo, args_list)

    # Log any failures
    failures = [r for r in results if r.get("error")]
    if failures:
        log.warning("%d / %d combinations failed", len(failures), total)
        for f in failures:
            log.warning("  %s: %s", {k: v for k, v in f.items() if k in PARAM_GRID}, f["error"])

    # Clean error key before building DataFrame
    for r in results:
        r.pop("error", None)

    # Compute composite scores and sort
    for r in results:
        r["composite_score"] = _compute_score(r, SCORING)

    results_df = pd.DataFrame(results)
    results_df.sort_values("composite_score", ascending=False, na_position="last", inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    log.info("Results saved to %s", OUTPUT_PATH)

    # Print top N
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 100)
    print(f"  TOP {TOP_N} PARAMETER SETS BY COMPOSITE SCORE")
    print("=" * 100)
    display_cols = [
        "composite_score", "sharpe", "calmar", "total_return", "max_drawdown",
        "profit_factor", "trades", *PARAM_GRID.keys(),
    ]
    print(results_df[display_cols].head(TOP_N).to_string(index=True))
    print("=" * 100)
    print(f"  Scoring weights: {SCORING['weights']}")
    print(f"  Filters: {SCORING['filters']}")
    print(f"  Total combinations: {total}  |  Workers: {n_workers}  |  Runtime: {elapsed:.1f}s")

    # Validation summary
    if "valid" in results_df.columns:
        n_valid = int(results_df["valid"].sum())
        n_invalid = total - n_valid
        print(f"\n  Validation: {n_valid}/{total} combos passed  |  {n_invalid} rejected")
        if n_valid == 0:
            print("  WARNING: No parameter combination passed validation.")

    print("=" * 100 + "\n")

    return results_df


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    name = args[0] if args else "first_hour_momentum"
    optimize(
        name,
        skip_edge_check="--skip-edge-check" in flags,
        skip_validation="--skip-validation" in flags,
    )
