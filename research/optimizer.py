"""
Parameter optimizer for strategy modules.

Runs a grid search over a parameter space, backtests each combination
in parallel using multiprocessing, and saves ranked results to
research/results.csv.

Usage:
    python -m research.optimizer [strategy_name]

If no strategy name is provided, defaults to "strategy".
"""

import importlib
import itertools
import logging
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

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
}

PARAM_GRID = {
    "atr_period": [10, 14, 20],
    "compression_lookback": [6, 8, 10],
    "compression_ratio": [1.0, 1.2, 1.4],
    "stop_atr_buffer": [0.8, 1.0, 1.2],
}

OUTPUT_PATH = Path("research/results.csv")
TOP_N = 10


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

    Parameters
    ----------
    args : tuple
        (params, strategy_name, data_file) where params is a dict of
        strategy parameters.

    Returns
    -------
    dict
        Result row with parameters, strategy name, and performance metrics.
    """
    params, strategy_name, data_file = args

    from engine.data_loader import load_csv
    from engine.backtester import run
    from engine.metrics import compute_metrics

    try:
        df = load_csv(data_file)
        strategy_module = importlib.import_module(f"strategies.{strategy_name}")
        signals = strategy_module.generate_signals(df.copy(), **params)
        bt_result = run(signals, **BACKTEST_CONFIG)
        metrics = compute_metrics(bt_result, initial_capital=BACKTEST_CONFIG["initial_capital"])

        mtm = metrics["mtm"]
        trades = metrics["trades"]

        return {
            "strategy_name": strategy_name,
            **params,
            "total_return": mtm["total_return"],
            "sharpe": mtm["sharpe"],
            "max_drawdown": mtm["max_drawdown"],
            "profit_factor": trades["profit_factor"],
            "trades": trades["total_trades"],
        }

    except Exception as exc:
        return {
            "strategy_name": strategy_name,
            **params,
            "total_return": None,
            "sharpe": None,
            "max_drawdown": None,
            "profit_factor": None,
            "trades": 0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def optimize(strategy_name: str = "strategy") -> pd.DataFrame:
    """Run grid search in parallel and return a DataFrame of results.

    Parameters
    ----------
    strategy_name : str
        Name of the strategy module inside strategies/ to optimise.
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

    # Build DataFrame and sort by Sharpe
    results_df = pd.DataFrame(results)
    results_df.sort_values("sharpe", ascending=False, na_position="last", inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    log.info("Results saved to %s", OUTPUT_PATH)

    # Print top N
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 90)
    print(f"  TOP {TOP_N} PARAMETER SETS BY SHARPE RATIO")
    print("=" * 90)
    display_cols = [
        "sharpe", "total_return", "max_drawdown", "profit_factor", "trades",
        *PARAM_GRID.keys(),
    ]
    print(results_df[display_cols].head(TOP_N).to_string(index=True))
    print("=" * 90)
    print(f"  Total combinations: {total}  |  Workers: {n_workers}  |  Runtime: {elapsed:.1f}s")
    print("=" * 90 + "\n")

    return results_df


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "strategy"
    optimize(name)
