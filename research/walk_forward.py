"""
Walk-forward analysis for strategy validation.

Splits data into rolling train/test windows, optimizes on train,
tests on out-of-sample, and aggregates OOS results to detect
overfitting.

Usage:
    python -m research.walk_forward [strategy_name]
"""

import importlib
import itertools
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest
from engine.metrics import compute_metrics

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
    "risk_per_trade": 0.005,
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
}

PARAM_GRID = {
    "orb_minutes": [15, 30, 45],
    "tp_atr_multiple": [1.5, 2.0, 2.5],
    "trade_window_end": ["10:30", "11:00", "11:30"],
}

OUTPUT_PATH = Path("research/walk_forward_results.csv")


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
            signals = strategy_module.generate_signals(df.copy(), **params)
            result = run_backtest(signals, **backtest_config)
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

def walk_forward(strategy_name: str = "opening_range_breakout") -> pd.DataFrame:
    """Run walk-forward analysis and return per-fold results."""
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

        # Optimize on training window
        best_params, is_sharpe = _grid_search(
            df_train, strategy_module, PARAM_GRID, BACKTEST_CONFIG,
        )

        # Test on out-of-sample window
        try:
            signals = strategy_module.generate_signals(df_test.copy(), **best_params)
            result = run_backtest(signals, **BACKTEST_CONFIG)
            metrics = compute_metrics(result, initial_capital=BACKTEST_CONFIG["initial_capital"])

            oos_sharpe = metrics["mtm"]["sharpe"]
            oos_return = metrics["mtm"]["total_return"]
            oos_max_dd = metrics["mtm"]["max_drawdown"]
            oos_trades = metrics["trades"]["total_trades"]
        except Exception as exc:
            log.warning("  Fold %d OOS failed: %s", i + 1, exc)
            oos_sharpe = oos_return = oos_max_dd = 0.0
            oos_trades = 0

        log.info("  IS Sharpe: %.2f | OOS Sharpe: %.2f | OOS Return: %.2f%% | Params: %s",
                 is_sharpe, oos_sharpe, oos_return, best_params)

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

    print(f"    Runtime:                {elapsed:>7.1f}s")
    print("=" * 90 + "\n")

    return results_df


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "opening_range_breakout"
    walk_forward(name)
