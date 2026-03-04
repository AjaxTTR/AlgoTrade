"""
Main entry point for the NQ Compression Breakout backtesting framework.

Usage:
    python main.py

Expects nq_15m_data.csv in the same directory.
Outputs: trade_log.csv, equity_curve.csv, drawdown_series.csv,
         performance_metrics.json, backtest_results.png
"""

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from data_loader import load_csv
from strategy import compression_breakout
from backtester import run
from metrics import compute_metrics, print_metrics, plot_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = "nq_15m_data.csv"

BACKTEST_CONFIG = {
    "initial_capital": 100_000.0,
    "risk_per_trade": 0.005,        # 0.5% of equity (decimal)
    "point_value": 20.0,            # NQ futures: $20 per point
    "commission_per_side": 2.0,     # per contract per side
    "slippage_points": 0.25,        # adverse points per fill
}

STRATEGY_CONFIG = {
    "atr_period": 14,
    "compression_lookback": 8,
    "compression_ratio": 1.2,
    "stop_atr_buffer": 1.0,
    "require_candle_confirm": False,
}

OUTPUT_DIR = Path(".")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_config(title: str, config: dict) -> None:
    """Print a configuration block."""
    print(f"\n  {title}")
    print("  " + "-" * 40)
    for key, value in config.items():
        label = key.replace("_", " ").title()
        print(f"    {label:30s} {value}")


def _export_trades(trades: list, path: Path) -> None:
    """Convert trade records to DataFrame and save as CSV."""
    if not trades:
        log.warning("No trades to export")
        return
    df = pd.DataFrame([asdict(t) for t in trades])
    df["direction"] = df["direction"].map({1: "LONG", -1: "SHORT"})
    df.to_csv(path, index=False)
    log.info("Trade log saved to %s (%d trades)", path, len(trades))


def _export_results(result, metrics: dict, output_dir: Path) -> None:
    """Save equity curves, drawdown, and metrics to disk."""
    # Equity curve (MTM + closed)
    equity_df = pd.DataFrame({
        "equity_mtm": result.equity_mtm,
        "equity_closed": result.equity_closed,
    })
    equity_path = output_dir / "equity_curve.csv"
    equity_df.to_csv(equity_path)
    log.info("Equity curve saved to %s", equity_path)

    # Drawdown series
    dd_path = output_dir / "drawdown_series.csv"
    result.drawdown_series.to_csv(dd_path, header=True)
    log.info("Drawdown series saved to %s", dd_path)

    # Performance metrics (JSON)
    metrics_path = output_dir / "performance_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Metrics saved to %s", metrics_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()

    print("\n" + "=" * 62)
    print("  NQ COMPRESSION BREAKOUT — BACKTEST")
    print("=" * 62)

    _print_config("Backtest Configuration", BACKTEST_CONFIG)
    _print_config("Strategy Configuration", STRATEGY_CONFIG)
    print()

    # --- Load data ---
    log.info("Loading data from %s", DATA_FILE)
    try:
        df = load_csv(DATA_FILE)
    except FileNotFoundError:
        log.error("%s not found. Place your OHLCV CSV in the project directory.", DATA_FILE)
        sys.exit(1)
    except ValueError as exc:
        log.error("Data validation failed: %s", exc)
        sys.exit(1)

    n_bars = len(df)
    log.info("Loaded %s bars  |  %s  ->  %s", f"{n_bars:,}", df.index[0], df.index[-1])

    # --- Generate signals ---
    log.info("Running compression breakout strategy")
    try:
        signals = compression_breakout(df, **STRATEGY_CONFIG)
    except Exception as exc:
        log.error("Strategy error: %s", exc)
        sys.exit(1)

    n_signals = int((signals["signal"] != 0).sum())
    log.info("Generated %d entry signals", n_signals)

    # --- Backtest ---
    log.info("Running backtest engine")
    try:
        result = run(signals, **BACKTEST_CONFIG)
    except Exception as exc:
        log.error("Backtest error: %s", exc)
        sys.exit(1)

    n_trades = len(result.trades)
    log.info("Completed %d trades", n_trades)

    # --- Metrics ---
    metrics = compute_metrics(result, initial_capital=BACKTEST_CONFIG["initial_capital"])
    print_metrics(metrics)

    # --- Export ---
    _export_trades(result.trades, OUTPUT_DIR / "trade_log.csv")
    _export_results(result, metrics, OUTPUT_DIR)

    # --- Plots ---
    log.info("Generating plots")
    plot_results(result, metrics)

    # --- Runtime summary ---
    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 62)
    print("  EXECUTION SUMMARY")
    print("=" * 62)
    print(f"    Bars processed:      {n_bars:>10,}")
    print(f"    Signals generated:   {n_signals:>10,}")
    print(f"    Trades completed:    {n_trades:>10,}")
    print(f"    Runtime:             {elapsed:>10.2f} s")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
