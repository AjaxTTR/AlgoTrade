"""
Main entry point for the NQ Compression Breakout backtesting framework.

Usage:
    python main.py

Expects nq_15m_data.csv in the same directory.
"""

import sys
from data_loader import load_csv
from strategy import compression_breakout
from backtester import run
from metrics import compute_metrics, print_metrics, plot_results


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE = "nq_15m_data.csv"
INITIAL_CAPITAL = 100_000.0
RISK_PCT = 0.5              # Risk 0.5% of equity per trade
POINT_VALUE = 20.0           # NQ futures: $20 per point
COMMISSION = 4.0             # Round-trip per contract

# Strategy parameters
ATR_PERIOD = 14
COMPRESSION_LOOKBACK = 12
COMPRESSION_RATIO = 0.75
STOP_ATR_BUFFER = 0.5
REQUIRE_CANDLE_CONFIRM = True


def main() -> None:
    # --- Load data ---
    print(f"Loading data from {DATA_FILE} ...")
    try:
        df = load_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"ERROR: {DATA_FILE} not found. Place your OHLCV CSV in the project directory.")
        sys.exit(1)

    print(f"  Loaded {len(df):,} bars  |  {df.index[0]}  ->  {df.index[-1]}")

    # --- Generate signals ---
    print("Running compression breakout strategy ...")
    signals = compression_breakout(
        df,
        atr_period=ATR_PERIOD,
        compression_lookback=COMPRESSION_LOOKBACK,
        compression_ratio=COMPRESSION_RATIO,
        stop_atr_buffer=STOP_ATR_BUFFER,
        require_candle_confirm=REQUIRE_CANDLE_CONFIRM,
    )

    entry_count = (signals["signal"] != 0).sum()
    print(f"  Generated {entry_count} entry signals")

    # --- Backtest ---
    print("Running backtest ...")
    result = run(
        signals,
        initial_capital=INITIAL_CAPITAL,
        risk_pct=RISK_PCT,
        point_value=POINT_VALUE,
        commission_per_contract=COMMISSION,
    )

    # --- Metrics ---
    metrics = compute_metrics(result, initial_capital=INITIAL_CAPITAL)
    print_metrics(metrics)

    # --- Plots ---
    print("Generating plots ...")
    plot_results(result, metrics)


if __name__ == "__main__":
    main()
