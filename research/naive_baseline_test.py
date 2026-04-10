"""
Naive baseline ladder test.

Runs five variants of the FH-momentum strategy on the same data with
the same backtester config (matching the locked aggressive prop config).
The only thing that varies between runs is the *signal-generation
filters*.  This isolates how much each filter actually contributes vs.
the bare directional edge.

Variants:
  1. Naive               — FH > 0, no filters
  2. + percentile        — FH > 75th expanding percentile
  3. + percentile + gap  — gap_momentum (current)
  4. regime_filtered     — regime_filtered_momentum (current)
  5. Combined portfolio  — both 3 and 4 (locked config baseline)

If (1) is profitable, the underlying edge is real.
If (1) is roughly as good as (3)/(4)/(5), the filters were noise.
If each step adds clear value, the complex strategy is justified.

Run from repo root:
    python -m research.naive_baseline_test
"""

import importlib
import logging

import numpy as np
import pandas as pd

from engine.backtester import run as run_backtest
from engine.data_loader import load_csv
from engine.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


DATA_FILE = "data/nq_15m_data.csv"
INITIAL_CAPITAL = 100_000.0


# Backtester config matches BASE_BACKTEST_CONFIG in run_portfolio.py
# (the locked aggressive prop portfolio config). Critical: max_dd_limit=0
# is DISABLED — otherwise the naive variant halts permanently after early
# losses and produces 9 trades instead of hundreds.
BACKTEST_CONFIG = {
    "initial_capital": INITIAL_CAPITAL,
    "risk_per_trade": 0.014,
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "spread_points": 0.25,
    "stop_atr_multiple": 1.0,
    "tp_atr_multiple": 2.0,
    "use_trailing_stop": False,
    "daily_dd_limit": 0.02,
    "max_daily_risk": 0.02,
    "max_dd_limit": 0.0,         # disabled (matches locked config)
    "use_volatility_sizing": True,
    "execution_delay_bars": 1,
    "max_bars_in_trade": 0,      # EOD exit via session_close
    "max_concurrent_trades": 1,
    "min_bars_between_entries": 2,
    "consec_loss_threshold": 2,
    "loss_scale_down": 0.5,
    "max_risk_per_trade": 0.0,   # disabled in locked config
}


VARIANTS = [
    {
        "name": "1. Naive (no filters)",
        "module": "strategies.naive_fh_momentum",
        "params": {},
    },
    {
        "name": "2. + 75th percentile",
        "module": "strategies.gap_momentum",
        "params": {"fh_percentile": 75.0, "gap_threshold_pct": -100.0},  # -100 disables gap filter
    },
    {
        "name": "3. + 75th pct + gap (gap_momentum)",
        "module": "strategies.gap_momentum",
        "params": {"fh_percentile": 75.0, "gap_threshold_pct": 0.10},
    },
    {
        "name": "4. regime_filtered_momentum",
        "module": "strategies.regime_filtered_momentum",
        "params": {
            "fh_percentile": 75.0,
            "fh_dead_zone_lo": 0.50,
            "fh_dead_zone_hi": 0.75,
            "trend_20d_max": 1.0,
            "vol_filter": True,
        },
    },
]


def run_variant(df: pd.DataFrame, name: str, module: str, params: dict) -> dict:
    log.info(f"Running: {name}")
    mod = importlib.import_module(module)
    signals = mod.generate_signals(df, **params)
    result = run_backtest(signals, strategy_name=name, **BACKTEST_CONFIG)
    metrics = compute_metrics(result, initial_capital=INITIAL_CAPITAL)

    trades = result.trades
    n = len(trades)
    if n == 0:
        return {
            "name": name, "trades": 0, "sharpe": 0.0, "return_pct": 0.0,
            "max_dd": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0,
        }

    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    wr = len(wins) / n * 100
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")

    return {
        "name": name,
        "trades": n,
        "sharpe": metrics["mtm"]["sharpe"],
        "return_pct": metrics["mtm"]["total_return"],
        "max_dd": metrics["mtm"]["max_drawdown"],
        "win_rate": wr,
        "profit_factor": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def main():
    log.info(f"Loading data from {DATA_FILE}")
    df = load_csv(DATA_FILE)
    log.info(f"Loaded {len(df)} bars: {df.index[0]} -> {df.index[-1]}")

    rows = []
    for v in VARIANTS:
        rows.append(run_variant(df, v["name"], v["module"], v["params"]))

    print()
    print("=" * 116)
    print(f"{'Variant':<40} {'Trades':>8} {'Sharpe':>8} {'Return%':>10} "
          f"{'MaxDD%':>10} {'WR%':>8} {'PF':>8} {'AvgWin':>10} {'AvgLoss':>10}")
    print("=" * 116)
    for r in rows:
        pf_str = f"{r['profit_factor']:>8.2f}" if r['profit_factor'] != float("inf") else f"{'inf':>8}"
        print(f"{r['name']:<40} {r['trades']:>8} {r['sharpe']:>8.2f} "
              f"{r['return_pct']:>10.1f} {r['max_dd']:>10.1f} "
              f"{r['win_rate']:>8.1f} {pf_str} "
              f"{r['avg_win']:>10.0f} {r['avg_loss']:>10.0f}")
    print("=" * 116)
    print()
    print("Reading the ladder:")
    print("  - If row 1 (naive) shows positive Sharpe + PF > 1, the underlying edge is real.")
    print("  - If row 2 doesn't beat row 1 by much, the percentile filter is mostly cosmetic.")
    print("  - If row 3 doesn't beat row 2, the gap filter is selection bias.")
    print("  - If row 4 doesn't beat row 1, the regime filters are noise.")


if __name__ == "__main__":
    main()
