"""
Prop firm scalability comparison.

Tests 6 configurations: {long-only, long+short} x {0.5%, 0.75%, 1.0% risk}
Runs backtest, Monte Carlo prop sim, and tier validation for each.

Usage:
    python -m research.prop_scaling_comparison
"""

import importlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest
from engine.metrics import compute_metrics, validate_strategy
from engine.prop_firm import simulate_prop_firm

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress noisy backtester DD warnings during batch runs
logging.getLogger("engine.backtester").setLevel(logging.ERROR)

DATA_FILE = "data/nq_15m_data.csv"
STRATEGY_NAME = "first_hour_momentum"

BASE_BACKTEST = {
    "initial_capital": 100_000.0,
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "use_trailing_stop": False,
    "daily_dd_limit": 0.02,
    "max_daily_risk": 0.02,
    "max_dd_limit": 0.0,
    "max_bars_in_trade": 8,
    "max_concurrent_trades": 1,
    "min_bars_between_entries": 2,
    "consec_loss_threshold": 2,
    "loss_scale_down": 0.5,
}

BASE_STRATEGY = {
    "session_start": "09:30",
    "session_end": "16:00",
    "fh_end": "10:30",
    "entry_cutoff": "15:45",
    "fh_percentile": 80.0,
    "atr_period": 14,
    "stop_atr_multiple": 1.5,
    "tp_atr_multiple": 2.0,
    "holding_bars": 8,
    "max_trades_per_day": 4,
    "pullback_atr_frac": 1.0,
    "min_bars_between_entries": 2,
}

PROP_FIRM_CONFIGS = {
    "FTMO Ph1 (10%/30d)": {
        "starting_balance": 100_000.0,
        "profit_target": 0.10,
        "daily_drawdown_limit": 0.05,
        "max_drawdown_limit": 0.10,
        "max_days": 30,
    },
    "FTMO Ph2 (5%/60d)": {
        "starting_balance": 100_000.0,
        "profit_target": 0.05,
        "daily_drawdown_limit": 0.05,
        "max_drawdown_limit": 0.10,
        "max_days": 60,
    },
    "Conserv (8%/45d)": {
        "starting_balance": 100_000.0,
        "profit_target": 0.08,
        "daily_drawdown_limit": 0.04,
        "max_drawdown_limit": 0.08,
        "max_days": 45,
    },
}

RISK_LEVELS = [0.005, 0.0075, 0.01]
MODES = [
    ("Long Only", False),
    ("Long+Short", True),
]

# Train/test split for tier validation
TRAIN_FRAC = 0.70


def _split_by_date(signals, frac):
    dates = signals.index.normalize().unique().sort_values()
    split_idx = int(len(dates) * frac)
    split_date = dates[split_idx]
    train = signals[signals.index < split_date].copy()
    test = signals[signals.index >= split_date].copy()
    return train, test


def _filter_tiers(signals, tiers):
    """Keep only signals belonging to given tier set."""
    out = signals.copy()
    mask = (out["signal"] != 0) & (~out["signal_tier"].isin(tiers))
    out.loc[mask, "signal"] = 0
    out.loc[mask, "stop_price"] = np.nan
    out.loc[mask, "tp_price"] = np.nan
    out.loc[mask, "size_factor"] = 1.0
    out.loc[mask, "signal_tier"] = 0
    return out


def main():
    t0 = time.perf_counter()

    print("\n" + "=" * 110)
    print("  PROP FIRM SCALABILITY COMPARISON")
    print("=" * 110)

    # Load data and strategy once
    df = load_csv(DATA_FILE)
    strategy = importlib.import_module(f"strategies.{STRATEGY_NAME}")
    n_years = (df.index[-1] - df.index[0]).days / 365.25

    results = []

    for mode_name, enable_short in MODES:
        # Generate signals once per mode
        strat_cfg = {**BASE_STRATEGY, "enable_short": enable_short}
        signals = strategy.generate_signals(df.copy(), **strat_cfg)

        n_signals = int((signals["signal"] != 0).sum())
        n_long_signals = int((signals["signal"] == 1).sum())
        n_short_signals = int((signals["signal"] == -1).sum())

        # Train/test split for this mode
        train_sig, test_sig = _split_by_date(signals, TRAIN_FRAC)

        for risk in RISK_LEVELS:
            label = f"{mode_name} @ {risk*100:.2f}%"
            bt_cfg = {**BASE_BACKTEST, "risk_per_trade": risk}

            # Full backtest
            result = run_backtest(signals, **bt_cfg)
            metrics = compute_metrics(result, initial_capital=bt_cfg["initial_capital"])
            validation = validate_strategy(metrics)

            mtm = metrics["mtm"]
            trades = metrics["trades"]
            n_trades = trades["total_trades"]

            # Train/test backtest for robustness check
            train_result = run_backtest(train_sig, **bt_cfg)
            test_result = run_backtest(test_sig, **bt_cfg)
            train_trades = train_result.trades
            test_trades = test_result.trades
            train_pnls = [t.pnl for t in train_trades]
            test_pnls = [t.pnl for t in test_trades]
            train_exp = np.mean(train_pnls) if train_pnls else 0
            test_exp = np.mean(test_pnls) if test_pnls else 0

            # Short-tier-only validation (if shorts enabled)
            short_test_exp = None
            if enable_short:
                short_test_signals = _filter_tiers(test_sig, {5, 6})
                short_test_result = run_backtest(short_test_signals, **bt_cfg)
                short_pnls = [t.pnl for t in short_test_result.trades]
                short_test_exp = np.mean(short_pnls) if short_pnls else 0

            # Monte Carlo prop firm sims
            prop_results = {}
            for pf_name, pf_cfg in PROP_FIRM_CONFIGS.items():
                pf = simulate_prop_firm(result, config=pf_cfg, n_simulations=5000, seed=42)
                prop_results[pf_name] = pf

            row = {
                "config": label,
                "mode": mode_name,
                "risk": risk * 100,
                "trades": n_trades,
                "trades_yr": round(n_trades / n_years, 1),
                "long_signals": n_long_signals,
                "short_signals": n_short_signals,
                "win_rate": trades["win_rate"],
                "pf": trades["profit_factor"],
                "sharpe": mtm["sharpe"],
                "sortino": mtm["sortino"],
                "calmar": mtm["calmar"],
                "total_return": mtm["total_return"],
                "max_dd": mtm["max_drawdown"],
                "valid": validation["passed"],
                "train_exp": round(train_exp, 2),
                "test_exp": round(test_exp, 2),
                "short_test_exp": round(short_test_exp, 2) if short_test_exp is not None else "-",
            }

            for pf_name, pf in prop_results.items():
                row[f"pass_{pf_name}"] = pf["pass_rate"]
                row[f"mc_dd50_{pf_name}"] = pf["drawdown_distribution"]["dd_p50"]
                row[f"mc_dd95_{pf_name}"] = pf["drawdown_distribution"]["dd_p95"]

            results.append(row)
            print(f"  Done: {label} -- {n_trades} trades, Sharpe {mtm['sharpe']:.2f}, DD {mtm['max_drawdown']:.2f}%")

    # ===================================================================
    # Print comparison table
    # ===================================================================
    print("\n" + "=" * 110)
    print("  CONFIGURATION COMPARISON")
    print("=" * 110)

    # Core metrics
    hdr = f"  {'Config':<25s} {'Trades':>7s} {'Tr/Yr':>6s} {'WR%':>6s} {'PF':>6s} {'Sharpe':>7s} {'Calmar':>7s} {'MaxDD%':>7s} {'Return%':>8s} {'Valid':>6s}"
    print(hdr)
    print("  " + "-" * 105)
    for r in results:
        v = "PASS" if r["valid"] else "FAIL"
        print(f"  {r['config']:<25s} {r['trades']:>7d} {r['trades_yr']:>6.1f} "
              f"{r['win_rate']:>6.1f} {r['pf']:>6.2f} {r['sharpe']:>7.2f} {r['calmar']:>7.2f} "
              f"{r['max_dd']:>7.2f} {r['total_return']:>8.2f} {v:>6s}")

    # Train/test robustness
    print(f"\n  {'Config':<25s} {'TrainExp$':>10s} {'TestExp$':>10s} {'ShortTestExp$':>14s}")
    print("  " + "-" * 65)
    for r in results:
        print(f"  {r['config']:<25s} {r['train_exp']:>10.2f} {r['test_exp']:>10.2f} "
              f"{str(r['short_test_exp']):>14s}")

    # Prop firm pass rates
    for pf_name in PROP_FIRM_CONFIGS:
        print(f"\n  {pf_name}")
        print(f"  {'Config':<25s} {'PassRate%':>10s} {'MC DD50%':>10s} {'MC DD95%':>10s}")
        print("  " + "-" * 60)
        for r in results:
            print(f"  {r['config']:<25s} {r[f'pass_{pf_name}']:>10.1f} "
                  f"{r[f'mc_dd50_{pf_name}']:>10.2f} {r[f'mc_dd95_{pf_name}']:>10.2f}")

    # Signal breakdown
    print(f"\n  {'Config':<25s} {'Long Sig':>10s} {'Short Sig':>10s}")
    print("  " + "-" * 50)
    for r in results:
        print(f"  {r['config']:<25s} {r['long_signals']:>10d} {r['short_signals']:>10d}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 110 + "\n")

    # Save to CSV
    out_path = Path("research/prop_scaling_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Results saved to {out_path}\n")


if __name__ == "__main__":
    main()
