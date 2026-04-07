"""
Execution realism stress test for the prop-firm portfolio.

Tests sensitivity to real-world execution costs by varying:
  1. Slippage: 0.25 -> 1.5 points (locked: 0.25)
  2. Commission: $2 -> $6 per side (locked: $2)
  3. Spread: 0 -> 1.0 points (locked: 0)
  4. Combined worst-case: simultaneous elevated costs

Each test runs the full portfolio backtest + prop firm simulation.

Usage:
    python -m research.execution_realism
"""

import importlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest, BacktestResult
from engine.prop_firm import simulate_prop_firm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("engine.backtester").setLevel(logging.ERROR)

DATA_FILE = "data/nq_15m_data.csv"
OUTPUT_DIR = Path("output/portfolio")

TOTAL_CAPITAL = 100_000.0

STRATEGY_CATALOGUE = {
    "gap_momentum": {
        "module": "strategies.gap_momentum",
        "strategy_params": {
            "fh_percentile": 75.0,
            "gap_threshold_pct": 0.10,
        },
        "backtest_overrides": {
            "stop_atr_multiple": 1.0,
            "tp_atr_multiple": 2.0,
        },
    },
    "regime_filtered_momentum": {
        "module": "strategies.regime_filtered_momentum",
        "strategy_params": {
            "fh_percentile": 75.0,
            "fh_dead_zone_lo": 0.50,
            "fh_dead_zone_hi": 0.75,
            "trend_20d_max": 1.0,
            "vol_filter": True,
        },
        "backtest_overrides": {
            "stop_atr_multiple": 1.0,
            "tp_atr_multiple": 2.0,
        },
    },
}

WEIGHTS = {"gap_momentum": 0.60, "regime_filtered_momentum": 0.40}
RISK_PER_TRADE = 0.014

BASE_BACKTEST_CONFIG = {
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "use_trailing_stop": False,
    "daily_dd_limit": 0.02,
    "max_daily_risk": 0.02,
    "max_dd_limit": 0.0,
    "max_bars_in_trade": 0,
    "max_concurrent_trades": 1,
    "min_bars_between_entries": 2,
    "consec_loss_threshold": 2,
    "loss_scale_down": 0.5,
    "max_risk_per_trade": 0.0,
}

PROP_CONFIG = {
    "starting_balance": TOTAL_CAPITAL,
    "profit_target": 0.10,
    "daily_drawdown_limit": 0.05,
    "max_drawdown_limit": 0.10,
    "max_days": 0,
}

PROP_SIMS = 3000


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    # Baseline
    {"label": "Baseline (locked)", "slippage_points": 0.25, "commission_per_side": 2.0, "spread_points": 0.0},

    # Slippage sweep
    {"label": "Slippage 0.50", "slippage_points": 0.50, "commission_per_side": 2.0, "spread_points": 0.0},
    {"label": "Slippage 0.75", "slippage_points": 0.75, "commission_per_side": 2.0, "spread_points": 0.0},
    {"label": "Slippage 1.00", "slippage_points": 1.00, "commission_per_side": 2.0, "spread_points": 0.0},
    {"label": "Slippage 1.50", "slippage_points": 1.50, "commission_per_side": 2.0, "spread_points": 0.0},

    # Commission sweep
    {"label": "Commission $3", "slippage_points": 0.25, "commission_per_side": 3.0, "spread_points": 0.0},
    {"label": "Commission $4", "slippage_points": 0.25, "commission_per_side": 4.0, "spread_points": 0.0},
    {"label": "Commission $5", "slippage_points": 0.25, "commission_per_side": 5.0, "spread_points": 0.0},
    {"label": "Commission $6", "slippage_points": 0.25, "commission_per_side": 6.0, "spread_points": 0.0},

    # Spread sweep
    {"label": "Spread 0.25", "slippage_points": 0.25, "commission_per_side": 2.0, "spread_points": 0.25},
    {"label": "Spread 0.50", "slippage_points": 0.25, "commission_per_side": 2.0, "spread_points": 0.50},
    {"label": "Spread 0.75", "slippage_points": 0.25, "commission_per_side": 2.0, "spread_points": 0.75},
    {"label": "Spread 1.00", "slippage_points": 0.25, "commission_per_side": 2.0, "spread_points": 1.00},

    # Combined worst-case scenarios
    {"label": "Moderate pessimistic", "slippage_points": 0.50, "commission_per_side": 3.0, "spread_points": 0.25},
    {"label": "High pessimistic", "slippage_points": 0.75, "commission_per_side": 4.0, "spread_points": 0.50},
    {"label": "Extreme pessimistic", "slippage_points": 1.00, "commission_per_side": 5.0, "spread_points": 0.75},
]


# ---------------------------------------------------------------------------
# Portfolio runner
# ---------------------------------------------------------------------------

def _run_portfolio(df, slippage_points, commission_per_side, spread_points):
    """Run both strategies with specified execution costs."""
    ref_index = None
    strategy_results = {}
    capital_map = {}

    for name, weight in WEIGHTS.items():
        capital = TOTAL_CAPITAL * weight
        capital_map[name] = capital
        cat = STRATEGY_CATALOGUE[name]

        mod = importlib.import_module(cat["module"])
        signals = mod.generate_signals(df.copy(), **cat["strategy_params"])

        bt_cfg = {
            **BASE_BACKTEST_CONFIG,
            **cat.get("backtest_overrides", {}),
            "risk_per_trade": RISK_PER_TRADE,
            "initial_capital": capital,
            "strategy_name": name,
            "slippage_points": slippage_points,
            "commission_per_side": commission_per_side,
        }
        # Add spread if backtester supports it
        if spread_points > 0:
            bt_cfg["spread_points"] = spread_points

        result = run_backtest(signals, **bt_cfg)
        strategy_results[name] = result
        if ref_index is None:
            ref_index = result.equity_mtm.index

    # Combine equity
    pf_mtm = pd.Series(TOTAL_CAPITAL, index=ref_index, dtype=np.float64)
    pf_closed = pd.Series(TOTAL_CAPITAL, index=ref_index, dtype=np.float64)

    for name, result in strategy_results.items():
        cap = capital_map[name]
        mtm_pnl = (result.equity_mtm - cap).reindex(ref_index, method="ffill").fillna(0.0)
        closed_pnl = (result.equity_closed - cap).reindex(ref_index, method="ffill").fillna(0.0)
        pf_mtm = pf_mtm + mtm_pnl
        pf_closed = pf_closed + closed_pnl

    all_trades = []
    for result in strategy_results.values():
        all_trades.extend(result.trades)
    all_trades.sort(key=lambda t: t.entry_time)

    return pf_mtm, pf_closed, all_trades


def _evaluate(pf_mtm, pf_closed, all_trades):
    """Compute key metrics."""
    from engine.metrics import _infer_bars_per_year, _equity_metrics

    n_trades = len(all_trades)
    if n_trades == 0:
        return {"sharpe": 0, "return_pct": 0, "max_dd": 0, "n_trades": 0,
                "profit_factor": 0, "win_rate": 0, "prop_pass_rate": 0,
                "total_pnl": 0, "avg_pnl": 0}

    bars_per_year = _infer_bars_per_year(pf_mtm.index)
    mtm_m = _equity_metrics(pf_mtm, TOTAL_CAPITAL, bars_per_year)

    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
    wr = len(wins) / n_trades * 100
    total_pnl = sum(t.pnl for t in all_trades)

    # Prop sim
    running_max = pf_mtm.cummax()
    dd = (pf_mtm - running_max) / running_max * 100
    bt_result = BacktestResult(
        equity_mtm=pf_mtm, equity_closed=pf_closed,
        drawdown_series=dd, trades=all_trades,
    )
    prop = simulate_prop_firm(bt_result, config=PROP_CONFIG, n_simulations=PROP_SIMS, seed=42)

    return {
        "sharpe": round(mtm_m["sharpe"], 3),
        "return_pct": round(mtm_m["total_return"], 2),
        "max_dd": round(mtm_m["max_drawdown"], 2),
        "n_trades": n_trades,
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 1),
        "prop_pass_rate": prop["pass_rate"],
        "total_pnl": round(total_pnl, 0),
        "avg_pnl": round(total_pnl / n_trades, 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()

    log.info("Loading data from %s", DATA_FILE)
    df = load_csv(DATA_FILE)
    log.info("Loaded %d bars", len(df))

    all_results = []

    for scenario in SCENARIOS:
        label = scenario["label"]
        log.info("Running: %s", label)

        try:
            pf_mtm, pf_closed, trades = _run_portfolio(
                df,
                slippage_points=scenario["slippage_points"],
                commission_per_side=scenario["commission_per_side"],
                spread_points=scenario["spread_points"],
            )
            metrics = _evaluate(pf_mtm, pf_closed, trades)
        except Exception as exc:
            log.warning("  %s failed: %s", label, exc)
            continue

        row = {
            "scenario": label,
            "slippage": scenario["slippage_points"],
            "commission": scenario["commission_per_side"],
            "spread": scenario["spread_points"],
            **metrics,
        }
        all_results.append(row)

        log.info("  Sharpe=%.2f, Return=%.1f%%, DD=%.1f%%, PF=%.2f, Prop=%.1f%%",
                 metrics["sharpe"], metrics["return_pct"], metrics["max_dd"],
                 metrics["profit_factor"], metrics["prop_pass_rate"])

    results_df = pd.DataFrame(all_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "execution_realism.csv"
    results_df.to_csv(out_path, index=False)
    log.info("Results saved to %s", out_path)

    # =========================================================================
    # Print results
    # =========================================================================
    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 120)
    print("  EXECUTION REALISM STRESS TEST -- PROP-FIRM PORTFOLIO")
    print("=" * 120)
    print(f"  Config: 60% gap_momentum + 40% regime_filtered | risk=1.4% | "
          f"stop=1.0x ATR | TP=2.0x ATR")

    # Get baseline for comparison
    baseline = results_df.iloc[0] if len(results_df) > 0 else None

    print(f"\n  {'Scenario':<25} {'Slip':>5} {'Comm':>5} {'Sprd':>5} "
          f"{'Sharpe':>7} {'Return':>8} {'MaxDD':>8} {'PF':>6} {'WR%':>5} "
          f"{'Prop%':>6} {'TotalPnL':>11} {'AvgPnL':>8}")
    print("  " + "-" * 116)

    for _, r in results_df.iterrows():
        is_baseline = r["scenario"] == "Baseline (locked)"
        marker = " <--" if is_baseline else ""
        print(f"  {r['scenario']:<25} {r['slippage']:>5.2f} {r['commission']:>5.1f} "
              f"{r['spread']:>5.2f} {r['sharpe']:>7.3f} {r['return_pct']:>7.2f}% "
              f"{r['max_dd']:>7.2f}% {r['profit_factor']:>6.3f} {r['win_rate']:>4.1f}% "
              f"{r['prop_pass_rate']:>5.1f}% ${r['total_pnl']:>10,.0f} "
              f"${r['avg_pnl']:>7,.0f}{marker}")

    # Impact analysis
    if baseline is not None and len(results_df) > 1:
        print("\n" + "=" * 120)
        print("  IMPACT ANALYSIS (change from baseline)")
        print("=" * 120)

        print(f"\n  {'Scenario':<25} {'dSharpe':>8} {'dReturn':>9} {'dPF':>6} "
              f"{'dProp%':>7} {'dPnL':>11}")
        print("  " + "-" * 70)

        for _, r in results_df.iterrows():
            if r["scenario"] == "Baseline (locked)":
                continue
            ds = r["sharpe"] - baseline["sharpe"]
            dr = r["return_pct"] - baseline["return_pct"]
            dpf = r["profit_factor"] - baseline["profit_factor"]
            dp = r["prop_pass_rate"] - baseline["prop_pass_rate"]
            dpnl = r["total_pnl"] - baseline["total_pnl"]
            print(f"  {r['scenario']:<25} {ds:>+7.3f} {dr:>+8.2f}% {dpf:>+5.3f} "
                  f"{dp:>+6.1f}% ${dpnl:>+10,.0f}")

    # Cost per trade analysis
    if baseline is not None:
        print("\n" + "=" * 120)
        print("  COST SENSITIVITY")
        print("=" * 120)

        base_pnl = baseline["total_pnl"]
        n_trades = baseline["n_trades"]

        # Find breakeven thresholds
        for param, values in [
            ("slippage", [r for _, r in results_df.iterrows() if "Slippage" in r["scenario"]]),
            ("commission", [r for _, r in results_df.iterrows() if "Commission" in r["scenario"]]),
            ("spread", [r for _, r in results_df.iterrows() if "Spread" in r["scenario"]]),
        ]:
            if not values:
                continue
            worst = min(values, key=lambda r: r["sharpe"])
            print(f"\n  {param.upper()}: worst tested ({worst['scenario']}) -> "
                  f"Sharpe={worst['sharpe']:.3f}, Prop={worst['prop_pass_rate']:.1f}%")
            if worst["sharpe"] > 0:
                print(f"    Edge survives even at extreme {param} levels")
            else:
                print(f"    Edge breaks at extreme {param} levels")

        # Combined worst case
        combined = [r for _, r in results_df.iterrows() if "pessimistic" in r["scenario"].lower()]
        if combined:
            print(f"\n  COMBINED SCENARIOS:")
            for r in combined:
                status = "VIABLE" if r["prop_pass_rate"] > 40 else "MARGINAL" if r["prop_pass_rate"] > 20 else "BROKEN"
                print(f"    {r['scenario']}: Sharpe={r['sharpe']:.3f}, "
                      f"Prop={r['prop_pass_rate']:.1f}%, PnL=${r['total_pnl']:,.0f} -> {status}")

    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 120 + "\n")

    return results_df


if __name__ == "__main__":
    main()
