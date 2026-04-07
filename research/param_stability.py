"""
Parameter stability sweep for the prop-firm portfolio.

Perturbs each key parameter around its locked value to check that
performance degrades gracefully (no cliff edges).  A stable system
should show smooth, gently sloping surfaces — not sharp spikes.

Sweeps:
  - risk_per_trade:   0.008 -> 0.020  (7 steps)
  - stop_atr_multiple: 0.6 -> 1.6    (6 steps)
  - tp_atr_multiple:  1.0 -> 3.0     (5 steps)
  - fh_percentile:    60 -> 90       (7 steps)

Each sweep holds all other params at locked defaults and runs the
full portfolio backtest + prop firm simulation.

Usage:
    python -m research.param_stability
"""

import importlib
import logging
import time
from dataclasses import asdict
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

# ---------------------------------------------------------------------------
# Locked defaults
# ---------------------------------------------------------------------------
LOCKED = {
    "risk_per_trade": 0.014,
    "stop_atr_multiple": 1.0,
    "tp_atr_multiple": 2.0,
    "fh_percentile": 75.0,
}

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
# Sweep definitions
# ---------------------------------------------------------------------------
SWEEPS = {
    "risk_per_trade": [0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020],
    "stop_atr_multiple": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    "tp_atr_multiple": [1.0, 1.5, 2.0, 2.5, 3.0],
    "fh_percentile": [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0],
}


# ---------------------------------------------------------------------------
# Portfolio runner with parameter overrides
# ---------------------------------------------------------------------------

def _run_portfolio(df, risk_per_trade=None, stop_atr_multiple=None,
                   tp_atr_multiple=None, fh_percentile=None):
    """Run both strategies with optional parameter overrides."""
    rpt = risk_per_trade or LOCKED["risk_per_trade"]
    stop = stop_atr_multiple or LOCKED["stop_atr_multiple"]
    tp = tp_atr_multiple or LOCKED["tp_atr_multiple"]
    fh_pct = fh_percentile or LOCKED["fh_percentile"]

    ref_index = None
    strategy_results = {}
    capital_map = {}

    for name, weight in WEIGHTS.items():
        capital = TOTAL_CAPITAL * weight
        capital_map[name] = capital
        cat = STRATEGY_CATALOGUE[name]

        # Override fh_percentile in strategy params
        strat_params = dict(cat["strategy_params"])
        strat_params["fh_percentile"] = fh_pct

        mod = importlib.import_module(cat["module"])
        signals = mod.generate_signals(df.copy(), **strat_params)

        bt_cfg = {
            **BASE_BACKTEST_CONFIG,
            "stop_atr_multiple": stop,
            "tp_atr_multiple": tp,
            "risk_per_trade": rpt,
            "initial_capital": capital,
            "strategy_name": name,
        }
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
    """Compute key metrics from portfolio results."""
    from engine.metrics import _infer_bars_per_year, _equity_metrics

    n_trades = len(all_trades)
    if n_trades == 0:
        return {"sharpe": 0, "return_pct": 0, "max_dd": 0, "n_trades": 0,
                "profit_factor": 0, "win_rate": 0, "prop_pass_rate": 0}

    bars_per_year = _infer_bars_per_year(pf_mtm.index)
    mtm_m = _equity_metrics(pf_mtm, TOTAL_CAPITAL, bars_per_year)

    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
    wr = len(wins) / n_trades * 100

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
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()

    log.info("Loading data from %s", DATA_FILE)
    df = load_csv(DATA_FILE)
    log.info("Loaded %d bars: %s -> %s", len(df), df.index[0], df.index[-1])

    all_results = []

    for param_name, values in SWEEPS.items():
        log.info("Sweeping %s: %s", param_name, values)

        for val in values:
            kwargs = {param_name: val}
            is_locked = (val == LOCKED[param_name])

            try:
                pf_mtm, pf_closed, trades = _run_portfolio(df, **kwargs)
                metrics = _evaluate(pf_mtm, pf_closed, trades)
            except Exception as exc:
                log.warning("  %s=%s failed: %s", param_name, val, exc)
                continue

            row = {
                "param": param_name,
                "value": val,
                "is_locked": is_locked,
                **metrics,
            }
            all_results.append(row)

            marker = " <-- LOCKED" if is_locked else ""
            log.info("  %s=%.4f -> Sharpe=%.2f, Return=%.1f%%, DD=%.1f%%, "
                     "Trades=%d, PF=%.2f, Prop=%.1f%%%s",
                     param_name, val, metrics["sharpe"], metrics["return_pct"],
                     metrics["max_dd"], metrics["n_trades"], metrics["profit_factor"],
                     metrics["prop_pass_rate"], marker)

    results_df = pd.DataFrame(all_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "param_stability.csv"
    results_df.to_csv(out_path, index=False)
    log.info("Results saved to %s", out_path)

    # =========================================================================
    # Print results
    # =========================================================================
    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 110)
    print("  PARAMETER STABILITY SWEEP -- PROP-FIRM PORTFOLIO")
    print("=" * 110)

    for param_name in SWEEPS:
        sub = results_df[results_df["param"] == param_name].copy()
        if sub.empty:
            continue

        print(f"\n  --- {param_name} ---")
        print(f"  {'Value':>10}  {'Sharpe':>7}  {'Return':>8}  {'MaxDD':>8}  "
              f"{'Trades':>6}  {'PF':>6}  {'WR%':>5}  {'Prop%':>6}  {'Note':>12}")
        print("  " + "-" * 100)

        for _, r in sub.iterrows():
            note = "<-- LOCKED" if r["is_locked"] else ""
            print(f"  {r['value']:>10.4f}  {r['sharpe']:>7.3f}  {r['return_pct']:>7.2f}%  "
                  f"{r['max_dd']:>7.2f}%  {r['n_trades']:>6.0f}  {r['profit_factor']:>6.3f}  "
                  f"{r['win_rate']:>4.1f}%  {r['prop_pass_rate']:>5.1f}%  {note:>12}")

        # Stability score: how much does the locked value's Sharpe degrade
        # across neighbors?
        locked_row = sub[sub["is_locked"]]
        if not locked_row.empty:
            locked_sharpe = locked_row.iloc[0]["sharpe"]
            sharpes = sub["sharpe"].values
            if len(sharpes) >= 3:
                max_sharpe = sharpes.max()
                min_sharpe = sharpes.min()
                spread = max_sharpe - min_sharpe
                locked_rank = (locked_sharpe - min_sharpe) / spread * 100 if spread > 0 else 100
                print(f"\n  Stability: locked Sharpe {locked_sharpe:.3f} is at "
                      f"{locked_rank:.0f}th percentile of sweep range "
                      f"[{min_sharpe:.3f}, {max_sharpe:.3f}]")

                # Check for cliff: is there a >50% Sharpe drop between adjacent values?
                cliff = False
                for i in range(1, len(sharpes)):
                    if sharpes[i-1] > 0 and sharpes[i] > 0:
                        pct_change = abs(sharpes[i] - sharpes[i-1]) / max(sharpes[i-1], 0.01) * 100
                        if pct_change > 50:
                            cliff = True
                if cliff:
                    print("  WARNING: Cliff edge detected -- >50% Sharpe change between adjacent values")
                else:
                    print("  OK: No cliff edges detected")

    # Summary
    print("\n" + "=" * 110)
    print("  STABILITY SUMMARY")
    print("=" * 110)

    for param_name in SWEEPS:
        sub = results_df[results_df["param"] == param_name]
        if sub.empty:
            continue
        locked_row = sub[sub["is_locked"]]
        if locked_row.empty:
            continue
        locked_sharpe = locked_row.iloc[0]["sharpe"]
        best_sharpe = sub["sharpe"].max()
        worst_sharpe = sub["sharpe"].min()

        all_positive = (sub["sharpe"] > 0).all()
        status = "STABLE" if all_positive else "MIXED"

        locked_prop = locked_row.iloc[0]["prop_pass_rate"]
        best_prop = sub["prop_pass_rate"].max()

        print(f"\n  {param_name}:")
        print(f"    Locked Sharpe: {locked_sharpe:.3f} (best: {best_sharpe:.3f}, worst: {worst_sharpe:.3f})")
        print(f"    Locked Prop:   {locked_prop:.1f}% (best: {best_prop:.1f}%)")
        print(f"    All folds positive Sharpe: {status}")

    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 110 + "\n")

    return results_df


if __name__ == "__main__":
    main()
