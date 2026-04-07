"""
Walk-forward validation for the prop-firm portfolio.

Runs the locked 2-strategy portfolio (gap_momentum + regime_filtered)
through rolling train/test windows.  No optimisation — validates that
the fixed config holds up out-of-sample across every period.

Each fold reports: Sharpe, return, drawdown, trade count, profit factor,
and a Monte Carlo prop firm pass rate estimate.

Usage:
    python -m research.walk_forward_portfolio
"""

import importlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest, BacktestResult
from engine.metrics import compute_metrics
from engine.prop_firm import simulate_prop_firm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
# Suppress noisy DD warnings during batch folds
logging.getLogger("engine.backtester").setLevel(logging.ERROR)

DATA_FILE = "data/nq_15m_data.csv"
OUTPUT_DIR = Path("output/portfolio")

# ---------------------------------------------------------------------------
# Locked portfolio config (matches run_portfolio.py)
# ---------------------------------------------------------------------------

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
RISK_PER_TRADE = 0.014  # locked aggressive config

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

# Walk-forward windows
TRAIN_MONTHS = 12
TEST_MONTHS = 6
STEP_MONTHS = 3
PROP_SIMS_PER_FOLD = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_folds(index, train_months, test_months, step_months):
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


def _run_portfolio_on_window(df_window: pd.DataFrame) -> tuple:
    """Run the locked portfolio on a data window.

    Returns (pf_mtm, pf_closed, all_trades, strategy_metrics).
    """
    ref_index = None
    strategy_results = {}
    capital_map = {}

    for name, weight in WEIGHTS.items():
        capital = TOTAL_CAPITAL * weight
        capital_map[name] = capital
        cat = STRATEGY_CATALOGUE[name]

        mod = importlib.import_module(cat["module"])
        signals = mod.generate_signals(df_window.copy(), **cat["strategy_params"])

        bt_cfg = {
            **BASE_BACKTEST_CONFIG,
            **cat.get("backtest_overrides", {}),
            "risk_per_trade": RISK_PER_TRADE,
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

    # Combine trades
    all_trades = []
    for result in strategy_results.values():
        all_trades.extend(result.trades)
    all_trades.sort(key=lambda t: t.entry_time)

    # Per-strategy trade counts
    strat_counts = {name: len(r.trades) for name, r in strategy_results.items()}

    return pf_mtm, pf_closed, all_trades, strat_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()

    log.info("Loading data from %s", DATA_FILE)
    df = load_csv(DATA_FILE)
    log.info("Loaded %d bars: %s -> %s", len(df), df.index[0], df.index[-1])

    folds = _build_folds(df.index, TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS)
    log.info("Walk-forward: %d folds (%dmo train / %dmo test / %dmo step)",
             len(folds), TRAIN_MONTHS, TEST_MONTHS, STEP_MONTHS)

    if not folds:
        log.error("Not enough data for even one fold")
        sys.exit(1)

    fold_results = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
        log.info("Fold %d/%d: train %s->%s | test %s->%s",
                 i + 1, len(folds),
                 train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d"),
                 test_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))

        df_test = df[test_start:test_end]
        if len(df_test) < 100:
            log.warning("  Fold %d: only %d bars — skipping", i + 1, len(df_test))
            continue

        try:
            pf_mtm, pf_closed, all_trades, strat_counts = _run_portfolio_on_window(df_test)
        except Exception as exc:
            log.warning("  Fold %d failed: %s", i + 1, exc)
            continue

        n_trades = len(all_trades)
        if n_trades == 0:
            log.warning("  Fold %d: no trades", i + 1)
            fold_results.append({
                "fold": i + 1,
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "sharpe": 0.0, "return_pct": 0.0, "max_dd": 0.0,
                "n_trades": 0, "profit_factor": 0.0, "win_rate": 0.0,
                "prop_pass_rate": 0.0,
                "gap_mom_trades": 0, "regime_filt_trades": 0,
            })
            continue

        # Metrics
        from engine.metrics import _infer_bars_per_year, _equity_metrics
        bars_per_year = _infer_bars_per_year(pf_mtm.index)
        mtm_m = _equity_metrics(pf_mtm, TOTAL_CAPITAL, bars_per_year)

        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
        wr = len(wins) / n_trades * 100

        # Prop firm sim on this fold's trades
        running_max = pf_mtm.cummax()
        dd = (pf_mtm - running_max) / running_max * 100
        fold_bt_result = BacktestResult(
            equity_mtm=pf_mtm, equity_closed=pf_closed,
            drawdown_series=dd, trades=all_trades,
        )
        prop = simulate_prop_firm(
            fold_bt_result, config=PROP_CONFIG,
            n_simulations=PROP_SIMS_PER_FOLD, seed=42,
        )

        log.info("  Sharpe: %.2f | Return: %.2f%% | DD: %.2f%% | "
                 "Trades: %d | PF: %.2f | Prop: %.1f%%",
                 mtm_m["sharpe"], mtm_m["total_return"], mtm_m["max_drawdown"],
                 n_trades, pf, prop["pass_rate"])

        fold_results.append({
            "fold": i + 1,
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "sharpe": round(mtm_m["sharpe"], 2),
            "return_pct": round(mtm_m["total_return"], 2),
            "max_dd": round(mtm_m["max_drawdown"], 2),
            "n_trades": n_trades,
            "profit_factor": round(pf, 2),
            "win_rate": round(wr, 1),
            "prop_pass_rate": prop["pass_rate"],
            "gap_mom_trades": strat_counts.get("gap_momentum", 0),
            "regime_filt_trades": strat_counts.get("regime_filtered_momentum", 0),
        })

    results_df = pd.DataFrame(fold_results)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "walk_forward_portfolio.csv"
    results_df.to_csv(out_path, index=False)
    log.info("Results saved to %s", out_path)

    # =========================================================================
    # Print results
    # =========================================================================
    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 120)
    print("  WALK-FORWARD VALIDATION -- PROP-FIRM PORTFOLIO")
    print("=" * 120)
    print(f"  Config: 60% gap_momentum + 40% regime_filtered | risk=1.4% | "
          f"stop=1.0x ATR | TP=2.0x ATR")
    print(f"  Folds: {len(results_df)} ({TRAIN_MONTHS}mo train / "
          f"{TEST_MONTHS}mo test / {STEP_MONTHS}mo step)")

    print(f"\n  {'Fold':>4}  {'Test Window':<25}  {'Sharpe':>7}  {'Return':>8}  "
          f"{'MaxDD':>7}  {'Trades':>6}  {'PF':>5}  {'WR%':>5}  "
          f"{'Prop%':>6}  {'GM':>4}  {'RF':>4}")
    print("  " + "-" * 116)

    for _, r in results_df.iterrows():
        window = f"{r['test_start']} -> {r['test_end']}"
        flag = " *" if r["sharpe"] < -0.5 else ""
        print(f"  {r['fold']:>4}  {window:<25}  {r['sharpe']:>7.2f}{flag}  "
              f"{r['return_pct']:>7.2f}%  {r['max_dd']:>7.2f}%  "
              f"{r['n_trades']:>6.0f}  {r['profit_factor']:>5.2f}  "
              f"{r['win_rate']:>4.0f}%  {r['prop_pass_rate']:>5.1f}%  "
              f"{r['gap_mom_trades']:>4.0f}  {r['regime_filt_trades']:>4.0f}")

    # Aggregates
    if len(results_df) > 0:
        active = results_df[results_df["n_trades"] > 0]
        n_active = len(active)

        print("\n  " + "-" * 116)
        print("  AGGREGATE OOS METRICS")
        print("  " + "-" * 50)

        avg_sharpe = active["sharpe"].mean()
        worst_sharpe = active["sharpe"].min()
        best_sharpe = active["sharpe"].max()
        positive = (active["sharpe"] > 0).sum()
        positive_pct = positive / n_active * 100 if n_active > 0 else 0
        catastrophic = (active["sharpe"] < -0.5).sum()
        avg_return = active["return_pct"].mean()
        worst_dd = active["max_dd"].min()
        avg_pf = active["profit_factor"].mean()
        avg_wr = active["win_rate"].mean()
        avg_prop = active["prop_pass_rate"].mean()
        total_trades = int(active["n_trades"].sum())

        # Prop pass consistency
        prop_above_60 = (active["prop_pass_rate"] >= 60).sum()
        prop_above_60_pct = prop_above_60 / n_active * 100 if n_active > 0 else 0

        print(f"    Avg OOS Sharpe:          {avg_sharpe:>8.2f}")
        print(f"    Best / Worst Sharpe:     {best_sharpe:>8.2f} / {worst_sharpe:.2f}")
        print(f"    Positive Sharpe folds:   {positive}/{n_active}  ({positive_pct:.0f}%)")
        print(f"    Catastrophic (< -0.5):   {catastrophic}")
        print(f"    Avg OOS Return/fold:     {avg_return:>7.2f}%")
        print(f"    Worst Drawdown:          {worst_dd:>7.2f}%")
        print(f"    Avg Profit Factor:       {avg_pf:>8.2f}")
        print(f"    Avg Win Rate:            {avg_wr:>7.1f}%")
        print(f"    Total Trades (all folds):{total_trades:>7d}")
        print(f"    Avg Prop Pass Rate:      {avg_prop:>7.1f}%")
        print(f"    Folds with Prop >= 60%:  {prop_above_60}/{n_active}  ({prop_above_60_pct:.0f}%)")

        # Verdict
        print("\n  " + "=" * 116)
        print("  VERDICT")
        print("  " + "=" * 116)

        stable = positive_pct >= 60 and catastrophic == 0
        prop_robust = prop_above_60_pct >= 50

        if stable and prop_robust:
            print("\n  ROBUST")
            print(f"    {positive_pct:.0f}% folds with positive Sharpe")
            print(f"    {prop_above_60_pct:.0f}% folds with viable prop pass rate (>= 60%)")
        elif stable:
            print("\n  STABLE (but prop pass rate inconsistent)")
            print(f"    {positive_pct:.0f}% positive Sharpe folds")
            print(f"    Only {prop_above_60_pct:.0f}% folds have viable prop pass rate")
        else:
            reasons = []
            if positive_pct < 60:
                reasons.append(f"only {positive_pct:.0f}% positive Sharpe (need >= 60%)")
            if catastrophic > 0:
                reasons.append(f"{catastrophic} catastrophic fold(s)")
            if not prop_robust:
                reasons.append(f"only {prop_above_60_pct:.0f}% folds with viable prop pass rate")
            print("\n  UNSTABLE")
            for r in reasons:
                print(f"    - {r}")

    print(f"\n    Runtime: {elapsed:.1f}s")
    print("=" * 120 + "\n")

    return results_df


if __name__ == "__main__":
    main()
