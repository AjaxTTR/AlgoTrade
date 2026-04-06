"""
Prop-firm-optimised portfolio runner.

Runs multiple independent strategies with unequal capital weights and
per-strategy risk parameters.  Primary objective: maximise probability
of passing a prop firm challenge (+10% target) before hitting drawdown
limits.  Sharpe/CAGR are reported but NOT the optimisation target.

Includes a grid search over weight / risk configurations that evaluates
each via Monte Carlo prop firm simulation (pass_rate, median_days).

Usage:
    python run_portfolio.py              # run best config
    python run_portfolio.py --grid       # grid search then run best
"""

import argparse
import importlib
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest, BacktestResult, TradeRecord
from engine.metrics import compute_metrics
from engine.prop_firm import simulate_prop_firm, simulate_prop_firm_2phase

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
OUTPUT_DIR = Path("output/portfolio")

TOTAL_CAPITAL = 100_000.0

# Prop firm challenge rules (FTMO-style)
PROP_CONFIG = {
    "starting_balance": TOTAL_CAPITAL,
    "profit_target": 0.10,          # +10%
    "daily_drawdown_limit": 0.05,   # 5% max daily loss
    "max_drawdown_limit": 0.10,     # 10% max trailing DD
    "max_days": 0,                  # 0 = unlimited calendar days
}

PROP_PHASE1_CONFIG = {
    "starting_balance": TOTAL_CAPITAL,
    "profit_target": 0.10,
    "daily_drawdown_limit": 0.05,
    "max_drawdown_limit": 0.10,
    "max_days": 0,
}

PROP_PHASE2_CONFIG = {
    "starting_balance": TOTAL_CAPITAL,
    "profit_target": 0.05,
    "daily_drawdown_limit": 0.05,
    "max_drawdown_limit": 0.10,
    "max_days": 0,
}

N_PROP_SIMS = 5000
PROP_SEED = 42

# ---------------------------------------------------------------------------
# Strategy catalogue — each entry defines the strategy module, its signal
# generation params, and backtester overrides.  Capital weight and
# risk_per_trade are set per-configuration (not here).
# ---------------------------------------------------------------------------

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
    "fh_mean_reversion": {
        "module": "strategies.fh_mean_reversion",
        "strategy_params": {
            "fh_percentile_lo": 20.0,
            "atr_percentile_min": 50.0,
        },
        "backtest_overrides": {
            "stop_atr_multiple": 1.0,
            "tp_atr_multiple": 2.0,
        },
    },
}

# Shared backtest config (applied to all strategies, then overridden)
BASE_BACKTEST_CONFIG = {
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "use_trailing_stop": False,
    "daily_dd_limit": 0.02,
    "max_daily_risk": 0.02,
    "max_dd_limit": 0.0,
    "max_bars_in_trade": 0,          # EOD exit via session_close
    "max_concurrent_trades": 1,
    "min_bars_between_entries": 2,
    "consec_loss_threshold": 2,
    "loss_scale_down": 0.5,
    "max_risk_per_trade": 0.0,       # disabled — DD limits control risk
}

# ---------------------------------------------------------------------------
# Default (best) configuration — used when running without --grid
# ---------------------------------------------------------------------------

DEFAULT_PORTFOLIO_CONFIG = {
    "strategies": {
        "gap_momentum": {"weight": 0.60, "risk_per_trade": 0.008},
        "regime_filtered_momentum": {"weight": 0.40, "risk_per_trade": 0.006},
    },
}

# ---------------------------------------------------------------------------
# Grid search configurations
# ---------------------------------------------------------------------------

GRID_CONFIGS = []

# Weight presets: (gap_momentum, regime_filtered, fh_mean_reversion)
_WEIGHT_PRESETS = [
    # Two-strategy configs (drop fh_mean_reversion)
    (0.60, 0.40, 0.00),
    (0.65, 0.35, 0.00),
    (0.70, 0.30, 0.00),
    (0.75, 0.25, 0.00),
    # Three-strategy configs (small fh_mean_reversion allocation)
    (0.55, 0.30, 0.15),
    (0.60, 0.30, 0.10),
    (0.60, 0.25, 0.15),
    (0.65, 0.25, 0.10),
    (0.50, 0.35, 0.15),
]

# Risk presets: (gap_momentum_risk, regime_risk, fh_mr_risk)
_RISK_PRESETS = [
    (0.008, 0.006, 0.005),
    (0.008, 0.008, 0.005),
    (0.010, 0.008, 0.005),
    (0.010, 0.010, 0.005),
    (0.012, 0.010, 0.005),
    (0.012, 0.008, 0.005),
    (0.010, 0.006, 0.005),
]

for _w, _r in product(_WEIGHT_PRESETS, _RISK_PRESETS):
    gw, rw, fw = _w
    gr, rr, fr = _r
    strats = {}
    if gw > 0:
        strats["gap_momentum"] = {"weight": gw, "risk_per_trade": gr}
    if rw > 0:
        strats["regime_filtered_momentum"] = {"weight": rw, "risk_per_trade": rr}
    if fw > 0:
        strats["fh_mean_reversion"] = {"weight": fw, "risk_per_trade": fr}
    GRID_CONFIGS.append({"strategies": strats})


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _run_single_strategy(
    name: str,
    catalogue_entry: dict,
    df: pd.DataFrame,
    capital: float,
    risk_per_trade: float,
) -> tuple[BacktestResult, dict, pd.DataFrame]:
    """Run one strategy: signals -> backtest -> metrics."""
    mod = importlib.import_module(catalogue_entry["module"])
    signals = mod.generate_signals(df.copy(), **catalogue_entry["strategy_params"])
    n_signals = int((signals["signal"] != 0).sum())

    bt_cfg = {
        **BASE_BACKTEST_CONFIG,
        **catalogue_entry.get("backtest_overrides", {}),
        "risk_per_trade": risk_per_trade,
        "initial_capital": capital,
        "strategy_name": name,
    }
    result = run_backtest(signals, **bt_cfg)
    metrics = compute_metrics(result, initial_capital=capital)

    return result, metrics, signals


def _run_portfolio(
    portfolio_config: dict,
    df: pd.DataFrame,
    total_capital: float,
    verbose: bool = True,
) -> tuple[pd.Series, pd.Series, list, dict, dict, dict]:
    """Execute a full portfolio configuration.

    Returns (pf_mtm, pf_closed, all_trades, strategy_results,
             strategy_metrics, capital_map).
    """
    strat_configs = portfolio_config["strategies"]

    # Compute capital allocation
    capital_map = {}
    for name, cfg in strat_configs.items():
        capital_map[name] = total_capital * cfg["weight"]

    strategy_results = {}
    strategy_metrics = {}

    for name, cfg in strat_configs.items():
        cat = STRATEGY_CATALOGUE[name]
        capital = capital_map[name]
        risk = cfg["risk_per_trade"]

        if verbose:
            log.info("Running %s (weight=%.0f%%, risk=%.1f%%, capital=$%.0f)",
                     name, cfg["weight"] * 100, risk * 100, capital)

        result, metrics, signals = _run_single_strategy(
            name, cat, df, capital, risk,
        )
        strategy_results[name] = result
        strategy_metrics[name] = metrics

        n_signals = int((signals["signal"] != 0).sum())
        n_trades = len(result.trades)
        if verbose:
            log.info("  %s: %d signals -> %d trades", name, n_signals, n_trades)

    # Combine equity curves
    ref_index = next(iter(strategy_results.values())).equity_mtm.index
    pf_mtm = pd.Series(total_capital, index=ref_index, dtype=np.float64)
    pf_closed = pd.Series(total_capital, index=ref_index, dtype=np.float64)

    for name, result in strategy_results.items():
        cap = capital_map[name]
        mtm_pnl = result.equity_mtm - cap
        closed_pnl = result.equity_closed - cap
        mtm_pnl = mtm_pnl.reindex(ref_index, method="ffill").fillna(0.0)
        closed_pnl = closed_pnl.reindex(ref_index, method="ffill").fillna(0.0)
        pf_mtm = pf_mtm + mtm_pnl
        pf_closed = pf_closed + closed_pnl

    pf_mtm.name = "equity_mtm"
    pf_closed.name = "equity_closed"

    # Combine trades
    all_trades = []
    for result in strategy_results.values():
        all_trades.extend(result.trades)
    all_trades.sort(key=lambda t: t.entry_time)

    return pf_mtm, pf_closed, all_trades, strategy_results, strategy_metrics, capital_map


# ---------------------------------------------------------------------------
# Prop firm evaluation on portfolio
# ---------------------------------------------------------------------------

def _build_portfolio_backtest_result(
    pf_mtm: pd.Series,
    pf_closed: pd.Series,
    all_trades: list,
) -> BacktestResult:
    """Build a synthetic BacktestResult from portfolio-level data
    so we can feed it to the prop firm simulator."""
    running_max = pf_mtm.cummax()
    dd = (pf_mtm - running_max) / running_max * 100
    return BacktestResult(
        equity_mtm=pf_mtm,
        equity_closed=pf_closed,
        drawdown_series=dd,
        trades=all_trades,
    )


def _evaluate_prop(
    pf_mtm: pd.Series,
    pf_closed: pd.Series,
    all_trades: list,
    n_sims: int = N_PROP_SIMS,
    seed: int = PROP_SEED,
) -> dict:
    """Run single-phase and two-phase prop firm simulations on the portfolio."""
    pf_result = _build_portfolio_backtest_result(pf_mtm, pf_closed, all_trades)

    single = simulate_prop_firm(
        pf_result, config=PROP_CONFIG, n_simulations=n_sims, seed=seed,
    )
    two_phase = simulate_prop_firm_2phase(
        pf_result,
        phase1_config=PROP_PHASE1_CONFIG,
        phase2_config=PROP_PHASE2_CONFIG,
        n_simulations=n_sims,
        seed=seed,
    )
    return {"single_phase": single, "two_phase": two_phase}


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def _config_label(cfg: dict) -> str:
    """Short label for a portfolio config."""
    parts = []
    for name, sc in cfg["strategies"].items():
        short = name.split("_")[0][0].upper() + name.split("_")[-1][0].upper()
        parts.append(f"{short}:{sc['weight']*100:.0f}%/{sc['risk_per_trade']*100:.1f}%")
    return " | ".join(parts)


def run_grid_search(df: pd.DataFrame) -> tuple[dict, list]:
    """Evaluate all grid configs via prop firm simulation.

    Returns (best_config, all_results_sorted).
    """
    log.info("Starting grid search over %d configurations", len(GRID_CONFIGS))
    results = []

    for i, cfg in enumerate(GRID_CONFIGS):
        label = _config_label(cfg)
        try:
            pf_mtm, pf_closed, all_trades, _, _, _ = _run_portfolio(
                cfg, df, TOTAL_CAPITAL, verbose=False,
            )
        except Exception as e:
            log.warning("Config %d failed: %s", i, e)
            continue

        n_trades = len(all_trades)
        if n_trades < 10:
            continue

        # Quick prop sim (fewer sims for speed during grid search)
        prop = _evaluate_prop(pf_mtm, pf_closed, all_trades,
                              n_sims=2000, seed=PROP_SEED)

        sp = prop["single_phase"]
        tp = prop["two_phase"]

        entry = {
            "config": cfg,
            "label": label,
            "n_trades": n_trades,
            "pass_rate_1p": sp["pass_rate"],
            "median_days_1p": sp.get("days_distribution", {}).get("ttp_p50", 999),
            "pass_rate_2p": tp["overall_pass_rate"],
            "median_days_2p": tp.get("total_days_dist", {}).get("total_p50", 999),
            "fail_rate_1p": sp["fail_rate"],
        }
        results.append(entry)

        if (i + 1) % 10 == 0:
            log.info("  Grid: %d / %d evaluated", i + 1, len(GRID_CONFIGS))

    log.info("Grid search complete: %d valid configs", len(results))

    # Sort: primary = highest pass_rate_1p, secondary = lowest median_days_1p
    results.sort(key=lambda r: (-r["pass_rate_1p"], r["median_days_1p"]))

    best = results[0]["config"] if results else DEFAULT_PORTFOLIO_CONFIG
    return best, results


def _print_grid_results(results: list, top_n: int = 15):
    """Print top grid search results."""
    print("\n" + "=" * 90)
    print("  PROP FIRM GRID SEARCH RESULTS")
    print("=" * 90)
    print(f"  {'#':>3s}  {'Pass%':>6s}  {'MedDays':>7s}  {'2P Pass%':>8s}  "
          f"{'2P Days':>7s}  {'Trades':>6s}  Configuration")
    print("  " + "-" * 86)

    for i, r in enumerate(results[:top_n]):
        print(f"  {i+1:>3d}  {r['pass_rate_1p']:>5.1f}%  {r['median_days_1p']:>7.0f}  "
              f"{r['pass_rate_2p']:>7.1f}%  {r['median_days_2p']:>7.0f}  "
              f"{r['n_trades']:>6d}  {r['label']}")

    print("=" * 90)


# ---------------------------------------------------------------------------
# Printing & export
# ---------------------------------------------------------------------------

def _print_portfolio_summary(
    pf_mtm: pd.Series,
    pf_closed: pd.Series,
    all_trades: list,
    total_capital: float,
    strategy_results: dict,
    strategy_metrics: dict,
    capital_map: dict,
    portfolio_config: dict,
):
    """Print full portfolio performance report."""
    from engine.metrics import _infer_bars_per_year, _equity_metrics

    bars_per_year = _infer_bars_per_year(pf_mtm.index)
    mtm_m = _equity_metrics(pf_mtm, total_capital, bars_per_year)
    closed_m = _equity_metrics(pf_closed, total_capital, bars_per_year)

    n_trades = len(all_trades)
    total_pnl = sum(t.pnl for t in all_trades)
    wins = [t for t in all_trades if t.pnl > 0]
    losses = [t for t in all_trades if t.pnl <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0.0
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    sep = "=" * 70

    print(f"\n{sep}")
    print("  PORTFOLIO PERFORMANCE (prop-firm optimised)")
    print(sep)

    # Config summary
    print("\n  ALLOCATION:")
    for name, cfg in portfolio_config["strategies"].items():
        cap = capital_map[name]
        print(f"    {name:30s}  weight={cfg['weight']*100:.0f}%  "
              f"risk={cfg['risk_per_trade']*100:.1f}%  "
              f"capital=${cap:,.0f}")

    print(f"\n  {'':30s} {'MTM':>10s} {'Closed':>10s}")
    print(f"  {'-' * 52}")
    print(f"  {'Total Return %':30s} {mtm_m['total_return']:>+10.2f} {closed_m['total_return']:>+10.2f}")
    print(f"  {'CAGR %':30s} {mtm_m['cagr']:>+10.2f} {closed_m['cagr']:>+10.2f}")
    print(f"  {'Max Drawdown %':30s} {mtm_m['max_drawdown']:>10.2f} {closed_m['max_drawdown']:>10.2f}")
    print(f"  {'Sharpe':30s} {mtm_m['sharpe']:>10.2f} {closed_m['sharpe']:>10.2f}")
    print(f"  {'Profit Factor':30s} {profit_factor:>10.2f} {'':>10s}")

    print(f"\n  {'-' * 52}")
    print(f"  {'Total Trades':30s} {n_trades:>10d}")
    print(f"  {'Total PnL':30s} {'${:>+,.2f}'.format(total_pnl):>10s}")
    print(f"  {'Win Rate %':30s} {win_rate:>10.1f}")
    print(f"  {'Avg Win ($)':30s} {avg_win:>+10.2f}")
    print(f"  {'Avg Loss ($)':30s} {avg_loss:>+10.2f}")

    # Per-strategy breakdown
    print(f"\n  {'-' * 52}")
    print("  PER-STRATEGY BREAKDOWN")
    print(f"  {'-' * 52}")

    for name in strategy_results:
        result = strategy_results[name]
        metrics = strategy_metrics[name]
        mtm_s = metrics["mtm"]
        trades_s = metrics["trades"]
        n_s = len(result.trades)
        pnl_s = sum(t.pnl for t in result.trades)
        n_sig = int((result.signals_df["signal"] != 0).sum())

        print(f"  {name}")
        print(f"    Signals: {n_sig:>5d}   Trades: {n_s:>5d}   "
              f"PnL: ${pnl_s:>+10,.2f}")
        print(f"    Return: {mtm_s['total_return']:>+7.2f}%   "
              f"Sharpe: {mtm_s['sharpe']:>5.2f}   "
              f"MaxDD: {mtm_s['max_drawdown']:>6.2f}%   "
              f"PF: {trades_s.get('profit_factor', 0.0):>5.2f}   "
              f"WR: {trades_s.get('win_rate', 0.0):>5.1f}%")
        print()

    # Correlations
    daily_returns = {}
    for name, result in strategy_results.items():
        daily_eq = result.equity_mtm.resample("D").last().dropna()
        daily_returns[name] = daily_eq.pct_change().dropna()

    if len(daily_returns) >= 2:
        print(f"  {'-' * 52}")
        print("  DAILY RETURN CORRELATIONS")
        print(f"  {'-' * 52}")
        corr_df = pd.DataFrame(daily_returns).corr()
        names_list = list(corr_df.columns)
        header = f"  {'':>25s}"
        for n in names_list:
            header += f" {n[:8]:>8s}"
        print(header)
        for i, n1 in enumerate(names_list):
            row = f"  {n1:>25s}"
            for j, n2 in enumerate(names_list):
                if j >= i:
                    row += f" {corr_df.loc[n1, n2]:>8.3f}"
                else:
                    row += f" {'':>8s}"
            print(row)

    print(f"\n{sep}")
    return mtm_m, closed_m


def _print_prop_results(prop: dict):
    """Print prop firm simulation results."""
    sp = prop["single_phase"]
    tp = prop["two_phase"]

    sep = "=" * 70
    print(f"\n{sep}")
    print("  PROP FIRM CHALLENGE EVALUATION")
    print(sep)

    # Single phase
    ci = sp["pass_rate_ci"]
    days_d = sp.get("days_distribution", {})
    print(f"\n  SINGLE PHASE (+10% target)")
    print(f"  {'-' * 52}")
    print(f"  {'Pass Rate':30s} {sp['pass_rate']:>9.1f}%")
    print(f"  {'95% CI':30s}  [{ci[0]:.1f}% - {ci[1]:.1f}%]")
    print(f"  {'Fail Rate':30s} {sp['fail_rate']:>9.1f}%")
    print(f"  {'Classification':30s} {sp['classification']:>10s}")
    if days_d:
        print(f"  {'Median Days to Pass':30s} {days_d.get('ttp_p50', 0):>10.0f}")
        print(f"  {'Days (P10/P25/P75/P90)':30s}  "
              f"{days_d.get('ttp_p10', 0):.0f} / {days_d.get('ttp_p25', 0):.0f} / "
              f"{days_d.get('ttp_p75', 0):.0f} / {days_d.get('ttp_p90', 0):.0f}")

    if sp["fail_reasons"]:
        print(f"\n  Failure breakdown:")
        for reason, count in sorted(sp["fail_reasons"].items(), key=lambda x: -x[1]):
            pct = count / sp["n_simulations"] * 100
            print(f"    {reason:26s} {count:>5,d}  ({pct:.1f}%)")

    # Two phase
    tp_ci = tp["overall_pass_rate_ci"]
    td = tp.get("total_days_dist", {})
    p1d = tp.get("p1_days_dist", {})
    p2d = tp.get("p2_days_dist", {})

    print(f"\n  TWO-PHASE (P1: +10%, P2: +5%, balance reset)")
    print(f"  {'-' * 52}")
    print(f"  {'Overall Pass Rate':30s} {tp['overall_pass_rate']:>9.1f}%")
    print(f"  {'95% CI':30s}  [{tp_ci[0]:.1f}% - {tp_ci[1]:.1f}%]")
    print(f"  {'Phase 1 Pass Rate':30s} {tp['phase1_pass_rate']:>9.1f}%")
    print(f"  {'Phase 2 Pass Rate (cond.)':30s} {tp['phase2_conditional_pass_rate']:>9.1f}%")
    print(f"  {'Failed in Phase 1':30s} {tp['fail_in_phase1_pct']:>9.1f}%")
    print(f"  {'  ...due to drawdown':30s} {tp['p1_dd_fail_pct']:>9.1f}%")
    print(f"  {'Failed in Phase 2':30s} {tp['fail_in_phase2_pct']:>9.1f}%")
    print(f"  {'  ...due to drawdown':30s} {tp['p2_dd_fail_pct']:>9.1f}%")

    if td:
        print(f"\n  TIMING (passing sims only)")
        print(f"  {'':35s} {'Med':>6s} {'Mean':>6s} {'P10':>6s} {'P90':>6s}")
        print(f"  {'Phase 1 days':35s} "
              f"{p1d.get('p1_p50', 0):>6.0f} {p1d.get('p1_mean', 0):>6.0f} "
              f"{p1d.get('p1_p10', 0):>6.0f} {p1d.get('p1_p90', 0):>6.0f}")
        print(f"  {'Phase 2 days':35s} "
              f"{p2d.get('p2_p50', 0):>6.0f} {p2d.get('p2_mean', 0):>6.0f} "
              f"{p2d.get('p2_p10', 0):>6.0f} {p2d.get('p2_p90', 0):>6.0f}")
        print(f"  {'TOTAL days (P1 + P2)':35s} "
              f"{td.get('total_p50', 0):>6.0f} {td.get('total_mean', 0):>6.0f} "
              f"{td.get('total_p10', 0):>6.0f} {td.get('total_p90', 0):>6.0f}")

    print(f"\n{sep}")


def _export_portfolio(
    pf_mtm: pd.Series,
    pf_closed: pd.Series,
    all_trades: list,
    portfolio_metrics: dict,
    strategy_metrics: dict,
    prop_results: dict,
    portfolio_config: dict,
    output_dir: Path,
):
    """Save all portfolio outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Equity curve
    equity_df = pd.DataFrame({"equity_mtm": pf_mtm, "equity_closed": pf_closed})
    equity_path = output_dir / "portfolio_equity.csv"
    equity_df.to_csv(equity_path)
    log.info("Portfolio equity saved to %s", equity_path)

    # Drawdown
    running_max = pf_mtm.cummax()
    dd = (pf_mtm - running_max) / running_max * 100
    dd_path = output_dir / "portfolio_drawdown.csv"
    dd.to_csv(dd_path, header=True)

    # Trade log
    if all_trades:
        trade_dicts = [asdict(t) for t in all_trades]
        trade_df = pd.DataFrame(trade_dicts)
        trade_df["direction"] = trade_df["direction"].map({1: "LONG", -1: "SHORT"})
        trade_path = output_dir / "portfolio_trades.csv"
        trade_df.to_csv(trade_path, index=False)
        log.info("Trade log saved to %s (%d trades)", trade_path, len(all_trades))

    # Metrics JSON (includes prop firm results and config)
    export = {
        "portfolio_config": portfolio_config,
        "portfolio_metrics": portfolio_metrics,
        "per_strategy": {name: m for name, m in strategy_metrics.items()},
        "prop_firm": {
            "single_phase": {
                "pass_rate": prop_results["single_phase"]["pass_rate"],
                "fail_rate": prop_results["single_phase"]["fail_rate"],
                "pass_rate_ci": prop_results["single_phase"]["pass_rate_ci"],
                "classification": prop_results["single_phase"]["classification"],
                "days_distribution": prop_results["single_phase"].get("days_distribution", {}),
                "fail_reasons": prop_results["single_phase"]["fail_reasons"],
            },
            "two_phase": {
                "overall_pass_rate": prop_results["two_phase"]["overall_pass_rate"],
                "overall_pass_rate_ci": prop_results["two_phase"]["overall_pass_rate_ci"],
                "phase1_pass_rate": prop_results["two_phase"]["phase1_pass_rate"],
                "phase2_conditional_pass_rate": prop_results["two_phase"]["phase2_conditional_pass_rate"],
                "total_days_dist": prop_results["two_phase"].get("total_days_dist", {}),
            },
        },
    }
    metrics_path = output_dir / "portfolio_metrics.json"
    metrics_path.write_text(json.dumps(export, indent=2, default=str))
    log.info("Metrics saved to %s", metrics_path)


def _plot_portfolio(
    pf_mtm: pd.Series,
    strategy_results: dict,
    capital_map: dict,
    output_dir: Path,
):
    """Generate portfolio equity chart with per-strategy breakdown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])

    ax1 = axes[0]
    ax1.plot(pf_mtm.index, pf_mtm.values, color="black", linewidth=1.5,
             label="Portfolio")

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for idx, (name, result) in enumerate(strategy_results.items()):
        c = colors[idx % len(colors)]
        ax1.plot(result.equity_mtm.index, result.equity_mtm.values,
                 color=c, linewidth=0.8, alpha=0.7, label=name)

    ax1.set_title("Portfolio Equity Curve (Prop-Firm Optimised)", fontsize=12,
                   fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    running_max = pf_mtm.cummax()
    dd = (pf_mtm - running_max) / running_max * 100
    ax2.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax2.plot(dd.index, dd.values, color="red", linewidth=0.5)
    ax2.set_title("Portfolio Drawdown", fontsize=10)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "portfolio_equity.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved to %s", plot_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prop-firm optimised portfolio runner")
    parser.add_argument("--grid", action="store_true",
                        help="Run grid search over weight/risk configurations")
    args = parser.parse_args()

    t_start = time.perf_counter()

    # Load data
    log.info("Loading data from %s", DATA_FILE)
    try:
        df = load_csv(DATA_FILE)
    except FileNotFoundError:
        log.error("%s not found.", DATA_FILE)
        sys.exit(1)
    log.info("Loaded %s bars  |  %s  ->  %s", f"{len(df):,}",
             df.index[0], df.index[-1])

    # Grid search or use default config
    if args.grid:
        best_config, grid_results = run_grid_search(df)
        _print_grid_results(grid_results)
        portfolio_config = best_config

        # Export grid results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        grid_export = []
        for r in grid_results:
            row = {**r}
            row.pop("config")
            grid_export.append(row)
        grid_df = pd.DataFrame(grid_export)
        grid_path = OUTPUT_DIR / "grid_search_results.csv"
        grid_df.to_csv(grid_path, index=False)
        log.info("Grid results saved to %s", grid_path)
    else:
        portfolio_config = DEFAULT_PORTFOLIO_CONFIG

    # Run best / default config with full output
    strat_configs = portfolio_config["strategies"]
    n_strategies = len(strat_configs)

    print("\n" + "=" * 70)
    print("  NQ PROP-FIRM PORTFOLIO")
    print("=" * 70)
    print(f"  Total capital:     ${TOTAL_CAPITAL:>12,.2f}")
    print(f"  Strategies:        {n_strategies:>12d}")
    for name, cfg in strat_configs.items():
        cap = TOTAL_CAPITAL * cfg["weight"]
        print(f"    {name:28s}  {cfg['weight']*100:>4.0f}%  "
              f"${cap:>10,.0f}  risk={cfg['risk_per_trade']*100:.1f}%")
    print()

    pf_mtm, pf_closed, all_trades, strategy_results, strategy_metrics, capital_map = \
        _run_portfolio(portfolio_config, df, TOTAL_CAPITAL, verbose=True)

    # Performance summary
    mtm_m, closed_m = _print_portfolio_summary(
        pf_mtm, pf_closed, all_trades, TOTAL_CAPITAL,
        strategy_results, strategy_metrics, capital_map, portfolio_config,
    )

    # Prop firm evaluation (full simulation count)
    log.info("Running prop firm simulations (%d sims)", N_PROP_SIMS)
    prop = _evaluate_prop(pf_mtm, pf_closed, all_trades,
                          n_sims=N_PROP_SIMS, seed=PROP_SEED)
    _print_prop_results(prop)

    # Export
    portfolio_metrics_export = {
        "mtm": mtm_m,
        "closed": closed_m,
        "total_trades": len(all_trades),
        "total_pnl": round(sum(t.pnl for t in all_trades), 2),
    }
    _export_portfolio(
        pf_mtm, pf_closed, all_trades,
        portfolio_metrics_export, strategy_metrics,
        prop, portfolio_config, OUTPUT_DIR,
    )
    _plot_portfolio(pf_mtm, strategy_results, capital_map, OUTPUT_DIR)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")
    print(f"  Output:  {OUTPUT_DIR}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
