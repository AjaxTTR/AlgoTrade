"""
Portfolio runner: execute multiple independent strategies simultaneously.

Runs each strategy through its own backtester instance with equal capital
allocation (total_capital / N strategies).  Trades are fully independent —
no signal overwrites, no shared position limits across strategies.

Outputs:
  - Total portfolio performance (combined equity curve)
  - Per-strategy performance breakdown
  - Combined trade log

Usage:
    python run_portfolio.py
"""

import importlib
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest, BacktestResult
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
OUTPUT_DIR = Path("output/portfolio")

TOTAL_CAPITAL = 100_000.0

# Strategy definitions: (module_name, strategy_params, backtest_overrides)
STRATEGIES = {
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
    "no_gap_breakout": {
        "module": "strategies.no_gap_breakout",
        "strategy_params": {
            "fh_percentile": 75.0,
            "gap_threshold_pct": 0.10,
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

# Shared backtest config (applied to all strategies)
BASE_BACKTEST_CONFIG = {
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "risk_per_trade": 0.005,
    "use_trailing_stop": False,
    "daily_dd_limit": 0.02,
    "max_daily_risk": 0.02,
    "max_dd_limit": 0.0,
    "max_bars_in_trade": 0,          # EOD exit via session_close
    "max_concurrent_trades": 1,       # 1 per strategy
    "min_bars_between_entries": 2,
    "consec_loss_threshold": 2,
    "loss_scale_down": 0.5,
    "max_risk_per_trade": 0.0,        # disabled — daily_dd_limit controls risk
}


# ---------------------------------------------------------------------------
# Per-strategy execution
# ---------------------------------------------------------------------------

def _run_single_strategy(
    name: str,
    config: dict,
    df: pd.DataFrame,
    capital: float,
) -> tuple[BacktestResult, dict, pd.DataFrame]:
    """Run one strategy end-to-end: signals -> backtest -> metrics.

    Returns (BacktestResult, metrics_dict, signals_df).
    """
    mod = importlib.import_module(config["module"])
    signals = mod.generate_signals(df.copy(), **config["strategy_params"])
    n_signals = int((signals["signal"] != 0).sum())

    bt_cfg = {
        **BASE_BACKTEST_CONFIG,
        **config.get("backtest_overrides", {}),
        "initial_capital": capital,
        "strategy_name": name,
    }
    result = run_backtest(signals, **bt_cfg)
    metrics = compute_metrics(result, initial_capital=capital)

    return result, metrics, signals


# ---------------------------------------------------------------------------
# Portfolio combination
# ---------------------------------------------------------------------------

def _combine_equity_curves(
    results: dict[str, BacktestResult],
    capital_per_strategy: float,
    total_capital: float,
) -> tuple[pd.Series, pd.Series]:
    """Combine per-strategy equity curves into a portfolio curve.

    Each strategy's equity is rebased to its change from initial capital,
    then all changes are summed:
        portfolio_equity[t] = total_capital + sum(strategy_equity[t] - strategy_capital)

    Returns (portfolio_mtm, portfolio_closed).
    """
    # Use the first strategy's index as the base timeline
    ref_index = next(iter(results.values())).equity_mtm.index

    pf_mtm = pd.Series(total_capital, index=ref_index, dtype=np.float64)
    pf_closed = pd.Series(total_capital, index=ref_index, dtype=np.float64)

    for name, result in results.items():
        # Strategy's PnL relative to its allocation
        mtm_pnl = result.equity_mtm - capital_per_strategy
        closed_pnl = result.equity_closed - capital_per_strategy

        # Align to reference index (forward fill for any mismatches)
        mtm_pnl = mtm_pnl.reindex(ref_index, method="ffill").fillna(0.0)
        closed_pnl = closed_pnl.reindex(ref_index, method="ffill").fillna(0.0)

        pf_mtm = pf_mtm + mtm_pnl
        pf_closed = pf_closed + closed_pnl

    pf_mtm.name = "equity_mtm"
    pf_closed.name = "equity_closed"
    return pf_mtm, pf_closed


def _combine_trades(results: dict[str, BacktestResult]) -> list:
    """Merge all trades across strategies, sorted by entry time."""
    all_trades = []
    for name, result in results.items():
        all_trades.extend(result.trades)
    all_trades.sort(key=lambda t: t.entry_time)
    return all_trades


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_strategy_summary(
    name: str,
    result: BacktestResult,
    metrics: dict,
    n_signals: int,
    capital: float,
):
    """Print a compact per-strategy summary."""
    mtm = metrics["mtm"]
    trades_m = metrics["trades"]
    n_trades = len(result.trades)
    total_pnl = sum(t.pnl for t in result.trades)

    print(f"  {name}")
    print(f"    Signals: {n_signals:>5d}   Trades: {n_trades:>5d}   "
          f"PnL: ${total_pnl:>+10,.2f}")
    print(f"    Return: {mtm['total_return']:>+7.2f}%   "
          f"Sharpe: {mtm['sharpe']:>5.2f}   "
          f"MaxDD: {mtm['max_drawdown']:>6.2f}%   "
          f"PF: {trades_m.get('profit_factor', 0.0):>5.2f}   "
          f"WR: {trades_m.get('win_rate', 0.0):>5.1f}%")


def _print_portfolio_summary(
    pf_mtm: pd.Series,
    pf_closed: pd.Series,
    all_trades: list,
    total_capital: float,
    strategy_results: dict,
    strategy_metrics: dict,
):
    """Print full portfolio performance report."""
    from engine.metrics import _infer_bars_per_year, _equity_metrics

    bars_per_year = _infer_bars_per_year(pf_mtm.index)
    mtm_m = _equity_metrics(pf_mtm, total_capital, bars_per_year)
    closed_m = _equity_metrics(pf_closed, total_capital, bars_per_year)

    # Drawdown
    running_max = pf_mtm.cummax()
    drawdown = (pf_mtm - running_max) / running_max * 100

    # Trade stats
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
    print("  PORTFOLIO PERFORMANCE SUMMARY")
    print(sep)

    print(f"\n  {'':30s} {'MTM':>10s} {'Closed':>10s}")
    print(f"  {'-' * 52}")
    print(f"  {'Total Return %':30s} {mtm_m['total_return']:>+10.2f} {closed_m['total_return']:>+10.2f}")
    print(f"  {'CAGR %':30s} {mtm_m['cagr']:>+10.2f} {closed_m['cagr']:>+10.2f}")
    print(f"  {'Max Drawdown %':30s} {mtm_m['max_drawdown']:>10.2f} {closed_m['max_drawdown']:>10.2f}")
    print(f"  {'Sharpe':30s} {mtm_m['sharpe']:>10.2f} {closed_m['sharpe']:>10.2f}")
    print(f"  {'Sortino':30s} {mtm_m['sortino']:>10.2f} {closed_m['sortino']:>10.2f}")
    print(f"  {'Calmar':30s} {mtm_m['calmar']:>10.2f} {closed_m['calmar']:>10.2f}")

    print(f"\n  {'-' * 52}")
    print(f"  {'Total Trades':30s} {n_trades:>10d}")
    print(f"  {'Total PnL':30s} {'${:>+,.2f}'.format(total_pnl):>10s}")
    print(f"  {'Win Rate %':30s} {win_rate:>10.1f}")
    print(f"  {'Profit Factor':30s} {profit_factor:>10.2f}")
    print(f"  {'Avg Win ($)':30s} {avg_win:>+10.2f}")
    print(f"  {'Avg Loss ($)':30s} {avg_loss:>+10.2f}")

    # Per-strategy breakdown
    print(f"\n  {'-' * 52}")
    print("  PER-STRATEGY BREAKDOWN")
    print(f"  {'-' * 52}")

    n_strats = len(strategy_results)
    cap_each = total_capital / n_strats

    for name in strategy_results:
        result = strategy_results[name]
        metrics = strategy_metrics[name]
        n_signals = int((result.signals_df["signal"] != 0).sum())
        _print_strategy_summary(name, result, metrics, n_signals, cap_each)
        print()

    # Correlation matrix of daily returns
    print(f"  {'-' * 52}")
    print("  DAILY RETURN CORRELATIONS")
    print(f"  {'-' * 52}")

    daily_returns = {}
    for name, result in strategy_results.items():
        daily_eq = result.equity_mtm.resample("D").last().dropna()
        daily_returns[name] = daily_eq.pct_change().dropna()

    if len(daily_returns) >= 2:
        corr_df = pd.DataFrame(daily_returns).corr()
        # Print upper triangle
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


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _export_portfolio(
    pf_mtm: pd.Series,
    pf_closed: pd.Series,
    all_trades: list,
    portfolio_metrics: dict,
    strategy_metrics: dict,
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

    # Metrics JSON
    export_metrics = {
        "portfolio": portfolio_metrics,
        "per_strategy": {},
    }
    for name, m in strategy_metrics.items():
        export_metrics["per_strategy"][name] = m
    metrics_path = output_dir / "portfolio_metrics.json"
    metrics_path.write_text(json.dumps(export_metrics, indent=2, default=str))
    log.info("Metrics saved to %s", metrics_path)


def _plot_portfolio(
    pf_mtm: pd.Series,
    strategy_results: dict[str, BacktestResult],
    capital_per_strategy: float,
    output_dir: Path,
):
    """Generate portfolio equity chart with per-strategy breakdown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])

    # Panel 1: Equity curves
    ax1 = axes[0]
    ax1.plot(pf_mtm.index, pf_mtm.values, color="black", linewidth=1.5,
             label="Portfolio")

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for idx, (name, result) in enumerate(strategy_results.items()):
        c = colors[idx % len(colors)]
        ax1.plot(result.equity_mtm.index, result.equity_mtm.values,
                 color=c, linewidth=0.8, alpha=0.7, label=name)

    ax1.set_title("Portfolio Equity Curve", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Drawdown
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
    t_start = time.perf_counter()

    n_strategies = len(STRATEGIES)
    capital_per_strategy = TOTAL_CAPITAL / n_strategies

    print("\n" + "=" * 70)
    print("  NQ PORTFOLIO BACKTEST")
    print("=" * 70)
    print(f"  Total capital:     ${TOTAL_CAPITAL:>12,.2f}")
    print(f"  Strategies:        {n_strategies:>12d}")
    print(f"  Capital each:      ${capital_per_strategy:>12,.2f}")
    print()

    # --- Load data once ---
    log.info("Loading data from %s", DATA_FILE)
    try:
        df = load_csv(DATA_FILE)
    except FileNotFoundError:
        log.error("%s not found.", DATA_FILE)
        sys.exit(1)
    log.info("Loaded %s bars  |  %s  ->  %s", f"{len(df):,}",
             df.index[0], df.index[-1])

    # --- Run each strategy independently ---
    strategy_results: dict[str, BacktestResult] = {}
    strategy_metrics: dict[str, dict] = {}

    for name, config in STRATEGIES.items():
        log.info("Running strategy: %s", name)
        result, metrics, signals = _run_single_strategy(
            name, config, df, capital_per_strategy,
        )
        strategy_results[name] = result
        strategy_metrics[name] = metrics
        n_signals = int((signals["signal"] != 0).sum())
        n_trades = len(result.trades)
        log.info("  %s: %d signals -> %d trades", name, n_signals, n_trades)

    # --- Combine into portfolio ---
    log.info("Combining equity curves")
    pf_mtm, pf_closed = _combine_equity_curves(
        strategy_results, capital_per_strategy, TOTAL_CAPITAL,
    )
    all_trades = _combine_trades(strategy_results)

    # --- Print results ---
    portfolio_mtm_metrics, portfolio_closed_metrics = _print_portfolio_summary(
        pf_mtm, pf_closed, all_trades, TOTAL_CAPITAL,
        strategy_results, strategy_metrics,
    )

    # --- Export ---
    portfolio_metrics_export = {
        "mtm": portfolio_mtm_metrics,
        "closed": portfolio_closed_metrics,
        "total_trades": len(all_trades),
        "total_pnl": round(sum(t.pnl for t in all_trades), 2),
    }
    _export_portfolio(
        pf_mtm, pf_closed, all_trades,
        portfolio_metrics_export, strategy_metrics, OUTPUT_DIR,
    )

    # --- Plot ---
    _plot_portfolio(pf_mtm, strategy_results, capital_per_strategy, OUTPUT_DIR)

    # --- Runtime ---
    elapsed = time.perf_counter() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")
    print(f"  Output:  {OUTPUT_DIR}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
