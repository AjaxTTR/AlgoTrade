"""
Performance metrics and visualisation.

All metric functions operate on a BacktestResult object returned by the
backtesting engine.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backtester import BacktestResult


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def compute_metrics(result: BacktestResult, initial_capital: float = 100_000.0) -> dict:
    """Compute a standard set of performance metrics.

    Parameters
    ----------
    result : BacktestResult
        Output from ``backtester.run()``.
    initial_capital : float
        Starting equity (must match the backtest).

    Returns
    -------
    dict
        Keys: total_return, cagr, max_drawdown, sharpe, win_rate,
        profit_factor, total_trades, avg_trade.
    """
    equity = result.equity_mtm
    trades = result.trades

    # --- Total return ---
    final_equity = equity.iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100

    # --- CAGR ---
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    years = days / 365.25 if days > 0 else 1.0
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    # --- Max drawdown ---
    max_dd = result.drawdown_series.min()

    # --- Sharpe ratio (annualised, based on bar returns) ---
    returns = equity.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        # Assume 15-min bars, ~27 bars/day, ~252 trading days
        bars_per_year = 27 * 252
        sharpe = returns.mean() / returns.std() * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # --- Trade-level metrics ---
    total_trades = len(trades)
    if total_trades > 0:
        pnls = np.array([t.pnl for t in trades])
        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]

        win_rate = len(winners) / total_trades * 100
        gross_profit = winners.sum() if len(winners) > 0 else 0.0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        avg_trade = pnls.mean()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade = 0.0

    return {
        "total_return": round(total_return, 2),
        "cagr": round(cagr, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "total_trades": total_trades,
        "avg_trade": round(avg_trade, 2),
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print performance metrics to stdout."""
    print("\n" + "=" * 50)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"  Total Return:    {metrics['total_return']:>10.2f} %")
    print(f"  CAGR:            {metrics['cagr']:>10.2f} %")
    print(f"  Max Drawdown:    {metrics['max_drawdown']:>10.2f} %")
    print(f"  Sharpe Ratio:    {metrics['sharpe']:>10.2f}")
    print(f"  Win Rate:        {metrics['win_rate']:>10.2f} %")
    print(f"  Profit Factor:   {metrics['profit_factor']:>10.2f}")
    print(f"  Total Trades:    {metrics['total_trades']:>10d}")
    print(f"  Avg Trade ($):   {metrics['avg_trade']:>10.2f}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(result: BacktestResult, metrics: dict) -> None:
    """Generate equity curve, drawdown, and trade distribution plots.

    Parameters
    ----------
    result : BacktestResult
        Backtest output containing equity curve, drawdown, and trades.
    metrics : dict
        Performance metrics dict (used for title annotation).
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    fig.suptitle(
        f"Backtest Results  |  Return: {metrics['total_return']}%  "
        f"Sharpe: {metrics['sharpe']}  MaxDD: {metrics['max_drawdown']}%",
        fontsize=13,
        fontweight="bold",
    )

    # --- Equity curves (MTM and closed-trade) ---
    ax1 = axes[0]
    ax1.plot(result.equity_mtm.index, result.equity_mtm.values, linewidth=1, color="#2196F3", label="Mark-to-Market")
    ax1.plot(result.equity_closed.index, result.equity_closed.values, linewidth=1, color="#FF9800", alpha=0.7, label="Closed-Trade")
    ax1.set_title("Equity Curve", fontsize=11)
    ax1.set_ylabel("Equity ($)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Drawdown ---
    ax2 = axes[1]
    ax2.fill_between(
        result.drawdown_series.index,
        result.drawdown_series.values,
        0,
        color="#F44336",
        alpha=0.4,
    )
    ax2.set_title("Drawdown (%)", fontsize=11)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)

    # --- Trade PnL distribution ---
    ax3 = axes[2]
    if result.trades:
        pnls = [t.pnl for t in result.trades]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, width=1.0, edgecolor="none")
        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.set_title("Trade P&L Distribution", fontsize=11)
        ax3.set_ylabel("P&L ($)")
        ax3.set_xlabel("Trade #")
    else:
        ax3.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax3.transAxes)

    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to backtest_results.png")
