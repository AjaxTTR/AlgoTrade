"""
Performance metrics and visualisation.

Computes a comprehensive set of trading performance statistics from both
mark-to-market and closed-trade equity curves.  All calculations are
vectorised via NumPy/Pandas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from engine.backtester import BacktestResult

# Small constant to prevent division-by-zero
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Frequency detection
# ---------------------------------------------------------------------------

def _infer_bars_per_year(index: pd.DatetimeIndex) -> float:
    """Estimate the number of bars per trading year from the datetime index.

    Uses the median timedelta between consecutive bars to determine bar
    frequency, then scales to a 252-day trading year.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamp index of the equity series.

    Returns
    -------
    float
        Estimated bars per trading year.
    """
    if len(index) < 2:
        return 252.0  # fallback: daily
    # Count actual bars per calendar day from the data, then scale to 252 trading days
    bars_per_day = pd.Series(index).groupby(index.date).count()
    avg_bars_per_day = bars_per_day.median()
    if avg_bars_per_day <= 0:
        return 252.0
    return avg_bars_per_day * 252


# ---------------------------------------------------------------------------
# Equity-level metrics (works on any equity Series)
# ---------------------------------------------------------------------------

def _equity_metrics(
    equity: pd.Series,
    initial_capital: float,
    bars_per_year: float,
) -> dict:
    """Compute return and risk metrics from an equity curve.

    Parameters
    ----------
    equity : pd.Series
        Equity curve with DatetimeIndex.
    initial_capital : float
        Starting capital.
    bars_per_year : float
        Number of bars in one trading year (for annualisation).

    Returns
    -------
    dict
        total_return, cagr, max_drawdown, sharpe, sortino, calmar.
    """
    final = equity.iloc[-1]
    total_return = (final - initial_capital) / (initial_capital + _EPS) * 100

    # CAGR
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    years = days / 365.25 if days > 0 else 1.0
    growth = final / (initial_capital + _EPS)
    cagr = (growth ** (1 / years) - 1) * 100 if years > 0 and growth > 0 else 0.0

    # Drawdown
    running_max = equity.cummax()
    drawdown_pct = (equity - running_max) / (running_max + _EPS) * 100
    max_dd = drawdown_pct.min()

    # Bar returns
    returns = equity.pct_change().dropna()
    ann_factor = np.sqrt(bars_per_year)

    # Sharpe ratio (annualised)
    sharpe = (returns.mean() / (returns.std() + _EPS)) * ann_factor if len(returns) > 1 else 0.0

    # Sortino ratio (annualised, downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 0.0
    sortino = (returns.mean() / (downside_std + _EPS)) * ann_factor if len(returns) > 1 else 0.0

    # Calmar ratio (CAGR / |max drawdown|)
    calmar = cagr / (abs(max_dd) + _EPS)

    return {
        "total_return": round(total_return, 2),
        "cagr": round(cagr, 2),
        "max_drawdown": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
    }


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------

def _consecutive_runs(outcomes: np.ndarray) -> tuple[int, int]:
    """Find max consecutive wins and losses from a boolean win array.

    Parameters
    ----------
    outcomes : np.ndarray
        Boolean array where True = win, False = loss.

    Returns
    -------
    tuple[int, int]
        (max_consecutive_wins, max_consecutive_losses)
    """
    if len(outcomes) == 0:
        return 0, 0

    # Detect change points; count run lengths between them
    changes = np.diff(outcomes.astype(np.int8))
    change_idx = np.nonzero(changes)[0]
    # Run boundaries: start, each change+1, end
    boundaries = np.concatenate([[0], change_idx + 1, [len(outcomes)]])
    run_lengths = np.diff(boundaries)
    run_values = outcomes[boundaries[:-1]]

    win_runs = run_lengths[run_values] if run_values.any() else np.array([0])
    loss_runs = run_lengths[~run_values] if (~run_values).any() else np.array([0])

    return int(win_runs.max()), int(loss_runs.max())


def _trade_metrics(trades: list, bar_interval_seconds: float = 900.0) -> dict:
    """Compute trade-level performance statistics.

    Parameters
    ----------
    trades : list[TradeRecord]
        Completed trade records from the backtester.

    Returns
    -------
    dict
        Comprehensive trade statistics.
    """
    total = len(trades)
    if total == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_trade": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "avg_duration_bars": 0.0,
            "min_duration": "0:00:00",
            "max_duration": "0:00:00",
            "avg_duration": "0:00:00",
        }

    pnls = np.array([t.pnl for t in trades])
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]

    win_rate = len(winners) / total * 100
    gross_profit = winners.sum() if len(winners) > 0 else 0.0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / (gross_loss + _EPS)

    avg_win = winners.mean() if len(winners) > 0 else 0.0
    avg_loss = losers.mean() if len(losers) > 0 else 0.0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    # Consecutive streaks
    outcomes = pnls > 0
    consec_w, consec_l = _consecutive_runs(outcomes)

    # Trade durations
    durations = pd.Series([t.exit_time - t.entry_time for t in trades])
    avg_dur = durations.mean()
    min_dur = durations.min()
    max_dur = durations.max()

    return {
        "total_trades": total,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 2),
        "avg_trade": round(pnls.mean(), 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "largest_win": round(pnls.max(), 2),
        "largest_loss": round(pnls.min(), 2),
        "consecutive_wins": consec_w,
        "consecutive_losses": consec_l,
        "avg_duration_bars": round(durations.dt.total_seconds().mean() / (bar_interval_seconds + _EPS), 1),
        "min_duration": str(min_dur),
        "max_duration": str(max_dur),
        "avg_duration": str(avg_dur),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics(result: BacktestResult, initial_capital: float = 100_000.0) -> dict:
    """Compute a comprehensive set of performance metrics.

    Metrics are computed for both the mark-to-market equity curve and the
    closed-trade equity curve, plus trade-level statistics.

    Parameters
    ----------
    result : BacktestResult
        Output from ``backtester.run()``.
    initial_capital : float
        Starting equity (must match the backtest).

    Returns
    -------
    dict
        Nested structure:
        - ``mtm``: equity metrics from mark-to-market curve
        - ``closed``: equity metrics from closed-trade curve
        - ``trades``: trade-level statistics
        - ``bars_per_year``: inferred annualisation factor
    """
    bars_per_year = _infer_bars_per_year(result.equity_mtm.index)

    # Infer median bar interval for duration calculations
    deltas = pd.Series(result.equity_mtm.index).diff().dropna().dt.total_seconds()
    bar_interval = deltas.median() if len(deltas) > 0 else 900.0

    mtm_metrics = _equity_metrics(result.equity_mtm, initial_capital, bars_per_year)
    closed_metrics = _equity_metrics(result.equity_closed, initial_capital, bars_per_year)
    trade_stats = _trade_metrics(result.trades, bar_interval_seconds=bar_interval)

    return {
        "mtm": mtm_metrics,
        "closed": closed_metrics,
        "trades": trade_stats,
        "bars_per_year": round(bars_per_year, 0),
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print performance metrics to stdout."""
    mtm = metrics["mtm"]
    closed = metrics["closed"]
    t = metrics["trades"]

    print("\n" + "=" * 62)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * 62)

    header = f"  {'':30s} {'MTM':>10s} {'Closed':>10s}"
    print(header)
    print("  " + "-" * 58)

    for key in ("total_return", "cagr", "max_drawdown", "sharpe", "sortino", "calmar"):
        label = key.replace("_", " ").title()
        unit = " %" if key in ("total_return", "cagr", "max_drawdown") else ""
        print(f"  {label + unit:30s} {mtm[key]:>10.2f} {closed[key]:>10.2f}")

    print("\n  " + "-" * 58)
    print(f"  {'Total Trades':30s} {t['total_trades']:>10d}")
    print(f"  {'Win Rate %':30s} {t['win_rate']:>10.2f}")
    print(f"  {'Profit Factor':30s} {t['profit_factor']:>10.2f}")
    print(f"  {'Expectancy ($)':30s} {t['expectancy']:>10.2f}")
    print(f"  {'Avg Trade ($)':30s} {t['avg_trade']:>10.2f}")
    print(f"  {'Avg Win ($)':30s} {t['avg_win']:>10.2f}")
    print(f"  {'Avg Loss ($)':30s} {t['avg_loss']:>10.2f}")
    print(f"  {'Largest Win ($)':30s} {t['largest_win']:>10.2f}")
    print(f"  {'Largest Loss ($)':30s} {t['largest_loss']:>10.2f}")
    print(f"  {'Consecutive Wins':30s} {t['consecutive_wins']:>10d}")
    print(f"  {'Consecutive Losses':30s} {t['consecutive_losses']:>10d}")
    print(f"  {'Avg Duration':30s} {t['avg_duration']:>18s}")

    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_results(result: BacktestResult, metrics: dict) -> None:
    """Generate publication-quality performance plots.

    Panels:
        1. Equity curves (MTM + closed-trade)
        2. Drawdown from MTM equity
        3. Trade P&L histogram
        4. Rolling Sharpe ratio

    Parameters
    ----------
    result : BacktestResult
        Backtest output.
    metrics : dict
        Metrics dict from ``compute_metrics()``.
    """
    mtm = metrics["mtm"]
    bars_per_year = metrics["bars_per_year"]

    # Style
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#CCCCCC",
        "font.family": "sans-serif",
        "font.size": 10,
    })

    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    fig.suptitle(
        f"Backtest Results  |  Return: {mtm['total_return']}%  "
        f"Sharpe: {mtm['sharpe']}  MaxDD: {mtm['max_drawdown']}%",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # ── 1. Equity curves ──
    ax1 = axes[0]
    ax1.plot(result.equity_mtm.index, result.equity_mtm.values,
             linewidth=1.0, color="#1565C0", label="Mark-to-Market")
    ax1.plot(result.equity_closed.index, result.equity_closed.values,
             linewidth=1.0, color="#FF8F00", alpha=0.75, label="Closed-Trade")
    ax1.set_title("Equity Curve", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax1.ticklabel_format(axis="y", style="plain")

    # ── 2. Drawdown ──
    ax2 = axes[1]
    ax2.fill_between(
        result.drawdown_series.index,
        result.drawdown_series.values,
        0,
        color="#D32F2F",
        alpha=0.35,
        linewidth=0,
    )
    ax2.plot(result.drawdown_series.index, result.drawdown_series.values,
             color="#D32F2F", linewidth=0.6, alpha=0.7)
    ax2.set_title("Drawdown", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")

    # ── 3. Trade P&L histogram ──
    ax3 = axes[2]
    if result.trades:
        pnls = np.array([t.pnl for t in result.trades])
        bins = min(50, max(10, len(pnls) // 3))
        ax3.hist(pnls, bins=bins, color="#1565C0", edgecolor="white",
                 linewidth=0.5, alpha=0.85)
        ax3.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
        ax3.axvline(pnls.mean(), color="#FF8F00", linewidth=1.2,
                    linestyle="-", label=f"Mean: ${pnls.mean():.0f}")
        ax3.legend(fontsize=9, framealpha=0.8)
    else:
        ax3.text(0.5, 0.5, "No trades", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=12, color="#999999")
    ax3.set_title("Trade P&L Distribution", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Frequency")
    ax3.set_xlabel("P&L ($)")

    # ── 4. Rolling Sharpe ratio ──
    ax4 = axes[3]
    returns = result.equity_mtm.pct_change().dropna()
    if len(returns) > 1:
        # Rolling window = ~60 trading days worth of bars
        window = max(20, int(bars_per_year / 4))
        roll_mean = returns.rolling(window, min_periods=window // 2).mean()
        roll_std = returns.rolling(window, min_periods=window // 2).std()
        rolling_sharpe = (roll_mean / (roll_std + _EPS)) * np.sqrt(bars_per_year)

        ax4.plot(rolling_sharpe.index, rolling_sharpe.values,
                 linewidth=0.9, color="#1565C0")
        ax4.axhline(0, color="#333333", linewidth=0.6, linestyle="--")
        ax4.fill_between(
            rolling_sharpe.index,
            rolling_sharpe.values,
            0,
            where=rolling_sharpe.values >= 0,
            color="#43A047",
            alpha=0.2,
        )
        ax4.fill_between(
            rolling_sharpe.index,
            rolling_sharpe.values,
            0,
            where=rolling_sharpe.values < 0,
            color="#D32F2F",
            alpha=0.2,
        )
    ax4.set_title("Rolling Sharpe Ratio (quarterly window)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Sharpe")
    ax4.set_xlabel("Date")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("Plot saved to backtest_results.png")
