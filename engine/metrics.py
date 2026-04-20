"""
Performance metrics and visualisation.

Computes a comprehensive set of trading performance statistics from both
mark-to-market and closed-trade equity curves.  All calculations are
vectorised via NumPy/Pandas.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
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
# Equity curve consistency (prop firm suitability)
# ---------------------------------------------------------------------------

def _consistency_metrics(
    equity: pd.Series,
    trades: list,
    bars_per_year: float,
) -> dict:
    """Measure equity curve consistency for prop firm suitability.

    Parameters
    ----------
    equity : pd.Series
        Mark-to-market equity curve.
    trades : list[TradeRecord]
        Completed trade records.
    bars_per_year : float
        Annualisation factor.

    Returns
    -------
    dict
        equity_volatility, max_consecutive_losses, rolling_dd_5d,
        rolling_dd_10d, return_consistency, consistency_score,
        prop_firm_score.
    """
    returns = equity.pct_change().dropna()

    # -- Equity volatility (annualised std of bar returns) --
    equity_vol = float(returns.std() * np.sqrt(bars_per_year)) if len(returns) > 1 else 0.0

    # -- Max consecutive losses from trade records --
    if len(trades) >= 1:
        pnls = np.array([t.pnl for t in trades])
        _, max_consec_losses = _consecutive_runs(pnls > 0)
    else:
        max_consec_losses = 0

    # -- Rolling drawdowns (5-day and 10-day windows on daily equity) --
    daily_equity = equity.groupby(equity.index.date).last()
    if len(daily_equity) >= 2:
        daily_ret = daily_equity.pct_change().dropna()

        def _rolling_max_dd(window):
            if len(daily_equity) < window:
                return 0.0
            roll_max = daily_equity.rolling(window, min_periods=window).max()
            roll_dd = (daily_equity - roll_max) / (roll_max + _EPS) * 100
            return float(roll_dd.min())

        rolling_dd_5d = _rolling_max_dd(5)
        rolling_dd_10d = _rolling_max_dd(10)

        # -- Return consistency (% of positive trading days) --
        return_consistency = float((daily_ret > 0).mean() * 100)
    else:
        rolling_dd_5d = 0.0
        rolling_dd_10d = 0.0
        return_consistency = 0.0

    # -- Consistency score (0-1) --
    # Component 1: Return consistency — 55% positive days = 0.5, 65% = 1.0
    rc_score = np.clip((return_consistency - 45) / 20, 0, 1)

    # Component 2: Volatility — lower is better, 0 at 50%+ annualised vol
    vol_score = np.clip(1 - equity_vol / 0.5, 0, 1) if equity_vol > 0 else 0.5

    # Component 3: Consecutive losses — 0 = perfect, 10+ = 0
    cl_score = np.clip(1 - max_consec_losses / 10, 0, 1)

    # Component 4: Rolling DD — 0% = perfect, -8%+ = 0
    dd_score = np.clip(1 - abs(rolling_dd_5d) / 8, 0, 1)

    consistency_score = float(
        0.35 * rc_score
        + 0.25 * vol_score
        + 0.20 * cl_score
        + 0.20 * dd_score
    )

    # -- Prop firm score (0-1) --
    # Blends consistency with risk-adjusted return quality
    sharpe = (returns.mean() / (returns.std() + _EPS)) * np.sqrt(bars_per_year) if len(returns) > 1 else 0.0
    sharpe_score = np.clip(sharpe / 3.0, 0, 1)  # 3.0 Sharpe = perfect

    calmar_raw = 0.0
    if len(equity) > 1:
        running_max = equity.cummax()
        max_dd = ((equity - running_max) / (running_max + _EPS) * 100).min()
        days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
        years = days / 365.25 if days > 0 else 1.0
        growth = equity.iloc[-1] / (equity.iloc[0] + _EPS)
        cagr = (growth ** (1 / years) - 1) * 100 if years > 0 and growth > 0 else 0.0
        calmar_raw = cagr / (abs(max_dd) + _EPS)
    calmar_score = np.clip(calmar_raw / 5.0, 0, 1)  # 5.0 calmar = perfect

    prop_firm_score = float(
        0.50 * consistency_score
        + 0.30 * sharpe_score
        + 0.20 * calmar_score
    )

    return {
        "equity_volatility": round(equity_vol, 4),
        "max_consecutive_losses": max_consec_losses,
        "rolling_dd_5d": round(rolling_dd_5d, 2),
        "rolling_dd_10d": round(rolling_dd_10d, 2),
        "return_consistency": round(return_consistency, 2),
        "consistency_score": round(consistency_score, 4),
        "prop_firm_score": round(prop_firm_score, 4),
    }


# ---------------------------------------------------------------------------
# Loss distribution analysis
# ---------------------------------------------------------------------------

def _loss_distribution_metrics(trades: list) -> dict:
    """Analyse loss distribution and tail risk.

    Parameters
    ----------
    trades : list[TradeRecord]
        Completed trade records.

    Returns
    -------
    dict
        Loss tail metrics, cap effectiveness, and exit reason breakdown.
    """
    if len(trades) == 0:
        return {
            "total_losses": 0,
            "avg_loss": 0.0,
            "median_loss": 0.0,
            "worst_loss": 0.0,
            "p5_loss": 0.0,
            "p1_loss": 0.0,
            "loss_std": 0.0,
            "loss_skew": 0.0,
            "max_loss_cap_exits": 0,
            "time_exits": 0,
            "vol_stop_tighten_effective": 0.0,
            "exit_reason_breakdown": {},
        }

    pnls = np.array([t.pnl for t in trades])
    losses = pnls[pnls < 0]
    n_losses = len(losses)

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    max_loss_cap_exits = reasons.get("max_loss_cap", 0)
    time_exits = reasons.get("time_exit", 0)

    if n_losses == 0:
        return {
            "total_losses": 0,
            "avg_loss": 0.0,
            "median_loss": 0.0,
            "worst_loss": 0.0,
            "p5_loss": 0.0,
            "p1_loss": 0.0,
            "loss_std": 0.0,
            "loss_skew": 0.0,
            "max_loss_cap_exits": max_loss_cap_exits,
            "time_exits": time_exits,
            "vol_stop_tighten_effective": 0.0,
            "exit_reason_breakdown": reasons,
        }

    loss_pcts = np.percentile(losses, [1, 5, 50])

    # Loss skewness (more negative = heavier left tail)
    if n_losses >= 3:
        loss_skew = float(pd.Series(losses).skew())
    else:
        loss_skew = 0.0

    # Tail concentration: % of total loss coming from worst 5% of trades
    sorted_losses = np.sort(losses)  # most negative first
    n_tail = max(1, int(len(sorted_losses) * 0.05))
    tail_loss = sorted_losses[:n_tail].sum()
    total_loss = losses.sum()
    tail_concentration = abs(tail_loss) / (abs(total_loss) + _EPS) * 100

    return {
        "total_losses": n_losses,
        "avg_loss": round(float(losses.mean()), 2),
        "median_loss": round(float(loss_pcts[2]), 2),
        "worst_loss": round(float(losses.min()), 2),
        "p5_loss": round(float(loss_pcts[1]), 2),
        "p1_loss": round(float(loss_pcts[0]), 2),
        "loss_std": round(float(losses.std()), 2),
        "loss_skew": round(loss_skew, 4),
        "tail_concentration_pct": round(tail_concentration, 2),
        "max_loss_cap_exits": max_loss_cap_exits,
        "time_exits": time_exits,
        "exit_reason_breakdown": reasons,
    }


# ---------------------------------------------------------------------------
# MAE/MFE excursion analysis
# ---------------------------------------------------------------------------

def _excursion_metrics(trades: list) -> dict:
    """Analyse max adverse/favourable excursion distributions.

    Parameters
    ----------
    trades : list[TradeRecord]
        Completed trade records with MAE/MFE fields.

    Returns
    -------
    dict
        Distribution stats for MAE, MFE, ratio, and stop efficiency.
    """
    if len(trades) == 0:
        return {
            "avg_mae": 0.0, "median_mae": 0.0, "p95_mae": 0.0,
            "avg_mfe": 0.0, "median_mfe": 0.0, "p95_mfe": 0.0,
            "avg_mae_mfe_ratio": 0.0, "median_mae_mfe_ratio": 0.0,
            "avg_stop_distance": 0.0,
            "mae_vs_stop_pct": 0.0,
            "stop_efficiency": 0.0,
            "edge_ratio": 0.0,
            "n_trades": 0,
        }

    maes = np.array([t.mae for t in trades])
    mfes = np.array([t.mfe for t in trades])
    ratios = np.array([t.mae_mfe_ratio for t in trades])
    stop_dists = np.array([t.stop_distance for t in trades])

    # MAE vs stop distance: how much of the stop is typically used
    valid_stops = stop_dists[stop_dists > 0]
    valid_maes = maes[stop_dists > 0]
    if len(valid_stops) > 0:
        mae_vs_stop = float(np.mean(valid_maes / valid_stops) * 100)
        # Stop efficiency: % of trades where MAE < 50% of stop distance
        stop_eff = float(np.mean(valid_maes < valid_stops * 0.5) * 100)
    else:
        mae_vs_stop = 0.0
        stop_eff = 0.0

    # Edge ratio: avg MFE / avg MAE (higher = better)
    avg_mae = float(np.mean(maes))
    avg_mfe = float(np.mean(mfes))
    edge_ratio = avg_mfe / (avg_mae + _EPS)

    mae_pcts = np.percentile(maes, [50, 95])
    mfe_pcts = np.percentile(mfes, [50, 95])

    return {
        "avg_mae": round(avg_mae, 2),
        "median_mae": round(float(mae_pcts[0]), 2),
        "p95_mae": round(float(mae_pcts[1]), 2),
        "avg_mfe": round(avg_mfe, 2),
        "median_mfe": round(float(mfe_pcts[0]), 2),
        "p95_mfe": round(float(mfe_pcts[1]), 2),
        "avg_mae_mfe_ratio": round(float(np.mean(ratios)), 4),
        "median_mae_mfe_ratio": round(float(np.median(ratios)), 4),
        "avg_stop_distance": round(float(np.mean(stop_dists)), 2),
        "mae_vs_stop_pct": round(mae_vs_stop, 2),
        "stop_efficiency": round(stop_eff, 2),
        "edge_ratio": round(edge_ratio, 4),
        "n_trades": len(trades),
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

    # Statistical significance of MTM returns
    returns = result.equity_mtm.pct_change().dropna()
    n_returns = len(returns)
    if n_returns >= 2 and returns.std() > 0:
        t_stat, p_value = sp_stats.ttest_1samp(returns, 0)
        std_err = returns.std() / np.sqrt(n_returns)
        mtm_metrics["t_statistic"] = round(float(t_stat), 4)
        mtm_metrics["p_value"] = round(float(p_value), 6)
        mtm_metrics["standard_error"] = round(float(std_err), 8)
        mtm_metrics["significance"] = bool(p_value < 0.05)
    else:
        mtm_metrics["t_statistic"] = 0.0
        mtm_metrics["p_value"] = 1.0
        mtm_metrics["standard_error"] = 0.0
        mtm_metrics["significance"] = False

    consistency = _consistency_metrics(result.equity_mtm, result.trades, bars_per_year)
    excursion = _excursion_metrics(result.trades)
    loss_dist = _loss_distribution_metrics(result.trades)

    return {
        "mtm": mtm_metrics,
        "closed": closed_metrics,
        "trades": trade_stats,
        "consistency": consistency,
        "excursion": excursion,
        "loss_distribution": loss_dist,
        "bars_per_year": round(bars_per_year, 0),
    }


# Default thresholds for strategy validation
DEFAULT_VALIDATION_THRESHOLDS = {
    "min_sharpe": 0.0,
    "min_profit_factor": 1.1,
    "max_p_value": 0.05,
    "max_drawdown": -30.0,       # % (e.g. -30 means reject if DD worse than -30%)
    "min_trades": 30,
    "min_win_rate": 30.0,        # %
    "max_avg_loss_ratio": 3.0,   # |avg_loss| / avg_win — reject lopsided losers
}


def validate_strategy(
    metrics: dict,
    thresholds: dict | None = None,
) -> dict:
    """Validate whether a strategy meets minimum quality thresholds.

    Runs each check independently so the caller gets a complete list of
    failures rather than stopping at the first one.

    Parameters
    ----------
    metrics : dict
        Output from ``compute_metrics()``.
    thresholds : dict | None
        Override individual thresholds.  Keys match
        ``DEFAULT_VALIDATION_THRESHOLDS``; missing keys use defaults.

    Returns
    -------
    dict
        passed : bool
            True only if every check passes.
        failures : list[dict]
            Each entry has ``rule``, ``threshold``, ``actual``, ``message``.
        checks_run : int
        checks_passed : int
    """
    th = {**DEFAULT_VALIDATION_THRESHOLDS, **(thresholds or {})}
    mtm = metrics.get("mtm", {})
    trades = metrics.get("trades", {})

    failures: list[dict] = []
    checks_run = 0

    def _check(rule: str, actual, threshold, compare, msg: str):
        nonlocal checks_run
        checks_run += 1
        if not compare(actual, threshold):
            failures.append({
                "rule": rule,
                "threshold": threshold,
                "actual": round(actual, 4) if isinstance(actual, float) else actual,
                "message": msg,
            })

    # 1. Sharpe ratio
    sharpe = mtm.get("sharpe", 0.0)
    _check(
        "min_sharpe", sharpe, th["min_sharpe"],
        lambda a, t: a >= t,
        f"Sharpe {sharpe:.2f} < {th['min_sharpe']:.2f} — no risk-adjusted edge",
    )

    # 2. Profit factor
    pf = trades.get("profit_factor", 0.0)
    _check(
        "min_profit_factor", pf, th["min_profit_factor"],
        lambda a, t: a >= t,
        f"Profit factor {pf:.2f} < {th['min_profit_factor']:.2f} — insufficient edge",
    )

    # 3. Statistical significance
    p_value = mtm.get("p_value", 1.0)
    _check(
        "max_p_value", p_value, th["max_p_value"],
        lambda a, t: a <= t,
        f"p-value {p_value:.4f} > {th['max_p_value']:.2f} — not statistically significant",
    )

    # 4. Max drawdown
    max_dd = mtm.get("max_drawdown", 0.0)
    _check(
        "max_drawdown", max_dd, th["max_drawdown"],
        lambda a, t: a >= t,  # DD is negative, so -40 < -30 fails
        f"Max DD {max_dd:.2f}% worse than {th['max_drawdown']:.1f}% limit",
    )

    # 5. Minimum trades
    n_trades = trades.get("total_trades", 0)
    _check(
        "min_trades", n_trades, th["min_trades"],
        lambda a, t: a >= t,
        f"Only {n_trades} trades — need >= {th['min_trades']} for statistical validity",
    )

    # 6. Win rate floor
    win_rate = trades.get("win_rate", 0.0)
    _check(
        "min_win_rate", win_rate, th["min_win_rate"],
        lambda a, t: a >= t,
        f"Win rate {win_rate:.1f}% < {th['min_win_rate']:.1f}% minimum",
    )

    # 7. Avg loss / avg win ratio (reject if losses dwarf wins)
    avg_win = trades.get("avg_win", 0.0)
    avg_loss = abs(trades.get("avg_loss", 0.0))
    if avg_win > 0:
        loss_ratio = avg_loss / avg_win
        _check(
            "max_avg_loss_ratio", loss_ratio, th["max_avg_loss_ratio"],
            lambda a, t: a <= t,
            f"|Avg loss|/Avg win = {loss_ratio:.2f} > {th['max_avg_loss_ratio']:.1f} — tail risk too high",
        )

    passed = len(failures) == 0
    return {
        "passed": passed,
        "failures": failures,
        "checks_run": checks_run,
        "checks_passed": checks_run - len(failures),
    }


def print_validation(validation: dict) -> None:
    """Pretty-print strategy validation results."""
    status = "PASS" if validation["passed"] else "FAIL"
    n_run = validation["checks_run"]
    n_passed = validation["checks_passed"]

    print("\n" + "=" * 62)
    print("  STRATEGY VALIDATION")
    print("=" * 62)
    print(f"  Status: {status}  ({n_passed}/{n_run} checks passed)")

    if validation["failures"]:
        print("\n  " + "-" * 58)
        print("  FAILURES:")
        for f in validation["failures"]:
            print(f"    [{f['rule']}] {f['message']}")
            print(f"      threshold={f['threshold']}  actual={f['actual']}")
    else:
        print("\n  All quality checks passed.")

    print("=" * 62 + "\n")


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

    sig_flag = "YES" if mtm.get("significance") else "NO"
    print(f"\n  {'t-statistic':30s} {mtm['t_statistic']:>10.4f}")
    print(f"  {'p-value':30s} {mtm['p_value']:>10.6f}")
    print(f"  {'Standard Error':30s} {mtm['standard_error']:>10.8f}")
    print(f"  {'Significant (p<0.05)':30s} {sig_flag:>10s}")

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

    if "consistency" in metrics:
        c = metrics["consistency"]
        print("\n  " + "-" * 58)
        print("  EQUITY CURVE CONSISTENCY")
        print("  " + "-" * 58)
        print(f"  {'Equity Volatility (ann.)':30s} {c['equity_volatility']:>10.4f}")
        print(f"  {'Max Consecutive Losses':30s} {c['max_consecutive_losses']:>10d}")
        print(f"  {'Rolling DD 5-day %':30s} {c['rolling_dd_5d']:>10.2f}")
        print(f"  {'Rolling DD 10-day %':30s} {c['rolling_dd_10d']:>10.2f}")
        print(f"  {'Return Consistency %':30s} {c['return_consistency']:>10.2f}")
        print(f"  {'Consistency Score':30s} {c['consistency_score']:>10.4f}")
        print(f"  {'Prop Firm Score':30s} {c['prop_firm_score']:>10.4f}")

    if "excursion" in metrics and metrics["excursion"]["n_trades"] > 0:
        e = metrics["excursion"]
        print("\n  " + "-" * 58)
        print("  MAE / MFE EXCURSION ANALYSIS")
        print("  " + "-" * 58)
        print(f"  {'':30s} {'  Avg':>8s} {'  Med':>8s} {' 95th':>8s}")
        print(f"  {'MAE (pts)':30s} {e['avg_mae']:>8.2f} {e['median_mae']:>8.2f} {e['p95_mae']:>8.2f}")
        print(f"  {'MFE (pts)':30s} {e['avg_mfe']:>8.2f} {e['median_mfe']:>8.2f} {e['p95_mfe']:>8.2f}")
        print(f"\n  {'MAE/MFE Ratio (avg)':30s} {e['avg_mae_mfe_ratio']:>10.4f}")
        print(f"  {'MAE/MFE Ratio (median)':30s} {e['median_mae_mfe_ratio']:>10.4f}")
        print(f"  {'Avg Stop Distance (pts)':30s} {e['avg_stop_distance']:>10.2f}")
        print(f"  {'MAE vs Stop %':30s} {e['mae_vs_stop_pct']:>10.2f}")
        print(f"  {'Stop Efficiency %':30s} {e['stop_efficiency']:>10.2f}")
        print(f"  {'Edge Ratio (MFE/MAE)':30s} {e['edge_ratio']:>10.4f}")

    if "loss_distribution" in metrics and metrics["loss_distribution"]["total_losses"] > 0:
        ld = metrics["loss_distribution"]
        print("\n  " + "-" * 58)
        print("  LOSS DISTRIBUTION & RISK CONTROL")
        print("  " + "-" * 58)
        print(f"  {'Total Losing Trades':30s} {ld['total_losses']:>10d}")
        print(f"  {'Avg Loss ($)':30s} {ld['avg_loss']:>10.2f}")
        print(f"  {'Median Loss ($)':30s} {ld['median_loss']:>10.2f}")
        print(f"  {'Worst Loss ($)':30s} {ld['worst_loss']:>10.2f}")
        print(f"  {'5th Percentile Loss ($)':30s} {ld['p5_loss']:>10.2f}")
        print(f"  {'1st Percentile Loss ($)':30s} {ld['p1_loss']:>10.2f}")
        print(f"  {'Loss Std Dev ($)':30s} {ld['loss_std']:>10.2f}")
        print(f"  {'Loss Skewness':30s} {ld['loss_skew']:>10.4f}")
        print(f"  {'Tail Concentration (5%) %':30s} {ld['tail_concentration_pct']:>10.2f}")
        if ld['max_loss_cap_exits'] > 0:
            print(f"  {'Max Loss Cap Exits':30s} {ld['max_loss_cap_exits']:>10d}")
        if ld['time_exits'] > 0:
            print(f"  {'Time-Based Exits':30s} {ld['time_exits']:>10d}")
        if ld['exit_reason_breakdown']:
            print("\n  Exit Reason Breakdown:")
            for reason, count in sorted(ld['exit_reason_breakdown'].items(), key=lambda x: -x[1]):
                pct = count / sum(ld['exit_reason_breakdown'].values()) * 100
                print(f"    {reason:26s} {count:>5d}  ({pct:.1f}%)")

    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

def run_monte_carlo(
    result: BacktestResult,
    n_simulations: int = 500,
    initial_capital: float = 100_000.0,
    seed: int | None = None,
) -> dict:
    """Monte Carlo robustness test via trade-order shuffling.

    Randomly permutes the sequence of closed trades and rebuilds the equity
    curve for each simulation.  This tests whether performance depends on
    the specific ordering of trades (and therefore market regime sequencing)
    or is robust to reordering.

    Parameters
    ----------
    result : BacktestResult
        Output from ``backtester.run()``.
    n_simulations : int
        Number of random permutations (default 500).
    initial_capital : float
        Starting equity used to build simulated curves.
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    dict
        final_returns : np.ndarray   — simulated final return % (n_simulations,)
        max_drawdowns : np.ndarray   — simulated max DD % (n_simulations,)
        percentiles   : dict         — 5th/50th/95th for return and max DD
        actual_return : float        — original strategy return %
        actual_max_dd : float        — original strategy max DD %
        n_trades      : int
        n_simulations : int
    """
    trades = result.trades
    if len(trades) < 2:
        empty = np.array([0.0])
        return {
            "final_returns": empty,
            "max_drawdowns": empty,
            "percentiles": {
                "return_p5": 0.0, "return_p50": 0.0, "return_p95": 0.0,
                "max_dd_p5": 0.0, "max_dd_p50": 0.0, "max_dd_p95": 0.0,
            },
            "actual_return": 0.0,
            "actual_max_dd": 0.0,
            "n_trades": len(trades),
            "n_simulations": 0,
        }

    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl for t in trades])
    n_trades = len(pnls)

    # Build shuffled PnL matrix: (n_simulations, n_trades)
    indices = np.tile(np.arange(n_trades), (n_simulations, 1))
    for row in indices:
        rng.shuffle(row)
    shuffled_pnls = pnls[indices]

    # Cumulative equity curves: (n_simulations, n_trades + 1)
    equity = np.empty((n_simulations, n_trades + 1), dtype=np.float64)
    equity[:, 0] = initial_capital
    np.cumsum(shuffled_pnls, axis=1, out=equity[:, 1:])
    equity[:, 1:] += initial_capital

    # Final returns (%)
    final_returns = (equity[:, -1] - initial_capital) / initial_capital * 100

    # Max drawdowns (%) — vectorised running max along trade axis
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdowns = (equity - running_max) / (running_max + _EPS) * 100
    max_drawdowns = drawdowns.min(axis=1)

    # Actual strategy values
    actual_final = result.equity_mtm.iloc[-1]
    actual_return = (actual_final - initial_capital) / (initial_capital + _EPS) * 100
    running_peak = result.equity_mtm.cummax()
    actual_max_dd = ((result.equity_mtm - running_peak) / (running_peak + _EPS) * 100).min()

    pcts_ret = np.percentile(final_returns, [5, 50, 95])
    pcts_dd = np.percentile(max_drawdowns, [5, 50, 95])

    return {
        "final_returns": final_returns,
        "max_drawdowns": max_drawdowns,
        "percentiles": {
            "return_p5": round(float(pcts_ret[0]), 2),
            "return_p50": round(float(pcts_ret[1]), 2),
            "return_p95": round(float(pcts_ret[2]), 2),
            "max_dd_p5": round(float(pcts_dd[0]), 2),
            "max_dd_p50": round(float(pcts_dd[1]), 2),
            "max_dd_p95": round(float(pcts_dd[2]), 2),
        },
        "actual_return": round(float(actual_return), 2),
        "actual_max_dd": round(float(actual_max_dd), 2),
        "n_trades": n_trades,
        "n_simulations": n_simulations,
    }


def print_monte_carlo(mc: dict) -> None:
    """Pretty-print Monte Carlo results."""
    p = mc["percentiles"]
    print("\n" + "=" * 62)
    print("  MONTE CARLO SIMULATION")
    print("=" * 62)
    print(f"  Simulations: {mc['n_simulations']}  |  Trades shuffled: {mc['n_trades']}")
    print("\n  " + "-" * 58)
    print(f"  {'':30s} {'  5th':>8s} {' 50th':>8s} {' 95th':>8s}")
    print(f"  {'Final Return %':30s} {p['return_p5']:>8.2f} {p['return_p50']:>8.2f} {p['return_p95']:>8.2f}")
    print(f"  {'Max Drawdown %':30s} {p['max_dd_p5']:>8.2f} {p['max_dd_p50']:>8.2f} {p['max_dd_p95']:>8.2f}")
    print("\n  " + "-" * 58)
    print(f"  {'Actual Return %':30s} {mc['actual_return']:>8.2f}")
    print(f"  {'Actual Max DD %':30s} {mc['actual_max_dd']:>8.2f}")

    # Rank actual vs simulations
    if mc["n_simulations"] > 0:
        ret_rank = (mc["final_returns"] <= mc["actual_return"]).mean() * 100
        dd_rank = (mc["max_drawdowns"] <= mc["actual_max_dd"]).mean() * 100
        print(f"\n  {'Actual return percentile':30s} {ret_rank:>7.1f}th")
        print(f"  {'Actual max DD percentile':30s} {dd_rank:>7.1f}th")
    print("=" * 62 + "\n")

