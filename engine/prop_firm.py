"""
Prop firm challenge simulation.

Evaluates whether a strategy can pass prop firm funding challenges
by stepping through trades against configurable rules, with Monte Carlo
shuffling to estimate pass rates across random trade orderings.
"""

import math
import numpy as np
import pandas as pd
from engine.backtester import BacktestResult

_EPS = 1e-9

# Default prop firm rules (modelled on typical FTMO-style challenge)
DEFAULT_CONFIG = {
    "starting_balance": 100_000.0,
    "profit_target": 0.10,        # 10% profit target
    "daily_drawdown_limit": 0.05, # 5% max daily loss
    "max_drawdown_limit": 0.10,   # 10% max trailing drawdown
    "max_days": 30,               # calendar days to hit target
}


def _simulate_single(
    pnls: np.ndarray,
    trade_dates: np.ndarray,
    config: dict,
) -> dict:
    """Step through one sequence of trades against prop firm rules.

    Returns dict with outcome, final equity, peak DD, and days elapsed.
    """
    balance = config["starting_balance"]
    equity_peak = balance
    profit_target_usd = balance * config["profit_target"]
    max_dd_frac = config["max_drawdown_limit"]
    daily_dd_frac = config["daily_drawdown_limit"]
    max_days = config["max_days"]

    day_start_equity = balance
    current_date = trade_dates[0] if len(trade_dates) > 0 else None
    start_date = current_date
    peak_dd_pct = 0.0

    for i in range(len(pnls)):
        t_date = trade_dates[i]

        # New trading day: reset daily tracking
        if t_date != current_date:
            current_date = t_date
            day_start_equity = balance

        # Apply trade PnL
        balance += pnls[i]

        # Track equity peak (trailing drawdown model)
        if balance > equity_peak:
            equity_peak = balance

        # Current drawdown from peak
        dd_from_peak = equity_peak - balance
        dd_pct = dd_from_peak / (equity_peak + _EPS) * 100
        if dd_pct > peak_dd_pct:
            peak_dd_pct = dd_pct

        # Daily drawdown check
        daily_loss = day_start_equity - balance
        if daily_dd_frac > 0 and day_start_equity > 0:
            if daily_loss / day_start_equity >= daily_dd_frac:
                return {
                    "outcome": "FAIL",
                    "reason": "daily_dd",
                    "final_equity": balance,
                    "peak_dd_pct": peak_dd_pct,
                    "days_elapsed": _days_between(start_date, t_date),
                    "trades_taken": i + 1,
                }

        # Max trailing drawdown check (from high-water mark)
        if max_dd_frac > 0 and dd_from_peak >= equity_peak * max_dd_frac:
            return {
                "outcome": "FAIL",
                "reason": "max_dd",
                "final_equity": balance,
                "peak_dd_pct": peak_dd_pct,
                "days_elapsed": _days_between(start_date, t_date),
                "trades_taken": i + 1,
            }

        # Profit target check
        profit = balance - config["starting_balance"]
        if profit >= profit_target_usd:
            return {
                "outcome": "PASS",
                "reason": "target_hit",
                "final_equity": balance,
                "peak_dd_pct": peak_dd_pct,
                "days_elapsed": _days_between(start_date, t_date),
                "trades_taken": i + 1,
            }

        # Time limit check
        if max_days > 0 and _days_between(start_date, t_date) >= max_days:
            return {
                "outcome": "FAIL",
                "reason": "time_exceeded",
                "final_equity": balance,
                "peak_dd_pct": peak_dd_pct,
                "days_elapsed": max_days,
                "trades_taken": i + 1,
            }

    # Ran out of trades without pass or fail
    profit = balance - config["starting_balance"]
    days = _days_between(start_date, trade_dates[-1]) if len(trade_dates) > 0 else 0
    return {
        "outcome": "PASS" if profit >= profit_target_usd else "FAIL",
        "reason": "target_hit" if profit >= profit_target_usd else "trades_exhausted",
        "final_equity": balance,
        "peak_dd_pct": peak_dd_pct,
        "days_elapsed": days,
        "trades_taken": len(pnls),
    }


def _days_between(d1, d2) -> int:
    """Calendar days between two date-like objects."""
    if d1 is None or d2 is None:
        return 0
    delta = pd.Timestamp(d2) - pd.Timestamp(d1)
    return max(0, delta.days)


def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (lower, upper) as percentages (0-100).
    """
    if n_total == 0:
        return (0.0, 0.0)
    p = n_success / n_total
    denom = 1 + z * z / n_total
    centre = p + z * z / (2 * n_total)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n_total)) / n_total)
    lo = max(0.0, (centre - spread) / denom)
    hi = min(1.0, (centre + spread) / denom)
    return (round(lo * 100, 2), round(hi * 100, 2))


def simulate_prop_firm(
    result: BacktestResult,
    config: dict | None = None,
    n_simulations: int = 5000,
    seed: int | None = None,
) -> dict:
    """Monte Carlo prop firm challenge simulation.

    Shuffles the trade order and replays against prop firm rules for
    each simulation. Trades keep their original PnL and calendar date
    distribution (dates are sampled with replacement from the original
    trade dates to preserve realistic day clustering).

    Parameters
    ----------
    result : BacktestResult
        Output from ``backtester.run()``.
    config : dict | None
        Prop firm rules. Uses DEFAULT_CONFIG if None.
    n_simulations : int
        Number of shuffled simulations (default 5000).
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    dict
        pass_rate, fail_rate, pass_rate_ci, avg_days_to_pass,
        classification, fail_reasons, equity_distribution,
        drawdown_distribution, actual, n_simulations, config.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    trades = result.trades

    if len(trades) < 2:
        return _empty_result(cfg)

    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl for t in trades])
    exit_dates = np.array([t.exit_time.date() for t in trades])

    # Assign shuffled trades to dates: spread them across the original
    # date range preserving the original number of trades per day.
    unique_dates = np.unique(exit_dates)
    trades_per_day = np.array([np.sum(exit_dates == d) for d in unique_dates])
    # Cumulative index boundaries for assigning shuffled trades to days
    day_boundaries = np.concatenate([[0], np.cumsum(trades_per_day)])

    # Precompute date mapping for shuffled trades (same for all sims)
    shuffled_dates_template = np.empty(len(pnls), dtype=exit_dates.dtype)
    for d_idx, d in enumerate(unique_dates):
        start, end = day_boundaries[d_idx], day_boundaries[d_idx + 1]
        shuffled_dates_template[start:end] = d

    # Run actual (unshuffled) first
    actual = _simulate_single(pnls, exit_dates, cfg)

    # Monte Carlo simulations
    sims = []
    for _ in range(n_simulations):
        shuffled_idx = rng.permutation(len(pnls))
        shuffled_pnls = pnls[shuffled_idx]
        sims.append(_simulate_single(shuffled_pnls, shuffled_dates_template, cfg))

    # Aggregate results
    outcomes = [s["outcome"] for s in sims]
    n_pass = sum(1 for o in outcomes if o == "PASS")
    pass_rate = n_pass / n_simulations * 100

    # Pass rate confidence interval (95% Wilson score)
    pass_rate_ci = _wilson_ci(n_pass, n_simulations, z=1.96)

    pass_days = [s["days_elapsed"] for s in sims if s["outcome"] == "PASS"]
    avg_days_to_pass = float(np.mean(pass_days)) if pass_days else 0.0

    # Percentile breakdown: drawdown distribution
    pctiles = [5, 25, 50, 75, 95]
    peak_dds = np.array([s["peak_dd_pct"] for s in sims])
    dd_pct_vals = np.percentile(peak_dds, pctiles)

    # Percentile breakdown: final equity distribution
    final_equities = np.array([s["final_equity"] for s in sims])
    eq_pct_vals = np.percentile(final_equities, pctiles)

    fail_reasons = {}
    for s in sims:
        if s["outcome"] == "FAIL":
            r = s["reason"]
            fail_reasons[r] = fail_reasons.get(r, 0) + 1

    # Classification
    if pass_rate >= 70:
        classification = "ROBUST"
    elif pass_rate >= 40:
        classification = "MARGINAL"
    else:
        classification = "UNVIABLE"

    return {
        "pass_rate": round(pass_rate, 2),
        "fail_rate": round(100 - pass_rate, 2),
        "pass_rate_ci": pass_rate_ci,
        "avg_days_to_pass": round(avg_days_to_pass, 1),
        "classification": classification,
        "fail_reasons": fail_reasons,
        "equity_distribution": {
            f"eq_p{p}": round(float(v), 2) for p, v in zip(pctiles, eq_pct_vals)
        },
        "drawdown_distribution": {
            f"dd_p{p}": round(float(v), 2) for p, v in zip(pctiles, dd_pct_vals)
        },
        "actual": actual,
        "n_simulations": n_simulations,
        "config": cfg,
    }


def _empty_result(cfg: dict) -> dict:
    """Return safe empty result for insufficient trades."""
    return {
        "pass_rate": 0.0,
        "fail_rate": 100.0,
        "pass_rate_ci": (0.0, 0.0),
        "avg_days_to_pass": 0.0,
        "classification": "UNVIABLE",
        "fail_reasons": {"insufficient_trades": 1},
        "equity_distribution": {
            "eq_p5": 0.0, "eq_p25": 0.0, "eq_p50": 0.0,
            "eq_p75": 0.0, "eq_p95": 0.0,
        },
        "drawdown_distribution": {
            "dd_p5": 0.0, "dd_p25": 0.0, "dd_p50": 0.0,
            "dd_p75": 0.0, "dd_p95": 0.0,
        },
        "actual": {"outcome": "FAIL", "reason": "insufficient_trades",
                   "final_equity": cfg["starting_balance"], "peak_dd_pct": 0.0,
                   "days_elapsed": 0, "trades_taken": 0},
        "n_simulations": 0,
        "config": cfg,
    }


def print_prop_firm(pf: dict) -> None:
    """Pretty-print prop firm simulation results."""
    cfg = pf["config"]
    dd = pf["drawdown_distribution"]
    eq = pf["equity_distribution"]
    actual = pf["actual"]
    ci = pf["pass_rate_ci"]

    print("\n" + "=" * 62)
    print("  PROP FIRM CHALLENGE SIMULATION")
    print("=" * 62)

    print(f"  Balance: ${cfg['starting_balance']:,.0f}  |  "
          f"Target: {cfg['profit_target']*100:.0f}%  |  "
          f"Max DD: {cfg['max_drawdown_limit']*100:.0f}%  |  "
          f"Daily DD: {cfg['daily_drawdown_limit']*100:.0f}%  |  "
          f"Days: {cfg['max_days']}")

    print("\n  " + "-" * 58)
    print(f"  {'Simulations':30s} {pf['n_simulations']:>10,d}")
    print(f"  {'Pass Rate':30s} {pf['pass_rate']:>9.1f}%")
    print(f"  {'95% CI':30s}  [{ci[0]:.1f}% - {ci[1]:.1f}%]")
    print(f"  {'Fail Rate':30s} {pf['fail_rate']:>9.1f}%")
    print(f"  {'Avg Days to Pass':30s} {pf['avg_days_to_pass']:>10.1f}")

    if pf["fail_reasons"]:
        print("\n  Failure breakdown:")
        for reason, count in sorted(pf["fail_reasons"].items(), key=lambda x: -x[1]):
            pct = count / pf["n_simulations"] * 100
            print(f"    {reason:26s} {count:>5,d}  ({pct:.1f}%)")

    print("\n  " + "-" * 58)
    header = f"  {'':30s} {'  5th':>8s} {' 25th':>8s} {' 50th':>8s} {' 75th':>8s} {' 95th':>8s}"
    print(header)
    print(f"  {'Peak Drawdown %':30s} "
          f"{dd['dd_p5']:>8.2f} {dd['dd_p25']:>8.2f} {dd['dd_p50']:>8.2f} "
          f"{dd['dd_p75']:>8.2f} {dd['dd_p95']:>8.2f}")
    print(f"  {'Final Equity ($)':30s} "
          f"{eq['eq_p5']:>8,.0f} {eq['eq_p25']:>8,.0f} {eq['eq_p50']:>8,.0f} "
          f"{eq['eq_p75']:>8,.0f} {eq['eq_p95']:>8,.0f}")

    print("\n  " + "-" * 58)
    a_status = actual["outcome"]
    a_reason = actual["reason"]
    print(f"  {'Actual Outcome':30s} {a_status:>10s}")
    print(f"  {'Actual Reason':30s} {a_reason:>10s}")
    print(f"  {'Actual Peak DD %':30s} {actual['peak_dd_pct']:>10.2f}")
    print(f"  {'Actual Days':30s} {actual['days_elapsed']:>10d}")
    print(f"  {'Actual Trades':30s} {actual['trades_taken']:>10d}")

    print("\n  " + "-" * 58)
    label = pf["classification"]
    print(f"  CLASSIFICATION: {label}")
    if label == "ROBUST":
        print("    Strategy consistently passes challenge under trade reordering.")
    elif label == "MARGINAL":
        print("    Strategy passes some orderings but is sensitive to sequencing.")
    else:
        print("    Strategy unlikely to pass challenge. Needs improvement.")

    print("=" * 62 + "\n")
