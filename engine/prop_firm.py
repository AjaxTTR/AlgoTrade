"""
Prop firm challenge simulation — MFFU (My Funded Futures) mechanics only.

Models the MFFU Phase 1 evaluation:
  - Absolute dollar profit target (e.g. $6,000 on the $100k account).
  - End-of-day trailing drawdown. The drawdown floor moves upward only on
    end-of-day *closed* balance; it is frozen during the trading day.
    Intraday breaches of the frozen floor still fail the account — MFFU
    just does not *raise* the floor intraday.
  - No daily loss limit.
  - 50% consistency rule: the single best day's profit may not exceed 50%
    of total profit at the moment the target is hit. Modelled as a
    pass-time filter: a run that hits the target but fails consistency
    is reclassified FAIL (reason = "consistency_rule").
  - No time limit on Phase 1.

Monte Carlo shuffles trade order to estimate pass rate across different
sequencings. See priority-list file for a known limitation in how
shuffled trades are re-dated.

Phase 2 (the funded-account payout structure) is NOT modelled here.
Phase 2 is an economic/payout question, not a pass/fail question, and
will live in its own function when we need it.
"""

import math
import numpy as np
import pandas as pd
from engine.backtester import BacktestResult

_EPS = 1e-9


# Verified MFFU Phase 1 ruleset. Only the $100k account is currently
# populated — the $50k and $150k numbers were not explicitly confirmed
# and should not be inferred. Add them when confirmed against MFFU docs.
MFFU_PHASE1_100K = {
    "starting_balance": 100_000.0,
    "profit_target_usd": 6_000.0,
    "trailing_dd_usd": 3_000.0,
    "consistency_rule_max_day_frac": 0.5,
    "max_days": 0,  # no time limit on Phase 1
}


def mffu_phase1_config(account_size: int) -> dict:
    """Return the locked MFFU Phase 1 rule dict for a given account size.

    Only the $100k account has verified numbers. Other sizes raise until
    their exact rules are confirmed — we refuse to guess, because a
    wrong profit target or DD value invalidates every sim run under it.
    """
    if account_size == 100_000:
        return dict(MFFU_PHASE1_100K)
    raise NotImplementedError(
        f"MFFU Phase 1 config for ${account_size:,} is not yet populated. "
        f"Confirm the exact profit target and trailing DD from MFFU docs "
        f"and add to MFFU_PHASE1_CONFIGS before using."
    )


def _simulate_single(
    pnls: np.ndarray,
    trade_dates: np.ndarray,
    config: dict,
) -> dict:
    """Step through one sequence of trades against MFFU Phase 1 rules.

    EOD trailing drawdown: floor is initialised at
    (starting_balance - trailing_dd_usd) and only updates at end-of-day on
    the closed balance. Intraday (within the day's trade sequence) breaches
    of the frozen floor fail the account.

    Returns dict with outcome, reason, final equity, peak DD in $, days
    elapsed, trades taken, and per-day PnL for consistency checking.
    """
    balance = config["starting_balance"]
    trailing_dd = config["trailing_dd_usd"]
    profit_target = config["profit_target_usd"]
    max_day_frac = config.get("consistency_rule_max_day_frac", None)
    max_days = config.get("max_days", 0)

    floor = balance - trailing_dd
    day_pnls: dict = {}  # date -> running pnl for that day
    peak_dd_usd = 0.0

    current_date = trade_dates[0] if len(trade_dates) > 0 else None
    start_date = current_date
    day_start_balance = balance

    n = len(pnls)
    for i in range(n):
        t_date = trade_dates[i]

        # Day rollover: lock in prior day's EOD balance and bump the floor
        if t_date != current_date:
            eod_balance = balance
            floor = max(floor, eod_balance - trailing_dd)
            current_date = t_date
            day_start_balance = balance

        balance += pnls[i]
        day_pnls[t_date] = day_pnls.get(t_date, 0.0) + pnls[i]

        under = (floor + trailing_dd) - balance
        if under > peak_dd_usd:
            peak_dd_usd = under

        # Intraday breach of frozen floor
        if balance <= floor:
            return {
                "outcome": "FAIL",
                "reason": "trailing_dd",
                "final_equity": balance,
                "peak_dd_usd": peak_dd_usd,
                "days_elapsed": _days_between(start_date, t_date),
                "trades_taken": i + 1,
                "day_pnls": day_pnls,
            }

        # Profit target
        profit = balance - config["starting_balance"]
        if profit >= profit_target:
            # Consistency check at pass time
            if max_day_frac is not None and profit > 0:
                max_day = max(day_pnls.values())
                if max_day / profit > max_day_frac:
                    return {
                        "outcome": "FAIL",
                        "reason": "consistency_rule",
                        "final_equity": balance,
                        "peak_dd_usd": peak_dd_usd,
                        "days_elapsed": _days_between(start_date, t_date),
                        "trades_taken": i + 1,
                        "day_pnls": day_pnls,
                        "max_day_frac_observed": max_day / profit,
                    }
            return {
                "outcome": "PASS",
                "reason": "target_hit",
                "final_equity": balance,
                "peak_dd_usd": peak_dd_usd,
                "days_elapsed": _days_between(start_date, t_date),
                "trades_taken": i + 1,
                "day_pnls": day_pnls,
            }

        # Optional time limit
        if max_days > 0 and _days_between(start_date, t_date) >= max_days:
            return {
                "outcome": "FAIL",
                "reason": "time_exceeded",
                "final_equity": balance,
                "peak_dd_usd": peak_dd_usd,
                "days_elapsed": max_days,
                "trades_taken": i + 1,
                "day_pnls": day_pnls,
            }

    # Ran out of trades without pass or fail
    profit = balance - config["starting_balance"]
    days = _days_between(start_date, trade_dates[-1]) if n > 0 else 0
    passed = profit >= profit_target
    if passed and max_day_frac is not None and profit > 0:
        max_day = max(day_pnls.values())
        if max_day / profit > max_day_frac:
            return {
                "outcome": "FAIL",
                "reason": "consistency_rule",
                "final_equity": balance,
                "peak_dd_usd": peak_dd_usd,
                "days_elapsed": days,
                "trades_taken": n,
                "day_pnls": day_pnls,
                "max_day_frac_observed": max_day / profit,
            }
    return {
        "outcome": "PASS" if passed else "FAIL",
        "reason": "target_hit" if passed else "trades_exhausted",
        "final_equity": balance,
        "peak_dd_usd": peak_dd_usd,
        "days_elapsed": days,
        "trades_taken": n,
        "day_pnls": day_pnls,
    }


def _days_between(d1, d2) -> int:
    if d1 is None or d2 is None:
        return 0
    delta = pd.Timestamp(d2) - pd.Timestamp(d1)
    return max(0, delta.days)


def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI for a binomial proportion. Returns (lo, hi) as percentages."""
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
    """Monte Carlo MFFU Phase 1 challenge simulation.

    Config defaults to MFFU_PHASE1_100K. Shuffles trade order and replays
    against MFFU rules. Trades keep their original PnL distribution; the
    date assignment for shuffled trades preserves the original number of
    trades per day but NOT which specific PnLs clustered together — this
    is a known limitation for EOD-trailing simulation and is tracked in
    the priority list.
    """
    cfg = {**MFFU_PHASE1_100K, **(config or {})}
    trades = result.trades

    if len(trades) < 2:
        return _empty_result(cfg)

    rng = np.random.default_rng(seed)
    pnls = np.array([t.pnl for t in trades])
    exit_dates = np.array([t.exit_time.date() for t in trades])

    unique_dates = np.unique(exit_dates)
    trades_per_day = np.array([np.sum(exit_dates == d) for d in unique_dates])
    day_boundaries = np.concatenate([[0], np.cumsum(trades_per_day)])

    shuffled_dates_template = np.empty(len(pnls), dtype=exit_dates.dtype)
    for d_idx, d in enumerate(unique_dates):
        start, end = day_boundaries[d_idx], day_boundaries[d_idx + 1]
        shuffled_dates_template[start:end] = d

    actual = _simulate_single(pnls, exit_dates, cfg)

    sims = []
    for _ in range(n_simulations):
        shuffled_idx = rng.permutation(len(pnls))
        shuffled_pnls = pnls[shuffled_idx]
        sims.append(_simulate_single(shuffled_pnls, shuffled_dates_template, cfg))

    outcomes = [s["outcome"] for s in sims]
    n_pass = sum(1 for o in outcomes if o == "PASS")
    pass_rate = n_pass / n_simulations * 100
    pass_rate_ci = _wilson_ci(n_pass, n_simulations)

    pass_days = [s["days_elapsed"] for s in sims if s["outcome"] == "PASS"]
    avg_days_to_pass = float(np.mean(pass_days)) if pass_days else 0.0

    if pass_days:
        ttp_pctiles = [10, 25, 50, 75, 90]
        ttp_vals = np.percentile(pass_days, ttp_pctiles)
        days_distribution = {
            f"ttp_p{p}": round(float(v), 1) for p, v in zip(ttp_pctiles, ttp_vals)
        }
        days_distribution["min"] = round(float(min(pass_days)), 1)
        days_distribution["max"] = round(float(max(pass_days)), 1)
    else:
        days_distribution = {}

    pctiles = [5, 25, 50, 75, 95]
    peak_dds = np.array([s["peak_dd_usd"] for s in sims])
    dd_pct_vals = np.percentile(peak_dds, pctiles)

    final_equities = np.array([s["final_equity"] for s in sims])
    eq_pct_vals = np.percentile(final_equities, pctiles)

    fail_reasons: dict = {}
    for s in sims:
        if s["outcome"] == "FAIL":
            r = s["reason"]
            fail_reasons[r] = fail_reasons.get(r, 0) + 1

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
        "drawdown_distribution_usd": {
            f"dd_p{p}": round(float(v), 2) for p, v in zip(pctiles, dd_pct_vals)
        },
        "days_distribution": days_distribution,
        "actual": actual,
        "n_simulations": n_simulations,
        "config": cfg,
    }


def _empty_result(cfg: dict) -> dict:
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
        "drawdown_distribution_usd": {
            "dd_p5": 0.0, "dd_p25": 0.0, "dd_p50": 0.0,
            "dd_p75": 0.0, "dd_p95": 0.0,
        },
        "days_distribution": {},
        "actual": {
            "outcome": "FAIL", "reason": "insufficient_trades",
            "final_equity": cfg["starting_balance"], "peak_dd_usd": 0.0,
            "days_elapsed": 0, "trades_taken": 0, "day_pnls": {},
        },
        "n_simulations": 0,
        "config": cfg,
    }


def print_prop_firm(pf: dict) -> None:
    """Pretty-print MFFU Phase 1 simulation results."""
    cfg = pf["config"]
    dd = pf["drawdown_distribution_usd"]
    eq = pf["equity_distribution"]
    actual = pf["actual"]
    ci = pf["pass_rate_ci"]

    max_days = cfg.get("max_days", 0)
    days_label = f"{max_days}" if max_days > 0 else "no limit"
    cons = cfg.get("consistency_rule_max_day_frac")
    cons_label = f"{cons*100:.0f}%" if cons is not None else "off"

    print("\n" + "=" * 62)
    print("  MFFU PHASE 1 CHALLENGE SIMULATION")
    print("=" * 62)

    print(f"  Balance: ${cfg['starting_balance']:,.0f}  |  "
          f"Target: ${cfg['profit_target_usd']:,.0f}  |  "
          f"Trailing DD: ${cfg['trailing_dd_usd']:,.0f} (EOD)  |  "
          f"Consistency: {cons_label}  |  "
          f"Days: {days_label}")

    print("\n  " + "-" * 58)
    print(f"  {'Simulations':30s} {pf['n_simulations']:>10,d}")
    print(f"  {'Pass Rate':30s} {pf['pass_rate']:>9.1f}%")
    print(f"  {'95% CI':30s}  [{ci[0]:.1f}% - {ci[1]:.1f}%]")
    print(f"  {'Fail Rate':30s} {pf['fail_rate']:>9.1f}%")
    print(f"  {'Avg Days to Pass':30s} {pf['avg_days_to_pass']:>10.1f}")

    if pf.get("days_distribution"):
        ttp = pf["days_distribution"]
        print(f"  {'Median Days to Pass':30s} {ttp.get('ttp_p50', 0):>10.1f}")
        print(f"  {'Days to Pass (10/25/75/90p)':30s}  "
              f"{ttp.get('ttp_p10', 0):.0f} / {ttp.get('ttp_p25', 0):.0f} / "
              f"{ttp.get('ttp_p75', 0):.0f} / {ttp.get('ttp_p90', 0):.0f}")

    if pf["fail_reasons"]:
        print("\n  Failure breakdown:")
        for reason, count in sorted(pf["fail_reasons"].items(), key=lambda x: -x[1]):
            pct = count / pf["n_simulations"] * 100
            print(f"    {reason:26s} {count:>5,d}  ({pct:.1f}%)")

    print("\n  " + "-" * 58)
    header = f"  {'':30s} {'  5th':>8s} {' 25th':>8s} {' 50th':>8s} {' 75th':>8s} {' 95th':>8s}"
    print(header)
    print(f"  {'Peak Drawdown ($)':30s} "
          f"{dd['dd_p5']:>8,.0f} {dd['dd_p25']:>8,.0f} {dd['dd_p50']:>8,.0f} "
          f"{dd['dd_p75']:>8,.0f} {dd['dd_p95']:>8,.0f}")
    print(f"  {'Final Equity ($)':30s} "
          f"{eq['eq_p5']:>8,.0f} {eq['eq_p25']:>8,.0f} {eq['eq_p50']:>8,.0f} "
          f"{eq['eq_p75']:>8,.0f} {eq['eq_p95']:>8,.0f}")

    print("\n  " + "-" * 58)
    print(f"  {'Actual Outcome':30s} {actual['outcome']:>10s}")
    print(f"  {'Actual Reason':30s} {actual['reason']:>10s}")
    print(f"  {'Actual Peak DD ($)':30s} {actual['peak_dd_usd']:>10,.0f}")
    print(f"  {'Actual Days':30s} {actual['days_elapsed']:>10d}")
    print(f"  {'Actual Trades':30s} {actual['trades_taken']:>10d}")

    print("\n  " + "-" * 58)
    label = pf["classification"]
    print(f"  CLASSIFICATION: {label}")
    if label == "ROBUST":
        print("    Strategy consistently passes MFFU Phase 1 under reordering.")
    elif label == "MARGINAL":
        print("    Strategy passes some orderings but is sensitive to sequencing.")
    else:
        print("    Strategy unlikely to pass MFFU Phase 1. Needs improvement.")

    print("=" * 62 + "\n")
