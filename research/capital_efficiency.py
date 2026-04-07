"""
Capital efficiency simulation: EV per unit time across risk profiles.

Models prop firm deployment as a repeating cycle — each account runs a
challenge, and on pass or fail is immediately redeployed with a new
challenge fee.  Simulates a 1-year (365-day) horizon to compare total
expected profit, cycles completed, and capital growth rate.

Uses actual Monte Carlo per-attempt outcomes from real trade PnLs.

Usage:
    python -m research.capital_efficiency
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.prop_firm import _simulate_single
from run_portfolio import (
    STRATEGY_CATALOGUE,
    TOTAL_CAPITAL,
    _run_single_strategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_FILE = "data/nq_15m_data.csv"
OUTPUT_DIR = Path("output/portfolio")

# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------

CHALLENGE_FEE = 500.0
PROFIT_SPLIT = 0.80
ACCOUNT_SIZE = 100_000.0
PROFIT_TARGET_PCT = 0.10
FUNDED_PAYOUT = ACCOUNT_SIZE * PROFIT_TARGET_PCT * PROFIT_SPLIT  # $8,000

PROP_CONFIG = {
    "starting_balance": ACCOUNT_SIZE,
    "profit_target": PROFIT_TARGET_PCT,
    "daily_drawdown_limit": 0.05,
    "max_drawdown_limit": 0.10,
    "max_days": 0,
}

# Portfolio weights (locked)
WEIGHTS = {
    "gap_momentum": 0.60,
    "regime_filtered_momentum": 0.40,
}

# ---------------------------------------------------------------------------
# Risk profiles to compare
# ---------------------------------------------------------------------------

RISK_PROFILES = {
    "SAFE (0.8%)":  0.008,
    "BALANCED (1.0%)": 0.010,
    "AGGR (1.3%)":  0.013,
    "AGGR (1.4%)":  0.014,
}

# Hybrid mixes: (safe_fraction, aggr_risk_key)
HYBRID_MIXES = [
    (1.0, None,            "100% SAFE"),
    (0.7, "AGGR (1.3%)",   "70S/30A @1.3%"),
    (0.5, "AGGR (1.3%)",   "50S/50A @1.3%"),
    (0.3, "AGGR (1.3%)",   "30S/70A @1.3%"),
    (0.0, "AGGR (1.3%)",   "100% AGGR @1.3%"),
    (0.7, "AGGR (1.4%)",   "70S/30A @1.4%"),
    (0.5, "AGGR (1.4%)",   "50S/50A @1.4%"),
    (0.0, "AGGR (1.4%)",   "100% AGGR @1.4%"),
    (0.0, "BALANCED (1.0%)", "100% BALANCED @1.0%"),
]

HORIZON_DAYS = 365
N_ACCOUNTS = 10
N_SIMS = 2000
SEED = 42


# ---------------------------------------------------------------------------
# Trade data
# ---------------------------------------------------------------------------

def _get_trade_data(df: pd.DataFrame, risk: float) -> tuple[np.ndarray, np.ndarray]:
    """Run portfolio at given risk and return (pnls, exit_dates) arrays."""
    all_pnls = []
    all_dates = []

    for name, weight in WEIGHTS.items():
        capital = TOTAL_CAPITAL * weight
        cat = STRATEGY_CATALOGUE[name]
        result, _, _ = _run_single_strategy(name, cat, df, capital, risk)
        for t in result.trades:
            all_pnls.append(t.pnl)
            all_dates.append(t.exit_time.date())

    order = np.argsort(all_dates)
    return np.array(all_pnls)[order], np.array(all_dates)[order]


def _build_shuffle_template(exit_dates: np.ndarray) -> np.ndarray:
    unique_dates = np.unique(exit_dates)
    trades_per_day = np.array([np.sum(exit_dates == d) for d in unique_dates])
    day_boundaries = np.concatenate([[0], np.cumsum(trades_per_day)])

    template = np.empty(len(exit_dates), dtype=exit_dates.dtype)
    for d_idx, d in enumerate(unique_dates):
        start, end = day_boundaries[d_idx], day_boundaries[d_idx + 1]
        template[start:end] = d
    return template


# ---------------------------------------------------------------------------
# Pre-compute outcome pools
# ---------------------------------------------------------------------------

def _build_outcome_pool(
    pnls: np.ndarray,
    dates_template: np.ndarray,
    rng: np.random.Generator,
    pool_size: int = 10000,
) -> list[dict]:
    """Run pool_size shuffled simulations and return list of outcome dicts.

    Each dict has: outcome (PASS/FAIL), days_elapsed, pnl (net PnL of attempt).
    We pre-compute a large pool so the cycling sim can just draw from it.
    """
    pool = []
    for _ in range(pool_size):
        idx = rng.permutation(len(pnls))
        result = _simulate_single(pnls[idx], dates_template, PROP_CONFIG)
        net_pnl = result["final_equity"] - ACCOUNT_SIZE
        pool.append({
            "outcome": result["outcome"],
            "days": result["days_elapsed"],
            "net_pnl": net_pnl,
        })
    return pool


# ---------------------------------------------------------------------------
# 1-year cycling simulation
# ---------------------------------------------------------------------------

def _simulate_one_year_account(
    pool: list[dict],
    rng: np.random.Generator,
    horizon: int = HORIZON_DAYS,
) -> dict:
    """Simulate one account cycling challenges over a 1-year horizon.

    Each attempt: pay challenge fee, run until pass/fail, collect payout
    if passed, then immediately start a new challenge.

    Returns summary of cycles, passes, total profit, etc.
    """
    day = 0
    total_profit = 0.0
    total_fees = 0.0
    n_cycles = 0
    n_passes = 0
    n_fails = 0
    payouts = 0.0

    pool_size = len(pool)

    while day < horizon:
        # Pay challenge fee
        total_fees += CHALLENGE_FEE

        # Draw a random outcome from pre-computed pool
        attempt = pool[rng.integers(pool_size)]
        attempt_days = max(attempt["days"], 1)  # at least 1 day per attempt

        # Check if this attempt fits within the horizon
        if day + attempt_days > horizon:
            # Partial: pro-rate? No — we assume you can't finish, so it's a
            # wasted fee (worst case). This is conservative.
            n_cycles += 1
            n_fails += 1
            break

        day += attempt_days
        n_cycles += 1

        if attempt["outcome"] == "PASS":
            n_passes += 1
            payouts += FUNDED_PAYOUT
        else:
            n_fails += 1

    total_profit = payouts - total_fees

    return {
        "n_cycles": n_cycles,
        "n_passes": n_passes,
        "n_fails": n_fails,
        "total_fees": total_fees,
        "total_payouts": payouts,
        "total_profit": total_profit,
        "days_used": day,
    }


def _simulate_fleet_one_year(
    pools: list[list[dict]],
    n_accounts: int,
    account_pool_assignments: list[int],
    rng: np.random.Generator,
    n_sims: int,
) -> dict:
    """Simulate a fleet of accounts over 1 year, n_sims times.

    account_pool_assignments: list of pool indices, one per account.
    """
    all_profits = np.zeros(n_sims)
    all_fees = np.zeros(n_sims)
    all_payouts = np.zeros(n_sims)
    all_cycles = np.zeros(n_sims)
    all_passes = np.zeros(n_sims)
    all_fails = np.zeros(n_sims)

    for sim_i in range(n_sims):
        for acc_i in range(n_accounts):
            pool_idx = account_pool_assignments[acc_i]
            result = _simulate_one_year_account(pools[pool_idx], rng)
            all_profits[sim_i] += result["total_profit"]
            all_fees[sim_i] += result["total_fees"]
            all_payouts[sim_i] += result["total_payouts"]
            all_cycles[sim_i] += result["n_cycles"]
            all_passes[sim_i] += result["n_passes"]
            all_fails[sim_i] += result["n_fails"]

    total_invested = np.mean(all_fees)
    avg_profit = float(np.mean(all_profits))

    return {
        "avg_cycles": float(np.mean(all_cycles)),
        "avg_passes": float(np.mean(all_passes)),
        "avg_fails": float(np.mean(all_fails)),
        "avg_fees": float(np.mean(all_fees)),
        "avg_payouts": float(np.mean(all_payouts)),
        "avg_profit": avg_profit,
        "median_profit": float(np.median(all_profits)),
        "p10_profit": float(np.percentile(all_profits, 10)),
        "p90_profit": float(np.percentile(all_profits, 90)),
        "avg_roi_pct": float(avg_profit / total_invested * 100) if total_invested > 0 else 0,
        "profit_per_day": avg_profit / HORIZON_DAYS,
        "profit_per_account_per_day": avg_profit / HORIZON_DAYS / n_accounts,
        "cycles_per_account": float(np.mean(all_cycles)) / n_accounts,
    }


# ---------------------------------------------------------------------------
# Single-profile EV/day analysis
# ---------------------------------------------------------------------------

def _analyze_profile(pool: list[dict]) -> dict:
    """Compute per-attempt stats from an outcome pool."""
    passes = [p for p in pool if p["outcome"] == "PASS"]
    fails = [p for p in pool if p["outcome"] != "PASS"]

    pass_rate = len(passes) / len(pool) * 100
    ev_per_attempt = pass_rate / 100 * FUNDED_PAYOUT - CHALLENGE_FEE

    pass_days = [p["days"] for p in passes]
    fail_days = [p["days"] for p in fails]
    all_days = [p["days"] for p in pool]

    median_pass_days = float(np.median(pass_days)) if pass_days else 0
    median_fail_days = float(np.median(fail_days)) if fail_days else 0
    median_all_days = float(np.median(all_days))

    # Expected days per attempt (weighted by outcome)
    avg_days_per_attempt = float(np.mean(all_days))

    # EV per day = EV per attempt / avg days per attempt
    ev_per_day = ev_per_attempt / avg_days_per_attempt if avg_days_per_attempt > 0 else 0

    # Expected cycles per year
    cycles_per_year = HORIZON_DAYS / avg_days_per_attempt if avg_days_per_attempt > 0 else 0

    # Annual EV
    annual_ev = ev_per_attempt * cycles_per_year

    return {
        "pass_rate": pass_rate,
        "ev_per_attempt": ev_per_attempt,
        "median_pass_days": median_pass_days,
        "median_fail_days": median_fail_days,
        "avg_days_per_attempt": avg_days_per_attempt,
        "ev_per_day": ev_per_day,
        "cycles_per_year": cycles_per_year,
        "annual_ev": annual_ev,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()

    log.info("Loading data from %s", DATA_FILE)
    try:
        df = load_csv(DATA_FILE)
    except FileNotFoundError:
        log.error("%s not found.", DATA_FILE)
        sys.exit(1)
    log.info("Loaded %s bars", f"{len(df):,}")

    rng = np.random.default_rng(SEED)

    # Step 1: Build outcome pools for each risk profile
    log.info("Building outcome pools (10,000 sims each)...")
    profile_pools = {}
    profile_stats = {}

    for label, risk in RISK_PROFILES.items():
        log.info("  %s (risk=%.1f%%)...", label, risk * 100)
        pnls, dates = _get_trade_data(df, risk)
        template = _build_shuffle_template(dates)
        pool = _build_outcome_pool(pnls, template, rng, pool_size=10000)
        profile_pools[label] = pool
        profile_stats[label] = _analyze_profile(pool)
        log.info("    %d trades | pass=%.1f%% | EV/attempt=$%.0f | EV/day=$%.1f",
                 len(pnls),
                 profile_stats[label]["pass_rate"],
                 profile_stats[label]["ev_per_attempt"],
                 profile_stats[label]["ev_per_day"])

    # =========================================================================
    # PART 1: Per-profile EV/day analysis
    # =========================================================================

    print("\n" + "=" * 110)
    print("  CAPITAL EFFICIENCY ANALYSIS -- EV PER UNIT TIME")
    print("=" * 110)
    print(f"  Horizon: {HORIZON_DAYS} days  |  Challenge fee: ${CHALLENGE_FEE:,.0f}  |  "
          f"Payout: ${FUNDED_PAYOUT:,.0f}  |  Pool: 10,000 sims per profile")

    print(f"\n  PART 1: PER-PROFILE EFFICIENCY (single account)")
    print(f"  {'-' * 106}")
    print(f"  {'Profile':>22s}  {'Pass%':>6s}  {'EV/Att':>8s}  {'AvgDays':>7s}  "
          f"{'EV/Day':>8s}  {'Cyc/Yr':>7s}  {'Annual EV':>10s}  {'PassDays':>8s}  {'FailDays':>8s}")
    print(f"  {'-' * 106}")

    for label in RISK_PROFILES:
        s = profile_stats[label]
        print(f"  {label:>22s}  {s['pass_rate']:>5.1f}%  "
              f"${s['ev_per_attempt']:>+7,.0f}  "
              f"{s['avg_days_per_attempt']:>7.0f}  "
              f"${s['ev_per_day']:>+7.1f}  "
              f"{s['cycles_per_year']:>7.1f}  "
              f"${s['annual_ev']:>+9,.0f}  "
              f"{s['median_pass_days']:>8.0f}  "
              f"{s['median_fail_days']:>8.0f}")

    # Identify best EV/day
    best_profile = max(profile_stats, key=lambda k: profile_stats[k]["ev_per_day"])
    print(f"\n  >> Best EV/day: {best_profile} "
          f"(${profile_stats[best_profile]['ev_per_day']:+.1f}/day/account)")

    # =========================================================================
    # PART 2: Fleet simulation over 1 year
    # =========================================================================

    print(f"\n  {'=' * 106}")
    print(f"  PART 2: FLEET SIMULATION ({N_ACCOUNTS} accounts, {HORIZON_DAYS}-day horizon, {N_SIMS} sims)")
    print(f"  {'=' * 106}")

    # Build pool list for indexing
    pool_keys = list(RISK_PROFILES.keys())
    pools_list = [profile_pools[k] for k in pool_keys]
    safe_pool_idx = pool_keys.index("SAFE (0.8%)")

    fleet_results = []

    for safe_frac, aggr_key, mix_label in HYBRID_MIXES:
        n_safe = round(N_ACCOUNTS * safe_frac)
        n_aggr = N_ACCOUNTS - n_safe

        # Build account assignments
        assignments = [safe_pool_idx] * n_safe
        if aggr_key is not None and n_aggr > 0:
            aggr_pool_idx = pool_keys.index(aggr_key)
            assignments += [aggr_pool_idx] * n_aggr

        # Pad if needed (100% safe case)
        while len(assignments) < N_ACCOUNTS:
            assignments.append(safe_pool_idx)

        log.info("  Simulating fleet: %s (%d safe + %d aggr)...",
                 mix_label, n_safe, n_aggr)

        result = _simulate_fleet_one_year(
            pools_list, N_ACCOUNTS, assignments, rng, N_SIMS,
        )
        result["label"] = mix_label
        result["n_safe"] = n_safe
        result["n_aggr"] = n_aggr
        fleet_results.append(result)

    # Print fleet results
    print(f"\n  {'Mix':>22s}  {'Cycles':>7s}  {'Passes':>7s}  {'Fails':>6s}  "
          f"{'Fees':>9s}  {'Payouts':>10s}  {'Net Profit':>11s}  "
          f"{'P10':>10s}  {'P90':>10s}  {'$/Day':>8s}  {'$/Acct/Day':>10s}")
    print(f"  {'-' * 120}")

    for r in fleet_results:
        print(f"  {r['label']:>22s}  "
              f"{r['avg_cycles']:>7.1f}  "
              f"{r['avg_passes']:>7.1f}  "
              f"{r['avg_fails']:>6.1f}  "
              f"${r['avg_fees']:>8,.0f}  "
              f"${r['avg_payouts']:>9,.0f}  "
              f"${r['avg_profit']:>+10,.0f}  "
              f"${r['p10_profit']:>+9,.0f}  "
              f"${r['p90_profit']:>+9,.0f}  "
              f"${r['profit_per_day']:>+7.0f}  "
              f"${r['profit_per_account_per_day']:>+9.1f}")

    # =========================================================================
    # PART 3: Comparison and ranking
    # =========================================================================

    print(f"\n  {'=' * 120}")
    print("  RANKING BY ANNUAL PROFIT (10 accounts)")
    print(f"  {'-' * 120}")

    ranked = sorted(fleet_results, key=lambda r: -r["avg_profit"])
    for i, r in enumerate(ranked):
        marker = " <<< BEST" if i == 0 else ""
        print(f"  {i+1:>2d}. {r['label']:>22s}  "
              f"Annual: ${r['avg_profit']:>+10,.0f}  |  "
              f"$/day: ${r['profit_per_day']:>+7.0f}  |  "
              f"Cycles/acct: {r['cycles_per_account']:>4.1f}  |  "
              f"P10: ${r['p10_profit']:>+9,.0f}  |  "
              f"P90: ${r['p90_profit']:>+9,.0f}{marker}")

    # Speed advantage analysis
    print(f"\n  {'-' * 120}")
    print("  SPEED VS SAFETY TRADEOFF")
    print(f"  {'-' * 120}")

    safe_only = next(r for r in fleet_results if r["label"] == "100% SAFE")
    for r in fleet_results:
        if r["label"] == "100% SAFE":
            continue
        delta = r["avg_profit"] - safe_only["avg_profit"]
        delta_cycles = r["avg_cycles"] - safe_only["avg_cycles"]
        sign = "+" if delta >= 0 else ""
        print(f"  {r['label']:>22s} vs SAFE:  "
              f"profit {sign}${delta:,.0f}  |  "
              f"cycles {sign}{delta_cycles:.1f}  |  "
              f"$/day: ${r['profit_per_day']:>+.0f} vs ${safe_only['profit_per_day']:>+.0f}")

    # =========================================================================
    # Save results
    # =========================================================================

    # Profile stats CSV
    profile_rows = []
    for label, s in profile_stats.items():
        profile_rows.append({"profile": label, **s})
    pd.DataFrame(profile_rows).to_csv(
        OUTPUT_DIR / "capital_efficiency_profiles.csv", index=False)

    # Fleet results CSV
    fleet_rows = []
    for r in fleet_results:
        fleet_rows.append({k: v for k, v in r.items()})
    pd.DataFrame(fleet_rows).to_csv(
        OUTPUT_DIR / "capital_efficiency_fleet.csv", index=False)

    log.info("Results saved to %s", OUTPUT_DIR)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
