"""
Deployment simulation: optimise capital allocation across SAFE and
AGGRESSIVE prop firm accounts.

Uses the discovered risk frontier to model real prop firm deployment
strategy — running multiple accounts simultaneously with different
risk profiles.

Runs actual Monte Carlo simulations per account using real trade PnLs
(not just pass rate assumptions).

Usage:
    python -m research.deployment_sim
"""

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.prop_firm import _simulate_single, _days_between
from run_portfolio import (
    STRATEGY_CATALOGUE,
    TOTAL_CAPITAL,
    _run_single_strategy,
    _build_portfolio_backtest_result,
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
# Prop firm economics
# ---------------------------------------------------------------------------

CHALLENGE_FEE = 500.0           # cost per challenge attempt ($100k account)
PROFIT_SPLIT = 0.80             # trader keeps 80% of profits
ACCOUNT_SIZE = 100_000.0        # funded account size
PROFIT_TARGET_PCT = 0.10        # 10% target = $10k profit
FUNDED_PAYOUT = ACCOUNT_SIZE * PROFIT_TARGET_PCT * PROFIT_SPLIT  # $8,000

# Prop firm rules (same as run_portfolio.py)
PROP_CONFIG = {
    "starting_balance": ACCOUNT_SIZE,
    "profit_target": PROFIT_TARGET_PCT,
    "daily_drawdown_limit": 0.05,
    "max_drawdown_limit": 0.10,
    "max_days": 0,
}

# ---------------------------------------------------------------------------
# Risk configs from frontier
# ---------------------------------------------------------------------------

SAFE_RISK = 0.008       # 0.8% — 90% pass, 268 median days
AGGR_RISK_A = 0.013     # 1.3% — 69.7% pass, 116 median days
AGGR_RISK_B = 0.014     # 1.4% — 65.3% pass, 91 median days

# Portfolio weights (locked)
WEIGHTS = {
    "gap_momentum": 0.60,
    "regime_filtered_momentum": 0.40,
}

# Deployment scenarios
ACCOUNT_COUNTS = [5, 10, 20]
SAFE_SPLITS = [1.0, 0.7, 0.5, 0.3, 0.0]  # fraction of accounts that are SAFE

# Monte Carlo: per deployment scenario
N_DEPLOY_SIMS = 2000
SEED = 42


# ---------------------------------------------------------------------------
# Trade data extraction
# ---------------------------------------------------------------------------

def _get_trade_data(df: pd.DataFrame, risk: float) -> tuple[np.ndarray, np.ndarray]:
    """Run portfolio at given risk and return (pnls, exit_dates) arrays."""
    ref_index = None
    all_pnls = []
    all_dates = []

    for name, weight in WEIGHTS.items():
        capital = TOTAL_CAPITAL * weight
        cat = STRATEGY_CATALOGUE[name]
        result, _, _ = _run_single_strategy(name, cat, df, capital, risk)

        for t in result.trades:
            all_pnls.append(t.pnl)
            all_dates.append(t.exit_time.date())

    # Sort by date
    order = np.argsort(all_dates)
    pnls = np.array(all_pnls)[order]
    dates = np.array(all_dates)[order]
    return pnls, dates


def _build_shuffle_template(exit_dates: np.ndarray):
    """Pre-compute date template for shuffled trades (preserves day structure)."""
    unique_dates = np.unique(exit_dates)
    trades_per_day = np.array([np.sum(exit_dates == d) for d in unique_dates])
    day_boundaries = np.concatenate([[0], np.cumsum(trades_per_day)])

    template = np.empty(len(exit_dates), dtype=exit_dates.dtype)
    for d_idx, d in enumerate(unique_dates):
        start, end = day_boundaries[d_idx], day_boundaries[d_idx + 1]
        template[start:end] = d
    return template


# ---------------------------------------------------------------------------
# Single deployment simulation
# ---------------------------------------------------------------------------

@dataclass
class DeploymentResult:
    n_accounts: int
    n_safe: int
    n_aggr: int
    safe_risk_pct: float
    aggr_risk_pct: float
    safe_split: float

    # Per-sim aggregated stats (arrays of length N_DEPLOY_SIMS)
    total_passes: np.ndarray
    safe_passes: np.ndarray
    aggr_passes: np.ndarray
    total_fails: np.ndarray
    total_cost: float          # fixed: n_accounts * CHALLENGE_FEE
    total_payouts: np.ndarray  # passes * FUNDED_PAYOUT per sim
    net_profit: np.ndarray     # payouts - cost per sim
    roi_pct: np.ndarray        # net_profit / cost * 100
    median_days_to_first_pass: np.ndarray
    median_days_to_all_pass: np.ndarray  # max days across passing accounts


def _simulate_deployment(
    n_accounts: int,
    n_safe: int,
    safe_pnls: np.ndarray,
    safe_dates_template: np.ndarray,
    aggr_pnls: np.ndarray,
    aggr_dates_template: np.ndarray,
    safe_risk_pct: float,
    aggr_risk_pct: float,
    safe_split: float,
    rng: np.random.Generator,
    n_sims: int,
) -> DeploymentResult:
    """Simulate deploying n_accounts across safe/aggressive splits."""
    n_aggr = n_accounts - n_safe

    total_cost = n_accounts * CHALLENGE_FEE

    all_total_passes = np.zeros(n_sims, dtype=int)
    all_safe_passes = np.zeros(n_sims, dtype=int)
    all_aggr_passes = np.zeros(n_sims, dtype=int)
    all_first_pass_days = np.full(n_sims, np.nan)
    all_last_pass_days = np.full(n_sims, np.nan)

    for sim_i in range(n_sims):
        pass_days = []

        # Run safe accounts
        for _ in range(n_safe):
            idx = rng.permutation(len(safe_pnls))
            result = _simulate_single(
                safe_pnls[idx], safe_dates_template, PROP_CONFIG,
            )
            if result["outcome"] == "PASS":
                all_safe_passes[sim_i] += 1
                all_total_passes[sim_i] += 1
                pass_days.append(result["days_elapsed"])

        # Run aggressive accounts
        for _ in range(n_aggr):
            idx = rng.permutation(len(aggr_pnls))
            result = _simulate_single(
                aggr_pnls[idx], aggr_dates_template, PROP_CONFIG,
            )
            if result["outcome"] == "PASS":
                all_aggr_passes[sim_i] += 1
                all_total_passes[sim_i] += 1
                pass_days.append(result["days_elapsed"])

        if pass_days:
            all_first_pass_days[sim_i] = min(pass_days)
            all_last_pass_days[sim_i] = max(pass_days)

    all_total_fails = n_accounts - all_total_passes
    all_payouts = all_total_passes.astype(float) * FUNDED_PAYOUT
    all_net = all_payouts - total_cost
    all_roi = all_net / total_cost * 100

    return DeploymentResult(
        n_accounts=n_accounts,
        n_safe=n_safe,
        n_aggr=n_aggr,
        safe_risk_pct=safe_risk_pct,
        aggr_risk_pct=aggr_risk_pct,
        safe_split=safe_split,
        total_passes=all_total_passes,
        safe_passes=all_safe_passes,
        aggr_passes=all_aggr_passes,
        total_fails=all_total_fails,
        total_cost=total_cost,
        total_payouts=all_payouts,
        net_profit=all_net,
        roi_pct=all_roi,
        median_days_to_first_pass=all_first_pass_days,
        median_days_to_all_pass=all_last_pass_days,
    )


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

    # Pre-compute trade data for each risk level
    log.info("Running backtests for SAFE (%.1f%%) and AGGRESSIVE configs...",
             SAFE_RISK * 100)

    safe_pnls, safe_dates = _get_trade_data(df, SAFE_RISK)
    safe_template = _build_shuffle_template(safe_dates)
    log.info("  SAFE: %d trades", len(safe_pnls))

    # Run both aggressive options
    aggr_configs = [
        ("1.3%", AGGR_RISK_A),
        ("1.4%", AGGR_RISK_B),
    ]

    all_results = []
    rng = np.random.default_rng(SEED)

    for aggr_label, aggr_risk in aggr_configs:
        aggr_pnls, aggr_dates = _get_trade_data(df, aggr_risk)
        aggr_template = _build_shuffle_template(aggr_dates)
        log.info("  AGGR %s: %d trades", aggr_label, len(aggr_pnls))

        for n_accounts in ACCOUNT_COUNTS:
            for safe_split in SAFE_SPLITS:
                n_safe = round(n_accounts * safe_split)
                n_aggr = n_accounts - n_safe

                log.info("  Simulating %d accounts (%d safe + %d aggr@%s)...",
                         n_accounts, n_safe, n_aggr, aggr_label)

                result = _simulate_deployment(
                    n_accounts=n_accounts,
                    n_safe=n_safe,
                    safe_pnls=safe_pnls,
                    safe_dates_template=safe_template,
                    aggr_pnls=aggr_pnls,
                    aggr_dates_template=aggr_template,
                    safe_risk_pct=SAFE_RISK * 100,
                    aggr_risk_pct=aggr_risk * 100,
                    safe_split=safe_split,
                    rng=rng,
                    n_sims=N_DEPLOY_SIMS,
                )
                all_results.append((aggr_label, result))

    # Build summary table
    rows = []
    for aggr_label, r in all_results:
        # Filter out NaN for day stats
        first_days = r.median_days_to_first_pass[~np.isnan(r.median_days_to_first_pass)]
        last_days = r.median_days_to_all_pass[~np.isnan(r.median_days_to_all_pass)]

        # P(zero passes) — total wipeout risk
        zero_pass_pct = np.mean(r.total_passes == 0) * 100

        rows.append({
            "aggr_risk": aggr_label,
            "n_accounts": r.n_accounts,
            "n_safe": r.n_safe,
            "n_aggr": r.n_aggr,
            "split_label": f"{r.n_safe}S/{r.n_aggr}A",
            "safe_split_pct": r.safe_split * 100,
            "total_cost": r.total_cost,
            "avg_passes": float(np.mean(r.total_passes)),
            "avg_safe_passes": float(np.mean(r.safe_passes)),
            "avg_aggr_passes": float(np.mean(r.aggr_passes)),
            "avg_fails": float(np.mean(r.total_fails)),
            "pass_rate_pct": float(np.mean(r.total_passes) / r.n_accounts * 100),
            "avg_payout": float(np.mean(r.total_payouts)),
            "avg_net_profit": float(np.mean(r.net_profit)),
            "median_net_profit": float(np.median(r.net_profit)),
            "p10_net_profit": float(np.percentile(r.net_profit, 10)),
            "p90_net_profit": float(np.percentile(r.net_profit, 90)),
            "avg_roi_pct": float(np.mean(r.roi_pct)),
            "median_roi_pct": float(np.median(r.roi_pct)),
            "zero_pass_pct": zero_pass_pct,
            "median_first_pass_days": float(np.median(first_days)) if len(first_days) > 0 else None,
            "median_all_pass_days": float(np.median(last_days)) if len(last_days) > 0 else None,
            "ev_per_account": float(np.mean(r.net_profit)) / r.n_accounts,
        })

    df_results = pd.DataFrame(rows)

    # Save CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "deployment_sim.csv"
    df_results.to_csv(csv_path, index=False)
    log.info("Results saved to %s", csv_path)

    # ---------------------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 110)
    print("  DEPLOYMENT SIMULATION — MULTI-ACCOUNT PROP FIRM STRATEGY")
    print("=" * 110)
    print(f"  SAFE: 0.8% risk (90% pass, ~268d)  |  Challenge fee: ${CHALLENGE_FEE:,.0f}  |  "
          f"Funded payout: ${FUNDED_PAYOUT:,.0f}  |  Sims: {N_DEPLOY_SIMS:,}")
    print(f"  Account size: ${ACCOUNT_SIZE:,.0f}  |  Profit target: "
          f"{PROFIT_TARGET_PCT*100:.0f}%  |  Profit split: {PROFIT_SPLIT*100:.0f}%")

    for aggr_label, aggr_risk in aggr_configs:
        subset = df_results[df_results["aggr_risk"] == aggr_label]

        print(f"\n  {'-' * 106}")
        print(f"  AGGRESSIVE CONFIG: {aggr_label} risk")
        print(f"  {'-' * 106}")

        for n_acc in ACCOUNT_COUNTS:
            acc_subset = subset[subset["n_accounts"] == n_acc]

            print(f"\n  +- {n_acc} ACCOUNTS (cost: ${n_acc * CHALLENGE_FEE:,.0f})")
            print(f"  | {'Split':>7s}  {'Passes':>7s}  {'Fails':>6s}  {'Pass%':>6s}  "
                  f"{'Payout':>9s}  {'Net':>9s}  {'P10 Net':>9s}  {'P90 Net':>9s}  "
                  f"{'ROI%':>7s}  {'Wipe%':>6s}  {'1st Pass':>8s}  {'EV/Acct':>8s}")
            print(f"  | {'-' * 100}")

            for _, r in acc_subset.iterrows():
                first_d = f"{r['median_first_pass_days']:.0f}d" if pd.notna(r['median_first_pass_days']) else "N/A"
                print(f"  | {r['split_label']:>7s}  "
                      f"{r['avg_passes']:>7.1f}  {r['avg_fails']:>6.1f}  "
                      f"{r['pass_rate_pct']:>5.1f}%  "
                      f"${r['avg_payout']:>8,.0f}  "
                      f"${r['avg_net_profit']:>+8,.0f}  "
                      f"${r['p10_net_profit']:>+8,.0f}  "
                      f"${r['p90_net_profit']:>+8,.0f}  "
                      f"{r['avg_roi_pct']:>+6.0f}%  "
                      f"{r['zero_pass_pct']:>5.1f}%  "
                      f"{first_d:>8s}  "
                      f"${r['ev_per_account']:>+7,.0f}")

            print(f"  +{'-' * 102}")

    # ---------------------------------------------------------------------------
    # Best allocation per account count
    # ---------------------------------------------------------------------------

    print(f"\n  {'=' * 106}")
    print("  OPTIMAL ALLOCATIONS (highest EV per account)")
    print(f"  {'=' * 106}")

    for n_acc in ACCOUNT_COUNTS:
        acc_sub = df_results[df_results["n_accounts"] == n_acc]
        best = acc_sub.loc[acc_sub["ev_per_account"].idxmax()]
        safest = acc_sub.loc[acc_sub["zero_pass_pct"].idxmin()]

        print(f"\n  {n_acc} ACCOUNTS:")
        print(f"    Max EV:     {best['split_label']} (aggr@{best['aggr_risk']})  "
              f"-> EV ${best['ev_per_account']:>+,.0f}/acct  |  "
              f"ROI {best['avg_roi_pct']:>+.0f}%  |  "
              f"Net ${best['avg_net_profit']:>+,.0f}  |  "
              f"Wipe risk: {best['zero_pass_pct']:.1f}%")
        print(f"    Safest:     {safest['split_label']} (aggr@{safest['aggr_risk']})  "
              f"-> EV ${safest['ev_per_account']:>+,.0f}/acct  |  "
              f"ROI {safest['avg_roi_pct']:>+.0f}%  |  "
              f"Net ${safest['avg_net_profit']:>+,.0f}  |  "
              f"Wipe risk: {safest['zero_pass_pct']:.1f}%")

    # ---------------------------------------------------------------------------
    # Expected value summary
    # ---------------------------------------------------------------------------

    print(f"\n  {'=' * 106}")
    print("  EXPECTED VALUE PER BATCH (best EV allocation)")
    print(f"  {'-' * 106}")
    print(f"  {'Batch':>7s}  {'Config':>20s}  {'Cost':>9s}  {'Avg Payout':>11s}  "
          f"{'Avg Net':>10s}  {'Med Net':>10s}  {'ROI':>7s}  {'EV/Acct':>9s}")
    print(f"  {'-' * 106}")

    for n_acc in ACCOUNT_COUNTS:
        acc_sub = df_results[df_results["n_accounts"] == n_acc]
        best = acc_sub.loc[acc_sub["ev_per_account"].idxmax()]
        label = f"{best['split_label']} @{best['aggr_risk']}"
        print(f"  {n_acc:>5d}x  {label:>20s}  "
              f"${best['total_cost']:>8,.0f}  "
              f"${best['avg_payout']:>10,.0f}  "
              f"${best['avg_net_profit']:>+9,.0f}  "
              f"${best['median_net_profit']:>+9,.0f}  "
              f"{best['avg_roi_pct']:>+6.0f}%  "
              f"${best['ev_per_account']:>+8,.0f}")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
