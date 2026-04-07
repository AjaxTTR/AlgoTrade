"""
Funded account simulation.

Models long-term profit extraction after passing a prop firm challenge.
Simulates day-by-day trading on a funded account with:
  - trailing drawdown limits (account blown = terminated)
  - daily drawdown limits
  - monthly profit withdrawals (profit split)
  - consistency rules (no single day > X% of monthly profit)
  - multiple risk profiles (0.5% - 0.8%)

Uses daily PnL blocks from real backtest data, shuffled via Monte Carlo.

Usage:
    python -m research.funded_sim
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
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
# Funded account rules
# ---------------------------------------------------------------------------

ACCOUNT_SIZE = 100_000.0
PROFIT_SPLIT = 0.80             # trader keeps 80%
PROFIT_TARGET_PCT = 0.10        # challenge target (for combined value calc)
CHALLENGE_PAYOUT = ACCOUNT_SIZE * PROFIT_TARGET_PCT * PROFIT_SPLIT  # $8,000
PAYOUT_INTERVAL_DAYS = 30       # monthly withdrawals
MIN_PAYOUT = 100.0              # minimum profit to trigger payout

# Funded account limits (typical FTMO-style funded rules)
FUNDED_CONFIG = {
    "max_trailing_dd": 0.05,    # 5% trailing drawdown = account blown
    "daily_dd_limit": 0.05,     # 5% max daily loss
    "consistency_rule": 0.40,   # no single day > 40% of payout-period profit
}

# Portfolio weights (locked)
WEIGHTS = {
    "gap_momentum": 0.60,
    "regime_filtered_momentum": 0.40,
}

# Risk profiles to sweep on the funded account
RISK_PROFILES = [0.005, 0.006, 0.007, 0.008]
RISK_LABELS = ["0.5%", "0.6%", "0.7%", "0.8%"]

# Simulation
HORIZON_MONTHS = 12
HORIZON_DAYS = HORIZON_MONTHS * 30  # ~360 trading days
N_SIMS = 3000
SEED = 42


# ---------------------------------------------------------------------------
# Build daily PnL blocks from backtest
# ---------------------------------------------------------------------------

def _get_daily_pnls(df: pd.DataFrame, risk: float) -> np.ndarray:
    """Run portfolio at given risk and return array of daily PnLs.

    Each element is the total PnL for one trading day (sum of all trades
    that closed on that day). Days with no trades have PnL = 0.
    """
    all_pnls = []
    all_dates = []

    for name, weight in WEIGHTS.items():
        capital = TOTAL_CAPITAL * weight
        cat = STRATEGY_CATALOGUE[name]
        result, _, _ = _run_single_strategy(name, cat, df, capital, risk)
        for t in result.trades:
            all_pnls.append(t.pnl)
            all_dates.append(t.exit_time.date())

    if not all_pnls:
        return np.array([0.0])

    # Aggregate to daily PnL
    trade_df = pd.DataFrame({"date": all_dates, "pnl": all_pnls})
    daily = trade_df.groupby("date")["pnl"].sum()

    # Fill in non-trading days with 0
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    daily = daily.reindex(full_range, fill_value=0.0)

    return daily.values


# ---------------------------------------------------------------------------
# Funded account simulation (single run)
# ---------------------------------------------------------------------------

def _simulate_funded(
    daily_pnl_pool: np.ndarray,
    rng: np.random.Generator,
    horizon_days: int,
    config: dict,
) -> dict:
    """Simulate one funded account over horizon_days.

    Daily PnLs are sampled with replacement from the pool (bootstrap).
    Monthly withdrawals happen every PAYOUT_INTERVAL_DAYS.

    Returns detailed result dict.
    """
    max_trail_dd = config["max_trailing_dd"]
    daily_dd_limit = config["daily_dd_limit"]
    consistency_pct = config["consistency_rule"]

    balance = ACCOUNT_SIZE
    equity_peak = balance  # trailing DD tracks from here
    total_withdrawn = 0.0
    monthly_profits = []
    monthly_pnls_raw = []
    payout_count = 0
    peak_dd_pct = 0.0

    # Track for consistency rule
    period_start_balance = balance
    period_day_pnls = []

    alive = True
    death_reason = None
    death_day = 0
    days_survived = 0

    # Daily equity curve for analysis
    equity_curve = np.zeros(horizon_days)

    # Sample daily PnLs for the full horizon
    sampled_indices = rng.integers(0, len(daily_pnl_pool), size=horizon_days)

    for day in range(horizon_days):
        day_pnl = daily_pnl_pool[sampled_indices[day]]

        # Apply PnL
        balance += day_pnl
        period_day_pnls.append(day_pnl)

        # Track equity peak (trailing DD)
        if balance > equity_peak:
            equity_peak = balance

        # Drawdown checks
        dd_from_peak = equity_peak - balance
        dd_pct = dd_from_peak / (equity_peak + 1e-9) * 100
        if dd_pct > peak_dd_pct:
            peak_dd_pct = dd_pct

        # Max trailing DD check
        if max_trail_dd > 0 and dd_from_peak >= equity_peak * max_trail_dd:
            alive = False
            death_reason = "max_trailing_dd"
            death_day = day + 1
            equity_curve[day] = balance
            break

        # Daily DD check (intraday — single day loss)
        if daily_dd_limit > 0 and day_pnl < 0:
            day_loss_pct = abs(day_pnl) / (balance - day_pnl + 1e-9)
            if day_loss_pct >= daily_dd_limit:
                alive = False
                death_reason = "daily_dd"
                death_day = day + 1
                equity_curve[day] = balance
                break

        equity_curve[day] = balance
        days_survived = day + 1

        # Monthly payout check
        if (day + 1) % PAYOUT_INTERVAL_DAYS == 0 and day > 0:
            period_profit = balance - period_start_balance

            # Consistency check: no single day > X% of period profit
            payout_eligible = True
            if period_profit > MIN_PAYOUT and consistency_pct > 0:
                max_day_pnl = max(period_day_pnls) if period_day_pnls else 0
                if period_profit > 0 and max_day_pnl > consistency_pct * period_profit:
                    # Consistency violation: reduce payout to exclude
                    # the excess portion (common firm approach)
                    adjusted_profit = period_profit - (max_day_pnl - consistency_pct * period_profit)
                    period_profit = max(0, adjusted_profit)

            if period_profit > MIN_PAYOUT:
                withdrawal = period_profit * PROFIT_SPLIT
                balance -= withdrawal
                total_withdrawn += withdrawal
                monthly_profits.append(withdrawal)
                payout_count += 1

                # Reset equity peak after withdrawal (trailing DD resets
                # to new balance on most funded programs)
                equity_peak = balance
            else:
                monthly_profits.append(0.0)

            monthly_pnls_raw.append(period_profit)

            # Reset period tracking
            period_start_balance = balance
            period_day_pnls = []

    return {
        "alive": alive,
        "death_reason": death_reason,
        "death_day": death_day,
        "days_survived": days_survived,
        "months_survived": days_survived / 30,
        "final_balance": balance,
        "total_withdrawn": total_withdrawn,
        "payout_count": payout_count,
        "monthly_profits": monthly_profits,
        "monthly_pnls_raw": monthly_pnls_raw,
        "peak_dd_pct": peak_dd_pct,
        "equity_curve": equity_curve[:days_survived],
        "total_extracted": total_withdrawn + (balance - ACCOUNT_SIZE),
    }


# ---------------------------------------------------------------------------
# Multi-sim aggregation
# ---------------------------------------------------------------------------

def _run_funded_sims(
    daily_pnl_pool: np.ndarray,
    rng: np.random.Generator,
    n_sims: int,
    config: dict,
) -> dict:
    """Run n_sims funded account simulations and aggregate."""
    results = []
    for _ in range(n_sims):
        r = _simulate_funded(daily_pnl_pool, rng, HORIZON_DAYS, config)
        results.append(r)

    alive_count = sum(1 for r in results if r["alive"])
    survival_rate = alive_count / n_sims * 100

    # Survival at each month
    survival_by_month = []
    for m in range(1, HORIZON_MONTHS + 1):
        day_cutoff = m * 30
        survived = sum(1 for r in results if r["days_survived"] >= day_cutoff)
        survival_by_month.append(survived / n_sims * 100)

    # Monthly income stats (across all sims, all months)
    all_monthly = []
    for r in results:
        all_monthly.extend(r["monthly_profits"])

    # Per-account total extracted
    all_extracted = [r["total_extracted"] for r in results]
    all_withdrawn = [r["total_withdrawn"] for r in results]
    all_payout_counts = [r["payout_count"] for r in results]
    all_peak_dds = [r["peak_dd_pct"] for r in results]
    all_days_survived = [r["days_survived"] for r in results]

    # Monthly income for surviving accounts only
    surviving_monthly = []
    for r in results:
        if r["alive"] and r["monthly_profits"]:
            surviving_monthly.extend(r["monthly_profits"])

    # Average monthly income per account (total withdrawn / months survived)
    avg_monthly_per_acct = []
    for r in results:
        if r["payout_count"] > 0:
            avg_monthly_per_acct.append(r["total_withdrawn"] / r["payout_count"])

    return {
        "n_sims": n_sims,
        "survival_rate": survival_rate,
        "survival_by_month": survival_by_month,
        "avg_extracted": float(np.mean(all_extracted)),
        "median_extracted": float(np.median(all_extracted)),
        "p10_extracted": float(np.percentile(all_extracted, 10)),
        "p90_extracted": float(np.percentile(all_extracted, 90)),
        "avg_withdrawn": float(np.mean(all_withdrawn)),
        "median_withdrawn": float(np.median(all_withdrawn)),
        "avg_payout_count": float(np.mean(all_payout_counts)),
        "avg_monthly_income": float(np.mean(surviving_monthly)) if surviving_monthly else 0,
        "median_monthly_income": float(np.median(surviving_monthly)) if surviving_monthly else 0,
        "avg_monthly_per_acct": float(np.mean(avg_monthly_per_acct)) if avg_monthly_per_acct else 0,
        "avg_peak_dd": float(np.mean(all_peak_dds)),
        "avg_days_survived": float(np.mean(all_days_survived)),
        "median_days_survived": float(np.median(all_days_survived)),
        "death_reasons": _death_breakdown(results, n_sims),
    }


def _death_breakdown(results: list, n_sims: int) -> dict:
    reasons = {}
    for r in results:
        if not r["alive"]:
            reason = r["death_reason"]
            reasons[reason] = reasons.get(reason, 0) + 1
    return {k: (v, round(v / n_sims * 100, 1)) for k, v in reasons.items()}


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

    # Build daily PnL pools for each risk level
    daily_pools = {}
    pool_stats = {}

    for risk, label in zip(RISK_PROFILES, RISK_LABELS):
        log.info("Building daily PnL pool for risk=%s...", label)
        pool = _get_daily_pnls(df, risk)
        daily_pools[label] = pool

        trading_days = np.sum(pool != 0)
        total_days = len(pool)
        avg_daily = float(np.mean(pool))
        std_daily = float(np.std(pool))
        win_days = np.sum(pool > 0)
        loss_days = np.sum(pool < 0)

        pool_stats[label] = {
            "total_days": total_days,
            "trading_days": int(trading_days),
            "avg_daily_pnl": avg_daily,
            "std_daily_pnl": std_daily,
            "win_days": int(win_days),
            "loss_days": int(loss_days),
            "day_win_rate": float(win_days / trading_days * 100) if trading_days > 0 else 0,
        }
        log.info("  %d total days, %d trading days, avg=$%.0f/day, std=$%.0f",
                 total_days, trading_days, avg_daily, std_daily)

    # Run simulations for each risk level
    all_results = {}

    for risk_label in RISK_LABELS:
        pool = daily_pools[risk_label]
        log.info("Simulating funded accounts at risk=%s (%d sims)...", risk_label, N_SIMS)
        result = _run_funded_sims(pool, rng, N_SIMS, FUNDED_CONFIG)
        all_results[risk_label] = result
        log.info("  Survival: %.1f%% | Avg extracted: $%.0f | Avg monthly: $%.0f",
                 result["survival_rate"], result["avg_extracted"],
                 result["avg_monthly_income"])

    # =========================================================================
    # Output
    # =========================================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 120)
    print("  FUNDED ACCOUNT SIMULATION -- POST-CHALLENGE PROFIT EXTRACTION")
    print("=" * 120)
    print(f"  Account: ${ACCOUNT_SIZE:,.0f}  |  Profit split: {PROFIT_SPLIT*100:.0f}%  |  "
          f"Payout: every {PAYOUT_INTERVAL_DAYS} days  |  "
          f"Horizon: {HORIZON_MONTHS} months  |  Sims: {N_SIMS:,}")
    print(f"  Max trailing DD: {FUNDED_CONFIG['max_trailing_dd']*100:.0f}%  |  "
          f"Daily DD: {FUNDED_CONFIG['daily_dd_limit']*100:.0f}%  |  "
          f"Consistency: no day > {FUNDED_CONFIG['consistency_rule']*100:.0f}% of period profit")

    # --- Daily PnL pool stats ---
    print(f"\n  DAILY PnL POOL STATS")
    print(f"  {'-' * 90}")
    print(f"  {'Risk':>6s}  {'TradeDays':>9s}  {'Avg$/Day':>9s}  {'Std$/Day':>9s}  "
          f"{'WinDays':>7s}  {'LossDays':>8s}  {'DayWR%':>7s}")
    print(f"  {'-' * 90}")

    for label in RISK_LABELS:
        s = pool_stats[label]
        print(f"  {label:>6s}  {s['trading_days']:>9d}  "
              f"${s['avg_daily_pnl']:>+8.0f}  "
              f"${s['std_daily_pnl']:>8.0f}  "
              f"{s['win_days']:>7d}  "
              f"{s['loss_days']:>8d}  "
              f"{s['day_win_rate']:>6.1f}%")

    # --- Survival & income summary ---
    print(f"\n  {'=' * 116}")
    print(f"  FUNDED ACCOUNT PERFORMANCE (per account, {HORIZON_MONTHS}-month horizon)")
    print(f"  {'=' * 116}")
    print(f"  {'Risk':>6s}  {'Surv%':>6s}  {'AvgDays':>7s}  "
          f"{'Extracted':>10s}  {'Med Extr':>10s}  {'P10':>10s}  {'P90':>10s}  "
          f"{'Payouts':>7s}  {'Mo Income':>10s}  {'Med Mo$':>8s}  {'PeakDD':>6s}")
    print(f"  {'-' * 116}")

    for label in RISK_LABELS:
        r = all_results[label]
        print(f"  {label:>6s}  {r['survival_rate']:>5.1f}%  "
              f"{r['avg_days_survived']:>7.0f}  "
              f"${r['avg_extracted']:>+9,.0f}  "
              f"${r['median_extracted']:>+9,.0f}  "
              f"${r['p10_extracted']:>+9,.0f}  "
              f"${r['p90_extracted']:>+9,.0f}  "
              f"{r['avg_payout_count']:>7.1f}  "
              f"${r['avg_monthly_income']:>9,.0f}  "
              f"${r['median_monthly_income']:>7,.0f}  "
              f"{r['avg_peak_dd']:>5.1f}%")

    # --- Failure breakdown ---
    print(f"\n  FAILURE BREAKDOWN")
    print(f"  {'-' * 60}")
    for label in RISK_LABELS:
        r = all_results[label]
        deaths = r["death_reasons"]
        if deaths:
            parts = [f"{reason}: {count} ({pct:.1f}%)"
                     for reason, (count, pct) in sorted(deaths.items(), key=lambda x: -x[1][0])]
            print(f"  {label:>6s}  {', '.join(parts)}")
        else:
            print(f"  {label:>6s}  No failures")

    # --- Survival curve ---
    print(f"\n  SURVIVAL RATE BY MONTH")
    print(f"  {'-' * 80}")
    header = f"  {'Risk':>6s}"
    for m in range(1, HORIZON_MONTHS + 1):
        header += f"  {'M'+str(m):>5s}"
    print(header)
    print(f"  {'-' * 80}")

    for label in RISK_LABELS:
        r = all_results[label]
        row = f"  {label:>6s}"
        for surv in r["survival_by_month"]:
            row += f"  {surv:>4.0f}%"
        print(row)

    # --- Combined value: challenge cycling + funded income ---
    print(f"\n  {'=' * 116}")
    print("  COMBINED VALUE: CHALLENGE PAYOUT + FUNDED INCOME (per passed account)")
    print(f"  {'-' * 116}")
    print(f"  Challenge payout (one-time): ${CHALLENGE_PAYOUT:,.0f}")
    print()
    print(f"  {'Risk':>6s}  {'Challenge$':>10s}  {'Funded$/yr':>10s}  "
          f"{'Total$/yr':>10s}  {'Surv%':>6s}  {'EV (risk-adj)':>13s}  "
          f"{'Mo Income':>10s}")
    print(f"  {'-' * 80}")

    challenge_payout = CHALLENGE_PAYOUT  # $8,000

    for label in RISK_LABELS:
        r = all_results[label]
        funded_annual = r["avg_extracted"]
        total_annual = challenge_payout + funded_annual
        # Risk-adjusted: weight by survival rate
        ev_adjusted = challenge_payout + funded_annual * (r["survival_rate"] / 100)

        print(f"  {label:>6s}  ${challenge_payout:>9,.0f}  "
              f"${funded_annual:>+9,.0f}  "
              f"${total_annual:>+9,.0f}  "
              f"{r['survival_rate']:>5.1f}%  "
              f"${ev_adjusted:>+12,.0f}  "
              f"${r['avg_monthly_income']:>9,.0f}")

    # --- Optimal risk recommendation ---
    print(f"\n  {'=' * 116}")
    print("  RECOMMENDATION")
    print(f"  {'-' * 116}")

    # Find best risk-adjusted EV
    best_label = None
    best_ev = -1e9
    for label in RISK_LABELS:
        r = all_results[label]
        funded_annual = r["avg_extracted"]
        ev = challenge_payout + funded_annual * (r["survival_rate"] / 100)
        if ev > best_ev:
            best_ev = ev
            best_label = label

    best_r = all_results[best_label]
    print(f"  Optimal funded risk: {best_label}")
    print(f"    Survival: {best_r['survival_rate']:.1f}% over {HORIZON_MONTHS} months")
    print(f"    Monthly income: ${best_r['avg_monthly_income']:,.0f} (median ${best_r['median_monthly_income']:,.0f})")
    print(f"    Total extracted/year: ${best_r['avg_extracted']:+,.0f}")
    print(f"    Risk-adjusted EV/account: ${best_ev:+,.0f}")
    print(f"    Strategy: challenge at 1.4% risk, then drop to {best_label} on funded account")

    # --- Save CSV ---
    rows = []
    for label in RISK_LABELS:
        r = all_results[label]
        funded_annual = r["avg_extracted"]
        ev_adj = challenge_payout + funded_annual * (r["survival_rate"] / 100)
        rows.append({
            "risk_pct": label,
            "survival_rate": r["survival_rate"],
            "avg_days_survived": round(r["avg_days_survived"], 1),
            "median_days_survived": r["median_days_survived"],
            "avg_extracted": round(r["avg_extracted"], 2),
            "median_extracted": round(r["median_extracted"], 2),
            "p10_extracted": round(r["p10_extracted"], 2),
            "p90_extracted": round(r["p90_extracted"], 2),
            "avg_withdrawn": round(r["avg_withdrawn"], 2),
            "avg_payout_count": round(r["avg_payout_count"], 1),
            "avg_monthly_income": round(r["avg_monthly_income"], 2),
            "median_monthly_income": round(r["median_monthly_income"], 2),
            "avg_peak_dd_pct": round(r["avg_peak_dd"], 2),
            "challenge_payout": challenge_payout,
            "total_annual_value": round(challenge_payout + funded_annual, 2),
            "risk_adjusted_ev": round(ev_adj, 2),
            **{f"survival_m{m}": round(s, 1)
               for m, s in enumerate(r["survival_by_month"], 1)},
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "funded_sim.csv", index=False)

    # --- Save survival curves ---
    surv_data = {"month": list(range(1, HORIZON_MONTHS + 1))}
    for label in RISK_LABELS:
        surv_data[f"survival_{label}"] = all_results[label]["survival_by_month"]
    pd.DataFrame(surv_data).to_csv(OUTPUT_DIR / "funded_survival.csv", index=False)

    log.info("Results saved to %s", OUTPUT_DIR)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
