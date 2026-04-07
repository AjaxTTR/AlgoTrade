"""
Deployment engine for prop firm capital cycling.

Day-by-day simulation of running multiple concurrent prop firm accounts
with aggressive risk, redeploying capital after each pass/fail.  Tracks
real capital (bankroll), enforces deployment constraints, and produces
capital growth curves.

Usage:
    python -m research.deployment_engine
"""

import logging
import sys
import time
from dataclasses import dataclass, field
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

WEIGHTS = {
    "gap_momentum": 0.60,
    "regime_filtered_momentum": 0.40,
}

# ---------------------------------------------------------------------------
# Deployment configurations to simulate
# ---------------------------------------------------------------------------

@dataclass
class DeployConfig:
    label: str
    risk: float                     # risk_per_trade
    starting_bankroll: float        # real capital to start
    max_concurrent: int             # max accounts running at once
    max_capital_deployed: float     # max total challenge fees outstanding
    max_concurrent_failures: int    # if this many fail on same day, pause 1 day
    horizon_days: int = 730         # 2 years


CONFIGS = [
    # Small scale: $10k bankroll
    DeployConfig("10k/5acct/1.4%",  0.014, 10_000,  5, 10_000, 3, 730),
    DeployConfig("10k/10acct/1.4%", 0.014, 10_000, 10, 10_000, 5, 730),
    DeployConfig("10k/5acct/1.3%",  0.013, 10_000,  5, 10_000, 3, 730),

    # Medium scale: $25k bankroll
    DeployConfig("25k/10acct/1.4%", 0.014, 25_000, 10, 25_000, 5, 730),
    DeployConfig("25k/20acct/1.4%", 0.014, 25_000, 20, 25_000, 8, 730),
    DeployConfig("25k/10acct/1.3%", 0.013, 25_000, 10, 25_000, 5, 730),

    # Large scale: $50k bankroll
    DeployConfig("50k/20acct/1.4%", 0.014, 50_000, 20, 50_000, 8, 730),
    DeployConfig("50k/20acct/1.3%", 0.013, 50_000, 20, 50_000, 8, 730),
]

N_SIMS = 1000
SEED = 42


# ---------------------------------------------------------------------------
# Trade data & outcome pools (reused pattern)
# ---------------------------------------------------------------------------

def _get_trade_data(df: pd.DataFrame, risk: float) -> tuple[np.ndarray, np.ndarray]:
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


def _build_outcome_pool(
    pnls: np.ndarray,
    dates_template: np.ndarray,
    rng: np.random.Generator,
    pool_size: int = 15000,
) -> list[dict]:
    pool = []
    for _ in range(pool_size):
        idx = rng.permutation(len(pnls))
        result = _simulate_single(pnls[idx], dates_template, PROP_CONFIG)
        pool.append({
            "outcome": result["outcome"],
            "days": max(result["days_elapsed"], 1),
        })
    return pool


# ---------------------------------------------------------------------------
# Account state
# ---------------------------------------------------------------------------

@dataclass
class Account:
    id: int
    start_day: int
    duration: int           # days this attempt will take
    outcome: str            # PASS or FAIL
    fee_paid: float = CHALLENGE_FEE

    @property
    def end_day(self) -> int:
        return self.start_day + self.duration


# ---------------------------------------------------------------------------
# Day-by-day deployment engine
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Result of a single multi-year deployment simulation."""
    bankroll_curve: np.ndarray      # bankroll at end of each day
    cumulative_invested: np.ndarray # total fees paid through each day
    cumulative_payouts: np.ndarray  # total payouts through each day
    total_passes: int
    total_fails: int
    total_cycles: int
    peak_bankroll: float
    max_drawdown_pct: float         # max DD in bankroll terms
    max_drawdown_abs: float
    final_bankroll: float
    days_to_double: int             # days to 2x starting bankroll (0 if never)
    days_to_100k: int               # days to reach $100k (0 if never/started above)


def _run_engine(
    cfg: DeployConfig,
    pool: list[dict],
    rng: np.random.Generator,
) -> SimResult:
    """Run one deployment simulation over the full horizon."""
    horizon = cfg.horizon_days
    pool_size = len(pool)

    bankroll = cfg.starting_bankroll
    bankroll_curve = np.zeros(horizon)
    cum_invested = np.zeros(horizon)
    cum_payouts = np.zeros(horizon)

    active_accounts: list[Account] = []
    total_invested = 0.0
    total_payouts = 0.0
    total_passes = 0
    total_fails = 0
    total_cycles = 0
    next_account_id = 0

    peak_bankroll = bankroll
    max_dd_abs = 0.0

    days_to_double = 0
    days_to_100k = 0
    double_target = cfg.starting_bankroll * 2
    hit_double = cfg.starting_bankroll >= double_target
    hit_100k = cfg.starting_bankroll >= 100_000

    paused_until = 0  # day index to resume after failure cluster

    for day in range(horizon):
        # --- Resolve completed accounts ---
        completed = [a for a in active_accounts if a.end_day <= day]
        daily_fails = 0

        for acc in completed:
            active_accounts.remove(acc)
            total_cycles += 1

            if acc.outcome == "PASS":
                total_passes += 1
                total_payouts += FUNDED_PAYOUT
                bankroll += FUNDED_PAYOUT
            else:
                total_fails += 1
                daily_fails += 1

        # --- Failure cluster check: pause if too many fail on same day ---
        if daily_fails >= cfg.max_concurrent_failures:
            paused_until = day + 1  # skip launching new accounts today

        # --- Launch new accounts ---
        if day >= paused_until:
            n_active = len(active_accounts)
            capital_deployed = n_active * CHALLENGE_FEE

            while (n_active < cfg.max_concurrent
                   and capital_deployed + CHALLENGE_FEE <= cfg.max_capital_deployed
                   and bankroll >= CHALLENGE_FEE):

                # Draw random outcome
                attempt = pool[rng.integers(pool_size)]

                # Pay fee
                bankroll -= CHALLENGE_FEE
                total_invested += CHALLENGE_FEE

                acc = Account(
                    id=next_account_id,
                    start_day=day,
                    duration=attempt["days"],
                    outcome=attempt["outcome"],
                )
                active_accounts.append(acc)
                next_account_id += 1
                n_active += 1
                capital_deployed += CHALLENGE_FEE

        # --- Track state ---
        bankroll_curve[day] = bankroll
        cum_invested[day] = total_invested
        cum_payouts[day] = total_payouts

        if bankroll > peak_bankroll:
            peak_bankroll = bankroll
        dd = peak_bankroll - bankroll
        if dd > max_dd_abs:
            max_dd_abs = dd

        if not hit_double and bankroll >= double_target:
            hit_double = True
            days_to_double = day + 1
        if not hit_100k and bankroll >= 100_000:
            hit_100k = True
            days_to_100k = day + 1

    max_dd_pct = (max_dd_abs / peak_bankroll * 100) if peak_bankroll > 0 else 0

    return SimResult(
        bankroll_curve=bankroll_curve,
        cumulative_invested=cum_invested,
        cumulative_payouts=cum_payouts,
        total_passes=total_passes,
        total_fails=total_fails,
        total_cycles=total_cycles,
        peak_bankroll=peak_bankroll,
        max_drawdown_pct=max_dd_pct,
        max_drawdown_abs=max_dd_abs,
        final_bankroll=bankroll,
        days_to_double=days_to_double if hit_double else 0,
        days_to_100k=days_to_100k if hit_100k else 0,
    )


# ---------------------------------------------------------------------------
# Multi-sim aggregation
# ---------------------------------------------------------------------------

@dataclass
class AggResult:
    label: str
    config: DeployConfig
    n_sims: int

    # Bankroll curves: (n_sims, horizon) matrix
    all_curves: np.ndarray

    # Aggregated scalars
    avg_final: float
    median_final: float
    p10_final: float
    p90_final: float
    avg_passes: float
    avg_fails: float
    avg_cycles: float
    avg_max_dd_pct: float
    avg_max_dd_abs: float
    avg_peak: float
    median_days_to_double: float
    median_days_to_100k: float
    pct_reached_double: float
    pct_reached_100k: float
    avg_net_profit: float
    median_net_profit: float


def _run_multi(
    cfg: DeployConfig,
    pool: list[dict],
    rng: np.random.Generator,
    n_sims: int,
) -> AggResult:
    horizon = cfg.horizon_days
    all_curves = np.zeros((n_sims, horizon))

    finals = []
    passes_list = []
    fails_list = []
    cycles_list = []
    dd_pct_list = []
    dd_abs_list = []
    peak_list = []
    d2x_list = []
    d100k_list = []
    net_list = []

    for i in range(n_sims):
        r = _run_engine(cfg, pool, rng)
        all_curves[i] = r.bankroll_curve
        finals.append(r.final_bankroll)
        passes_list.append(r.total_passes)
        fails_list.append(r.total_fails)
        cycles_list.append(r.total_cycles)
        dd_pct_list.append(r.max_drawdown_pct)
        dd_abs_list.append(r.max_drawdown_abs)
        peak_list.append(r.peak_bankroll)
        d2x_list.append(r.days_to_double)
        d100k_list.append(r.days_to_100k)
        net_list.append(r.final_bankroll - cfg.starting_bankroll)

    finals = np.array(finals)
    d2x = np.array(d2x_list)
    d100k = np.array(d100k_list)
    net = np.array(net_list)

    reached_double = d2x > 0
    reached_100k = d100k > 0

    return AggResult(
        label=cfg.label,
        config=cfg,
        n_sims=n_sims,
        all_curves=all_curves,
        avg_final=float(np.mean(finals)),
        median_final=float(np.median(finals)),
        p10_final=float(np.percentile(finals, 10)),
        p90_final=float(np.percentile(finals, 90)),
        avg_passes=float(np.mean(passes_list)),
        avg_fails=float(np.mean(fails_list)),
        avg_cycles=float(np.mean(cycles_list)),
        avg_max_dd_pct=float(np.mean(dd_pct_list)),
        avg_max_dd_abs=float(np.mean(dd_abs_list)),
        avg_peak=float(np.mean(peak_list)),
        median_days_to_double=float(np.median(d2x[reached_double])) if reached_double.any() else 0,
        median_days_to_100k=float(np.median(d100k[reached_100k])) if reached_100k.any() else 0,
        pct_reached_double=float(reached_double.mean() * 100),
        pct_reached_100k=float(reached_100k.mean() * 100),
        avg_net_profit=float(np.mean(net)),
        median_net_profit=float(np.median(net)),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_growth_curves(results: list[AggResult], output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

    ax1 = axes[0]
    colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A",
              "#C62828", "#00838F", "#4E342E", "#283593"]

    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        days = np.arange(r.config.horizon_days)

        median_curve = np.median(r.all_curves, axis=0)
        p10_curve = np.percentile(r.all_curves, 10, axis=0)
        p90_curve = np.percentile(r.all_curves, 90, axis=0)

        ax1.plot(days, median_curve, color=c, linewidth=1.5, label=r.label)
        ax1.fill_between(days, p10_curve, p90_curve, color=c, alpha=0.1)

    ax1.set_title("Deployment Engine: Capital Growth (Median + P10/P90 band)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Bankroll ($)")
    ax1.set_xlabel("Days")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=100_000, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # Drawdown subplot: show worst-case (P10) for each config
    ax2 = axes[1]
    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        median_curve = np.median(r.all_curves, axis=0)
        running_max = np.maximum.accumulate(median_curve)
        dd_pct = np.where(running_max > 0,
                          (running_max - median_curve) / running_max * 100, 0)
        days = np.arange(r.config.horizon_days)
        ax2.plot(days, -dd_pct, color=c, linewidth=0.8, alpha=0.7)

    ax2.set_title("Median Bankroll Drawdown", fontsize=10)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Days")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "deployment_growth.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Growth curve plot saved to %s", plot_path)


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

    # Build outcome pools for each unique risk level
    unique_risks = sorted(set(c.risk for c in CONFIGS))
    pools = {}

    for risk in unique_risks:
        log.info("Building outcome pool for risk=%.1f%%...", risk * 100)
        pnls, dates = _get_trade_data(df, risk)
        template = _build_shuffle_template(dates)
        pools[risk] = _build_outcome_pool(pnls, template, rng, pool_size=15000)
        # Quick stats
        p = pools[risk]
        pass_rate = sum(1 for x in p if x["outcome"] == "PASS") / len(p) * 100
        avg_days = np.mean([x["days"] for x in p])
        log.info("  %d outcomes | pass=%.1f%% | avg_days=%.0f", len(p), pass_rate, avg_days)

    # Run all configurations
    all_results: list[AggResult] = []

    for cfg in CONFIGS:
        log.info("Simulating %s (%d sims)...", cfg.label, N_SIMS)
        pool = pools[cfg.risk]
        result = _run_multi(cfg, pool, rng, N_SIMS)
        all_results.append(result)
        log.info("  Final: $%.0f (median) | DD: %.1f%% | Double: %.0f days (%.0f%%)",
                 result.median_final, result.avg_max_dd_pct,
                 result.median_days_to_double, result.pct_reached_double)

    # =========================================================================
    # Output
    # =========================================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Print results ---
    print("\n" + "=" * 130)
    print("  DEPLOYMENT ENGINE -- PROP FIRM CAPITAL CYCLING")
    print("=" * 130)
    print(f"  Challenge fee: ${CHALLENGE_FEE:,.0f}  |  Payout: ${FUNDED_PAYOUT:,.0f}  |  "
          f"Sims: {N_SIMS:,}  |  Horizon: {CONFIGS[0].horizon_days} days")

    # Summary table
    print(f"\n  {'Config':>22s}  {'Start':>7s}  {'Accts':>5s}  "
          f"{'Final$':>9s}  {'Med$':>9s}  {'P10$':>9s}  {'P90$':>9s}  "
          f"{'Cycles':>6s}  {'Pass':>5s}  {'Fail':>5s}  "
          f"{'MaxDD%':>6s}  {'MaxDD$':>8s}  "
          f"{'2x Days':>7s}  {'2x%':>5s}  {'100k D':>6s}  {'100k%':>5s}")
    print(f"  {'-' * 126}")

    for r in all_results:
        d2x = f"{r.median_days_to_double:.0f}" if r.pct_reached_double > 0 else "N/A"
        d100k = f"{r.median_days_to_100k:.0f}" if r.pct_reached_100k > 0 else "N/A"

        print(f"  {r.label:>22s}  "
              f"${r.config.starting_bankroll/1000:>5.0f}k  "
              f"{r.config.max_concurrent:>5d}  "
              f"${r.avg_final:>8,.0f}  "
              f"${r.median_final:>8,.0f}  "
              f"${r.p10_final:>8,.0f}  "
              f"${r.p90_final:>8,.0f}  "
              f"{r.avg_cycles:>6.0f}  "
              f"{r.avg_passes:>5.0f}  "
              f"{r.avg_fails:>5.0f}  "
              f"{r.avg_max_dd_pct:>5.1f}%  "
              f"${r.avg_max_dd_abs:>7,.0f}  "
              f"{d2x:>7s}  "
              f"{r.pct_reached_double:>4.0f}%  "
              f"{d100k:>6s}  "
              f"{r.pct_reached_100k:>4.0f}%")

    # --- Scaling analysis ---
    print(f"\n  {'=' * 126}")
    print("  SCALING ANALYSIS: TIME TO MILESTONES")
    print(f"  {'-' * 126}")

    for r in all_results:
        curves = r.all_curves  # (n_sims, horizon)

        milestones = []
        for target_label, target_val in [("2x", r.config.starting_bankroll * 2),
                                          ("5x", r.config.starting_bankroll * 5),
                                          ("10x", r.config.starting_bankroll * 10),
                                          ("$100k", 100_000),
                                          ("$250k", 250_000),
                                          ("$500k", 500_000)]:
            if target_val <= r.config.starting_bankroll:
                continue

            # For each sim, find first day bankroll >= target
            hit_days = []
            for sim_i in range(r.n_sims):
                hits = np.where(curves[sim_i] >= target_val)[0]
                if len(hits) > 0:
                    hit_days.append(hits[0] + 1)

            if hit_days:
                pct_hit = len(hit_days) / r.n_sims * 100
                med_days = np.median(hit_days)
                p10_days = np.percentile(hit_days, 10)
                p90_days = np.percentile(hit_days, 90)
                milestones.append((target_label, pct_hit, med_days, p10_days, p90_days))

        if milestones:
            print(f"\n  {r.label}:")
            print(f"    {'Target':>8s}  {'Hit%':>6s}  {'Median':>7s}  {'P10':>7s}  {'P90':>7s}")
            print(f"    {'-' * 40}")
            for tl, pct, med, p10, p90 in milestones:
                print(f"    {tl:>8s}  {pct:>5.0f}%  {med:>6.0f}d  {p10:>6.0f}d  {p90:>6.0f}d")

    # --- Capital efficiency comparison ---
    print(f"\n  {'=' * 126}")
    print("  CAPITAL EFFICIENCY (net profit / starting bankroll)")
    print(f"  {'-' * 126}")
    print(f"  {'Config':>22s}  {'Net Profit':>11s}  {'Med Net':>10s}  "
          f"{'ROI%':>7s}  {'Ann ROI%':>8s}  {'$/Day':>8s}")
    print(f"  {'-' * 80}")

    for r in all_results:
        years = r.config.horizon_days / 365
        ann_roi = (r.avg_net_profit / r.config.starting_bankroll) / years * 100
        per_day = r.avg_net_profit / r.config.horizon_days

        print(f"  {r.label:>22s}  "
              f"${r.avg_net_profit:>+10,.0f}  "
              f"${r.median_net_profit:>+9,.0f}  "
              f"{r.avg_net_profit / r.config.starting_bankroll * 100:>+6.0f}%  "
              f"{ann_roi:>+7.0f}%  "
              f"${per_day:>+7.0f}")

    # --- Best configs ---
    print(f"\n  {'=' * 126}")
    print("  OPTIMAL CONFIGURATIONS")
    print(f"  {'-' * 126}")

    # Group by starting bankroll
    for bankroll in sorted(set(c.starting_bankroll for c in CONFIGS)):
        group = [r for r in all_results if r.config.starting_bankroll == bankroll]
        best_profit = max(group, key=lambda r: r.avg_net_profit)
        best_speed = min(
            [r for r in group if r.pct_reached_double > 50],
            key=lambda r: r.median_days_to_double,
            default=None,
        )

        print(f"\n  ${bankroll/1000:.0f}k BANKROLL:")
        print(f"    Max profit:  {best_profit.label}  "
              f"-> ${best_profit.avg_net_profit:>+,.0f} net  |  "
              f"${best_profit.avg_net_profit / best_profit.config.horizon_days:>+.0f}/day")
        if best_speed:
            print(f"    Fastest 2x:  {best_speed.label}  "
                  f"-> {best_speed.median_days_to_double:.0f} days  |  "
                  f"{best_speed.pct_reached_double:.0f}% reach 2x")

    # --- Save CSV ---
    rows = []
    for r in all_results:
        years = r.config.horizon_days / 365
        rows.append({
            "config": r.label,
            "starting_bankroll": r.config.starting_bankroll,
            "max_concurrent": r.config.max_concurrent,
            "risk_pct": r.config.risk * 100,
            "horizon_days": r.config.horizon_days,
            "avg_final": round(r.avg_final, 2),
            "median_final": round(r.median_final, 2),
            "p10_final": round(r.p10_final, 2),
            "p90_final": round(r.p90_final, 2),
            "avg_passes": round(r.avg_passes, 1),
            "avg_fails": round(r.avg_fails, 1),
            "avg_cycles": round(r.avg_cycles, 1),
            "avg_max_dd_pct": round(r.avg_max_dd_pct, 2),
            "avg_max_dd_abs": round(r.avg_max_dd_abs, 2),
            "avg_net_profit": round(r.avg_net_profit, 2),
            "median_net_profit": round(r.median_net_profit, 2),
            "ann_roi_pct": round((r.avg_net_profit / r.config.starting_bankroll) / years * 100, 1),
            "profit_per_day": round(r.avg_net_profit / r.config.horizon_days, 2),
            "median_days_to_double": r.median_days_to_double,
            "pct_reached_double": round(r.pct_reached_double, 1),
            "median_days_to_100k": r.median_days_to_100k,
            "pct_reached_100k": round(r.pct_reached_100k, 1),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "deployment_engine.csv", index=False)

    # --- Save median growth curves ---
    curve_data = {}
    for r in all_results:
        curve_data[f"{r.label}_median"] = np.median(r.all_curves, axis=0)
        curve_data[f"{r.label}_p10"] = np.percentile(r.all_curves, 10, axis=0)
        curve_data[f"{r.label}_p90"] = np.percentile(r.all_curves, 90, axis=0)
    curve_df = pd.DataFrame(curve_data)
    curve_df.index.name = "day"
    curve_df.to_csv(OUTPUT_DIR / "deployment_growth_curves.csv")

    # --- Plot ---
    _plot_growth_curves(all_results, OUTPUT_DIR)

    elapsed = time.perf_counter() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 130 + "\n")


if __name__ == "__main__":
    main()
