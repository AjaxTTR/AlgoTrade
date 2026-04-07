"""
Risk frontier analysis for the prop-firm portfolio.

Sweeps risk_per_trade from 0.5% to 1.5% (0.1% increments) across the
locked 60/40 gap_momentum + regime_filtered portfolio.  Records pass rate,
median days to target, and failure rate for each level.

Usage:
    python -m research.risk_frontier
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
    BASE_BACKTEST_CONFIG,
    PROP_CONFIG,
    TOTAL_CAPITAL,
    _run_single_strategy,
    _build_portfolio_backtest_result,
)
from engine.prop_firm import simulate_prop_firm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_FILE = "data/nq_15m_data.csv"
OUTPUT_DIR = Path("output/portfolio")
N_SIMS = 5000
SEED = 42

# Fixed portfolio weights (locked config)
WEIGHTS = {
    "gap_momentum": 0.60,
    "regime_filtered_momentum": 0.40,
}

# Risk levels to sweep: 0.5% to 1.5% in 0.1% increments
RISK_LEVELS = [round(r / 1000, 4) for r in range(5, 16)]  # 0.005 .. 0.015


def _run_portfolio_at_risk(
    df: pd.DataFrame,
    risk_per_trade: float,
) -> tuple[pd.Series, pd.Series, list]:
    """Run the 2-strategy portfolio with uniform risk_per_trade."""
    ref_index = None
    strategy_results = {}
    capital_map = {}

    for name, weight in WEIGHTS.items():
        capital = TOTAL_CAPITAL * weight
        capital_map[name] = capital
        cat = STRATEGY_CATALOGUE[name]

        result, metrics, signals = _run_single_strategy(
            name, cat, df, capital, risk_per_trade,
        )
        strategy_results[name] = result
        if ref_index is None:
            ref_index = result.equity_mtm.index

    # Combine equity curves
    pf_mtm = pd.Series(TOTAL_CAPITAL, index=ref_index, dtype=np.float64)
    pf_closed = pd.Series(TOTAL_CAPITAL, index=ref_index, dtype=np.float64)

    for name, result in strategy_results.items():
        cap = capital_map[name]
        mtm_pnl = result.equity_mtm - cap
        closed_pnl = result.equity_closed - cap
        mtm_pnl = mtm_pnl.reindex(ref_index, method="ffill").fillna(0.0)
        closed_pnl = closed_pnl.reindex(ref_index, method="ffill").fillna(0.0)
        pf_mtm = pf_mtm + mtm_pnl
        pf_closed = pf_closed + closed_pnl

    all_trades = []
    for result in strategy_results.values():
        all_trades.extend(result.trades)
    all_trades.sort(key=lambda t: t.entry_time)

    return pf_mtm, pf_closed, all_trades


def main():
    t_start = time.perf_counter()

    log.info("Loading data from %s", DATA_FILE)
    try:
        df = load_csv(DATA_FILE)
    except FileNotFoundError:
        log.error("%s not found.", DATA_FILE)
        sys.exit(1)
    log.info("Loaded %s bars  |  %s -> %s", f"{len(df):,}",
             df.index[0], df.index[-1])

    results = []

    for risk in RISK_LEVELS:
        risk_pct = risk * 100
        log.info("Risk %.1f%% — running backtest + prop sim...", risk_pct)

        pf_mtm, pf_closed, all_trades = _run_portfolio_at_risk(df, risk)
        n_trades = len(all_trades)

        if n_trades < 10:
            log.warning("  Only %d trades at %.1f%% risk — skipping", n_trades, risk_pct)
            continue

        pf_result = _build_portfolio_backtest_result(pf_mtm, pf_closed, all_trades)
        prop = simulate_prop_firm(
            pf_result, config=PROP_CONFIG, n_simulations=N_SIMS, seed=SEED,
        )

        days_dist = prop.get("days_distribution", {})
        ci = prop["pass_rate_ci"]

        # Failure breakdown
        fail_reasons = prop.get("fail_reasons", {})
        max_dd_fails = fail_reasons.get("max_dd", 0)
        daily_dd_fails = fail_reasons.get("daily_dd", 0)
        other_fails = sum(v for k, v in fail_reasons.items()
                          if k not in ("max_dd", "daily_dd"))

        entry = {
            "risk_pct": risk_pct,
            "pass_rate": prop["pass_rate"],
            "pass_rate_ci_lo": ci[0],
            "pass_rate_ci_hi": ci[1],
            "fail_rate": prop["fail_rate"],
            "median_days": days_dist.get("ttp_p50", None),
            "p10_days": days_dist.get("ttp_p10", None),
            "p90_days": days_dist.get("ttp_p90", None),
            "n_trades": n_trades,
            "fail_max_dd": max_dd_fails,
            "fail_daily_dd": daily_dd_fails,
            "fail_other": other_fails,
            "classification": prop["classification"],
        }
        results.append(entry)

        log.info("  Pass: %.1f%%  |  Median days: %s  |  Trades: %d",
                 prop["pass_rate"],
                 f"{days_dist.get('ttp_p50', 'N/A'):.0f}"
                 if days_dist.get("ttp_p50") else "N/A",
                 n_trades)

    if not results:
        log.error("No valid results.")
        sys.exit(1)

    # Build results DataFrame
    df_results = pd.DataFrame(results)

    # Classify risk zones
    def _zone(row):
        if row["pass_rate"] >= 85:
            return "SAFE"
        elif row["pass_rate"] >= 65:
            return "BALANCED"
        else:
            return "AGGRESSIVE"

    df_results["zone"] = df_results.apply(_zone, axis=1)

    # Save CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "risk_frontier.csv"
    df_results.to_csv(csv_path, index=False)
    log.info("Results saved to %s", csv_path)

    # Print table
    print("\n" + "=" * 95)
    print("  RISK FRONTIER — PROP FIRM PASS RATE vs RISK PER TRADE")
    print("=" * 95)
    print(f"  Portfolio: 60% gap_momentum + 40% regime_filtered  |  "
          f"Target: +{PROP_CONFIG['profit_target']*100:.0f}%  |  "
          f"Max DD: {PROP_CONFIG['max_drawdown_limit']*100:.0f}%  |  "
          f"Sims: {N_SIMS:,}")
    print()
    print(f"  {'Risk%':>6s}  {'Pass%':>6s}  {'95% CI':>15s}  {'Fail%':>6s}  "
          f"{'Med Days':>8s}  {'P10':>5s}  {'P90':>5s}  "
          f"{'Trades':>6s}  {'MaxDD':>6s}  {'Zone':>12s}")
    print("  " + "-" * 91)

    for _, r in df_results.iterrows():
        med = f"{r['median_days']:.0f}" if pd.notna(r["median_days"]) else "N/A"
        p10 = f"{r['p10_days']:.0f}" if pd.notna(r["p10_days"]) else "N/A"
        p90 = f"{r['p90_days']:.0f}" if pd.notna(r["p90_days"]) else "N/A"
        ci_str = f"[{r['pass_rate_ci_lo']:.1f}-{r['pass_rate_ci_hi']:.1f}]"
        max_dd_pct = r["fail_max_dd"] / N_SIMS * 100

        print(f"  {r['risk_pct']:>5.1f}%  {r['pass_rate']:>5.1f}%  {ci_str:>15s}  "
              f"{r['fail_rate']:>5.1f}%  {med:>8s}  {p10:>5s}  {p90:>5s}  "
              f"{r['n_trades']:>6.0f}  {max_dd_pct:>5.1f}%  {r['zone']:>12s}")

    # Zone summary
    print("\n  " + "-" * 91)
    print("  ZONE DEFINITIONS:")
    print("    SAFE:        pass >= 85%  (high confidence, slower)")
    print("    BALANCED:    pass 65-84%  (moderate risk/speed tradeoff)")
    print("    AGGRESSIVE:  pass < 65%   (fastest, but high failure risk)")

    # Identify transitions
    safe = df_results[df_results["zone"] == "SAFE"]
    balanced = df_results[df_results["zone"] == "BALANCED"]
    aggressive = df_results[df_results["zone"] == "AGGRESSIVE"]

    print("\n  KEY LEVELS:")
    if not safe.empty:
        best_safe = safe.loc[safe["median_days"].idxmin()] if safe["median_days"].notna().any() else safe.iloc[0]
        print(f"    Best SAFE:       {best_safe['risk_pct']:.1f}% risk  "
              f"-> {best_safe['pass_rate']:.1f}% pass, "
              f"{best_safe['median_days']:.0f} median days")
    if not balanced.empty:
        mid = balanced.iloc[len(balanced) // 2]
        print(f"    Mid BALANCED:    {mid['risk_pct']:.1f}% risk  "
              f"-> {mid['pass_rate']:.1f}% pass, "
              f"{mid['median_days']:.0f} median days")
    if not aggressive.empty:
        fastest = aggressive.loc[aggressive["median_days"].idxmin()] if aggressive["median_days"].notna().any() else aggressive.iloc[0]
        print(f"    Fastest AGGR:    {fastest['risk_pct']:.1f}% risk  "
              f"-> {fastest['pass_rate']:.1f}% pass, "
              f"{fastest['median_days']:.0f} median days")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
