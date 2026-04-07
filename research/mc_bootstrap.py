"""
Monte Carlo bootstrap expansion for the prop-firm portfolio.

Goes beyond the standard MC (trade-order shuffling) by adding:
  1. Bootstrap resampling — sample trades WITH replacement to test
     sensitivity to the specific trade population (not just ordering).
  2. Block bootstrap — sample contiguous trade blocks to preserve
     serial correlation in win/loss streaks.
  3. Subsample stress test — run prop sim on random 50%/70%/90% of
     trades to test edge fragility.

All tests use the locked portfolio config (1.4% risk, 60/40 split).

Usage:
    python -m research.mc_bootstrap
"""

import importlib
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest, BacktestResult
from engine.prop_firm import _simulate_single, _wilson_ci

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("engine.backtester").setLevel(logging.ERROR)

DATA_FILE = "data/nq_15m_data.csv"
OUTPUT_DIR = Path("output/portfolio")

TOTAL_CAPITAL = 100_000.0

STRATEGY_CATALOGUE = {
    "gap_momentum": {
        "module": "strategies.gap_momentum",
        "strategy_params": {
            "fh_percentile": 75.0,
            "gap_threshold_pct": 0.10,
        },
        "backtest_overrides": {
            "stop_atr_multiple": 1.0,
            "tp_atr_multiple": 2.0,
        },
    },
    "regime_filtered_momentum": {
        "module": "strategies.regime_filtered_momentum",
        "strategy_params": {
            "fh_percentile": 75.0,
            "fh_dead_zone_lo": 0.50,
            "fh_dead_zone_hi": 0.75,
            "trend_20d_max": 1.0,
            "vol_filter": True,
        },
        "backtest_overrides": {
            "stop_atr_multiple": 1.0,
            "tp_atr_multiple": 2.0,
        },
    },
}

WEIGHTS = {"gap_momentum": 0.60, "regime_filtered_momentum": 0.40}
RISK_PER_TRADE = 0.014

BASE_BACKTEST_CONFIG = {
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "use_trailing_stop": False,
    "daily_dd_limit": 0.02,
    "max_daily_risk": 0.02,
    "max_dd_limit": 0.0,
    "max_bars_in_trade": 0,
    "max_concurrent_trades": 1,
    "min_bars_between_entries": 2,
    "consec_loss_threshold": 2,
    "loss_scale_down": 0.5,
    "max_risk_per_trade": 0.0,
}

PROP_CONFIG = {
    "starting_balance": TOTAL_CAPITAL,
    "profit_target": 0.10,
    "daily_drawdown_limit": 0.05,
    "max_drawdown_limit": 0.10,
    "max_days": 0,
}

N_SIMS = 5000


# ---------------------------------------------------------------------------
# Portfolio runner
# ---------------------------------------------------------------------------

def _run_portfolio(df):
    """Run both strategies and return combined trades."""
    all_trades = []
    for name, weight in WEIGHTS.items():
        capital = TOTAL_CAPITAL * weight
        cat = STRATEGY_CATALOGUE[name]
        mod = importlib.import_module(cat["module"])
        signals = mod.generate_signals(df.copy(), **cat["strategy_params"])
        bt_cfg = {
            **BASE_BACKTEST_CONFIG,
            **cat.get("backtest_overrides", {}),
            "risk_per_trade": RISK_PER_TRADE,
            "initial_capital": capital,
            "strategy_name": name,
        }
        result = run_backtest(signals, **bt_cfg)
        all_trades.extend(result.trades)
    all_trades.sort(key=lambda t: t.entry_time)
    return all_trades


# ---------------------------------------------------------------------------
# MC methods
# ---------------------------------------------------------------------------

def _run_mc_shuffle(pnls, dates_template, config, rng, n_sims):
    """Standard MC: shuffle trade order, keep same trades."""
    results = []
    for _ in range(n_sims):
        idx = rng.permutation(len(pnls))
        results.append(_simulate_single(pnls[idx], dates_template, config))
    return results


def _run_mc_bootstrap(pnls, dates_template, config, rng, n_sims):
    """Bootstrap: sample N trades WITH replacement from the population."""
    n = len(pnls)
    results = []
    for _ in range(n_sims):
        idx = rng.choice(n, size=n, replace=True)
        results.append(_simulate_single(pnls[idx], dates_template, config))
    return results


def _run_mc_block_bootstrap(pnls, dates_template, config, rng, n_sims, block_size=10):
    """Block bootstrap: sample contiguous blocks WITH replacement.

    Preserves serial correlation (win/loss streaks).
    """
    n = len(pnls)
    n_blocks = max(1, n // block_size)
    target_len = n_blocks * block_size

    results = []
    for _ in range(n_sims):
        # Sample block start indices with replacement
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        indices = np.concatenate([np.arange(s, s + block_size) for s in starts])
        indices = indices[:target_len]
        # Use matching dates template (truncated to same length)
        sim_dates = dates_template[:len(indices)]
        results.append(_simulate_single(pnls[indices], sim_dates, config))
    return results


def _run_mc_subsample(pnls, dates_template, config, rng, n_sims, frac):
    """Subsample: randomly pick frac% of trades (without replacement)."""
    n = len(pnls)
    k = max(2, int(n * frac))
    results = []
    for _ in range(n_sims):
        idx = rng.choice(n, size=k, replace=False)
        idx.sort()
        sim_dates = dates_template[:k]
        results.append(_simulate_single(pnls[idx], sim_dates, config))
    return results


def _summarize(sims, label):
    """Summarize MC results into a dict."""
    n = len(sims)
    n_pass = sum(1 for s in sims if s["outcome"] == "PASS")
    pass_rate = n_pass / n * 100
    ci = _wilson_ci(n_pass, n)

    pass_days = [s["days_elapsed"] for s in sims if s["outcome"] == "PASS"]
    med_days = float(np.median(pass_days)) if pass_days else 0.0

    peak_dds = [s["peak_dd_pct"] for s in sims]
    med_dd = float(np.median(peak_dds))
    p95_dd = float(np.percentile(peak_dds, 95))

    fail_reasons = {}
    for s in sims:
        if s["outcome"] == "FAIL":
            r = s["reason"]
            fail_reasons[r] = fail_reasons.get(r, 0) + 1

    return {
        "method": label,
        "n_sims": n,
        "pass_rate": round(pass_rate, 2),
        "ci_lo": ci[0],
        "ci_hi": ci[1],
        "med_days": round(med_days, 1),
        "med_dd": round(med_dd, 2),
        "p95_dd": round(p95_dd, 2),
        "fail_max_dd": fail_reasons.get("max_dd", 0),
        "fail_daily_dd": fail_reasons.get("daily_dd", 0),
        "fail_exhausted": fail_reasons.get("trades_exhausted", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()

    log.info("Loading data from %s", DATA_FILE)
    df = load_csv(DATA_FILE)
    log.info("Loaded %d bars", len(df))

    trades = _run_portfolio(df)
    n_trades = len(trades)
    log.info("Portfolio trades: %d", n_trades)

    pnls = np.array([t.pnl for t in trades])
    exit_dates = np.array([t.exit_time.date() for t in trades])

    # Build date template
    unique_dates = np.unique(exit_dates)
    trades_per_day = np.array([np.sum(exit_dates == d) for d in unique_dates])
    day_boundaries = np.concatenate([[0], np.cumsum(trades_per_day)])
    dates_template = np.empty(len(pnls), dtype=exit_dates.dtype)
    for d_idx, d in enumerate(unique_dates):
        s, e = day_boundaries[d_idx], day_boundaries[d_idx + 1]
        dates_template[s:e] = d

    rng = np.random.default_rng(42)
    all_results = []

    # 1. Standard MC (baseline)
    log.info("Running standard MC shuffle (%d sims)...", N_SIMS)
    sims = _run_mc_shuffle(pnls, dates_template, PROP_CONFIG, rng, N_SIMS)
    all_results.append(_summarize(sims, "Shuffle (standard)"))

    # 2. Bootstrap resampling
    log.info("Running bootstrap resampling (%d sims)...", N_SIMS)
    sims = _run_mc_bootstrap(pnls, dates_template, PROP_CONFIG, rng, N_SIMS)
    all_results.append(_summarize(sims, "Bootstrap (with replacement)"))

    # 3. Block bootstrap (block=5, 10, 20)
    for bs in [5, 10, 20]:
        log.info("Running block bootstrap (block=%d, %d sims)...", bs, N_SIMS)
        sims = _run_mc_block_bootstrap(pnls, dates_template, PROP_CONFIG, rng, N_SIMS, block_size=bs)
        all_results.append(_summarize(sims, f"Block bootstrap (k={bs})"))

    # 4. Subsample stress tests
    for frac in [0.50, 0.70, 0.90]:
        label = f"Subsample {int(frac*100)}% ({int(n_trades*frac)} trades)"
        log.info("Running %s (%d sims)...", label, N_SIMS)
        sims = _run_mc_subsample(pnls, dates_template, PROP_CONFIG, rng, N_SIMS, frac)
        all_results.append(_summarize(sims, label))

    results_df = pd.DataFrame(all_results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "mc_bootstrap.csv"
    results_df.to_csv(out_path, index=False)
    log.info("Results saved to %s", out_path)

    # =========================================================================
    # Print results
    # =========================================================================
    elapsed = time.perf_counter() - t_start

    print("\n" + "=" * 110)
    print("  MONTE CARLO BOOTSTRAP EXPANSION -- PROP-FIRM PORTFOLIO")
    print("=" * 110)
    print(f"  Config: 60% gap_momentum + 40% regime_filtered | risk=1.4% | "
          f"stop=1.0x ATR | TP=2.0x ATR")
    print(f"  Total trades: {n_trades} | Simulations per method: {N_SIMS}")
    print(f"  Trade stats: {(pnls>0).sum()} wins ({(pnls>0).mean()*100:.1f}%), "
          f"avg win=${pnls[pnls>0].mean():,.0f}, avg loss=${pnls[pnls<=0].mean():,.0f}")

    print(f"\n  {'Method':<40} {'Pass%':>6} {'CI':>16} {'MedDays':>8} "
          f"{'MedDD%':>7} {'P95DD%':>7} {'MaxDD':>6} {'DlyDD':>6} {'Exhaust':>8}")
    print("  " + "-" * 108)

    for _, r in results_df.iterrows():
        ci_str = f"[{r['ci_lo']:.1f}-{r['ci_hi']:.1f}]"
        print(f"  {r['method']:<40} {r['pass_rate']:>5.1f}% {ci_str:>16} "
              f"{r['med_days']:>8.1f} {r['med_dd']:>6.2f}% {r['p95_dd']:>6.2f}% "
              f"{r['fail_max_dd']:>6.0f} {r['fail_daily_dd']:>6.0f} {r['fail_exhausted']:>8.0f}")

    # Interpretation
    print("\n" + "=" * 110)
    print("  INTERPRETATION")
    print("=" * 110)

    baseline = all_results[0]
    bootstrap = all_results[1]

    spread = abs(baseline["pass_rate"] - bootstrap["pass_rate"])
    if spread < 5:
        print(f"\n  Shuffle vs Bootstrap spread: {spread:.1f}pp -> STABLE")
        print("    Edge is not dependent on specific trade population.")
    elif spread < 15:
        print(f"\n  Shuffle vs Bootstrap spread: {spread:.1f}pp -> MODERATE sensitivity")
        print("    Some dependence on specific trades, but within acceptable range.")
    else:
        print(f"\n  Shuffle vs Bootstrap spread: {spread:.1f}pp -> HIGH sensitivity")
        print("    Edge heavily depends on a few outlier trades. Fragile.")

    # Block bootstrap check: does preserving streaks hurt?
    block_rates = [r["pass_rate"] for r in all_results if "Block" in r["method"]]
    if block_rates:
        avg_block = np.mean(block_rates)
        block_spread = abs(baseline["pass_rate"] - avg_block)
        if block_spread < 5:
            print(f"\n  Shuffle vs Block Bootstrap spread: {block_spread:.1f}pp -> NO streak sensitivity")
            print("    Win/loss clustering does not meaningfully impact pass rate.")
        else:
            print(f"\n  Shuffle vs Block Bootstrap spread: {block_spread:.1f}pp -> STREAK SENSITIVE")
            print("    Trade ordering and streak patterns matter for pass rate.")

    # Subsample degradation
    sub_results = [r for r in all_results if "Subsample" in r["method"]]
    if sub_results:
        sub_90 = next((r for r in sub_results if "90%" in r["method"]), None)
        sub_50 = next((r for r in sub_results if "50%" in r["method"]), None)
        if sub_90 and sub_50:
            print(f"\n  Subsample degradation: 100% -> 90% -> 50%")
            print(f"    {baseline['pass_rate']:.1f}% -> {sub_90['pass_rate']:.1f}% -> {sub_50['pass_rate']:.1f}%")
            if sub_50["pass_rate"] > 30:
                print("    Edge survives aggressive subsampling. Robust.")
            else:
                print("    Edge degrades significantly with fewer trades. Frequency-dependent.")

    print(f"\n  Runtime: {elapsed:.1f}s")
    print("=" * 110 + "\n")

    return results_df


if __name__ == "__main__":
    main()
