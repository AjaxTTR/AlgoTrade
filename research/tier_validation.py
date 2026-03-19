"""
Tier-by-tier robustness validation.

Splits data into train (70%) and test (30%) periods, then evaluates each
signal tier independently to detect overfitting.

Usage:
    python -m research.tier_validation
"""

import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data_loader import load_csv
from engine.backtester import run as run_backtest

# Suppress noisy DD warnings during batch runs
logging.getLogger("engine.backtester").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Config (synced with main.py)
# ---------------------------------------------------------------------------
DATA_FILE = "data/nq_15m_data.csv"
STRATEGY_NAME = "first_hour_momentum"

BACKTEST_CONFIG = {
    "initial_capital": 100_000.0,
    "risk_per_trade": 0.005,
    "point_value": 20.0,
    "commission_per_side": 2.0,
    "slippage_points": 0.25,
    "use_trailing_stop": False,
    "daily_dd_limit": 0.02,
    "max_daily_risk": 0.02,
    "max_dd_limit": 0.0,
    "max_bars_in_trade": 8,
    "max_concurrent_trades": 1,
    "min_bars_between_entries": 2,
    "consec_loss_threshold": 2,
    "loss_scale_down": 0.5,
}

STRATEGY_CONFIG = {
    "session_start": "09:30",
    "session_end": "16:00",
    "fh_end": "10:30",
    "entry_cutoff": "15:45",
    "fh_percentile": 80.0,
    "atr_period": 14,
    "stop_atr_multiple": 1.5,
    "tp_atr_multiple": 2.0,
    "holding_bars": 8,
    "max_trades_per_day": 4,
    "pullback_atr_frac": 1.0,
    "min_bars_between_entries": 2,
    "enable_short": True,
}

TRAIN_FRAC = 0.70
TIER_NAMES = {
    1: "Long Early Entry (10:00)",
    2: "Long Standard Entry (10:30)",
    3: "Long Pullback Re-entry",
    4: "Long Midday Continuation",
    5: "Short Standard Entry (10:30)",
    6: "Short Pullback Re-entry",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_by_date(signals: pd.DataFrame, frac: float):
    """Chronological train/test split at the given fraction of trading days."""
    dates = signals.index.normalize().unique().sort_values()
    split_idx = int(len(dates) * frac)
    split_date = dates[split_idx]
    train = signals[signals.index < split_date].copy()
    test = signals[signals.index >= split_date].copy()
    return train, test, split_date


def _filter_tier(signals: pd.DataFrame, tier: int) -> pd.DataFrame:
    """Zero out signals that don't belong to the given tier."""
    out = signals.copy()
    mask = (out["signal"] != 0) & (out["signal_tier"] != tier)
    out.loc[mask, "signal"] = 0
    out.loc[mask, "stop_price"] = np.nan
    out.loc[mask, "tp_price"] = np.nan
    out.loc[mask, "size_factor"] = 1.0
    out.loc[mask, "signal_tier"] = 0
    return out


def _trade_stats(trades):
    """Compute summary stats from a list of TradeRecord."""
    if not trades:
        return {
            "n": 0, "win_rate": 0.0, "pf": 0.0, "expectancy": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "avg_pnl": 0.0,
            "total_pnl": 0.0, "sharpe_trade": 0.0,
            "max_win": 0.0, "max_loss": 0.0,
            "edge_ratio": 0.0,
        }
    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    n = len(pnls)
    wr = len(wins) / n * 100
    gp = wins.sum() if len(wins) else 0.0
    gl = abs(losses.sum()) if len(losses) else 0.0
    pf = gp / gl if gl > 0 else float("inf") if gp > 0 else 0.0
    avg_w = wins.mean() if len(wins) else 0.0
    avg_l = losses.mean() if len(losses) else 0.0
    exp = (wr / 100 * avg_w) + ((1 - wr / 100) * avg_l)

    maes = np.array([t.mae for t in trades])
    mfes = np.array([t.mfe for t in trades])
    avg_mae = maes.mean() if len(maes) else 0.0
    avg_mfe = mfes.mean() if len(mfes) else 0.0
    edge_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0.0

    sharpe_t = pnls.mean() / pnls.std() if pnls.std() > 0 else 0.0

    return {
        "n": n,
        "win_rate": round(wr, 1),
        "pf": round(pf, 2),
        "expectancy": round(exp, 2),
        "avg_win": round(avg_w, 2),
        "avg_loss": round(avg_l, 2),
        "avg_pnl": round(pnls.mean(), 2),
        "total_pnl": round(pnls.sum(), 2),
        "sharpe_trade": round(sharpe_t, 3),
        "max_win": round(pnls.max(), 2),
        "max_loss": round(pnls.min(), 2),
        "edge_ratio": round(edge_ratio, 3),
    }


def _degradation(train_val, test_val):
    """Percentage change from train to test (negative = degradation)."""
    if train_val == 0:
        return 0.0
    return (test_val - train_val) / abs(train_val) * 100


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load data
    print("\n" + "=" * 78)
    print("  TIER-BY-TIER ROBUSTNESS VALIDATION")
    print("=" * 78)

    try:
        df = load_csv(DATA_FILE)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  Error loading data: {exc}")
        sys.exit(1)

    # Generate signals
    strategy = importlib.import_module(f"strategies.{STRATEGY_NAME}")
    signals = strategy.generate_signals(df, **STRATEGY_CONFIG)

    # Train/test split
    train_sig, test_sig, split_date = _split_by_date(signals, TRAIN_FRAC)

    print(f"\n  Data range:  {signals.index[0].date()}  ->  {signals.index[-1].date()}")
    print(f"  Split date:  {split_date.date()}")
    print(f"  Train:       {train_sig.index[0].date()} -> {train_sig.index[-1].date()}"
          f"  ({len(train_sig):,} bars)")
    print(f"  Test:        {test_sig.index[0].date()} -> {test_sig.index[-1].date()}"
          f"  ({len(test_sig):,} bars)")

    # Count signals per tier
    all_signals = signals[signals["signal"] != 0]
    train_signals = train_sig[train_sig["signal"] != 0]
    test_signals = test_sig[test_sig["signal"] != 0]

    print(f"\n  Total signals: {len(all_signals)}"
          f"  (train: {len(train_signals)}, test: {len(test_signals)})")

    tiers_present = sorted(all_signals["signal_tier"].unique())
    for tier in tiers_present:
        n_train = (train_signals["signal_tier"] == tier).sum()
        n_test = (test_signals["signal_tier"] == tier).sum()
        name = TIER_NAMES.get(tier, f"Tier {tier}")
        print(f"    Tier {tier} ({name}):  train={n_train}  test={n_test}")

    # Run backtest per tier, per split
    results = {}  # {tier: {"train": stats, "test": stats}}

    for tier in tiers_present:
        train_filtered = _filter_tier(train_sig, tier)
        test_filtered = _filter_tier(test_sig, tier)

        train_result = run_backtest(train_filtered, **BACKTEST_CONFIG)
        test_result = run_backtest(test_filtered, **BACKTEST_CONFIG)

        results[tier] = {
            "train": _trade_stats(train_result.trades),
            "test": _trade_stats(test_result.trades),
        }

    # Also run combined (all tiers) for reference
    train_all = run_backtest(train_sig, **BACKTEST_CONFIG)
    test_all = run_backtest(test_sig, **BACKTEST_CONFIG)
    results["all"] = {
        "train": _trade_stats(train_all.trades),
        "test": _trade_stats(test_all.trades),
    }

    # -----------------------------------------------------------------------
    # Print detailed report
    # -----------------------------------------------------------------------
    metrics_to_show = [
        ("n",           "Trades",           "{:>6d}",    "{:>6d}"),
        ("win_rate",    "Win Rate %",       "{:>6.1f}",  "{:>6.1f}"),
        ("pf",          "Profit Factor",    "{:>6.2f}",  "{:>6.2f}"),
        ("expectancy",  "Expectancy ($)",   "{:>9.0f}",  "{:>9.0f}"),
        ("avg_pnl",     "Avg Trade ($)",    "{:>9.0f}",  "{:>9.0f}"),
        ("sharpe_trade","Trade Sharpe",     "{:>6.3f}",  "{:>6.3f}"),
        ("edge_ratio",  "Edge Ratio",       "{:>6.3f}",  "{:>6.3f}"),
        ("total_pnl",   "Total PnL ($)",    "{:>11,.0f}","{:>11,.0f}"),
        ("max_win",     "Max Win ($)",      "{:>9,.0f}", "{:>9,.0f}"),
        ("max_loss",    "Max Loss ($)",     "{:>9,.0f}", "{:>9,.0f}"),
    ]

    for tier in list(tiers_present) + ["all"]:
        if tier == "all":
            label = "ALL TIERS COMBINED"
        else:
            label = f"TIER {tier}: {TIER_NAMES.get(tier, '?')}"

        train_s = results[tier]["train"]
        test_s = results[tier]["test"]

        print(f"\n  {'-' * 74}")
        print(f"  {label}")
        print(f"  {'-' * 74}")
        print(f"  {'Metric':<20s} {'Train':>11s} {'Test':>11s} {'Degrad %':>10s}  Note")
        print(f"  {'-' * 74}")

        for key, label_str, fmt_train, fmt_test in metrics_to_show:
            tv = train_s[key]
            sv = test_s[key]

            train_str = fmt_train.format(tv)
            test_str = fmt_test.format(sv)

            # Degradation (skip for count fields and loss fields)
            if key in ("n", "max_win", "max_loss", "total_pnl"):
                deg_str = "       -"
                note = ""
            else:
                deg = _degradation(tv, sv)
                deg_str = f"{deg:>+8.1f}"
                if abs(deg) < 15:
                    note = "OK"
                elif abs(deg) < 30:
                    note = "WATCH"
                elif key in ("win_rate", "pf", "expectancy", "sharpe_trade", "edge_ratio") and deg < -30:
                    note = "OVERFIT?"
                else:
                    note = "CONCERN"

            print(f"  {label_str:<20s} {train_str:>11s} {test_str:>11s} {deg_str:>10s}  {note}")

    # -----------------------------------------------------------------------
    # Summary verdict
    # -----------------------------------------------------------------------
    print(f"\n  {'=' * 74}")
    print(f"  ROBUSTNESS VERDICT")
    print(f"  {'=' * 74}")

    flags = []
    for tier in tiers_present:
        name = TIER_NAMES.get(tier, f"Tier {tier}")
        tr = results[tier]["train"]
        te = results[tier]["test"]

        if tr["n"] < 10:
            flags.append(f"  Tier {tier} ({name}): INSUFFICIENT DATA — only {tr['n']} train trades")
            continue
        if te["n"] < 5:
            flags.append(f"  Tier {tier} ({name}): INSUFFICIENT DATA — only {te['n']} test trades")
            continue

        issues = []

        # Win rate degradation
        wr_deg = _degradation(tr["win_rate"], te["win_rate"])
        if wr_deg < -25:
            issues.append(f"win rate dropped {wr_deg:+.0f}%")

        # Profit factor degradation
        if tr["pf"] > 0 and tr["pf"] != float("inf"):
            pf_deg = _degradation(tr["pf"], te["pf"])
            if pf_deg < -30:
                issues.append(f"profit factor dropped {pf_deg:+.0f}%")

        # Expectancy sign flip
        if tr["expectancy"] > 0 and te["expectancy"] <= 0:
            issues.append("expectancy turned negative")

        # Edge ratio collapse
        if tr["edge_ratio"] > 0:
            er_deg = _degradation(tr["edge_ratio"], te["edge_ratio"])
            if er_deg < -40:
                issues.append(f"edge ratio collapsed {er_deg:+.0f}%")

        # Trade Sharpe
        if tr["sharpe_trade"] > 0:
            sh_deg = _degradation(tr["sharpe_trade"], te["sharpe_trade"])
            if sh_deg < -50:
                issues.append(f"trade Sharpe dropped {sh_deg:+.0f}%")

        if issues:
            flags.append(f"  Tier {tier} ({name}): POTENTIAL OVERFIT — {'; '.join(issues)}")
        else:
            verdict = "ROBUST" if te["expectancy"] > 0 else "MARGINAL (negative test expectancy)"
            flags.append(f"  Tier {tier} ({name}): {verdict}")

    for f in flags:
        print(f)

    # Overall
    all_tr = results["all"]["train"]
    all_te = results["all"]["test"]
    if all_te["expectancy"] > 0 and all_te["pf"] > 1.0:
        overall = "PASS — combined strategy holds on test data"
    elif all_te["expectancy"] > 0:
        overall = "MARGINAL — positive expectancy but weak profit factor on test"
    else:
        overall = "FAIL — negative expectancy on test data"

    print(f"\n  Overall: {overall}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
