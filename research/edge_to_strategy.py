"""
Convert statistical edges into backtestable strategy signals.

Reads edge_analysis_results.csv (or a DataFrame from build_summary /
rank_edges), translates each qualifying edge into concrete entry rules,
and generates a combined signal DataFrame compatible with the backtester.

Usage:
    python -m research.edge_to_strategy [--top N] [--min-score 0.5]

The output is a strategies-compatible generate_signals function that can
be passed directly to engine.backtester.run().
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EDGE_RESULTS_PATH = Path("output/edge_analysis/edge_analysis_results.csv")
RANKED_EDGES_PATH = Path("output/edge_analysis/ranked_edges.csv")

# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

@dataclass
class StrategyRule:
    """A single tradable rule derived from a statistical edge."""
    name: str
    analysis: str           # source analysis (time_of_day_bias, day_of_week, etc.)
    key: str                # edge identifier (e.g. "14:45", "Monday")
    direction: int          # 1 = long, -1 = short
    mean_return: float      # expected return from edge analysis
    t_stat: float
    p_adj: float
    stability: float
    sample_size: int
    entry_condition: str    # human-readable condition description
    filters: dict           # machine-readable filter parameters


# ---------------------------------------------------------------------------
# Edge parsers — one per analysis type
# ---------------------------------------------------------------------------

def _parse_time_of_day(row: dict) -> StrategyRule | None:
    """Convert a time-of-day edge into a rule."""
    key = row["key"]
    mean_ret = row["mean_return"]

    # key is a time slot like "14:45" or "09:30"
    parts = key.split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None

    direction = 1 if mean_ret > 0 else -1
    dir_label = "LONG" if direction == 1 else "SHORT"

    # Entry: at the start of this time slot
    # Exit: at the end of the bar (next bar open)
    entry_time = f"{hour:02d}:{minute:02d}"

    return StrategyRule(
        name=f"tod_{entry_time}_{dir_label.lower()}",
        analysis="time_of_day_bias",
        key=key,
        direction=direction,
        mean_return=mean_ret,
        t_stat=row.get("t_stat", 0),
        p_adj=row.get("p_adj", 1.0),
        stability=row.get("stability", 0),
        sample_size=int(row.get("sample_size", 0)),
        entry_condition=f"{dir_label} at {entry_time} bar open",
        filters={
            "entry_time": entry_time,
            "hold_bars": 1,
        },
    )


def _parse_day_of_week(row: dict) -> StrategyRule | None:
    """Convert a day-of-week edge into a rule."""
    key = row["key"]
    mean_ret = row["mean_return"]

    dow_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4,
    }
    if key not in dow_map:
        return None

    direction = 1 if mean_ret > 0 else -1
    dir_label = "LONG" if direction == 1 else "SHORT"

    return StrategyRule(
        name=f"dow_{key.lower()}_{dir_label.lower()}",
        analysis="day_of_week",
        key=key,
        direction=direction,
        mean_return=mean_ret,
        t_stat=row.get("t_stat", 0),
        p_adj=row.get("p_adj", 1.0),
        stability=row.get("stability", 0),
        sample_size=int(row.get("sample_size", 0)),
        entry_condition=f"{dir_label} at {key} open, hold full session",
        filters={
            "day_of_week": dow_map[key],
            "hold_bars": None,  # session-length hold
        },
    )


def _parse_volatility_regime(row: dict) -> StrategyRule | None:
    """Convert a volatility regime edge into a rule."""
    key = row["key"]
    mean_ret = row["mean_return"]

    valid_regimes = {"Low", "Medium", "High"}
    if key not in valid_regimes:
        return None

    direction = 1 if mean_ret > 0 else -1
    dir_label = "LONG" if direction == 1 else "SHORT"

    return StrategyRule(
        name=f"vol_{key.lower()}_{dir_label.lower()}",
        analysis="volatility_regime",
        key=key,
        direction=direction,
        mean_return=mean_ret,
        t_stat=row.get("t_stat", 0),
        p_adj=row.get("p_adj", 1.0),
        stability=row.get("stability", 0),
        sample_size=int(row.get("sample_size", 0)),
        entry_condition=f"{dir_label} when ATR regime is {key}",
        filters={
            "vol_regime": key,
            "hold_bars": 1,
        },
    )


def _parse_interaction(row: dict) -> StrategyRule | None:
    """Convert an interaction edge into a rule.

    Interaction keys follow the pattern "FactorA × FactorB".
    """
    key = row["key"]
    mean_ret = row["mean_return"]

    if " × " not in key:
        return None

    factor_a, factor_b = key.split(" × ", 1)
    direction = 1 if mean_ret > 0 else -1
    dir_label = "LONG" if direction == 1 else "SHORT"

    # Detect interaction type from factor names
    filters = {"hold_bars": 1}

    # Time × Vol: e.g. "14-16 × Low_Vol"
    time_blocks = {"09:30-10", "10-12", "12-14", "14-16"}
    vol_regimes = {"Low_Vol", "Med_Vol", "High_Vol"}
    dow_names = {"Mon", "Tue", "Wed", "Thu", "Fri"}
    fh_dirs = {"FH_Down", "FH_Flat", "FH_Up"}
    rest_dirs = {"Rest_Down", "Rest_Flat", "Rest_Up"}

    if factor_a in time_blocks and factor_b in vol_regimes:
        filters["time_block"] = factor_a
        filters["vol_regime"] = factor_b
        condition = f"{dir_label} during {factor_a} when vol={factor_b}"

    elif factor_a in dow_names and factor_b in vol_regimes:
        dow_full_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
        filters["day_of_week"] = dow_full_map[factor_a]
        filters["vol_regime"] = factor_b
        condition = f"{dir_label} on {factor_a} when vol={factor_b}"

    elif factor_a in fh_dirs and factor_b == "rest_return":
        # First-hour direction predicts rest-of-day
        filters["fh_direction"] = factor_a
        filters["hold_bars"] = None  # rest-of-day hold
        condition = f"{dir_label} after {factor_a}, hold rest of day"

    elif factor_a in fh_dirs and factor_b in rest_dirs:
        # Full grid: FH_dir × rest_dir — daily return
        filters["fh_direction"] = factor_a
        filters["rest_direction"] = factor_b
        filters["hold_bars"] = None
        condition = f"{dir_label} when {factor_a} and {factor_b}"

    else:
        condition = f"{dir_label} when {factor_a} and {factor_b}"
        filters["factor_a"] = factor_a
        filters["factor_b"] = factor_b

    safe_key = key.replace(" × ", "_x_").replace(" ", "_").replace("-", "")
    return StrategyRule(
        name=f"int_{safe_key}_{dir_label.lower()}",
        analysis="interactions",
        key=key,
        direction=direction,
        mean_return=mean_ret,
        t_stat=row.get("t_stat", 0),
        p_adj=row.get("p_adj", 1.0),
        stability=row.get("stability", 0),
        sample_size=int(row.get("sample_size", 0)),
        entry_condition=condition,
        filters=filters,
    )


_PARSERS = {
    "time_of_day_bias": _parse_time_of_day,
    "day_of_week": _parse_day_of_week,
    "volatility_regime": _parse_volatility_regime,
    "interactions": _parse_interaction,
    "autocorrelation_by_time": _parse_time_of_day,
}


# ---------------------------------------------------------------------------
# Edge → Rules conversion
# ---------------------------------------------------------------------------

def edges_to_rules(
    edges_df: pd.DataFrame,
    min_p_adj: float = 0.05,
    min_stability: float = 0.0,
    min_sample_size: int = 50,
) -> list[StrategyRule]:
    """Convert qualifying edges from a summary DataFrame into strategy rules.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Edge analysis summary with columns: analysis, key, mean_return,
        sample_size, t_stat, p_value, p_adj, stability.
    min_p_adj : float
        Maximum adjusted p-value to include (default 0.05).
    min_stability : float
        Minimum stability score (default 0.0).
    min_sample_size : int
        Minimum observations (default 50).

    Returns
    -------
    list[StrategyRule]
        Parsed rules ready for signal generation.
    """
    rules = []

    for _, row in edges_df.iterrows():
        # Apply filters
        p = row.get("p_adj", row.get("p_value", 1.0))
        if pd.isna(p):
            p = row.get("p_value", 1.0)
        if p > min_p_adj:
            continue

        stab = row.get("stability", 0)
        if pd.notna(stab) and stab < min_stability:
            continue

        n = row.get("sample_size", 0)
        if n < min_sample_size:
            continue

        # Dispatch to the right parser
        analysis = row.get("analysis", "")
        parser = _PARSERS.get(analysis)
        if parser is None:
            log.debug("No parser for analysis type: %s", analysis)
            continue

        rule = parser(row.to_dict())
        if rule is not None:
            rules.append(rule)

    log.info("Converted %d edges into %d rules", len(edges_df), len(rules))
    return rules


# ---------------------------------------------------------------------------
# Signal generation from rules
# ---------------------------------------------------------------------------

def _add_derived_columns(df: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Add columns needed by rule filters (ATR, vol regime, time features)."""
    out = df.copy()

    # ATR
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift(1)).abs(),
        (out["low"] - out["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

    # Rolling vol for regime classification
    out["returns"] = out["close"].pct_change()
    out["rolling_vol_20"] = out["returns"].rolling(20).std()

    # Time features
    out["hour"] = out.index.hour
    out["minute"] = out.index.minute
    out["mins_since_midnight"] = out["hour"] * 60 + out["minute"]
    out["day_of_week"] = out.index.dayofweek
    out["date"] = out.index.date

    # Time blocks (matching edge_analysis convention)
    out["time_block"] = out["hour"].map({
        9: "09:30-10", 10: "10-12", 11: "10-12",
        12: "12-14", 13: "12-14", 14: "14-16", 15: "14-16",
    })

    return out


def _classify_vol_regime(df: pd.DataFrame) -> pd.Series:
    """Classify each bar into a volatility regime using rolling vol terciles.

    Uses an expanding tercile boundary to avoid lookahead: at each bar,
    the regime is based on percentile rank within the prior 100 bars.
    """
    vol = df["rolling_vol_20"]
    regime = pd.Series("Med_Vol", index=df.index, dtype=str)

    # Use rolling percentile rank (no lookahead)
    pct_rank = vol.rolling(100, min_periods=50).rank(pct=True)
    regime[pct_rank <= 0.333] = "Low_Vol"
    regime[pct_rank >= 0.667] = "High_Vol"

    return regime


def _compute_first_hour_return(df: pd.DataFrame) -> pd.Series:
    """Compute first-hour return direction per day.

    Returns a per-bar Series with values "FH_Up", "FH_Flat", "FH_Down"
    forward-filled across the rest of the day.
    """
    rth = df.between_time("09:30", "16:00")
    first_hour = rth.between_time("09:30", "10:30")

    fh_daily = first_hour.groupby("date").agg(
        fh_open=("open", "first"), fh_close=("close", "last"),
    )
    fh_daily["fh_return"] = fh_daily["fh_close"] / fh_daily["fh_open"] - 1

    # Classify into terciles using expanding quantiles (no lookahead)
    fh_ret = fh_daily["fh_return"]
    expanding_q33 = fh_ret.expanding(min_periods=20).quantile(0.333)
    expanding_q67 = fh_ret.expanding(min_periods=20).quantile(0.667)

    fh_dir = pd.Series("FH_Flat", index=fh_daily.index)
    fh_dir[fh_ret < expanding_q33] = "FH_Down"
    fh_dir[fh_ret > expanding_q67] = "FH_Up"

    # Map back to bar level
    bar_fh_dir = df["date"].map(fh_dir)
    return bar_fh_dir


def _eval_rule(rule: StrategyRule, df: pd.DataFrame) -> np.ndarray:
    """Evaluate a single rule against the data. Returns signal array (1/-1/0)."""
    n = len(df)
    signals = np.zeros(n, dtype=np.int8)
    f = rule.filters

    # Build the condition mask
    mask = pd.Series(True, index=df.index)

    # Time-of-day filter
    if "entry_time" in f:
        h, m = (int(x) for x in f["entry_time"].split(":"))
        target_mins = h * 60 + m
        mask = mask & (df["mins_since_midnight"] == target_mins)

    # Day-of-week filter
    if "day_of_week" in f:
        mask = mask & (df["day_of_week"] == f["day_of_week"])

    # Volatility regime filter
    if "vol_regime" in f:
        mask = mask & (df["vol_regime"] == f["vol_regime"])

    # Time block filter
    if "time_block" in f:
        mask = mask & (df["time_block"] == f["time_block"])

    # First-hour direction filter
    if "fh_direction" in f:
        mask = mask & (df["fh_direction"] == f["fh_direction"])
        # Only enter after first hour (10:30 onwards)
        mask = mask & (df["mins_since_midnight"] >= 630)  # 10:30

    # Apply direction
    signals[mask.values] = rule.direction

    return signals


def generate_signals(
    df: pd.DataFrame,
    rules: list[StrategyRule],
    atr_period: int = 14,
    tp_atr_multiple: float = 1.5,
    session_start: str = "09:30",
    session_end: str = "16:00",
    max_trades_per_day: int = 3,
    **kwargs,
) -> pd.DataFrame:
    """Generate combined signals from a list of edge-derived rules.

    This is the backtester-compatible interface matching the contract:
        generate_signals(df, **params) -> DataFrame with signal, stop_price columns.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    rules : list[StrategyRule]
        Rules from edges_to_rules().
    atr_period : int
        ATR lookback period.
    tp_atr_multiple : float
        ATR multiplier for take-profit.
    session_start, session_end : str
        Session boundaries in HH:MM.
    max_trades_per_day : int
        Maximum signals per day.

    Returns
    -------
    pd.DataFrame
        DataFrame with signal, stop_price, tp_price, session_close columns.
    """
    out = _add_derived_columns(df, atr_period=atr_period)

    # Add regime classification
    out["vol_regime"] = _classify_vol_regime(out)

    # Add first-hour direction (for interaction rules that need it)
    has_fh_rules = any("fh_direction" in r.filters for r in rules)
    if has_fh_rules:
        out["fh_direction"] = _compute_first_hour_return(out)

    # Evaluate each rule and combine via priority (strongest t-stat first)
    n = len(out)
    combined = np.zeros(n, dtype=np.int8)
    rule_source = np.full(n, "", dtype=object)

    # Sort rules by |t_stat| descending so strongest edges take priority
    sorted_rules = sorted(rules, key=lambda r: abs(r.t_stat), reverse=True)

    for rule in sorted_rules:
        rule_signals = _eval_rule(rule, out)
        # Only fill where combined is still 0 (first-writer-wins)
        new_mask = (combined == 0) & (rule_signals != 0)
        combined[new_mask] = rule_signals[new_mask]
        rule_source[new_mask] = rule.name

    # Restrict to RTH session
    ss_h, ss_m = (int(x) for x in session_start.split(":"))
    se_h, se_m = (int(x) for x in session_end.split(":"))
    ss_mins = ss_h * 60 + ss_m
    se_mins = se_h * 60 + se_m
    outside_session = (out["mins_since_midnight"] < ss_mins) | (out["mins_since_midnight"] >= se_mins)
    combined[outside_session.values] = 0

    # Cap trades per day
    dates = out["date"]
    any_signal = combined != 0
    daily_cum = pd.Series(any_signal, index=out.index).groupby(dates).cumsum()
    combined[daily_cum.values > max_trades_per_day] = 0

    out["signal"] = combined
    out["rule_source"] = rule_source

    # Stop price: ATR-based stop (1x ATR from entry)
    atr_vals = out["atr"].values
    stop_price = np.full(n, np.nan)
    long_mask = combined == 1
    short_mask = combined == -1
    close_vals = out["close"].values
    stop_price[long_mask] = close_vals[long_mask] - atr_vals[long_mask]
    stop_price[short_mask] = close_vals[short_mask] + atr_vals[short_mask]
    out["stop_price"] = stop_price

    # Take-profit: ATR-based
    tp_price = np.full(n, np.nan)
    tp_price[long_mask] = close_vals[long_mask] + atr_vals[long_mask] * tp_atr_multiple
    tp_price[short_mask] = close_vals[short_mask] - atr_vals[short_mask] * tp_atr_multiple
    out["tp_price"] = tp_price

    # Session close marker
    bar_mins = out["mins_since_midnight"]
    past_end = bar_mins >= se_mins
    prev_past = past_end.shift(1, fill_value=False)
    dates_arr = np.array(dates)
    prev_date = pd.Series(dates_arr, index=out.index).shift(1)
    new_day = pd.Series(dates_arr, index=out.index) != prev_date
    out["session_close"] = past_end & (~prev_past | new_day)

    # Clean up temp columns
    drop_cols = [
        "returns", "rolling_vol_20", "hour", "minute",
        "mins_since_midnight", "day_of_week", "date", "time_block",
        "vol_regime", "rule_source",
    ]
    if "fh_direction" in out.columns:
        drop_cols.append("fh_direction")
    out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore", inplace=True)

    return out


# ---------------------------------------------------------------------------
# Rule summary
# ---------------------------------------------------------------------------

def print_rules(rules: list[StrategyRule]) -> None:
    """Pretty-print strategy rules."""
    print("\n" + "=" * 90)
    print("  EDGE-DERIVED STRATEGY RULES")
    print("=" * 90)
    print(f"  Total rules: {len(rules)}\n")

    for i, r in enumerate(rules, 1):
        dir_str = "LONG" if r.direction == 1 else "SHORT"
        stab_str = f"{r.stability:.2f}" if r.stability else "N/A"
        print(f"  #{i:2d}  [{r.analysis:25s}]  {r.name}")
        print(f"       Direction:  {dir_str}")
        print(f"       Condition:  {r.entry_condition}")
        print(f"       Edge:       mean={r.mean_return*10000:+.2f}bps  "
              f"t={r.t_stat:+.2f}  q={r.p_adj:.4f}  "
              f"n={r.sample_size}  stab={stab_str}")
        print(f"       Filters:    {r.filters}")
        print()

    print("=" * 90 + "\n")


def rules_to_dataframe(rules: list[StrategyRule]) -> pd.DataFrame:
    """Convert rules to a DataFrame for saving/inspection."""
    rows = []
    for r in rules:
        rows.append({
            "name": r.name,
            "analysis": r.analysis,
            "key": r.key,
            "direction": r.direction,
            "mean_return_bps": round(r.mean_return * 10000, 2),
            "t_stat": round(r.t_stat, 2),
            "p_adj": round(r.p_adj, 4),
            "stability": round(r.stability, 3) if r.stability else None,
            "sample_size": r.sample_size,
            "entry_condition": r.entry_condition,
            "filters": str(r.filters),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_edge_strategy(
    edges_path: str | Path | None = None,
    min_p_adj: float = 0.05,
    min_stability: float = 0.0,
    min_sample_size: int = 50,
) -> list[StrategyRule]:
    """Load edges and convert to strategy rules.

    Parameters
    ----------
    edges_path : str | Path | None
        Path to edge CSV. Tries ranked_edges.csv first, then
        edge_analysis_results.csv.
    min_p_adj : float
        Max p-adjusted value to include.
    min_stability : float
        Min stability score.
    min_sample_size : int
        Min observations per edge.

    Returns
    -------
    list[StrategyRule]
    """
    # Find the best available edge file
    if edges_path is not None:
        path = Path(edges_path)
    elif RANKED_EDGES_PATH.exists():
        path = RANKED_EDGES_PATH
    elif EDGE_RESULTS_PATH.exists():
        path = EDGE_RESULTS_PATH
    else:
        log.error(
            "No edge results found. Run edge analysis first:\n"
            "  python -m research.edge_analysis"
        )
        return []

    log.info("Loading edges from %s", path)
    edges_df = pd.read_csv(path)
    log.info("Loaded %d edges", len(edges_df))

    rules = edges_to_rules(
        edges_df,
        min_p_adj=min_p_adj,
        min_stability=min_stability,
        min_sample_size=min_sample_size,
    )

    if not rules:
        log.warning("No rules generated — check edge quality and filter thresholds")
        return rules

    # Save rules
    rules_df = rules_to_dataframe(rules)
    out_path = Path("output/edge_analysis/strategy_rules.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rules_df.to_csv(out_path, index=False)
    log.info("Rules saved to %s", out_path)

    print_rules(rules)

    return rules


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert edges to strategy rules")
    parser.add_argument("--edges", type=str, default=None, help="Path to edge CSV")
    parser.add_argument("--min-p", type=float, default=0.05, help="Max p_adj")
    parser.add_argument("--min-stability", type=float, default=0.0, help="Min stability")
    parser.add_argument("--min-n", type=int, default=50, help="Min sample size")
    parser.add_argument("--backtest", action="store_true", help="Run backtest after conversion")
    args = parser.parse_args()

    rules = build_edge_strategy(
        edges_path=args.edges,
        min_p_adj=args.min_p,
        min_stability=args.min_stability,
        min_sample_size=args.min_n,
    )

    if rules and args.backtest:
        from engine.data_loader import load_csv
        from engine.backtester import run as run_backtest
        from engine.metrics import compute_metrics, print_metrics

        log.info("Running backtest with %d edge-derived rules...", len(rules))
        df = load_csv("data/nq_15m_data.csv")
        signals = generate_signals(df, rules=rules)

        n_signals = int((signals["signal"] != 0).sum())
        log.info("Generated %d signals from %d rules", n_signals, len(rules))

        result = run_backtest(
            signals,
            initial_capital=100_000.0,
            risk_per_trade=0.015,
            point_value=20.0,
            commission_per_side=2.0,
            slippage_points=0.25,
        )
        metrics = compute_metrics(result, initial_capital=100_000.0)
        print_metrics(metrics)
