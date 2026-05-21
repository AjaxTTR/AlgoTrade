"""
Gate 2 v2 — walk-forward MFFU Phase 1 viability test.

Takes a parameterised signal function and an explicit parameter grid,
runs walk-forward chunks across the training slice, tunes parameters
in-sample on each chunk, measures pass rate out-of-sample on the
chunk's test tail, and aggregates across chunks via median (with
cross-chunk variance as tiebreaker).

The cross-chunk aggregation rule is the load-bearing anti-cherry-pick
guardrail. Without it walk-forward becomes window-shopping: for any
parameter grid you can find a chunk that pass-rate-spikes if you allow
yourself to pick the chunk after the fact. Median + variance tiebreak
forces "the methodology has to work most of the time, with low chunk-
to-chunk dispersion."

Conventions (locked 2026-05-05, malleable on user command):
  * Walk-forward: 18-month train / 6-month test, slid 6 months forward
    per chunk, across the 2017-2022 training slice.
  * Max tunable parameters: 3, in {sizing, stop/exit, filter}.
  * Grid cap: 27 combinations (3 values per parameter).
  * Per-chunk selection: highest in-sample pass rate.
  * Cross-chunk aggregation: median OOS pass rate; ties broken by
    lowest cross-chunk variance.

Sealed test slice (2023-2024) is NEVER touched by Gate 2. That slice
is reserved for one-shot post-both-gates validation.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from engine.backtester import BacktestResult, run
from engine.gate1 import DEFAULT_GATE1_WRAPPER
from engine.prop_firm import simulate_prop_firm, MFFU_PHASE1_100K


# Locked walk-forward defaults — malleable on user command.
DEFAULT_WALK_FORWARD = {
    "train_months": 18,
    "test_months": 6,
    "slide_months": 6,
    "train_start": "2017-01-01",
    "train_end": "2022-12-31",   # sealed slice 2023-2024 untouched
}


# Gate 2 thresholds carried forward from feedback_gate2_thresholds.md.
# days_to_pass remains malleable per hypothesis.
DEFAULT_GATE2_THRESHOLDS = {
    "min_pass_rate": 50.0,             # %
    "max_drawdown_fail_rate": 30.0,    # %
    "consistency_pass_frac": 0.80,     # >= 80% of passing sims must satisfy
    "consistency_max_day_frac": 0.50,  # max_day_frac <= 0.50
}


@dataclass
class WalkForwardChunk:
    """One walk-forward chunk: in-sample training tail + OOS test tail."""
    chunk_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class ChunkResult:
    """Per-chunk Gate 2 outcome."""
    chunk: WalkForwardChunk
    selected_params: dict
    in_sample_pass_rate: float
    oos_pass_rate: float
    oos_dd_fail_rate: float
    oos_consistency_pass_frac: float
    oos_n_trades: int
    grid_in_sample_pass_rates: dict   # combo_repr -> in_sample_pass_rate


@dataclass
class Gate2Result:
    """Full Gate 2 walk-forward report."""
    chunks: list[ChunkResult] = field(default_factory=list)
    median_oos_pass_rate: float = 0.0
    cross_chunk_variance: float = 0.0
    n_chunks_above_threshold: int = 0
    thresholds: dict = field(default_factory=dict)
    walk_forward_config: dict = field(default_factory=dict)
    param_grid: dict = field(default_factory=dict)


def build_walk_forward_chunks(
    train_start: str | pd.Timestamp = DEFAULT_WALK_FORWARD["train_start"],
    train_end: str | pd.Timestamp = DEFAULT_WALK_FORWARD["train_end"],
    train_months: int = DEFAULT_WALK_FORWARD["train_months"],
    test_months: int = DEFAULT_WALK_FORWARD["test_months"],
    slide_months: int = DEFAULT_WALK_FORWARD["slide_months"],
) -> list[WalkForwardChunk]:
    """Generate walk-forward chunk boundaries.

    Each chunk is (train_months in-sample) + (test_months OOS), slid
    forward by slide_months. Chunks ending after train_end are dropped.
    """
    start = pd.Timestamp(train_start)
    end = pd.Timestamp(train_end)
    chunks: list[WalkForwardChunk] = []
    chunk_train_start = start
    cid = 0
    while True:
        chunk_train_end = chunk_train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        chunk_test_start = chunk_train_end + pd.Timedelta(days=1)
        chunk_test_end = chunk_test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if chunk_test_end > end:
            break
        chunks.append(WalkForwardChunk(
            chunk_id=cid,
            train_start=chunk_train_start,
            train_end=chunk_train_end,
            test_start=chunk_test_start,
            test_end=chunk_test_end,
        ))
        cid += 1
        chunk_train_start = chunk_train_start + pd.DateOffset(months=slide_months)
    return chunks


def _slice_signals(signals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return signals.loc[(signals.index >= start) & (signals.index <= end)].copy()


def _grid_combinations(param_grid: dict) -> list[dict]:
    """Cartesian product of param_grid (dict of name -> list of values).

    Caller is responsible for keeping the grid within the ≤27-combo cap.
    """
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    return [dict(zip(keys, vals)) for vals in product(*value_lists)]


def _check_grid_size(param_grid: dict, max_combos: int = 27) -> None:
    if not param_grid:
        return
    n = 1
    for vals in param_grid.values():
        n *= len(vals)
    if n > max_combos:
        raise ValueError(
            f"Parameter grid has {n} combinations (cap = {max_combos}). "
            f"Reduce grid scope per locked Gate 2 spec."
        )
    if len(param_grid) > 3:
        raise ValueError(
            f"Parameter grid has {len(param_grid)} parameters (cap = 3). "
            f"Reduce per locked Gate 2 spec."
        )


def _consistency_pass_frac(simulate_pf_result: dict, max_day_frac: float) -> float:
    """Fraction of PASSing sims that satisfy the consistency rule.

    The MFFU consistency check is already enforced inside
    simulate_prop_firm — sims that violated it are reclassified FAIL
    with reason='consistency_rule'. So among the simulator's PASSes,
    100% have passed consistency by construction. This function exists
    for explicit reporting; it returns 1.0 when there are passing sims.

    If we ever decouple consistency from the simulator's pass criterion
    (e.g. to report it as a soft flag), this function becomes load-
    bearing. Kept here so the Gate 2 thresholds dict has a clear hook.
    """
    n_total = simulate_pf_result.get("n_simulations", 0)
    if n_total == 0:
        return 0.0
    pass_rate = simulate_pf_result.get("pass_rate", 0.0)
    return 1.0 if pass_rate > 0 else 0.0


def gate2_evaluate(
    signal_fn: Callable[..., pd.DataFrame],
    raw_data: pd.DataFrame,
    param_grid: dict,
    wrapper_overrides: dict | None = None,
    walk_forward_overrides: dict | None = None,
    thresholds: dict | None = None,
    n_simulations: int = 5000,
    seed: int | None = 42,
) -> Gate2Result:
    """Run Gate 2 v2 walk-forward MFFU pass-rate measurement.

    Parameters
    ----------
    signal_fn : Callable
        signal_fn(df, **params) -> signals DataFrame with columns
        required by engine.backtester.run. Must take all keys in
        param_grid as kwargs.
    raw_data : pd.DataFrame
        OHLCV (and any pre-computed features) indexed by Timestamp.
        Must span the full walk-forward window.
    param_grid : dict
        Parameter name -> list of candidate values. Max 3 keys, max 27
        Cartesian combinations.
    wrapper_overrides : dict, optional
        Backtest engine overrides (sizing, stop, etc.). Defaults to
        Gate 1's locked wrapper but Deployment-Mode-flavored: Gate 2
        is a viability test, so deployment-style sizing is appropriate
        here. Caller can pass compounding sizing, halts, etc.
    walk_forward_overrides : dict, optional
        Override DEFAULT_WALK_FORWARD (train_months, test_months,
        slide_months, train_start, train_end).
    thresholds : dict, optional
        Override DEFAULT_GATE2_THRESHOLDS.
    n_simulations : int
        MC sims per pass-rate evaluation (default 5000).
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    Gate2Result
    """
    _check_grid_size(param_grid)

    wf = {**DEFAULT_WALK_FORWARD, **(walk_forward_overrides or {})}
    th = {**DEFAULT_GATE2_THRESHOLDS, **(thresholds or {})}
    wrapper = {**DEFAULT_GATE1_WRAPPER, **(wrapper_overrides or {})}

    chunks = build_walk_forward_chunks(
        train_start=wf["train_start"],
        train_end=wf["train_end"],
        train_months=wf["train_months"],
        test_months=wf["test_months"],
        slide_months=wf["slide_months"],
    )

    combos = _grid_combinations(param_grid)
    chunk_results: list[ChunkResult] = []

    for chunk in chunks:
        train_data = _slice_signals(raw_data, chunk.train_start, chunk.train_end)
        test_data = _slice_signals(raw_data, chunk.test_start, chunk.test_end)
        if len(train_data) == 0 or len(test_data) == 0:
            continue

        # In-sample tuning: pick combo with highest in-sample pass rate
        in_sample_grid: dict = {}
        best_combo = None
        best_in_sample_rate = -1.0

        for combo in combos:
            train_signals = signal_fn(train_data, **combo)
            # Combo keys are routed into BOTH signal_fn and the wrapper.
            # signal_fn consumes the keys it cares about and ignores the rest
            # via **_; the backtester's run() accepts arbitrary **kwargs, so
            # signal-side keys passed here are absorbed harmlessly. This lets
            # the grid tune wrapper-level knobs (e.g. stop_atr_multiple)
            # alongside signal-level ones (e.g. entry_z).
            train_bt = run(train_signals, **{**wrapper, **combo})
            train_pf = simulate_prop_firm(
                train_bt, n_simulations=n_simulations, seed=seed,
            )
            in_rate = train_pf["pass_rate"]
            in_sample_grid[repr(combo)] = round(in_rate, 2)
            if in_rate > best_in_sample_rate:
                best_in_sample_rate = in_rate
                best_combo = combo

        # OOS measurement on test tail with selected combo
        test_signals = signal_fn(test_data, **best_combo) if best_combo is not None else test_data
        test_bt = run(test_signals, **{**wrapper, **(best_combo or {})})
        test_pf = simulate_prop_firm(
            test_bt, n_simulations=n_simulations, seed=seed,
        )

        oos_rate = test_pf["pass_rate"]
        dd_fail = test_pf["fail_reasons"].get("trailing_dd", 0)
        dd_fail_rate = (dd_fail / test_pf["n_simulations"] * 100) if test_pf["n_simulations"] > 0 else 0.0
        cons_pass = _consistency_pass_frac(test_pf, th["consistency_max_day_frac"])

        chunk_results.append(ChunkResult(
            chunk=chunk,
            selected_params=best_combo or {},
            in_sample_pass_rate=round(best_in_sample_rate, 2),
            oos_pass_rate=round(oos_rate, 2),
            oos_dd_fail_rate=round(dd_fail_rate, 2),
            oos_consistency_pass_frac=round(cons_pass, 3),
            oos_n_trades=len(test_bt.trades),
            grid_in_sample_pass_rates=in_sample_grid,
        ))

    # Cross-chunk aggregation: median + variance
    if chunk_results:
        oos_rates = np.array([c.oos_pass_rate for c in chunk_results])
        median_oos = float(np.median(oos_rates))
        variance = float(np.var(oos_rates))
        n_above = int(np.sum(oos_rates >= th["min_pass_rate"]))
    else:
        median_oos = 0.0
        variance = 0.0
        n_above = 0

    return Gate2Result(
        chunks=chunk_results,
        median_oos_pass_rate=round(median_oos, 2),
        cross_chunk_variance=round(variance, 3),
        n_chunks_above_threshold=n_above,
        thresholds=th,
        walk_forward_config=wf,
        param_grid=param_grid,
    )


def print_gate2(result: Gate2Result) -> None:
    """Pretty-print a Gate 2 walk-forward report."""
    th = result.thresholds
    wf = result.walk_forward_config

    print("\n" + "=" * 78)
    print("  GATE 2 v2 - WALK-FORWARD MFFU PHASE 1 PASS-RATE")
    print("=" * 78)
    print(f"  Walk-forward: {wf['train_months']}mo train / {wf['test_months']}mo test, "
          f"slid {wf['slide_months']}mo")
    print(f"  Slice: {wf['train_start']} to {wf['train_end']} (sealed 2023-2024 untouched)")
    print(f"  Param grid: {result.param_grid}")
    print(f"  Chunks: {len(result.chunks)}")
    print()

    print("  Per-chunk OOS results:")
    print("  " + "-" * 74)
    print(f"  {'#':>3s}  {'Test window':22s}  {'PassRate':>9s}  {'DD-Fail':>8s}  {'N':>5s}  {'Selected params'}")
    for c in result.chunks:
        win = f"{c.chunk.test_start.date()}..{c.chunk.test_end.date()}"
        print(f"  {c.chunk.chunk_id:>3d}  {win:22s}  "
              f"{c.oos_pass_rate:>8.1f}%  {c.oos_dd_fail_rate:>7.1f}%  "
              f"{c.oos_n_trades:>5d}  {c.selected_params}")

    print("  " + "-" * 74)
    print(f"  Median OOS pass rate:           {result.median_oos_pass_rate:>8.1f}%")
    print(f"  Cross-chunk variance:           {result.cross_chunk_variance:>8.2f}")
    print(f"  Chunks at/above {th['min_pass_rate']:.0f}% threshold:    "
          f"{result.n_chunks_above_threshold}/{len(result.chunks)}")
    print()

    verdict = ("STRONG" if result.median_oos_pass_rate >= th["min_pass_rate"]
               else "WEAK")
    print(f"  Verdict (median vs {th['min_pass_rate']:.0f}% reference):       [{verdict}]")
    print("  User decides whether to progress to sealed test slice.")
    print("=" * 78 + "\n")
