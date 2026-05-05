# Gate 2 v2 — walk-forward MFFU Phase 1 viability

**Status:** Locked 2026-05-05.

## Purpose

Gate 2 answers: "does the strategy, with parameters tuned under
walk-forward discipline on the training slice, survive MFFU Phase 1
mechanics?" The output is a measured median OOS pass rate across
walk-forward chunks.

A hypothesis must show STRONG signal at Gate 1 before earning a Gate 2
run. The sealed test slice (2023–2024) is touched only after both
gates and only once.

## Walk-forward structure (locked, malleable on user command)

| Element | Value |
|---|---|
| Training slice | 2017-01-01 to 2022-12-31 (sealed slice 2023-2024 untouched) |
| Chunk train portion | 18 months |
| Chunk test (OOS) portion | 6 months |
| Slide cadence | 6 months forward per chunk |
| Resulting chunks | ~9 across the 6-year slice |

Codified in `engine/gate2.py::DEFAULT_WALK_FORWARD`.

## Parameter grid scope (locked, malleable on user command)

| Element | Value |
|---|---|
| Max tunable parameters | 3 |
| Allowed classes | 1 sizing + 1 stop/exit + 1 filter |
| Grid size cap | ≤ 27 combinations (3 values × 3 params) |

Enforced at runtime via `_check_grid_size` — exceeds these and the
function raises.

## Selection rules

**Per-chunk:** for each walk-forward chunk, evaluate every parameter
combination on the in-sample (18-month) portion. Pick the combination
with the highest in-sample MFFU Phase 1 pass rate as that chunk's
selected parameter set.

**Cross-chunk aggregation (load-bearing anti-cherry-pick rule):**
- Headline metric: **median** OOS pass rate across all chunks
- Tiebreaker: **lowest cross-chunk variance**

This is the rule that prevents "best-chunk window-shopping." Without
it, walk-forward becomes a way to launder cherry-picking with a process
gloss. With it, a strategy must work *most of the time* across the
training slice with *low chunk-to-chunk dispersion* to clear the bar.

## Reference thresholds (locked, malleable on user command)

| Threshold | Value |
|---|---|
| Pass rate | ≥ 50% (median across chunks per the rule above) |
| Drawdown-fail rate | ≤ 30% |
| Consistency | max_day_frac ≤ 0.50 on ≥ 80% of passing sims |

Days-to-pass remains malleable per hypothesis (per
`memory/feedback_gate2_thresholds.md`).

The MFFU 50% consistency rule is enforced inside `simulate_prop_firm`
at the simulator level — sims violating it are reclassified FAIL with
reason `consistency_rule`. The Gate 2 consistency threshold above is
therefore largely redundant by construction; it exists as an explicit
hook in case we later decouple consistency from the simulator's pass
criterion.

## Monte Carlo prerequisite — day-clustering fix

`engine/prop_firm.py::simulate_prop_firm` now permutes **whole-day
PnL blocks** rather than individual trade PnLs. The prior
implementation broke intra-day clustering, which biased pass rates
optimistic for EOD-trailing-DD and consistency rules. Day-level
permutation is the methodologically correct shuffle.

This fix landed in the same session as the v2 lock and is a
prerequisite for any Gate 2 run.

## Malleability discipline

Same rule as Gate 1: thresholds, walk-forward shape, and grid
constraints are forward-looking malleable. Change before a run, never
after seeing the result. Pre-registration captures whatever was active
at commit time.

The `max_tunable_parameters` cap is the single most malleable knob in
practice — some hypotheses will need 1, some will need 3. The 3-cap is
a project-default hard ceiling, not a per-hypothesis target.

## Where this lives in code

- `engine/gate2.py`
  - `DEFAULT_WALK_FORWARD` — locked walk-forward shape
  - `DEFAULT_GATE2_THRESHOLDS` — locked reference thresholds
  - `build_walk_forward_chunks(...)` — chunk boundaries
  - `gate2_evaluate(signal_fn, raw_data, param_grid, ...)` →
    `Gate2Result`
  - `print_gate2(result)` — pretty-print
- `engine/prop_firm.py::simulate_prop_firm` — Monte Carlo MFFU sim
  with day-level permutation (post-fix)

## Sealed slice + locked vault, unchanged

- Sealed test slice (2023–2024): one-shot, post-both-gates only.
- Locked vault (2005–2015, not yet acquired): pre-deployment only.

Both remain untouched by Gate 1 and Gate 2. Burning either changes the
methodology contract.
