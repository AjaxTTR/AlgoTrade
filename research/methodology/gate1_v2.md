# Gate 1 v2 — measurement, not pass/fail

**Status:** Locked 2026-05-05. Replaces the v1 statistical-bar
framing (HAC/Bonferroni CIs) and the early-v2 path-survival framing.

## Purpose

Gate 1 v2 is a **measurement layer**. It runs a hypothesis's signal
through a fixed, project-default execution wrapper at unit sizing and
emits six descriptive elements as STRONG/WEAK indicators. The user
makes the call on whether the pattern justifies progressing to Gate 2.

There is no pass/fail. The gate is a report, not a verdict.

## Why this shape

- **Signal-existence question, not viability question.** Gate 1 asks
  "is there a measurable pattern?" Gate 2 asks "does it survive prop-
  firm mechanics under tuning?" Conflating the two collapses the
  pipeline.
- **Stub-rule overshoot.** Earlier v2 drafts measured raw signal
  direction with no execution mechanics. That's too pure — no real
  prop strategy trades without a stop. The default wrapper is the
  minimum viable trade rule, not a tuning surface.
- **Curve-fit defense moved to Gate 2.** Gate 1 cannot prevent curve
  fitting by being stricter; only Gate 2's pre-registered parameter
  grid + walk-forward erosion penalty can do that. So Gate 1 is
  loosened deliberately, and the load-bearing guardrail moves
  downstream.

## Default execution wrapper (project-wide, malleable on user command)

| Element | Value |
|---|---|
| Entry | Next-bar open after signal (`execution_delay_bars=1`) |
| Stop | 1.5× ATR(14) at entry |
| Take profit | 2× ATR(14) at entry (fixed, no trail) |
| Time exit | Session close if neither stop nor TP hit |
| Sizing | Unit (1 contract NQ / 1 micro MNQ) |
| Costs | $5 RT full NQ ($2.50/side) — `commission_per_side=2.5` |
| Spread | 0.25 points (NQ tick) |

Codified in `engine/gate1.py::DEFAULT_GATE1_WRAPPER`. Per-hypothesis
overrides supported via `wrapper_overrides` kwarg but each must be
documented in the pre-reg with reason.

## Six elements (reported as STRONG/WEAK)

| # | Element | Reference threshold | STRONG when |
|---|---|---|---|
| 1 | Profit factor net of costs | 1.2 | ≥ threshold |
| 2 | Total net P&L | $0 | > 0 |
| 3 | Sample size (trade count) | 30 | ≥ threshold |
| 4 | Single-day concentration (max-day P&L / total) | 0.30 | ≤ threshold |
| 5 | Half-stability (both halves PF > 1.0 AND net positive) | n/a | both true |
| 6 | Profit / max DD MTM (USD) | 1.5 | ≥ threshold |

Codified in `engine/gate1.py::DEFAULT_GATE1_THRESHOLDS`.

## Diagnostics (reported but not classified)

- 30-day rolling path-survival rate vs MFFU Phase 1 at unit sizing
  (informative, not binding — Gate 2 is where path-survival becomes
  the real test)
- Sharpe, Calmar, expectancy in R-multiples
- Trade count distribution by month

## Malleability discipline

Reference thresholds and the default wrapper are **forward-looking
malleable**. The user can change a threshold or a wrapper element
*before* a hypothesis runs — that's a methodology evolution and is
fine. What is **not** fine is changing a threshold *after* seeing a
result, because that retroactively re-decides whether the hypothesis
was strong or weak.

Pre-registration captures whatever was active at commit time. The
result is judged against that snapshot. If you want to change a
default mid-flight, the right move is to abandon the in-flight pre-reg
and start a new one with the new defaults — not to retroactively patch.

## Where this lives in code

- `engine/gate1.py`
  - `DEFAULT_GATE1_WRAPPER` — locked execution wrapper
  - `DEFAULT_GATE1_THRESHOLDS` — locked reference thresholds
  - `gate1_evaluate(signals, wrapper_overrides=None, thresholds=None)`
    → `Gate1Result`
  - `print_gate1(result)` — pretty-print

## Things demoted from earlier drafts

- **HAC + Bonferroni statistical bar** (v1) — institutional, mis-
  calibrated for prop-firm context.
- **Stub-rule-only measurement** (early v2) — overshot; ignored that
  real strategies need a minimum execution wrapper to express their
  edge.
- **Path-survival as binding criterion** (early v2) — too punishing
  for raw signals; collapsed Gate 1 into a mini-Gate-2.
- **Consistency rule as Gate 1 element** (early v2) — moved to Gate 2
  where the simulator already enforces it.
