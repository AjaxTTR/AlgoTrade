# Hypothesis: Open Momentum / Continuation (NQ screening)

> This is a **screening pre-registration**, not a strategy test. We are
> measuring whether a conditional behaviour exists in NQ 15-min data
> before any strategy is designed. Fields below are locked before code
> runs; Test log section is populated after.

## Type
Measurement screen (distribution-based). Not a strategy.

## Claim being tested
On NQ during RTH, the signed return over the first 30 minutes of the
session (09:30–10:00 ET) predicts the signed return over the rest of
the session (10:00–16:00 ET) in the **same direction** (momentum /
continuation). Larger opening moves predict larger continuation moves.

## Proposed mechanism
The 09:30–10:00 ET window concentrates the session's information
intake: overnight news is digested, opening auction imbalances clear,
and the dominant participant flow for the day establishes direction.
Institutional execution programs that begin at the open typically
continue working through the day, reinforcing the initial move.
Behavioural anchoring on the early-session direction further sustains
the trend until late-session profit-taking. Under this story, the sign
and magnitude of the 30-min opening return should carry information
about the continuation through to the close.

This is the opposite prediction to the overnight reversal mechanism
tested previously (where the open *fades* the overnight move). Here
we are measuring whether *intra-session* direction persists once the
session is underway.

## Data scope
- **Instrument:** NQ futures, 15-min bars
- **Slice used:** training only — 2018-01-02 → 2022-12-30
- **Test slice (2023–2024) is sealed.** Not touched in this screen.

## Session convention (locked)
- **Opening window = 09:30 ET RTH open → 10:00 ET** (2 bars)
- `open_return = (price_at_10:00 - price_at_09:30) / price_at_09:30`
  where `price_at_09:30` is the open of the 09:30 bar and
  `price_at_10:00` is the open of the 10:00 bar (equivalently, the
  close of the 09:45 bar).
- **Continuation window = 10:00 ET → 16:00 ET RTH close**
- `fwd_close = (price_at_16:00 - price_at_10:00) / price_at_10:00`
  where `price_at_16:00` is the close of the 15:45 bar.
- One `(open_return, fwd_close)` pair per trading day.
- Days missing either the 09:30, 10:00, or 15:45 bar (early closes,
  partial-session days, holiday-shortened sessions) are dropped.

## Feature definition (locked)
`open_return` as defined above. One value per trading day.

## Forward-return horizons (locked)
- **Primary (binding for pass/fail):** `fwd_close` — 10:00 ET → 16:00 ET
- **Diagnostic (reported, not binding):** `fwd_2h` — 10:00 ET → 12:00 ET
- **Diagnostic (reported, not binding):** `fwd_4h` — 10:00 ET → 14:00 ET

Only `fwd_close` is used to evaluate pass criteria. Diagnostic
horizons are reported for shape information but cannot rescue or kill
the screen.

## Bucketing scheme (locked)
- **5 buckets (quintiles)** by `open_return` (signed, not absolute)
- **Expanding percentile**: each day is bucketed using ONLY the
  distribution of `open_return` on prior days. No full-sample
  percentile fit.
- **Burn-in: 252 trading days.** The first ~year of the training slice
  is used to build the percentile history but is not included in any
  bucket aggregation.

## De-meaning (locked)
Forward returns are de-meaned against the unconditional mean forward
return (computed on the training slice, post burn-in) at the same
horizon before bucket aggregation. This strips out unconditional drift
and leaves the conditional effect.

## Pre-registered expected direction per bucket (momentum / continuation)
| Bucket | Opening move (09:30→10:00) | Expected forward return (10:00→16:00) |
|---|---|---|
| Q1 | biggest down-open | **negative** (continues down) |
| Q2 | mild down-open | slightly negative |
| Q3 | flat open | ~zero |
| Q4 | mild up-open | slightly positive |
| Q5 | biggest up-open | **positive** (continues up) |

Prediction is **monotonic from Q1 negative to Q5 positive** — the
opposite shape to the overnight reversal screen's prediction.

## Stability check (locked)
Training slice split into halves by date, identical to overnight
screen:
- **Half A:** 2018-01-02 → 2020-06-30
- **Half B:** 2020-07-01 → 2022-12-30

The pre-registered pattern must hold in BOTH halves independently to
count as stable.

## Significance standard (locked)
Per bucket on the primary horizon: **95% bootstrap confidence interval**
on the mean forward return, resampled with replacement at the day level
(non-overlapping daily observations). 10,000 bootstrap iterations.

## Pass criteria (ALL four must hold)
1. Bucket means are **monotonic** from Q1 → Q5 (no zigzag) on the
   primary horizon (`fwd_close`).
2. **Q1 and Q5 bootstrap 95% CIs exclude zero** on the primary horizon.
3. The **signs of Q1 and Q5 match the pre-registered prediction**
   (Q1 negative, Q5 positive) on the primary horizon.
4. The monotonic pattern holds in **both halves** (Half A and Half B)
   independently on the primary horizon.

## Fail criteria (ANY one kills the screen)
- Buckets flat or non-monotonic on the primary horizon.
- Q1 or Q5 CI on the primary horizon contains zero.
- Q1 or Q5 sign on the primary horizon opposite to prediction.
- Pattern present in one half only.

## Decision rule
- **Pass:** feature and bucketing scheme are frozen exactly as-is.
  Next session, run the identical code on the 2023–2024 test slice
  exactly once. No tuning between screen and test.
- **Fail:** log outcome, archive this hypothesis file as-is, move on.
  No re-parameterizing (different opening window length, different
  bucket count, different horizons, sign-only collapse) and re-running
  on the training slice. Each of those counts as a new hypothesis and
  would require fresh pre-registration on a fresh slice.

## Multiple-testing accounting
Pass/fail evaluated on 5 buckets × 1 primary horizon = 5 cells.
Diagnostic horizons (`fwd_2h`, `fwd_4h`) add 10 more cells reported
but not binding. When and if this graduates to a strategy and a formal
significance statement is required, the 15 reported cells must be
included in the multiple-testing correction, alongside cells from
prior failed screens (overnight_effect, htf_alignment_edge).

## Budget note
This screen spends one further unit of training-slice judgment budget,
bringing total spent to 3. Two effects from the original list remain
unscreened (large-move reversal, ATR regime-conditioned variants).
Each additional screen on the training slice compounds researcher-
degrees-of-freedom contamination. If this screen fails, the strong
methodological prior is to either (a) source a fresh slice before
spending more budget, or (b) pivot to the framework-derived two-gate
template rather than continue naive screens.

---

## Test log (populated AFTER pre-registered sections above are locked — do not edit above)

### Screen run
- **Date:** _to be filled after run_
- **Notebook:** _to be filled after run_
- **Commit hash of code at time of run:** _to be filled after run_
- **Training slice used:** 2018-01-02 → 2022-12-30
- **Trading days with valid `open_return` and `fwd_close`:** _to be filled_
- **Days after 252-day burn-in, with assigned bucket:** _to be filled_
- **Days excluded (missing 09:30, 10:00, or 15:45 bar):** _to be filled_

#### Per-bucket counts
_to be filled_

#### Main bucket table — de-meaned forward returns (full training slice, post burn-in)
_to be filled_

#### Bootstrap 95% CIs on the primary horizon (`fwd_close`)
_to be filled_

#### Stability — Half A vs Half B
_to be filled_

### Decision
- **Passed / failed:** _to be filled_
- **Which criterion was the binding one:** _to be filled_
- **Commentary:** _to be filled_

### If passed — formal test on 2023–2024
_to be filled_
