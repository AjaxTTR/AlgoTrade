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
- **Date:** 2026-04-30
- **Notebook:** `research/notebooks/screen_open_momentum.ipynb`
- **Commit hash of code at time of run:** `6bc2fc2` (pre-registration commit; no code changes before execution)
- **Training slice used:** 2018-01-02 → 2022-12-30 (116,696 15-min bars)
- **Trading days with valid `open_return`:** 1,270
- **Days after 252-day burn-in, with assigned bucket:** 1,018 (2018-12-31 → 2022-12-30)
- **Days excluded from `fwd_close` horizon:** 44 (early-close sessions missing the 15:45 bar)
- **Days excluded from `fwd_2h` horizon:** 3
- **Days excluded from `fwd_4h` horizon:** 44

#### Per-bucket counts
| Bucket | n |
|---|---|
| Q1 | 234 |
| Q2 | 189 |
| Q3 | 177 |
| Q4 | 182 |
| Q5 | 236 |

#### Main bucket table — de-meaned forward returns (full training slice, post burn-in)
| Bucket | fwd_close | fwd_2h | fwd_4h |
|---|---|---|---|
| Q1 | −0.018% | −0.011% | −0.052% |
| Q2 | −0.044% | −0.050% | −0.003% |
| Q3 | −0.018% | +0.025% | +0.009% |
| Q4 | −0.070% | −0.068% | −0.055% |
| Q5 | +0.116% | +0.086% | +0.088% |

#### Bootstrap 95% CIs on the primary horizon (`fwd_close`)
| Bucket | Mean | 95% CI | Excludes 0? |
|---|---|---|---|
| Q1 | −0.018% | [−0.205%, +0.172%] | No |
| Q2 | −0.044% | [−0.185%, +0.091%] | No |
| Q3 | −0.018% | [−0.160%, +0.123%] | No |
| Q4 | −0.070% | [−0.223%, +0.082%] | No |
| Q5 | +0.116% | [−0.033%, +0.263%] | No (just barely) |

No bucket on the primary horizon excludes zero. Q5 came closest — the lower edge of its CI is at −0.033%, brushing zero from above.

#### Stability — Half A (n=381, 2018-12-31 → 2020-06-30) vs Half B (n=637, 2020-07-01 → 2022-12-30)
| Bucket | Half A fwd_close mean | Half B fwd_close mean |
|---|---|---|
| Q1 | **+0.196%** | **−0.124%** |
| Q2 | +0.002% | −0.077% |
| Q3 | −0.067% | +0.013% |
| Q4 | −0.065% | −0.073% |
| Q5 | +0.152% | +0.095% |

Q1 flips sign between halves (+19.6 bps → −12.4 bps), the same fingerprint observed in the overnight screen. In Half A, big down-opens *bounced* (reversal regime); in Half B, big down-opens *continued down* (continuation regime). Q5 is positive in both halves, but its magnitude is order-of-noise once CIs are considered.

### Decision
- **Passed / failed:** **FAIL**
- **Which criterion was the binding one:** Three of four failed independently.
  1. Monotonicity on full slice: zigzag (Q1 −0.02, Q2 −0.04, Q3 −0.02, Q4 −0.07, Q5 +0.12). Not monotonic increasing.
  2. Extreme-bucket significance: Q1 and Q5 `fwd_close` CIs both contain zero (Q5 barely; Q1 widely).
  3. Predicted signs: PASS — Q1 negative, Q5 positive on full slice. The only criterion satisfied.
  4. Stability: Half A and Half B disagree on Q1 sign. Half A is reversal-shaped (Q1 strongly positive); Half B is continuation-shaped (Q1 strongly negative).
- **Commentary:**
  The open momentum / continuation hypothesis, as pre-registered on NQ
  15-min futures with 30-min opening window and 09:30→10:00 ET reference
  prices, **does not hold on the 2018-2022 training slice**. Q5 (biggest
  up-opens) showed the predicted continuation pattern in both halves and
  on the full slice, but its 95% CI on `fwd_close` straddles zero by ~3
  bps on the lower edge — too close to call from this slice alone. The
  middle buckets show no coherent pattern.

  **Cross-screen pattern (flagged for the record, not acted on):**
  Q1 sign-flip between halves is now observed on **two** screens
  (overnight effect, open momentum). Both show:
  - Half A (2018 → mid-2020): big down-moves *bounce up* (reversal).
  - Half B (mid-2020 → 2022): big down-moves *continue down* (momentum).
  This is consistent with a structural shift around the COVID dislocation
  and the subsequent rate-hike cycle. Any pre-registered claim that
  requires stability across this 2020-07 boundary on the current
  training slice is therefore working with a dataset that contains two
  qualitatively different regimes averaged together. This is a property
  of the data, not of the screens.

  Q5 positivity is consistent across both halves and across all three
  horizons in this screen (~+10 bps de-meaned). Acting on it as
  evidence of a one-sided up-open continuation effect would be data-
  derived hypothesizing off this screen — not legitimate without a
  fresh slice or a fresh pre-registration. Deferred.

  **Budget note:** This screen spent the third unit of training-slice
  judgment budget. Three for three on FAIL on the same training slice.
  The methodological prior strengthens further toward (a) sourcing a
  fresh slice before any further screening on this dataset, or (b)
  pivoting to a framework-derived two-gate hypothesis with explicit
  regime-handling, rather than continuing naive screens.

### If passed — formal test on 2023–2024
Not applicable. Screen failed; test slice remains sealed.
