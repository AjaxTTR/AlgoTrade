# Hypothesis: Overnight → Intraday Reversal (NQ screening)

> This is a **screening pre-registration**, not a strategy test. We are
> measuring whether a conditional behaviour exists in NQ 15-min data
> before any strategy is designed. Fields below are locked before code
> runs; Test log section is populated after.

## Type
Measurement screen (distribution-based). Not a strategy.

## Claim being tested
On NQ during RTH, days with large overnight moves reverse during the
first few hours of the trading day; days with small overnight moves
show no meaningful directional effect.

## Proposed mechanism
Overnight information (macro releases, non-US session repricing, news
flow) accumulates while US cash equities are shut. When NQ gaps large
at the RTH open, a portion of that overnight move reflects
liquidity-constrained repricing by a narrow set of participants.
Once the full US session opens and liquidity returns, the move is
partially faded as broader participants transact. Small overnight
moves lack this liquidity-premium component and show no systematic
intraday response.

## Data scope
- **Instrument:** NQ futures, 15-min bars
- **Slice used:** training only — 2018-01-02 → 2022-12-30 (33,724 bars)
- **Test slice (2023–2024) is sealed.** Not touched in this screen.

## Session convention (locked)
- **Overnight = RTH close (16:00 ET) → RTH open (09:30 ET)**
- `overnight_return = (RTH_open_t - RTH_close_{t-1}) / RTH_close_{t-1}`
- RTH open = the 15-min bar opening at 09:30 ET on day t
- RTH close = the 15-min bar closing at 16:00 ET on day t-1
- Forward returns measured from the 09:30 ET RTH open, within RTH only

## Feature definition (locked)
`overnight_return` as defined above. One value per trading day.
Days without a prior RTH close (first day of dataset, post-holiday
edge cases) are dropped.

## Forward-return horizons (locked — exactly these four)
1. 30 minutes from RTH open (2 bars)
2. 1 hour from RTH open (4 bars)
3. 2 hours from RTH open (8 bars)
4. RTH close from RTH open

## Bucketing scheme (locked)
- **5 buckets (quintiles)** by `overnight_return`
- **Expanding percentile**: each day is bucketed using ONLY the
  distribution of `overnight_return` on prior days. No full-sample
  percentile fit.
- **Burn-in: 252 trading days.** The first ~year of the training slice
  is used to build the percentile history but is not included in any
  bucket aggregation.

## De-meaning (locked)
Forward returns are de-meaned against the unconditional mean forward
return (computed on the training slice, post burn-in) at the same
horizon before bucket aggregation. This strips out unconditional drift
and leaves the conditional effect.

## Pre-registered expected direction per bucket
| Bucket | Overnight move | Expected forward return |
|---|---|---|
| Q1 | biggest gap down | **positive** (bounce) |
| Q2 | mild gap down | slightly positive |
| Q3 | flat | ~zero |
| Q4 | mild gap up | slightly negative |
| Q5 | biggest gap up | **negative** (fade) |

Prediction applies across all four forward-return horizons. If the
effect exists at some horizons and not others, that is itself a
partial result and will be documented.

## Stability check (locked)
Training slice split into halves by date:
- **Half A:** 2018-01-02 → mid-2020 (approx. 2018-01 to 2020-06)
- **Half B:** mid-2020 → 2022-12-30 (approx. 2020-07 to 2022-12)

The pre-registered pattern must hold in BOTH halves independently to
count as stable.

## Significance standard (locked)
Per bucket-horizon cell: **95% bootstrap confidence interval** on the
mean forward return, resampled with replacement at the day level
(non-overlapping assumption preserved). 10,000 bootstrap iterations.

## Pass criteria (ALL four must hold)
1. Bucket means are **monotonic** from Q1 → Q5 (no zigzag) on at
   least the 1-hour horizon (the primary horizon for this mechanism).
2. **Q1 and Q5 bootstrap 95% CIs exclude zero** on the 1-hour horizon.
3. The **signs of Q1 and Q5 match the pre-registered prediction**
   (Q1 positive, Q5 negative) on the 1-hour horizon.
4. The monotonic pattern holds in **both halves** (Half A and Half B)
   independently on the 1-hour horizon.

## Fail criteria (ANY one kills the screen)
- Buckets flat or non-monotonic on the 1-hour horizon.
- Q1 or Q5 CI on the 1-hour horizon contains zero.
- Q1 or Q5 sign on the 1-hour horizon opposite to prediction.
- Pattern present in one half only.

## Decision rule
- **Pass:** feature and bucketing scheme are frozen exactly as-is.
  Next session, run the identical code on the 2023–2024 test slice
  exactly once. No tuning between screen and test.
- **Fail:** log outcome, archive this hypothesis file as-is, move on.
  No re-parameterizing (different bucket count, different horizons,
  different session convention) and re-running on the training slice.
  Those counts as new hypotheses and would require fresh
  pre-registration.

## Multiple-testing accounting
This screen evaluates 5 buckets × 4 horizons = 20 cells. The pass
criteria above are evaluated on the 1-hour horizon only (primary),
but all 20 cells will be reported. When and if this graduates to a
strategy and a formal significance statement is required, the 20
tests run here must be included in the multiple-testing correction.

## Budget note
This screen spends a portion of the training-slice judgment budget.
Two other effects (open momentum, large-move reversal) were
considered and deferred. They are candidates for future screens but
each additional screen on the training slice compounds researcher-
degrees-of-freedom contamination and should be budgeted accordingly.

---

## Test log (populated AFTER pre-registered sections above are locked — do not edit above)

### Screen run
- **Date:** 2026-04-23
- **Notebook:** `research/notebooks/screen_overnight_effect.ipynb`
- **Commit hash of code at time of run:** `b678aa1` (pre-registration commit; no code changes before execution)
- **Training slice used:** 2018-01-02 → 2022-12-30 (116,696 15-min bars)
- **Trading days with valid `overnight_return`:** 1,226
- **Days after 252-day burn-in, with assigned bucket:** 974 (2019-01-15 → 2022-12-30)
- **Days excluded from `fwd_close` horizon:** 44 (early-close sessions missing the 15:45 bar)

#### Per-bucket counts
| Bucket | n |
|---|---|
| Q1 | 213 |
| Q2 | 191 |
| Q3 | 181 |
| Q4 | 172 |
| Q5 | 217 |

#### Main bucket table — de-meaned forward returns (full training slice, post burn-in)
| Bucket | fwd_30m | fwd_1h | fwd_2h | fwd_close |
|---|---|---|---|---|
| Q1 | −0.015% | −0.011% | −0.054% | −0.122% |
| Q2 | +0.046% | +0.075% | +0.087% | +0.022% |
| Q3 | −0.009% | −0.043% | −0.031% | +0.047% |
| Q4 | +0.003% | +0.002% | +0.026% | −0.003% |
| Q5 | −0.021% | −0.021% | −0.018% | +0.066% |

#### Bootstrap 95% CIs on the 1-hour horizon (primary)
| Bucket | Mean | 95% CI | Excludes 0? |
|---|---|---|---|
| Q1 | −0.011% | [−0.122%, +0.102%] | No |
| Q2 | +0.075% | [−0.012%, +0.164%] | No |
| Q3 | −0.043% | [−0.114%, +0.027%] | No |
| Q4 | +0.002% | [−0.080%, +0.082%] | No |
| Q5 | −0.021% | [−0.111%, +0.071%] | No |

No bucket-horizon cell on the 1h horizon excludes zero. Same holds for every horizon except one: on `fwd_close`, Q1 CI is [−0.328%, +0.081%] (still includes zero).

#### Stability — Half A (n=358, 2019-01-15 → 2020-06-30) vs Half B (n=616, 2020-07-02 → 2022-12-30)
| Bucket | Half A 1h mean | Half B 1h mean |
|---|---|---|
| Q1 | **+0.174%** | **−0.100%** |
| Q2 | +0.050% | +0.090% |
| Q3 | +0.016% | −0.075% |
| Q4 | +0.032% | −0.013% |
| Q5 | −0.013% | −0.027% |

Q1 flips sign between halves (+17 bps → −10 bps). The effect is not stable.

### Decision
- **Passed / failed:** **FAIL**
- **Which criterion was the binding one:** All four failed independently.
  1. Monotonicity on full slice: zigzag pattern, not monotonic.
  2. Extreme-bucket significance: Q1 and Q5 1h CIs both contain zero.
  3. Predicted signs: Q1 is negative (predicted positive). Q5 is negative (matched), but this co-occurs with Q1 being negative, so the "reversal" interpretation fails.
  4. Stability: Q1 reverses sign between halves.
- **Commentary:**
  The overnight → intraday reversal hypothesis, as pre-registered on NQ
  15-min futures with RTH-close-to-RTH-open session convention, **does
  not hold on the 2018-2022 training slice**. The lack of significance
  on every bucket-horizon cell rules out even a weak version of the
  claim.

  **Observations flagged for the record (not acted on):**
  - Q2 (mild gap-down days) is the only cell with a consistent positive
    sign in both halves at short horizons. Interesting but acting on it
    from this screen would be data-derived hypothesizing. Deferred.
  - Half B (post-2020-07) shows Q1 *continuing* down on all four
    horizons (−0.07%, −0.10%, −0.13%, −0.12%), i.e. gap-down
    momentum, not reversal. This is opposite to the literature
    mechanism. Possibly a structural shift (zero-rate era → rising
    rate era, post-COVID vol regime), possibly that the equities
    overnight-reversal effect does not port to near-24h index futures.
    Not actionable from this screen.
  - The unconditional mean forward return at every horizon is ~+2–3
    bps — small positive drift, consistent with NQ's overall upward
    trend over 2019-2022.

  **Budget note:** This screen spent one unit of training-slice
  judgment budget. Effects 2 (open momentum) and 3 (large-move
  reversal) remain unscreened. The cleanest next step methodologically
  would be to read more deeply before pre-registering either, since
  the naive port of a well-known equities effect failed to replicate
  here — suggesting the futures-vs-equities distinction matters more
  than the original prompt assumed.

### If passed — formal test on 2023–2024
Not applicable. Screen failed; test slice remains sealed.
