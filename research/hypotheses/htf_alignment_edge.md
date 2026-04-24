# Hypothesis: HTF Trend Alignment — 1H × 4H Dual-Timeframe Edge (NQ screening)

> This is a **screening pre-registration**, not a strategy test. We are
> measuring whether a conditional behaviour exists in NQ 15-min data
> before any strategy is designed. Fields below are locked before code
> runs; Test log section is populated after.

## Type
Measurement screen (distribution-based, categorical conditioning). Not
a strategy. No entries, exits, PnL, sizing, or prop-firm mechanics are
involved in this screen.

## Source / framework origin
Derived from the "Higher-Timeframe Structure" component of the
Discretionary Overlay in Rogul (2026),
`Research_papers/ssrn-6469419.pdf`, Section 3.1. The paper's overlay
also references liquidity zones and regime classification; those are
deliberately **not** tested here. Order-block analysis was dropped on
methodological grounds (no peer-reviewed empirical support, heavy
post-hoc bias). Regime classification is a separate candidate screen.
This screen isolates the HTF-trend-alignment component alone, so its
independent contribution can be measured before being combined with
anything else.

## Claim being tested
On NQ 15-min data during RTH, forward returns are conditional on the
agreement of trend direction across two higher timeframes (1H and 4H).
Specifically: bars where both the 1H and 4H trend slopes are positive
show positive mean forward returns; bars where both are negative show
negative mean forward returns; bars with conflicting or flat
timeframes show no systematic effect.

## Proposed mechanism
Time-series momentum is documented across liquid instruments
(Moskowitz, Ooi & Pedersen, 2012). A dual-timeframe filter requiring
faster (1H, ~day-scale) and slower (4H, ~multi-day-scale) trend
agreement is hypothesised to isolate bars where momentum is
coherent across horizons — which should produce stronger
directional forward-return bias than single-timeframe alignment.
Conflicting-timeframe bars represent transition states where
directional bias is structurally ambiguous and no edge is expected.

## Data scope
- **Instrument:** NQ futures, 15-min bars
- **Slice used:** training only — 2018-01-02 → 2022-12-30
- **Test slice (2023–2024) is sealed.** Not touched in this screen.

## Feature construction (locked)

### HTF bar construction
- **1H bars:** resample the 15-min CSV using standard OHLCV aggregation
  (open=first, high=max, low=min, close=last, volume=sum). Right-closed,
  label = bar open.
- **4H bars:** same aggregation, 4H rule. Right-closed, label = bar open.

### MA slope features
- `MA_1H_t = close.rolling(20).mean()` on 1H series. Horizon ≈ last
  trading day.
- `MA_4H_t = close.rolling(16).mean()` on 4H series. Horizon ≈ last
  2.8 days of session time.
- `slope_1H_t = MA_1H_t − MA_1H_{t−1}`
- `slope_4H_t = MA_4H_t − MA_4H_{t−1}`

### Lookahead safety
Both slope series are shifted forward by one of their own bars
(`.shift(1)`) before reindexing onto the 15-min timeline. At any 15-min
timestamp, the feature reflects only the most recent **closed** 1H
and 4H bars. Reindexing uses forward-fill.

### State variables
- `state_1H_t = sign(slope_1H_t)` ∈ {+1, 0, −1}
- `state_4H_t = sign(slope_4H_t)` ∈ {+1, 0, −1}
- **Composite HTF state (primary conditioning variable):**
  ```
  HTF_state_t = +1  if state_1H_t == +1 AND state_4H_t == +1
              = −1  if state_1H_t == −1 AND state_4H_t == −1
              =  0  otherwise (conflict or either flat)
  ```

## Session convention (locked)
- **Observation window:** 15-min bars whose open time falls within
  RTH and allows the primary forward horizon to complete within the
  same RTH session. Concretely: bar open time in
  [09:30 ET, 15:00 ET] inclusive, so that a 1h forward (4 bars) closes
  no later than 16:00 ET.
- **Holidays / early closes:** days with a truncated session are
  excluded entirely if the 15:00 ET bar does not exist.

## Forward-return horizons (locked — exactly these three)
Measured as simple returns on `close`:
1. **15 min (1 bar):** `(close_{t+1} − close_t) / close_t` — **co-primary horizon** (execution-cadence match)
2. **1 hour (4 bars):** `(close_{t+4} − close_t) / close_t` — **co-primary horizon** (persistence test)
3. **4 hours (16 bars):** `(close_{t+16} − close_t) / close_t` — diagnostic only; bars near RTH close will have this horizon cross into the overnight session, so this is reported for pattern but not used for pass criteria.

Both the 15m and 1h horizons are co-primary: the filter must produce
signal at the shortest hold (15m, matching execution cadence) **and**
at a longer hold (1h, testing persistence). A filter that passes only
at 1h but not at 15m is ambiguously useful for a 15m-execution
strategy; a filter that passes at 15m but not at 1h suggests the
effect is noise-bounded and won't survive any realistic hold time.

## De-meaning (locked)
Forward returns are de-meaned against the unconditional mean forward
return computed on the observation window (same session mask,
training slice only), per horizon, before bucket aggregation. This
strips NQ's unconditional upward drift 2018-2022 and leaves the
conditional effect.

## Bucketing
- **Primary conditioning:** `HTF_state_t` ∈ {+1, 0, −1}. Three buckets.
- **Diagnostic reporting:** full 2×2 interaction of `state_1H_t` ×
  `state_4H_t` (9 cells including flats). Reported for transparency;
  pass criteria are evaluated only on the composite.

## Pre-registered expected direction per composite bucket

Prediction applies to both the 15m and 1h co-primary horizons.

| Composite state | 1H | 4H | Expected de-meaned forward return (15m and 1h) |
|---|---|---|---|
| +1 | + | + | **positive** |
| 0 | any conflict / flat | — | **no systematic effect** (CI straddles zero) |
| −1 | − | − | **negative** |

## Stability check (locked)
Training slice split into halves by date:
- **Half A:** 2018-01-02 → 2020-06-30
- **Half B:** 2020-07-01 → 2022-12-30

The pre-registered pattern must hold in **both halves** independently
on **both co-primary horizons (15m and 1h)**.

## Significance standard (locked)
- Per composite-bucket cell on **each co-primary horizon (15m and 1h)**:
  **95% confidence interval on the de-meaned mean forward return, using
  Newey-West HAC standard errors**.
- **Multiple-testing adjustment for co-primary horizons:** with two
  co-primary significance tests (15m and 1h), we apply a **Bonferroni
  correction**: each horizon's CI is computed at **97.5% (z = 2.24)**
  rather than 95%, so the joint family-wise error rate stays at 5%.
  A bucket must clear this tighter bar on **both** horizons to satisfy
  the significance criterion. We accept Bonferroni as conservative
  (the two horizons are positively correlated, so joint type-I error
  is actually lower than Bonferroni implies), but not knowing the
  exact correlation, conservative is the right direction to err.
- **Kernel:** Bartlett.
- **Truncation lag:** 23 bars (= one full trading day within the
  observation window, so within-day autocorrelation is captured
  without over-smoothing cross-day dependence which is near-zero on
  NQ intraday returns).
- **CI formula:** `mean ± 1.96 × HAC_SE`.
- No random seed required — Newey-West is deterministic given the
  data and lag.
- Rationale: with N in the tens of thousands of bars per bucket, the
  Central Limit Theorem makes parametric inference reliable; the HAC
  correction handles the known within-day autocorrelation of 15-min
  returns without the compute overhead of non-parametric resampling.
  This is the standard approach in financial econometrics (Harvey &
  Liu 2014).

## Pass criteria (ALL four must hold)
1. **+1 bucket** (both agree bullish): de-meaned mean forward return
   is **positive** and its Bonferroni-adjusted 97.5% Newey-West HAC CI
   **excludes zero** on **both** the 15m and 1h horizons.
2. **−1 bucket** (both agree bearish): de-meaned mean forward return
   is **negative** and its Bonferroni-adjusted 97.5% Newey-West HAC CI
   **excludes zero** on **both** the 15m and 1h horizons.
3. **0 bucket** (conflict) separation: on the 1h horizon, the conflict
   bucket's mean is strictly between the two agreement buckets' means.
   (The point of the filter is that agreement separates. If the
   conflict bucket has the same or larger effect than one of the
   agreement buckets, the filter is not doing what's claimed.)
   Evaluated on 1h only because separation magnitude is cleaner there
   than at the noisy 15m horizon.
4. **Stability:** the signs of the +1 and −1 bucket means are
   preserved in **both Half A and Half B** independently, on **both**
   the 15m and 1h horizons. Half-level CIs are not required to exclude
   zero (halving the data halves the power), only that the sign
   persists.

## Fail criteria (ANY one kills the screen)
- Either agreement bucket's 97.5% CI contains zero on the full slice,
  on either the 15m or the 1h horizon.
- Either agreement bucket's mean sign is opposite to prediction on
  either horizon.
- Either agreement bucket flips sign between Half A and Half B on
  either horizon.
- Conflict bucket's 1h mean is not strictly between the two agreement
  buckets' means (i.e. agreement fails to separate).

## Decision rule
- **Pass:** feature construction and bucketing frozen exactly as-is.
  This screen result does not itself graduate to the sealed test
  slice — a screen measures existence, not tradability. A subsequent
  `TEMPLATE_twogate.md` pre-registration is required to convert this
  measured edge into a strategy and run the two-gate pipeline.
- **Fail:** log outcome, archive this hypothesis file as-is, move on.
  No re-parameterizing (different MA lengths, different timeframes,
  different composite rules, different horizons) and re-running on
  the training slice. Those count as new hypotheses and would require
  fresh pre-registration.

## Multiple-testing accounting
This screen reports:
- 3 composite buckets × 3 horizons = 9 cells (primary)
- 9 interaction cells (2×2 + flats, diagnostic)
- 3 composite buckets × 2 halves = 6 stability cells

Pass criteria are evaluated on the 15m and 1h co-primary horizons,
Bonferroni-adjusted for those two tests. The 4h horizon and the 2×2
interaction are reported as diagnostics only. If this graduates to a
strategy under `TEMPLATE_twogate.md` and a formal significance
statement is required, the full set of tests run here must be included
in any further multiple-testing correction.

## Budget note
This screen spends one unit of training-slice judgment budget. Prior
spend: 1 (overnight_effect, FAIL on 2026-04-23). Running this brings
training-slice budget spent to 2. Remaining candidate screens
(open momentum, large-move reversal, ATR regime classification) are
deferred — each additional training-slice screen compounds
researcher-degrees-of-freedom contamination.

---

## Test log (populated AFTER pre-registered sections above are locked — do not edit above)

### Screen run
- **Date:** 2026-04-24
- **Notebook:** `research/notebooks/screen_htf_alignment.ipynb`
- **Commit hash of pre-reg at time of run:** `f848afb`
- **Training slice used:** 2018-01-02 → 2022-12-30 (116,696 15-min bars raw; 33,724 RTH bars)
- **Observation bars after session mask + non-NaN features:** 28,816
- **Observation range:** 2018-01-02 09:30 ET → 2022-12-30 15:00 ET

#### Per-bucket counts
| Composite state | n |
|---|---|
| +1 (both bullish) | 11,099 |
| 0 (conflict/flat) | 10,283 |
| −1 (both bearish) | 7,434 |

#### Primary table — de-meaned forward returns, full training slice
| Composite state | fwd_15m (co-primary) | fwd_1h (co-primary) |
|---|---|---|
| +1 | +0.00145% | +0.01086% |
| 0 | +0.00147% | −0.00184% |
| −1 | −0.00420% | −0.01366% |

(Unconditional means pre-de-meaning: fwd_15m +0.00105%, fwd_1h +0.00276%.)

#### Newey-West HAC 97.5% CIs on 15m horizon (co-primary)
| Composite state | Mean | 97.5% CI | Excludes 0? |
|---|---|---|---|
| +1 | +0.00145% | [−0.00197%, +0.00488%] | No |
| 0 | +0.00147% | [−0.00412%, +0.00705%] | No |
| −1 | −0.00420% | [−0.01043%, +0.00204%] | No |

#### Newey-West HAC 97.5% CIs on 1h horizon (co-primary)
| Composite state | Mean | 97.5% CI | Excludes 0? |
|---|---|---|---|
| +1 | +0.01086% | [−0.00148%, +0.02319%] | No (miss by ~1.5 bp) |
| 0 | −0.00184% | [−0.02089%, +0.01720%] | No |
| −1 | −0.01366% | [−0.03678%, +0.00946%] | No |

#### Diagnostic 2×2 interaction table (1h horizon, de-meaned)
| 1H \ 4H | +1 | 0 | −1 |
|---|---|---|---|
| +1 | +0.01086% (n=11,099) | +0.04334% (n=34) | +0.01370% (n=4,922) |
| 0 | −0.06636% (n=18) | +0.10217% (n=22) | −0.25836% (n=20) |
| −1 | −0.01618% (n=5,248) | +0.06266% (n=19) | −0.01366% (n=7,434) |

The `(1H+, 4H−)` cell at +0.01370% is indistinguishable from the `(1H+, 4H+)` agreement cell at +0.01086%. Same on the bearish side: `(1H−, 4H+)` at −0.01618% vs `(1H−, 4H−)` at −0.01366%. The 1H state drives essentially all the separation; the 4H filter adds no incremental signal over the 1H alone. The `state_4h == 0` cells are small-sample noise (MA slopes on 4H are almost never exactly zero).

#### Stability — Half A vs Half B (15m horizon)
| Composite state | Half A mean | Half B mean | Signs preserved? |
|---|---|---|---|
| +1 | +0.00055% | +0.00241% | Yes |
| 0 | +0.00250% | +0.00038% | n/a (no required sign) |
| −1 | −0.00350% | −0.00478% | Yes |

#### Stability — Half A vs Half B (1h horizon)
| Composite state | Half A mean | Half B mean | Signs preserved? |
|---|---|---|---|
| +1 | +0.00865% | +0.01320% | Yes |
| 0 | +0.00921% | −0.01340% | n/a (flipped; no required sign) |
| −1 | −0.02127% | −0.00735% | Yes |

### Decision
- **Passed / failed:** **FAIL**
- **Which criteria were binding:** Criteria 1 and 2 — neither agreement bucket's CI excluded zero at the Bonferroni-adjusted 97.5% level on either co-primary horizon. Criteria 3 (conflict bucket between agreement buckets on 1h) and 4 (signs preserved across both halves on both horizons) both passed. The directional pattern was correct; magnitudes were insufficient.
- **Commentary:**

  The HTF dual-timeframe trend filter, as pre-registered, does not
  produce a conditional forward-return effect on NQ 15-min bars that
  clears the Bonferroni-adjusted significance bar for two co-primary
  horizons. Directionally the pattern is consistent with time-series
  momentum (+1 bucket positive, −1 bucket negative, both stable across
  training halves), but the effect size is roughly the same order of
  magnitude as its standard error. The 1h bucket for `+1` misses the
  CI-excludes-zero bar by ~1.5 bp, which means a looser threshold
  (unadjusted 95% one-horizon) would have cleared — but that would
  have been a post-hoc adjustment, which is exactly what
  pre-registration prevents.

  **Observations flagged for the record (not acted on):**
  - **The 4H filter is doing no work.** The 2×2 interaction table
    shows `(1H+, 4H+)` and `(1H+, 4H−)` have nearly identical means
    (+0.0109% vs +0.0137%). Same on the bearish side. The dual-filter
    requirement does not improve separation over a 1H-only filter.
    This is a genuine diagnostic finding: if a follow-up were
    pre-registered, a simpler 1H-only filter would be the honest
    version.
  - **The magnitudes are plausibly real but low-powered.** At n ≈ 10k
    bars per bucket with 15-min return std ≈ 0.24%, detecting a 1-bp
    effect at 97.5% requires meaningfully more data or a lower
    variance regime. The budget cost of another screen to try to
    detect this more finely is not justified given (a) the 4H
    filter's redundancy and (b) two consecutive training-slice
    screens having now failed.
  - **Neutral-bucket (0) behaviour at 1h flipped between halves**
    (+0.9 bp Half A, −1.3 bp Half B). No pre-registered prediction,
    but worth noting — suggests the conflict bucket is dominated by
    regime-dependent noise rather than structural information.

  **Budget note:** This screen spent the second training-slice
  judgment unit. Prior spend: 1 (overnight_effect FAIL). Running
  total: 2 screens spent, both FAIL. Each additional training-slice
  screen compounds contamination; the next move warrants a pause to
  reconsider whether further naive-feature screens are the right
  budget allocation, or whether a different research path (framework-
  derived hypothesis under `TEMPLATE_twogate.md`, or deeper reading
  before committing) is a better use of remaining budget.

### If passed — next step
Not applicable. Screen failed; test slice remains sealed. Hypothesis
file archived as-is. No re-parameterization permitted.

### If passed — next step
Pass does NOT grant test-slice access. It authorises drafting a
strategy-shaped pre-registration under `TEMPLATE_twogate.md` that uses
this HTF filter as one component of a trade signal. Gate 1 (edge
existence) can cite this screen as evidence; Gate 2 (MFFU Phase 1
viability) still has to be earned on its own terms.
