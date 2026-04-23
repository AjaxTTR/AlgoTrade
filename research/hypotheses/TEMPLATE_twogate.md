# Hypothesis: [short descriptive name]

> Copy this template to a new file in `research/hypotheses/` named after
> the hypothesis. Fill it out BEFORE any code is written. The upper
> sections (everything above the Test log) are frozen at commit time;
> the Test log is populated after runs complete.
>
> This is the **two-gate** template. Use it for hypotheses that propose
> a strategy (not raw measurement screens). For pure measurement
> screens — where no strategy is being tested yet, just a conditional
> behaviour — use a screen-shaped pre-registration like
> `overnight_effect.md`.
>
> The two gates:
>
>   **Gate 1 — Edge exists.** On the training slice, does the proposed
>   edge measure positive and stable? No prop-firm mechanics involved.
>   Pure edge detection.
>
>   **Gate 2 — MFFU Phase 1 viable.** Assuming Gate 1 passes, does a
>   strategy that uses this edge survive MFFU Phase 1 mechanics
>   (EOD trailing DD, consistency rule, profit target) under Monte
>   Carlo trade-order shuffling?
>
> Only if both gates pass does the strategy earn the right to be run
> once on the sealed 2023–2024 test slice.

---

## Hypothesis (1-2 sentences, plain English)
What's the proposed edge and how would a strategy exploit it?

## Proposed mechanism
Why should this work? Market microstructure, behavioural bias, risk
premium, liquidity, inventory cycle? If no mechanism can be stated,
that's a red flag — document it.

## Source
- Paper / research / own observation:
- If paper-derived: replication status (paper reproduced on its own
  instrument first? Skipped with reason?).

## MFFU compatibility checklist (pre-flight)
Answer these BEFORE designing signal rules. Any "no" here disqualifies
the hypothesis or forces a design change:

- [ ] Does the strategy hold any position past 4:10 PM ET? (MFFU forces
      close at 4:10 PM — overnight holds are disallowed.)
- [ ] Does the strategy rely on infrequent large payoff days?
      (50% consistency rule punishes this — flag explicitly.)
- [ ] Does the strategy produce enough trades per month to plausibly
      hit a $6k target without a jackpot day? (Rough floor: ≥8
      independent trade days expected per pass attempt.)

## Gate 1 — Edge existence

### Signal definition (locked)
Exact formula / rule. No ambiguity. If the paper is vague, make the
call here and commit it; do not leave "we'll decide once we see the
data".

### Data scope
- **Slice:** training only — 2018-01-02 → 2022-12-30.
- **Test slice (2023–2024) is sealed.** Not touched in Gate 1 or Gate 2.

### Forward-return horizon(s) (locked)

### Bucketing / conditioning scheme (locked)
Expanding percentile with burn-in, or fixed thresholds stated here.
Specify exactly. No post-hoc re-bucketing.

### De-meaning (locked)
Yes / no. If yes, against what baseline?

### Significance standard (locked)
- Bootstrap method (day-level / trade-level):
- Iterations:
- Seed:
- CI width:

### Gate 1 pass criteria (ALL must hold)
1.
2.
3.

### Gate 1 fail criteria (ANY one kills)
-

### Stability check
- Half-split boundary:
- Pattern must hold in both halves: yes / no

## Gate 2 — MFFU Phase 1 viability

Only reached if Gate 1 passes. Gate 2 requires translating the edge
into an executable strategy.

### Strategy spec (locked before running)
- Entry rule:
- Exit rule:
- Stop rule:
- Position sizing:
- Max concurrent positions:
- Session filter (RTH-only? forced flat by 4:10 PM?):

### Tunable parameter grid (locked)
The parameters you will search, their ranges, and the step size. If
you cannot enumerate the full grid here, the hypothesis is not ready.

| Parameter | Range | Step | Rationale |
|---|---|---|---|
| | | | |

### Mechanical selection rule (locked)
Which parameter combo from the grid gets carried to the sealed test
slice? State the exact metric and tie-breaker. Examples:
- "Highest MFFU Phase 1 pass rate; tie broken by lowest median
  days-to-pass."
- "Highest Sharpe on training slice, subject to MFFU pass rate ≥ X%."

No post-hoc "actually this one looked better for reason Y" allowed.

### MFFU config used
- Account size: $100,000
- Profit target: $6,000
- Trailing DD (EOD): $3,000
- Consistency rule: 50%
- Source: `engine.prop_firm.MFFU_PHASE1_100K`

### Gate 2 pass criteria (pre-registered thresholds)
Suggested defaults — tighten if the edge is strong enough to justify,
loosen only with explicit reasoning recorded here:
- [ ] Monte Carlo pass rate ≥ 50% (with 95% Wilson CI lower bound > 40%)
- [ ] P(fail on trailing DD) ≤ 30%
- [ ] Observed max_day_frac at pass ≤ 0.5 on ≥ 80% of passing sims
- [ ] Median days-to-pass ≤ 30 calendar days

### Gate 2 fail criteria (ANY one kills)
-

## Test slice (sealed — touched ONLY if Gate 1 AND Gate 2 pass)

### One-shot test protocol
- Run the locked strategy with the mechanically-selected parameters
  exactly once on 2023–2024.
- No re-tuning, no parameter jiggling, no "let me just try X".
- Report the raw metrics. Success criteria pre-registered here:

### Sealed-slice success criteria
- [ ] MFFU Phase 1 pass rate ≥ [number] (state here, do not leave blank)
- [ ] Additional criteria:

---

## Test log (populated AFTER pre-registered sections above are locked — do not edit above)

### Gate 1 run
- **Date:**
- **Notebook:** `research/notebooks/[filename].ipynb`
- **Commit hash at time of run:**
- **Training slice bars used:**
- **Result summary:**
- **Gate 1 outcome:** PASS / FAIL
- **If fail — which criterion:**

### Gate 2 run (only if Gate 1 passed)
- **Date:**
- **Notebook:**
- **Parameter search results:**

| Combo | Pass rate | 95% CI | Med days | DD fails % | Consistency fails % |
|---|---|---|---|---|---|
| | | | | | |

- **Mechanically selected combo:**
- **Gate 2 outcome:** PASS / FAIL
- **If fail — which criterion:**

### Sealed-slice run (only if Gate 1 AND Gate 2 passed)
- **Date:**
- **Notebook:**
- **Result:**
- **Outcome vs pre-registered success criteria:**

### Final decision
- **Outcome:** (deploy-candidate / park / archive)
- **Reasoning:**
- **Budget spent:** 1 training-slice screen unit + [1 test-slice use if
  reached; otherwise 0].
