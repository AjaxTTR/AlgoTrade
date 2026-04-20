# Hypothesis: [short descriptive name]

> Copy this template to a new file in `research/hypotheses/` named after
> the hypothesis (e.g. `overnight_drift.md`, `vix_filter.md`). Fill it
> out BEFORE any code is written. The purpose is pre-registration —
> committing to expectations before seeing results.

## Paper
- **Title:**
- **Authors, year:**
- **Link / DOI / PDF location:**
- **Journal / venue:** (top-tier peer-reviewed? working paper? preprint?)

## Hypothesis (1-2 sentences, plain English)
What's the claimed edge?

## Proposed mechanism (why does this allegedly work?)
Market microstructure? Behavioral bias? Risk premium? Information
asymmetry? Liquidity provision? Inventory cycles?

**If the paper does not explain WHY, that is itself a red flag** —
statistical patterns without causal mechanisms are disproportionately
likely to be data-mining artifacts.

## Paper's test setup
- **Instrument(s):**
- **Frequency (daily / intraday / etc.):**
- **Period tested:**
- **Sample size (trades or N):**
- **Transaction cost assumption:**
- **Survivorship / selection bias discussion:** (did authors address it?)

## Signal definition (VERBATIM from paper)
Copy the formula, algorithm, or pseudocode exactly as stated in the
paper. If the paper uses mathematical notation, transcribe the full
equation with variable definitions.

```
[paste exact definition here]
```

**Ambiguity flags:** note EXACTLY which parts of the definition are
vague or under-specified. These are degrees of freedom we will have
to nail down, and each one inflates the risk of over-fitting when
we implement.

## Entry / exit rules (VERBATIM)
Same principle — exact quotes from the paper.

```
[paste rules here]
```

## Reported results
| Metric | Value |
|---|---|
| Sharpe ratio | |
| Annualised return | |
| Max drawdown | |
| t-statistic (if given) | |
| Win rate | |
| Number of trades | |
| Period | |

**Credibility check** (Harvey, Liu & Zhu 2016 framework):
- [ ] t-stat > 3.0?  (lower → higher false-discovery risk)
- [ ] Sharpe > 1.0 on paper data?
- [ ] Robustness tests shown across subperiods?
- [ ] Transaction costs realistic for the instrument / frequency?

## Author-acknowledged caveats
What did the authors say the limits were? Worth reading critically —
authors often downplay. Look specifically for:
- Survivorship bias
- Look-ahead bias (accidental use of future info)
- Transaction cost assumptions
- Period-specificity (only works in certain decades?)
- Sensitivity to parameter choices

## Your read (the researcher's own assessment)
- **Why does this interest you?**
- **Do you believe the mechanism?**
- **Any suspicion it's data-mined?**
- **Would you trade this on your own money?** (honest answer — if no, why are we testing it?)

## Adaptation to our context (pre-registered BEFORE coding)
- **Our instrument (NQ futures) vs theirs:**
- **Our frequency (15-min) vs theirs:**
- **Our period (2017-2024 active, 2025-2026 Yahoo OOS) vs theirs:**
- **Other adaptations we'll need:**

Every adaptation introduces one degree of freedom that inflates the
risk of false-positive results. List each one explicitly.

## Pre-registered expectations (fill in BEFORE running any code)

### Replication test (paper's own setup, as close as our data allows)
- **Sharpe we'd accept as successful replication:**
- **Sharpe below which we conclude the paper doesn't replicate:**

### Port to NQ 15-min
- **Sharpe we'd consider "edge exists on NQ":**
- **Sharpe below which we'd drop the hypothesis:**

### Decision rule
- [ ] If replication fails → archive, do not port
- [ ] If replication succeeds but NQ port fails → archive with note
- [ ] If NQ port succeeds but only marginally (Sharpe < 0.5) → park,
      revisit if we run out of higher-conviction hypotheses
- [ ] If NQ port clearly succeeds (meets pre-registered threshold) →
      promote to live integration research (Priority 0.7 Edge Mode
      validation, then Deployment Mode)

## Test log (populated as work progresses — DO NOT edit pre-registered sections above)

### Replication run
- **Date:**
- **Notebook:** `research/notebooks/[filename].ipynb`
- **Result Sharpe:**
- **Did it match pre-registered expectation?** (yes / no / marginal)
- **Commentary:**

### NQ port run
- **Date:**
- **Notebook:**
- **Result Sharpe:**
- **Did it match pre-registered expectation?**
- **Commentary:**

### Final decision
- **Outcome:** (promote / park / archive)
- **Reasoning:**
- **If archived — what would make us revisit?**
