# Engine priority list

Known limitations / deferred work on `engine/` modules. Captured here so
we don't lose them when context rotates.

## High priority (tackle when we approach live strategy evaluation)

### 1. Monte Carlo shuffle in `simulate_prop_firm` does not preserve PnL-to-day clustering

**Location:** `engine/prop_firm.py::simulate_prop_firm`.

**What it does today:** on each sim, trade PnLs are permuted, then
re-assigned to a date template built from the original per-day trade
*counts*. This preserves "how many trades on day 1 vs day 2" but loses
"which specific PnLs landed together on the same day".

**Why it matters more under EOD trailing DD:** which PnLs cluster on
which days drives *when the EOD floor steps up*. Two runs with the
same set of PnLs but different clustering have genuinely different
pass probabilities under MFFU mechanics. Under the old intraday
trailing model the effect was smaller because the floor reset
trade-by-trade relative to the running peak; under EOD the floor is
frozen intraday and updates only at day close, so day composition is
a first-order driver of outcomes.

**What a fix would look like:** instead of shuffling PnLs against a
day-count template, shuffle at the day level — resample *days* (each
day carrying its actual bundle of trades) with replacement to a target
length. This preserves the natural PnL-to-day clustering and the
dependence structure within a day. Open question: how to handle sims
that run short of the profit target before exhausting sampled days
(extend with more sampled days, or cap at a horizon).

**When to fix:** before running Gate 2 for real on a strategy that
clears Gate 1. Pre-fix results are directionally informative but not
trustworthy as absolute pass-rate numbers.

## Medium priority

### 2. MFFU Phase 1 configs for $50k and $150k accounts not populated

**Location:** `engine/prop_firm.py::mffu_phase1_config`.

Only the $100k account ruleset is verified. Other sizes raise
`NotImplementedError`. Populate with exact confirmed numbers when a
run against a different account size is actually needed.

### 3. Phase 2 (funded account) economics not modelled

MFFU Phase 2 is a payout-structure question (100% of first $10k, then
90/10 split), not a pass/fail question. Deferred until we have a
strategy worth reasoning about expected payout for. Will be its own
function; do not bolt onto `simulate_prop_firm`.

---

## Next-session audit (user-driven review of this session's changes)

All changes below were made in the 2026-04-23 session under the new
two-gate + MFFU pipeline. Before running any hypothesis against them,
the user should line-read and sign off on each:

1. **`engine/prop_firm.py` rewrite.** Verify the EOD trailing DD logic
   matches MFFU's published rule (particularly: floor frozen intraday,
   updates only on EOD closed balance, `<=` treated as breach). Verify
   the consistency-rule check at pass time is applied in the direction
   you want (currently: hits target AND max-day/total > 0.5 ⇒ FAIL).
   Sanity-check the 5 test cases run inline this session against your
   own mental model.
2. **`MFFU_PHASE1_100K` constant.** Confirm every number matches the
   MFFU docs verbatim: balance $100k, target $6k, trailing DD $3k,
   consistency 50%, no time limit. One wrong number invalidates every
   Gate 2 sim run under it.
3. **`TEMPLATE_twogate.md` pass/fail thresholds.** The Gate 2
   suggested defaults (pass rate ≥ 50%, DD-fail ≤ 30%, max_day_frac
   ≤ 0.5 on ≥ 80% of passing sims, median days-to-pass ≤ 30) were
   chosen as reasonable starting points, not derived from data. Audit
   whether these are the right bars for your risk tolerance before
   committing a real hypothesis file against them.
4. **`CLAUDE.md` two-gate section.** Read end-to-end; confirm the
   description matches the collaboration model you want Claude to
   enforce next session.
5. **`feedback_two_gate_pipeline.md` memory.** This is what Claude
   will recall unprompted in future sessions. Confirm it captures the
   intent accurately.

The audit exists because all of the above was built fast and applies
to every future hypothesis. Finding a mistake here after committing a
pre-reg is expensive; finding it before is free.

---

## Developing hypothesis — current state

Captured here (not a pre-registration) so the next session has a
single place to pick up the thread.

**Source:** Rogul paper (`Research_papers/ssrn-6469419.pdf`). Hybrid
systematic-discretionary framework, tested on a small sample (N=42
trades across 3 non-contiguous months), parameters withheld by the
authors, headline result +$6,285 / PF 1.82 / Sharpe 1.65.

**Evidence quality:** thin. Parameter opacity + small-N + non-
contiguous sampling make this closer to an idea-source than a
reproducible result. User acknowledges this and has elected to treat
the paper as a *framework* to derive an independent hypothesis from,
rather than as something to replicate. Independent measurement on the
training slice is the load-bearing step, not the paper's numbers.

**Why this path at all:** Effect 1 (overnight → intraday reversal)
failed Gate 1 on 2026-04-23. Effects 2 (open momentum) and 3
(large-move reversal) remain unscreened but each additional naive
equities-port-to-futures screen burns training-slice budget. A
framework-derived hypothesis, if specified cleanly, is a different
kind of test — potentially a better use of remaining budget.

**What's NOT yet decided (open for next session):**
- Which specific conditional behaviour from the Rogul framework
  becomes the Gate 1 edge definition.
- Whether to run Effect 2 or 3 first as a cheaper naive-port screen
  before committing to a framework-derived hypothesis.
- The tunable-parameter grid + mechanical selection rule for Gate 2
  (cannot be specified until a strategy spec exists).

**What IS locked:**
- Any strategy hypothesis goes through `TEMPLATE_twogate.md`.
- Gate 2 uses `MFFU_PHASE1_100K` exclusively.
- Test slice 2023–2024 remains sealed.
- Budget: 1 training-slice screen already spent (Effect 1 FAIL).

**Claude's role on resume:** implement what the user pre-registers;
flag methodological issues; do NOT pitch which of the candidate paths
to take next (strategy-idea generation remains user-owned).
