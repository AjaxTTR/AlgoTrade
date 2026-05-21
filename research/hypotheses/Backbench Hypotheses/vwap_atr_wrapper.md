# Hypothesis: VWAP-z entry × symmetric ATR wrapper — Gate 2 viability

> Pre-registered 2026-05-21. Locked before any code is written.

---

## ⚠️ Result-informed disclosure

**This pre-reg is result-informed and the user has explicitly accepted
that cost.** The design — VWAP-z entries with a symmetric 1.5×/2.0× ATR
wrapper — emerged as the cross-baseline (Run 2) in
`vwap_reversion.md` and `volatility_gated_vwap.md`. The primary runs
in those pre-regs (asymmetric exit designs) all FAILed; the ATR
baseline consistently measured 6/6 STRONG on Gate 1 but with a t-stat
of 1.24 and a per-trade expectancy 95% CI of [−$24.14, +$114.99] —
encouraging but not statistically significant.

The decision to promote the baseline to a primary in its own pre-reg
is being made AFTER seeing that result. Per the contamination
discipline, this is a soft-contamination event: the entry × exit
combination was not selected blind. The Gate 2 grid (locked below) is
explicitly offset from the observed-good values so the walk-forward
does not just confirm the corner where the result lived.

**Budget bookkeeping: screen #7 on the contaminated 2018–2022 training
slice.** Six prior screens (overnight, HTF, open momentum, ATWS, VWAP
Reversion, Volatility-Gated VWAP) — five FAIL primary, one PARK with
inconclusive baseline.

Any positive Gate 2 result must be interpreted with these
caveats. The cleanest validation remains the locked vault (2005–2015,
not yet acquired) and the sealed test slice (2023–2024, untouched).

---

## Hypothesis

LONG-only entries on a moderately deep negative VWAP-z dislocation,
exited with a symmetric ATR-shaped stop and target, capture enough of
the institutional liquidity-vacuum reversion on NQ to survive MFFU
Phase 1 mechanics across a walk-forward.

## Proposed mechanism

Inherited from `vwap_reversion.md`:

- Daily-reset VWAP measures intraday institutional fair value.
- Price ≥ 2σ below VWAP triggers mechanical buying from VWAP-benchmarked
  execution algorithms.
- Symmetric ATR exits cap loss and let winners run a fixed,
  volatility-normalised distance without an anchor that drifts toward
  the trade (the failure mode of EMA and VWAP-touch exits).

The new claim here is **viability under MFFU Phase 1 mechanics**, not
edge existence — which the prior Gate 1 reading neither confirmed nor
ruled out.

## Source

- VWAP Reversion Gate 1 Run 2 result (cited as Gate 1 reading for this
  pre-reg; see below).
- Three-paper structural prior unchanged (Bali, Mirpoorian, Cantarutti).

Replication status: not replicated on source instruments. Structural
priors only.

## Gate 1 — already measured

This pre-reg **cites the existing Run 2 measurement** from
`vwap_reversion.md` and `volatility_gated_vwap.md` as its Gate 1
reading. The Run 2 code path is identical across those pre-regs and
this one:

- Entries: `signal_zscore <= -2.0`, 10:30–15:00 NY, LONG only, 1 contract.
- Exits: 1.5× ATR(14) stop, 2.0× ATR(14) target, EOD-flat 15:45 NY.
- Costs: $5 round-trip, 0.25 spread.
- Concurrency: max 1.

**Gate 1 result cited from prior runs:**

| Element | Value | Verdict |
|---|---|---|
| Profit factor net | 1.202 | STRONG |
| Net P&L | +$9,900.88 | STRONG |
| Sample size | 221 | STRONG |
| Single-day concentration | 0.173 (max $1,712.50) | STRONG |
| Half-stability | yes | STRONG |
| Profit / Max DD | 2.652 (max DD $3,734) | STRONG |
| **Overall** | **6 / 6 STRONG** | |

| Supplementary | Value |
|---|---|
| Win rate | 52.0% |
| Expectancy / trade | +$44.80 |
| 95% bootstrap CI | [−$24.14, +$114.99] |
| t-stat | +1.240 |

**The CI brackets zero.** The Gate 1 result is consistent with a small
real edge AND with no edge at all (with 221 trades and t=1.24, roughly
1 in 9 random walks could produce this signature). Gate 2 is being run
because the user judges the indicator pattern strong enough to invest
the walk-forward measurement cost, NOT because Gate 1 has confirmed
edge.

No Gate 1 re-run; that would consume a degree of freedom without
producing new information.

## Gate 2 — MFFU Phase 1 walk-forward viability

### Strategy spec (locked)

- **Entry:** `signal_zscore <= entry_z`, 10:30–15:00 NY, LONG only, 1
  contract.
- **Exit:** symmetric ATR wrapper — `stop_atr_multiple × ATR(14)` below
  entry, `tp_atr_multiple × ATR(14)` above entry, EOD-flat 15:45 NY.
- **Max concurrent positions:** 1.
- **Costs:** $5 round-trip, 0.25 spread.
- **Sizing:** 1 contract (Gate 2 measures viability of unit edge).

### Tunable parameter grid (locked, OFFSET FROM OBSERVED-GOOD VALUES)

3 parameters × 3 levels = **27 combinations**. At the cap.

| Parameter | Range | Step | Rationale |
|---|---|---|---|
| `entry_z` | [-1.75, -2.00, -2.25] | 0.25 | Shoulder depth. Centered on the locked Gate 1 base case to test stability of that specific entry threshold. |
| `stop_atr_multiple` | [1.0, 1.25, 1.5] | 0.25 | Stop tightness. **The observed-good value (1.5) is on the WIDE edge of the grid, not the center.** Forces walk-forward to consider tighter stops. |
| `tp_atr_multiple` | [1.5, 2.0, 2.5] | 0.5 | Target distance. The observed-good value (2.0) is mid-grid; 1.5 and 2.5 test sensitivity. |

Window times (10:30–15:00), EOD-flat (15:45), direction (LONG),
concurrency cap (1), and costs are **fixed, not tunable.**

### Walk-forward shape (locked)

Per `gate2_v2.md`: **18-month train / 6-month test, slid 6 months
forward across 2017–2022 (~9 chunks).** Per chunk: in-sample tuning by
MFFU pass rate, OOS measurement on the chunk's 6-month test tail.

### Mechanical selection rule (locked)

**Median OOS MFFU Phase 1 pass rate across chunks, ties broken by
lowest cross-chunk variance of OOS pass rate.** Per `gate2_v2.md`.

### MFFU config

- Account: $100,000 | Target: $6,000 | Trailing DD: $3,000 | Consistency: 50%
- `engine.prop_firm.MFFU_PHASE1_100K`
- Whole-day PnL block permutation.
- 5,000 Monte Carlo simulations per pass-rate evaluation.

### Gate 2 pass criteria

- [ ] Median OOS MC pass rate ≥ 50% with 95% Wilson lower bound > 40%
- [ ] P(fail on trailing DD) ≤ 30% on the selected combo
- [ ] max_day_frac ≤ 0.5 on ≥ 80% of passing sims
- [ ] **Median days-to-pass ≤ 45 calendar days** (unchanged from prior
      VWAP pre-regs)

### Gate 2 fail criteria (ANY one kills)

- Median OOS MC pass rate < 50% on selected combo
- Wilson lower bound ≤ 40%
- P(fail trailing DD) > 30%
- max_day_frac > 0.5 on ≥ 20% of passing sims
- Median days-to-pass > 45

## Test slice (sealed — only if Gate 2 passes)

### One-shot test protocol

- Run mechanically-selected combo exactly once on 2023-01-03 →
  2024-12-31.
- No re-tuning, no second look.

### Sealed-slice success criteria

- [ ] MFFU Phase 1 pass rate ≥ 50%
- [ ] Per-trade expectancy 95% bootstrap CI lower bound > $0 net
- [ ] Median days-to-pass ≤ 45
- [ ] Pass-rate drop ≤ 15 pp vs Gate 2 median OOS

---

## Test log (populated AFTER pre-reg sections above are locked)

### Gate 2 run
- **Date:** 2026-05-21
- **Notebook:** not run as a notebook — executed inline via Bash (Python
  one-liner driving `gate2_evaluate`). Result recorded here for the
  audit trail.
- **Walk-forward shape used:** 18-month train / 6-month test, slid 6
  months, `train_start=2018-01-02`, `train_end=2022-12-30`. Yielded
  **6 chunks** (not the documented 9 — project data starts 2018-01-02,
  not 2017-01-01).
- **Chunks evaluated:** 6
- **MC sims per evaluation:** 5,000, seed=42

**Per-chunk OOS results:**

| # | Test window | OOS pass rate | DD-fail % | OOS trades | Selected combo (entry_z, stop_atr, tp_atr) |
|---|---|---|---|---|---|
| 0 | 2019-07-02 → 2020-01-01 | 0.0% | 75.2% | 44 | -2.25 / 1.5 / 2.0 |
| 1 | 2020-01-02 → 2020-07-01 | 0.0% | 41.6% | 10 | -1.75 / 1.5 / 2.5 |
| 2 | 2020-07-02 → 2021-01-01 | 0.0% | 0.0% | 3 | -2.25 / 1.5 / 2.5 |
| 3 | 2021-01-02 → 2021-07-01 | 0.0% | 97.9% | 26 | -1.75 / 1.0 / 2.0 |
| 4 | 2021-07-02 → 2022-01-01 | 0.0% | 18.8% | 14 | -1.75 / 1.25 / 2.0 |
| 5 | 2022-01-02 → 2022-07-01 | 0.0% | 0.0% | 0 | -2.25 / 1.0 / 2.0 |

- **Median OOS pass rate:** 0.0%
- **Cross-chunk variance:** 0.00 (degenerate — every chunk is 0%)
- **Chunks ≥ 50% threshold:** 0 / 6
- **Mechanically selected combo:** undefined — selection rule is
  median OOS pass rate with ties broken by lowest cross-chunk
  variance, but every combo / chunk hit 0%. No combo survives.
- **Gate 2 outcome:** **FAIL**
- **If FAIL — which criterion:** every Gate 2 pass criterion failed:
  median pass rate (0% < 50%); Wilson lower bound (0% ≤ 40%);
  per-chunk trade density was catastrophically thin (0–44 OOS
  trades / 6-month window). Chunk 5 produced zero OOS trades under any
  combo — 2022 H1 vol profile delivered no -2σ VWAP dislocations.
- **Parameter selection wandered** — `entry_z` swung between -1.75 and
  -2.25 across chunks; `stop_atr` between 1.0 and 1.5. No stable
  optimum, consistent with in-sample tuning fitting to noise.

### Sealed-slice run
- **NOT TOUCHED.** Gate 2 failed, sealed slice (2023–2024) preserved.

### Final decision
- **Outcome:** **archive**
- **Reasoning:** The result-informed promotion of the symmetric ATR
  baseline to a primary hypothesis confirmed that aggregate Gate 1
  metrics (6/6 STRONG, profit factor 1.20, profit/DD 2.65) are not
  prop-firm-aware — they don't see the trailing-DD path or the 50%
  consistency rule. The same numbers that looked "encouraging but
  inconclusive" on Gate 1 produced 0/6 chunk pass rate under MFFU
  Phase 1 mechanics. This is exactly the Gate 1 → Gate 2 leak that
  the two-gate design is built to surface; the discipline worked.

  Cumulative tally on the negative-shock-on-NQ-15min entry family:
  3 primary Gate 1 FAILs (ATWS, VWAP Reversion, Volatility-Gated VWAP)
  + 1 Gate 2 FAIL (this run). The CI on the underlying Run 2 reading
  always bracketed zero. The empirical pattern is now unambiguous:
  whatever statistical hint of edge the entries carry is too thin to
  be extractable on this instrument/timeframe under MFFU mechanics.
- **Budget spent:** screen #7 on the contaminated 2018–2022 training
  slice. Sealed slice (2023–2024) and locked vault (2005–2015) both
  remain untouched.
- **What this opens up next session:** entry family is archived on
  this instrument/timeframe. Next-session candidates: pivot timeframe
  (5-min or 60-min — different microstructure), acquire locked vault
  for fresh data substrate, or pivot to a structurally different
  hypothesis (not negative-shock-mean-reversion-flavour).
