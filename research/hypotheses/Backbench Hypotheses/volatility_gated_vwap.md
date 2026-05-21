# Hypothesis: Volatility-Gated VWAP Reversion — two-stage exit with hard ATR floor and prior-bar-low trail

> Pre-registered 2026-05-21. Locked before any code is written. Sections
> above the Test log are frozen at commit time; the Test log is
> populated after runs complete.

---

## Source & contamination bookkeeping

This hypothesis was **pre-written** by the user before the ATWS or VWAP
Reversion screens ran. The two-stage trailing-exit design is not a
result-informed redesign of either prior hypothesis. The user is
choosing to deploy it on the 2018–2022 training slice now, informed by
the prior two screens showing that asymmetric exits failed and
ATR-shaped floors did not. Soft contamination cost is acknowledged.

**Budget bookkeeping: screen #6 on the contaminated 2018–2022 slice.**
Prior five screens (overnight, HTF, open momentum, ATWS, VWAP
Reversion) all failed or returned weak on the primary run. The
contamination floor is now meaningful and is recorded.

---

## Hypothesis (1-2 sentences, plain English)

The VWAP-reversion entry signal carries some edge, but anchor-touch
and time-only exits give it back. A two-stage exit — hard ATR
invalidation floor for the full life of the trade, plus a prior-bar-low
trailing stop that activates once price has recovered above a
trail-trigger z-threshold — should retain the edge by letting winners
run during high-momentum snap-backs while bounding adverse outcomes.

## Proposed mechanism

Two structural claims:

1. **The hard ATR floor caps the loss tail.** This was the
   structurally missing piece in VWAP Reversion Run 1 (no
   adverse-price stop → 45% time-exits at full intra-window loss).
   2.5× ATR is a wide stop that respects the dislocation depth at
   entry (price is already -2σ extended) while still capping the
   per-trade worst case.

2. **The trailing exit lets the structural reversion run.** Cantarutti
   says reversion probability decays with time; Bali says downside
   shocks resolve fast and violently when they do resolve. A position
   that has snapped back above the trail-trigger threshold has already
   begun reverting — the trail captures whatever further upside the
   institutional liquidity step-in produces, exiting only when the
   prior-bar-low is broken (i.e. the snap-back stalls).

The "None" case in the Gate 2 grid (VWAP-touch limit + hard ATR
floor) is a deliberate control: if the None case performs
indistinguishably from the trailing cases, the trailing logic is not
the source of any observed edge.

## MFFU compatibility checklist

- [x] No position held past 4:10 PM ET. EOD-flat at 15:45 NY.
- [x] No reliance on infrequent jackpot days — trail exits and ATR
      floor produce many small-to-medium outcomes.
- [x] Plausible ≥ 8 independent trade days per pass attempt. Same
      entry rule as VWAP Reversion, which produced 221 trades over 5
      years on the ATR baseline.

## Gate 1 — Edge measurement (v2)

### Signal definition (locked)

Same VWAP feature set as `vwap_reversion.md`:

- `typical_price = (high + low + close) / 3`
- `vwap` = cumulative `(typical_price * volume) / volume` since 09:30
  NY reset, RTH only.
- `vwap_dev = close - vwap`
- `intraday_std` = expanding std of `vwap_dev` since reset
  (min_periods=2).
- `vwap_z = vwap_dev / intraday_std`
- `signal_zscore = vwap_z.shift(1)`
- `target_price = vwap.shift(1)` (used only when trail_trigger_z = None)

No new feature engineering required.

### Data scope

- **Slice:** training only — 2018-01-02 → 2022-12-30.
- **Test slice (2023–2024) is sealed.**
- **Locked vault (2005–2015) not yet acquired.**

### Significance standard (locked)

- Trade-level bootstrap, 10,000 iterations, seed 42, 95% CI on
  per-trade expectancy.

### Execution wrappers — TWO runs

**Shared entry condition:**
- `signal_zscore <= -2.0`
- 10:30 NY ≤ bar_time ≤ 15:00 NY
- LONG only, 1 contract

**Shared EOD rule:** EOD-flat at 15:45 NY market exit.
**Shared costs:** $5 round-trip, 0.25 spread.
**Shared sizing:** 1 contract, no scaling.
**Shared concurrency cap:** max 1 concurrent position.
**Shared time-stop:** `max_hold_bars = 6` (15-min bars → 90 min).
**Shared hard ATR floor:** **2.5× ATR(14)** below entry fill, active
for the **full life of the trade** regardless of stage.

**Run 1 — Two-Stage Exit (primary):**
- **Stage 1 (Passive Hold):** From entry until `vwap_z >
  trail_trigger_z` at bar close. During this phase, exits are:
  hard ATR stop, time-stop, EOD.
- **Stage 2 (Active Trail) — sticky, fires once and stays active:**
  Once `vwap_z > trail_trigger_z` at any bar's close, mark the
  position trail-active. Trail stop level = highest prior-bar low
  observed since activation (ratchets up, never down). Exit at market
  if current bar's low ≤ trail stop level.
- **Both exit floors live in Stage 2:** trail break AND hard ATR floor
  AND time-stop AND EOD all remain checked every bar — whichever
  fires first wins.
- **Re-crossing below `trail_trigger_z` after activation does NOT
  reset.** Trail mode is sticky.
- **No upside profit target.** Only profit-side exit is the trail
  break.

**Run 2 — Default ATR wrapper (cross-baseline):**
- TP: 2.0 × ATR(14) above entry fill.
- Stop: 1.5 × ATR(14) below entry fill.
- Same entries, EOD, costs, sizing, concurrency.

### Gate 1 base-case parameter values (locked)

| Parameter | Gate 1 value |
|---|---|
| `entry_z` | **-2.00** |
| `max_hold_bars` | **6** |
| `trail_trigger_z` | **-1.0** |
| Hard ATR stop multiple | **2.5** |
| Entry window | 10:30–15:00 NY |
| EOD-flat | 15:45 NY |

### Reported metrics (BOTH runs)

1. Total trade count
2. Win rate
3. Gross profit vs gross loss (and net profit factor)
4. t-statistic on per-trade P&L; expectancy with 95% bootstrap CI
5. Six Gate 1 v2 STRONG/WEAK indicators
6. Exit-reason breakdown (trail / hard-stop / time / EOD / target)
7. **User's analytical evaluation focus: Profit / Max DD ratio.**
   (Gate 2 selection rule remains MFFU pass-rate-based per
   `gate2_v2.md` — this is the Gate 1 reading the user is judging on,
   not a methodology change.)

### Methodological flags carried into result interpretation

- **Hard ATR floor is locked at 2.5× and not in the Gate 2 grid.**
  Stability of result to this choice is not tested under the locked
  pre-reg. Future hypotheses may explore it; this one does not.
- **Trail is pegged to prior-bar low on 15-min bars.** This is a
  relatively loose trail — the prior bar's low can be meaningfully
  below current price. By design (per user spec: "avoid chopping out
  on minor pullbacks"), but worth noting that snap-back gains can be
  given back substantially before the trail fires.
- **Sticky-Stage-2 is by design.** Once the trail activates, the hard
  ATR floor is still live but the structural premise of the trade has
  shifted from "reversion expected" to "ride the snap-back". If price
  re-crosses below `trail_trigger_z`, the trade does NOT revert to
  Stage 1 logic.

### Gate 1 user-decision indicators

Six STRONG/WEAK readings inform user go/no-go on Gate 2. User has
flagged Profit/Max DD ratio as the analytical metric of primary
interest.

## Gate 2 — MFFU Phase 1 viability

Reached only if Run 1 Gate 1 measurement is judged strong by the user.

### Strategy spec (locked)

Two-stage exit as defined for Run 1 above, with grid-tunable
parameters as below. Hard ATR multiple, entry window, EOD time, sizing,
and direction are **fixed, not tunable.**

### Tunable parameter grid (locked)

3 parameters × 3 levels = **27 combinations.** At the cap.

| Parameter | Range | Step | Rationale |
|---|---|---|---|
| `entry_z` | [-1.75, -2.00, -2.25] | 0.25 | Shoulder depth. |
| `max_hold_bars` | [4, 6, 8] | 2 | Reversion-window tightness. 60/90/120 min. |
| `trail_trigger_z` | [None, -1.0, -0.5] | n/a | None = control (VWAP-touch TP + hard ATR floor, no trail). -1.0 and -0.5 = trail activates at moderate / deep recovery. |

**`trail_trigger_z = None` behaviour (control case):** VWAP-touch limit
TP (high ≥ vwap.shift(1)) replaces the trail. Hard ATR floor, time-stop,
and EOD all remain active. Purpose: ensures any positive result on the
trailing cases is attributable to the trail, not just to the ATR
floor.

### Walk-forward shape (locked)

Per `gate2_v2.md`: 18-month train / 6-month test, slid 6 months across
2017–2022 (~9 chunks). Per chunk: tune by in-sample MFFU Phase 1 pass
rate; measure OOS pass rate on the test tail.

### Mechanical selection rule (locked)

**Median OOS MFFU Phase 1 pass rate, ties broken by lowest
cross-chunk variance.** Unchanged from gate2_v2.md. Profit/Max DD is
the user's analytical Gate 1 metric, NOT the Gate 2 selection rule.

### MFFU config

- Account: $100,000 | Target: $6,000 | Trailing DD: $3,000 | Consistency: 50%
- `engine.prop_firm.MFFU_PHASE1_100K`
- Whole-day PnL block permutation.

### Gate 2 pass criteria

- [ ] MC pass rate ≥ 50% (Wilson lower bound > 40%)
- [ ] P(fail trailing DD) ≤ 30%
- [ ] max_day_frac ≤ 0.5 on ≥ 80% of passing sims
- [ ] Median days-to-pass ≤ 45

### Gate 2 fail criteria (ANY one kills)

- MC pass rate < 50% on selected combo
- Wilson lower bound ≤ 40%
- P(fail trailing DD) > 30%
- max_day_frac > 0.5 on ≥ 20% of passing sims
- Median days-to-pass > 45

## Test slice (sealed — only if Gate 1 AND Gate 2 pass)

### Sealed-slice success criteria

- [ ] MFFU Phase 1 pass rate ≥ 50%
- [ ] Per-trade expectancy 95% bootstrap CI lower bound > $0 net
- [ ] Median days-to-pass ≤ 45
- [ ] Pass-rate drop ≤ 15 pp vs Gate 2 median OOS

---

## Test log (populated AFTER pre-registered sections are locked — do not edit above)

### Gate 1 run
- **Date:**
- **Notebook:** `research/notebooks/volatility_gated_vwap_gate1.ipynb`
- **Commit hash at time of run:**
- **Run 1 result summary (Two-Stage Exit):**
- **Run 2 result summary (Default ATR baseline):**
- **Six v2 indicators (Run 1):**
- **Six v2 indicators (Run 2):**
- **Profit / Max DD (Run 1):**
- **Exit-reason breakdown (Run 1):**
- **Gate 1 user-decision outcome:** PROGRESS / PARK / ARCHIVE
- **Reasoning:**

### Gate 2 run (only if Gate 1 progressed)
- **Date:**
- **Notebook:**
- **Parameter search results:**

| Combo (entry_z, max_hold_bars, trail_trigger_z) | Median OOS pass rate | Cross-chunk variance | 95% Wilson CI | Med days | DD fails % | Consistency fails % |
|---|---|---|---|---|---|---|
| | | | | | | |

- **Mechanically selected combo:**
- **Gate 2 outcome:** PASS / FAIL

### Sealed-slice run (only if Gate 1 AND Gate 2 passed)
- **Date:**
- **Notebook:**
- **Result:**

### Final decision
- **Outcome:** (deploy-candidate / park / archive)
- **Reasoning:**
- **Budget spent:** screen #6 on contaminated training slice.
