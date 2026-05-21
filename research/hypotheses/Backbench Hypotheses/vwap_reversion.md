# Hypothesis: VWAP Reversion — LONG-only NQ on intraday VWAP dislocation

> Pre-registered 2026-05-21. Locked before any code is written. Sections
> above the Test log are frozen at commit time; the Test log is
> populated after runs complete.

---

## Locked confirmations (2026-05-21)

- `max_hold_bars = 6` for Gate 1 base case (90 min on 15-min bars).
  User rationale: ideal median representation of the time-stop.
- Run 2 (default ATR cross-baseline) **retained**. User rationale:
  needed to isolate whether alpha derives from the entry signal or the
  exit mechanics.

Pre-reg locked. No edits below this point until results are in.

---

## Hypothesis (1-2 sentences, plain English)

Sharp intraday sell-offs on NQ that drive price ≥ 2σ below the day's
volume-weighted average price (VWAP) trigger mechanical buying from
institutional execution algorithms benchmarked to VWAP, producing a
reliable mean-reversion back toward the anchor. A LONG-only intraday
strategy entering on a moderate-to-deep negative VWAP-z-score and
exiting on VWAP touch should produce positive expectancy on the NQ
training slice.

## Proposed mechanism

The mechanism is **microstructural and explicit**, not statistical
pattern-finding:

Multi-billion-dollar funds executing large NQ orders are mandated by
mandate or compensation structure to benchmark their average fill
against VWAP. When NQ trades materially below intraday VWAP, any LONG
order being worked toward a VWAP benchmark is mechanically a buy: every
contract bought below VWAP improves the fund's benchmark deviation.
This is not discretionary — it is algorithmic and price-insensitive
within tight bounds. The deeper the deviation below VWAP, the more
aggressive the institutional bid.

VWAP has two structural properties that distinguish it from EMA-based
anchors (the ATWS failure mode):

- **Cumulative-volume inertia.** By the entry window's start (10:30
  NY), the VWAP denominator already contains an hour of session
  volume. Each new bar adds a small fraction. Anchor drift toward an
  open position is dampened — the "target running to meet the trade"
  dynamic that gutted ATWS Run 1's wins is structurally weaker here.
- **Daily reset.** No carry-over from prior days. The anchor measures
  *today's* institutional fair value, not a recency-weighted average
  of recent regimes.

## Source

User-derived hypothesis built on the same three-paper structural prior
as ATWS:

1. **Bali, Demirtas, Levy — SSRN 1440781 (Asymmetric Rubber Band).**
   Reversion speed is higher on downside dislocations than upside. →
   Justifies LONG-only restriction.
2. **Mirpoorian et al. — SSRN 6506218 (Adaptive Anchor).**
   Static-lookback anchors bleed during trends. → VWAP is adaptive but
   anchored to *intraday volume* not to *recent price*, which is the
   property Mirpoorian flags as load-bearing.
3. **Cantarutti — SSRN 5310321 (First Hitting Time vs Half-Life).**
   Reversion probability decays with time in trade. → Justifies
   max_hold_bars time-stop and EOD-flat.

Plus the explicit institutional-execution mechanism above, which ATWS
did not have.

**ATWS prior:** the same entry conditions on an EMA anchor (ATWS Run 2,
default ATR wrapper) produced 159 trades, t-stat 0.731, 95% bootstrap
CI on per-trade expectancy [−$51.80, +$116.47]. **The CI brackets zero
— this is consistent with edge existing AND with no edge being
present.** VWAP is being motivated structurally, not on the ATWS Run 2
result.

Replication status: **not replicated on source instruments.** Papers
contribute structural priors; VWAP-reversion construction is novel to
this project. Flag carried into Test log interpretation.

## MFFU compatibility checklist (pre-flight)

- [x] Does the strategy hold any position past 4:10 PM ET? **No** —
      EOD-flat hard rule at 15:45 NY.
- [x] Does the strategy rely on infrequent large payoff days? **No** —
      VWAP-touch TP and -2σ-shoulder entries produce many small wins.
      Distribution shape compatible with 50% consistency rule.
- [x] Plausible ≥ 8 independent trade days per pass attempt? **Yes,
      expected.** -2σ on intraday VWAP with a 4.5-hour entry window
      should produce ≥1 trade per typical-to-active day. Concrete count
      is a Gate 1 measurement output.

## Gate 1 — Edge measurement (v2: measurement, not pass/fail)

Per `research/methodology/gate1_v2.md`, Gate 1 reports six STRONG/WEAK
indicators against malleable reference thresholds. The user makes the
call on whether to progress to Gate 2.

### Signal definition (locked)

Features computed in `engine/features.py` via a new `build_vwap_features()`:

- `typical_price = (high + low + close) / 3`
- `vwap` = cumulative-sum-since-09:30-NY-reset of `(typical_price *
  volume)` divided by cumulative-sum-since-reset of `volume`. **Daily
  reset at 09:30 NY (RTH only).** Globex / overnight volume excluded
  — anchor measures US institutional fair value only.
- `vwap_dev = close - vwap`
- `intraday_std` = expanding standard deviation of `vwap_dev` since the
  09:30 reset (`min_periods=2`).
- `vwap_z = vwap_dev / intraday_std`
- `signal_zscore = vwap_z.shift(1)` (entry uses prior bar's z — no
  look-ahead).
- `target_price = vwap.shift(1)` (TP anchor frozen at prior bar —
  prevents intra-bar look-ahead on the limit-touch exit).

### Data scope

- **Slice:** training only — 2018-01-02 → 2022-12-30.
- **Test slice (2023–2024) is sealed.** Not touched in Gate 1 or Gate 2.
- **Locked vault (2005–2015) not yet acquired** — reserved for one-shot
  validation post-deployment-decision.

### Bucketing / conditioning scheme (locked)

Fixed threshold: entry whenever `signal_zscore <= -2.0` and time
filter holds. No expanding-percentile bucketing.

### De-meaning (locked)

No. Strategy P&L is reported raw, net of costs.

### Significance standard (locked)

- Trade-level bootstrap of per-trade P&L.
- 10,000 iterations.
- Seed: 42.
- 95% CI on mean trade P&L (expectancy) reported alongside t-stat.

### Execution wrappers

Both runs share entries and EOD-flat. They differ only in TP / stop
logic.

**Shared entry condition:**
- `signal_zscore <= -2.0`
- 10:30 NY ≤ bar_time ≤ 15:00 NY
- LONG only, 1 contract

**Shared EOD rule:**
- If position open at 15:45 NY, exit at market on that bar's close.

**Shared costs:**
- $5 round-trip commission per trade.

**Shared sizing:**
- 1 contract per signal, no scaling, no compounding.

**Shared concurrency cap:**
- `max_concurrent_trades = 1`.

**Run 1 — VWAP-Touch + Time-Stop (primary):**
- TP: intra-bar limit — exit if `bar.high >= target_price`.
- **No adverse-price stop.** Only exits are TP, time-stop, and EOD.
- Time stop: force exit at market after `max_hold_bars = 6` bars
  (15-min bars → 90 minutes maximum hold). **Base case 6 pending user
  confirm.**
- This is a structural departure from ATWS Run 1 (which had a regime
  z-threshold stop). Justification: Cantarutti finding — reversion
  probability decays with time, so time itself is the invalidation
  signal. Mechanically, the trade gives the market a fixed window to
  revert; if it does not, the structural premise has failed.

**Run 2 — Default ATR wrapper (cross-hypothesis baseline) — confirm or strike:**
- TP: 2.0 × ATR(14) above entry fill.
- Stop: 1.5 × ATR(14) below entry fill.
- Pure cross-hypothesis baseline. Same entries; ATR-shaped exits.
- Allows comparing the VWAP-touch design against a symmetric volatility
  baseline, paralleling the ATWS pre-reg structure.

### Reported metrics (BOTH runs)

1. Total trade count
2. Win rate
3. Gross profit vs gross loss (and net profit factor)
4. t-statistic on per-trade P&L; expectancy (mean trade P&L) with 95%
   bootstrap CI

Plus the six Gate 1 v2 STRONG/WEAK indicators (profit factor net,
total net P&L, sample size, single-day concentration, half-stability,
profit / maxDD).

### Methodological flags carried into result interpretation

- **No adverse-price stop in Run 1.** A trade entered at -2σ that
  continues adverse holds for up to 6 bars (90 min). Per-trade worst
  case is bounded only by intra-window NQ volatility. This is by
  design (Cantarutti), but it interacts with Gate 2's trailing DD —
  expect higher per-trade loss tail vs ATWS Run 1.
- **Expanding-std denominator is thin at session start.** At 10:30 NY
  the std is computed from ~4 fifteen-min observations. Z-scores
  computed against a 4-sample std are noisy. The std stabilises through
  the day; the Gate 2 `start_time` grid sweep partially tests this.
- **Daily-reset VWAP excludes Globex.** A strategy run on the same
  signal but including overnight volume in VWAP would behave
  measurably differently. The reset boundary is a load-bearing locked
  choice, not a tunable.

### Gate 1 user-decision indicators

Six STRONG/WEAK readings inform user go/no-go on Gate 2. No automatic
pass criterion is pre-committed.

## Gate 2 — MFFU Phase 1 viability

Reached only if Run 1 Gate 1 measurement is judged strong by the user.

### Strategy spec (locked before running)

Carried verbatim from Run 1 above, with grid-tunable parameters as
listed below:
- Entry: `signal_zscore <= entry_z`, `start_time` ≤ bar_time ≤ 15:00
  NY, LONG only.
- TP: intra-bar limit on `high >= target_price` (vwap.shift(1)).
- Stop: time-stop at `max_hold_bars` bars, market exit.
- Position sizing: 1 contract.
- Max concurrent positions: 1.
- Session filter: RTH, EOD-flat at 15:45 NY.

### Tunable parameter grid (locked)

3 parameters × 3 levels = **27 combinations.** At the cap.

| Parameter | Range | Step | Rationale |
|---|---|---|---|
| `entry_z` | [-1.75, -2.00, -2.25] | 0.25 | Tests shoulder depth — shallower = more trades / weaker dislocation; deeper = fewer / stronger. |
| `max_hold_bars` | [4, 6, 8] | 2 | Tests reversion-window tightness. 4 = 60 min, 6 = 90 min, 8 = 120 min. Cantarutti predicts shorter is structurally cleaner; sample-size argues longer. |
| `start_time` | [10:00, 10:30, 11:00] | 30 min | Tests sensitivity to the expanding-std warm-up. Later start = thicker std denominator = more stable z; earlier = more entries but noisier threshold. |

VWAP definition, typical-price formula, daily-reset boundary,
end_time (15:00), EOD-flat (15:45), TP rule (VWAP-touch), direction
(LONG), and sizing (1 contract) are **fixed, not tunable.**

### Walk-forward shape (locked)

Per `research/methodology/gate2_v2.md`: 18-month train / 6-month test,
slid 6 months forward across 2017–2022 (≈9 chunks). Per chunk: tune
within the grid above by in-sample MFFU Phase 1 pass rate; measure OOS
pass rate on the chunk's 6-month test tail.

### Mechanical selection rule (locked)

Cross-chunk aggregation: **median OOS MFFU Phase 1 pass rate** across
chunks. Ties broken by **lowest cross-chunk variance** of OOS pass
rate. The combo selected by this rule is the one carried — only if
Gate 2 thresholds are also met — to the sealed test slice.

### MFFU config used

- Account size: $100,000
- Profit target: $6,000
- Trailing DD (EOD): $3,000
- Consistency rule: 50%
- Source: `engine.prop_firm.MFFU_PHASE1_100K`
- Monte Carlo permutation: whole-day PnL blocks (per 2026-05-05 fix).

### Gate 2 pass criteria (pre-registered thresholds)

- [ ] Monte Carlo pass rate ≥ 50% (with 95% Wilson CI lower bound > 40%)
- [ ] P(fail on trailing DD) ≤ 30%
- [ ] Observed max_day_frac at pass ≤ 0.5 on ≥ 80% of passing sims
- [ ] **Median days-to-pass ≤ 45 calendar days** — malleable per
      hypothesis per `feedback_gate2_thresholds.md`. Set to 45 for the
      same reasoning as ATWS pre-reg (MFFU industry loosening; this
      is a structurally low-frequency setup).

### Gate 2 fail criteria (ANY one kills)

- Monte Carlo pass rate < 50% on the selected combo.
- Wilson lower bound on pass rate ≤ 40%.
- P(fail on trailing DD) > 30% on the selected combo.
- max_day_frac > 0.5 on ≥ 20% of passing sims.
- Median days-to-pass > 45.

## Test slice (sealed — touched ONLY if Gate 1 AND Gate 2 pass)

### One-shot test protocol

- Run the locked strategy with the mechanically-selected parameter
  combo exactly once on 2023-01-03 → 2024-12-31.
- No re-tuning, no parameter jiggling, no second look.

### Sealed-slice success criteria

- [ ] MFFU Phase 1 pass rate ≥ 50%.
- [ ] Per-trade expectancy 95% bootstrap CI lower bound > $0 net of
      costs.
- [ ] Median days-to-pass ≤ 45.
- [ ] No structural deterioration vs Gate 2: pass rate drop ≤ 15
      percentage points relative to the selected combo's median OOS
      pass rate from the walk-forward.

---

## Budget bookkeeping

**Screen #5 on the contaminated 2018–2022 training slice.** Prior four
(overnight reversal, HTF alignment, open momentum, ATWS) all FAILed or
returned weak. User has acknowledged the degree-of-freedom cost and
the contamination floor.

Structural-anchor change (EMA → VWAP) is judged by the user to
constitute a meaningful hypothesis shift rather than an ATWS parameter
tweak. Recorded for audit.

---

## Test log (populated AFTER pre-registered sections above are locked — do not edit above)

### Gate 1 run
- **Date:**
- **Notebook:** `research/notebooks/vwap_gate1.ipynb`
- **Commit hash at time of run:**
- **Training slice bars used:**
- **Run 1 result summary (VWAP-Touch + Time-Stop):**
- **Run 2 result summary (Default ATR wrapper) — if retained:**
- **Six v2 indicators (Run 1):**
- **Six v2 indicators (Run 2) — if retained:**
- **Gate 1 user-decision outcome:** PROGRESS / PARK / ARCHIVE
- **Reasoning:**

### Gate 2 run (only if Gate 1 progressed)
- **Date:**
- **Notebook:**
- **Parameter search results:**

| Combo (entry_z, max_hold_bars, start_time) | Median OOS pass rate | Cross-chunk variance | 95% Wilson CI | Med days | DD fails % | Consistency fails % |
|---|---|---|---|---|---|---|
| | | | | | | |

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
  reached; otherwise 0]. **5th screen on contaminated training slice.**
