# Hypothesis: ATWS — Asymmetric Trailing-Window Shock (z-score mean reversion, LONG-only NQ)

> Pre-registered 2026-05-21. Locked before any code is written. Sections
> above the Test log are frozen at commit time; the Test log is
> populated after runs complete.

---

## Hypothesis (1-2 sentences, plain English)

Violent downside dislocations on NQ — defined as a negative z-score
stretch of the close against a short adaptive anchor — are followed by
faster, more reliable mean reversion than equivalent upside stretches
(fear resolves faster than greed). A LONG-only intraday strategy that
enters on a moderate-shoulder negative z-stretch and exits when price
re-touches the anchor should produce positive expectancy on the NQ
training slice.

## Proposed mechanism

Sharp intraday sell-offs create transient liquidity vacuums: passive
buyers step back, stops cascade, and a small population of market
makers / liquidity providers absorbs the imbalance at a discount.
Reversion to the local trend is mechanical rather than informational —
the dislocation is microstructural, not driven by new fundamentals.
Upside stretches do not exhibit the same asymmetry because greed
unwinds slower and is more often news-driven.

The asymmetry, the adaptive-anchor choice, and the strict time-stop are
each anchored in published findings (see Source).

## Source

Three-paper synthesis (user-supplied 2026-05-21):

1. **Bali, Demirtas, Levy — SSRN 1440781 (Asymmetric Rubber Band).**
   Mean-reversion speed (θ) is significantly higher on sharp drops than
   on rallies. → Justifies LONG-only restriction.
2. **Mirpoorian et al. — SSRN 6506218 (Adaptive Anchor).**
   Static-lookback reversion strategies bleed during trends; reversion
   is only profitable when the anchor adapts to recent velocity. →
   Justifies EMA over SMA.
3. **Cantarutti — SSRN 5310321 (First Hitting Time vs Half-Life).**
   Probability of successful reversion decays with time spent in trade.
   → Justifies strict EOD-flat and tight invalidation stop.

Replication status: **not replicated on source instruments.** The
papers contribute structural assumptions (asymmetry, EMA, time-decay),
not a headline number being reproduced. The ATWS construction itself is
novel to this project. Flag carried into Test log interpretation.

## MFFU compatibility checklist (pre-flight)

- [x] Does the strategy hold any position past 4:10 PM ET? **No** —
      EOD-flat hard rule at 15:45 NY (see Strategy spec).
- [x] Does the strategy rely on infrequent large payoff days? **No** —
      moderate-shoulder entries (z ≤ -1.5) and a fixed anchor TP
      produce many small wins, not jackpot days. The 50% consistency
      rule is the natural direction of this distribution.
- [x] Plausible ≥ 8 independent trade days per pass attempt? **Yes,
      expected.** Moving from -2.5 (tail) to -1.5 (shoulder) was
      explicitly chosen to lift trade frequency into a regime where
      statistical inference is meaningful. Concrete count is a Gate 1
      measurement output.

## Gate 1 — Edge measurement (v2: measurement, not pass/fail)

Per `research/methodology/gate1_v2.md`, Gate 1 reports six STRONG/WEAK
indicators against malleable reference thresholds. The user makes the
call on whether to progress to Gate 2. The hypothesis does not
pre-commit to numerical pass/fail thresholds at Gate 1 — only to the
measurement protocol and to the requested supplementary metrics below.

### Signal definition (locked)

Features computed in `engine/features.py`:

- `anchor_mean = EMA(close, span=20)`
- `rolling_std = close.rolling(20).std()`
- `z_score = (close - anchor_mean) / rolling_std`
- `signal_zscore = z_score.shift(1)` (entry uses prior bar's z)
- `target_price = anchor_mean.shift(1)` (exit anchor frozen at prior
  bar — prevents intra-bar look-ahead for the limit-touch TP)

### Data scope

- **Slice:** training only — 2018-01-02 → 2022-12-30.
- **Test slice (2023–2024) is sealed.** Not touched in Gate 1 or Gate 2.
- **Locked vault (2005–2015) is not yet acquired** and is reserved for
  one-shot validation post-deployment-decision.

### Forward-return horizon(s) (locked)

N/A — ATWS is a strategy hypothesis, not a forward-return screen.
Outcomes are realised per-trade P&L on the executed wrapper.

### Bucketing / conditioning scheme (locked)

Fixed threshold: entry whenever `signal_zscore <= -1.5` and time
filter holds. No expanding-percentile bucketing.

### De-meaning (locked)

No. Strategy P&L is reported raw, net of costs.

### Significance standard (locked)

- Trade-level bootstrap of per-trade P&L.
- 10,000 iterations.
- Seed: 42.
- 95% CI on mean trade P&L (expectancy) reported alongside t-stat.

### Execution wrappers — TWO runs (cross-hypothesis comparability)

Both runs share the same entry and the same EOD-flat. They differ only
in TP / stop logic. Gate 1 metrics are reported for both.

**Shared entry condition:**
- `signal_zscore <= -1.5`
- 10:00 NY ≤ bar_time ≤ 13:00 NY
- LONG only, 1 contract

**Shared EOD rule:**
- If position open at 15:45 NY, exit at market on that bar's close.
  Prop-firm overnight margin + gap-risk hard constraint, and the
  Cantarutti time-decay finding makes carrying further structurally
  unjustified.

**Shared costs:**
- $5 round-trip commission per trade, applied as the Gate 1 default.

**Shared sizing:**
- 1 contract per signal, no scaling, no compounding. (Sizing decisions
  deferred to Gate 4 prop-sim.)

**Run 1 — Dynamic Regime Exits (primary):**
- TP: intra-bar limit — exit if `bar.high >= target_price`.
- Stop: evaluate at bar close — if `z_score <= -2.25`, exit at the
  **open of the next bar** at market. (Close-of-bar evaluation,
  next-bar-open execution — no intra-bar look-ahead.)

**Run 2 — Default ATR wrapper (baseline):**
- TP: 2.0 × ATR(14) above entry fill.
- Stop: 1.5 × ATR(14) below entry fill.
- Pure cross-hypothesis baseline so ATWS results are comparable to the
  prior FAILed screens (overnight, HTF, open momentum), which all ran
  under the same default wrapper.

### Reported metrics (BOTH runs)

Per the user's directive:
1. Total trade count
2. Win rate
3. Gross profit vs gross loss (and net profit factor)
4. t-statistic on per-trade P&L; expectancy (mean trade P&L) with 95%
   bootstrap CI

Plus the six Gate 1 v2 STRONG/WEAK indicators (profit factor net,
total net P&L, sample size, single-day concentration, half-stability,
profit / maxDD).

### Expected behaviour

The strategy utilises dynamic, data-driven exits. While initial
thresholds are stationed at a theoretical 1:2 volatility distance
(entry at -1.5σ, regime-invalidation stop at -2.25σ → 0.75σ adverse
move), the **realised** Risk-to-Reward ratio will be variable, driven
by:
- continuous updating of the EMA target anchor (TP distance from entry
  is path-dependent),
- close-of-bar stop evaluation with next-bar-open execution (stop fills
  may slip beyond -2.25σ),
- EOD-flat truncation (trades that neither hit TP nor stop close at
  whatever P&L holds at 15:45 NY).

Do not read the result against a fixed 1:2 R:R expectation.

### Gate 1 user-decision indicators

Per Gate 1 v2, six STRONG/WEAK readings inform a user go/no-go on
Gate 2. No automatic pass criterion is pre-committed here. The
asymmetry hypothesis (Bali) requires Run 1 to materially outperform a
symmetric LONG-only baseline on the same entries; Run 2 provides that
baseline directly.

## Gate 2 — MFFU Phase 1 viability

Reached only if Run 1 Gate 1 measurement is judged strong by the user.

### Strategy spec (locked before running)

Carried verbatim from Run 1 above:
- Entry: `signal_zscore <= -anchor_z`, 10:00–13:00 NY, LONG only.
- TP: intra-bar limit on `high >= target_price`.
- Stop: close-of-bar `z_score <= (entry_z - stop_distance_z)`, exit at
  next bar's open.
- Position sizing: 1 contract (Gate 2 measures viability of unit edge).
- Max concurrent positions: 1.
- Session filter: RTH, EOD-flat at 15:45 NY.

### Tunable parameter grid (locked)

3 parameters × 3 levels = **27 combinations.** At the cap.

| Parameter | Range | Step | Rationale |
|---|---|---|---|
| `anchor_window` (EMA span) | [15, 20, 25] | 5 | Tests anchor responsiveness — narrower hugs trend faster (Mirpoorian) but noisier; wider is smoother but slower to re-anchor post-shock. |
| `entry_z` | [-1.25, -1.50, -1.75] | 0.25 | Tests shoulder depth — shallower = more trades / weaker dislocation; deeper = fewer / stronger. |
| `stop_distance_z` | [0.50, 0.75, 1.00] | 0.25 | σ distance below the triggered entry at which regime is judged invalidated. With entry -1.50 and distance 0.75, dynamic stop fires at z ≤ -2.25. Tests regime-invalidation tightness. |

Time-window (10:00–13:00), EOD time (15:45), TP rule (anchor-touch),
direction (LONG), and sizing (1 contract) are **fixed, not tunable.**
Locking them now prevents grid-blowout and preserves the asymmetry
hypothesis as a structural prior.

### Walk-forward shape (locked)

Per `research/methodology/gate2_v2.md`: 18-month train / 6-month test,
slid 6 months forward across 2017–2022 (≈9 chunks). Per chunk: tune
within the grid above by in-sample MFFU Phase 1 pass rate; measure OOS
pass rate on the chunk's 6-month test tail.

### Mechanical selection rule (locked)

Cross-chunk aggregation: **median OOS MFFU Phase 1 pass rate** across
chunks. Ties broken by **lowest cross-chunk variance** of OOS pass rate
(the load-bearing anti-cherry-pick guardrail per Gate 2 v2 spec). The
combo selected by this rule is the one carried — only if Gate 2 thresholds
are also met — to the sealed test slice.

### MFFU config used

- Account size: $100,000
- Profit target: $6,000
- Trailing DD (EOD): $3,000
- Consistency rule: 50%
- Source: `engine.prop_firm.MFFU_PHASE1_100K`
- Monte Carlo permutation: whole-day PnL blocks (per 2026-05-05 fix —
  trade-level shuffle was biased optimistic for EOD-trailing rules).

### Gate 2 pass criteria (pre-registered thresholds)

- [ ] Monte Carlo pass rate ≥ 50% (with 95% Wilson CI lower bound > 40%)
- [ ] P(fail on trailing DD) ≤ 30%
- [ ] Observed max_day_frac at pass ≤ 0.5 on ≥ 80% of passing sims
- [ ] **Median days-to-pass ≤ 45 calendar days** — malleable per
      hypothesis per `feedback_gate2_thresholds.md`. Set to 45 rather
      than default 30 because (a) MFFU's time-limit terms have loosened
      and 45 reflects current industry norms, (b) the asymmetric-shock
      setup is structurally low-frequency on a single instrument and a
      30-day floor would punish the design rather than measure it.

### Gate 2 fail criteria (ANY one kills)

- Monte Carlo pass rate < 50% on the selected combo.
- Wilson lower bound on pass rate ≤ 40%.
- P(fail on trailing DD) > 30% on the selected combo.
- max_day_frac > 0.5 on ≥ 20% of passing sims (concentration risk).
- Median days-to-pass > 45.

## Test slice (sealed — touched ONLY if Gate 1 AND Gate 2 pass)

### One-shot test protocol

- Run the locked strategy with the mechanically-selected parameter
  combo exactly once on 2023-01-03 → 2024-12-31.
- No re-tuning, no parameter jiggling, no second look.
- Report the raw metrics defined under "Sealed-slice success criteria".

### Sealed-slice success criteria

- [ ] MFFU Phase 1 pass rate ≥ 50% (matching the Gate 2 threshold).
- [ ] Per-trade expectancy 95% bootstrap CI lower bound > $0 net of
      costs.
- [ ] Median days-to-pass ≤ 45.
- [ ] No structural deterioration vs Gate 2: pass rate drop ≤ 15
      percentage points relative to the selected combo's median OOS
      pass rate from the walk-forward.

---

## Test log (populated AFTER pre-registered sections above are locked — do not edit above)

### Gate 1 run
- **Date:**
- **Notebook:** `research/notebooks/[filename].ipynb`
- **Commit hash at time of run:**
- **Training slice bars used:**
- **Run 1 result summary (Dynamic Regime Exits):**
- **Run 2 result summary (Default ATR wrapper):**
- **Six v2 indicators (Run 1):**
- **Six v2 indicators (Run 2):**
- **Gate 1 user-decision outcome:** PROGRESS / PARK / ARCHIVE
- **Reasoning:**

### Gate 2 run (only if Gate 1 progressed)
- **Date:**
- **Notebook:**
- **Parameter search results:**

| Combo (anchor_window, entry_z, stop_distance_z) | Median OOS pass rate | Cross-chunk variance | 95% Wilson CI | Med days | DD fails % | Consistency fails % |
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
  reached; otherwise 0]. **This is the 4th screen on the contaminated
  training slice.**
