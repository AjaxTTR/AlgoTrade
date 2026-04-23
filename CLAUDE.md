# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Python research library for NQ (Nasdaq 100) futures intraday strategies. Built as a reusable backtesting + validation engine that notebooks and strategy modules import from. No live execution, no external APIs, no global CLI harness — research is driven from Jupyter notebooks.

## Repository Structure

```
engine/
  data_loader.py                     # CSV ingestion, timezone handling, session filtering
  backtester.py                      # Event-loop backtest engine with multi-position tracking
  features.py                        # Feature computation (ATR, percentiles, FH metrics)
  metrics.py                         # Performance metrics (Sharpe, CAGR, drawdown, MCPT)
  prop_firm.py                       # Prop firm challenge simulation (Monte Carlo)
strategies/                          # User-authored strategy modules (empty — awaiting hypotheses)
research/
  notebooks/                         # All exploratory research lives here
  hypotheses/                        # Pre-registration templates and per-hypothesis specs
  experiment_log.csv                 # Pre-registered expectation + final outcome per hypothesis
data/
  nq_15m_data.csv                    # 7 years of NQ 15-min OHLCV (not in git)
configs/                             # Reserved for shared config files (currently empty)
venv/                                # Project virtual environment (not in git)
requirements.txt                     # Pinned dependencies
CLAUDE.md                            # This file
```

## How research runs

There is no CLI entry point. Research happens in notebooks under `research/notebooks/`, which import directly from `engine/`:

```python
from engine.data_loader import load_csv
from engine.backtester import run
from engine.metrics import compute_metrics, print_metrics
from engine.prop_firm import simulate_prop_firm
```

Launch: `venv/Scripts/jupyter lab` — then open a notebook. Charts render inline. `research/notebooks/00_smoke_test.ipynb` is the reference template.

Data requirement: `data/nq_15m_data.csv` with columns `timestamp, open, high, low, close, volume`.

## Collaboration model

See `memory/feedback_collaboration_model.md` for the full spec. Summary:

- **User** owns hypothesis generation, research direction, go/no-go decisions.
- **Claude** owns implementation, methodological critique (contamination, lookahead, multiple testing, overfitting), teaching quant concepts on request, and engineering hygiene.
- **Claude does NOT pitch strategy ideas, design edges from observed patterns, or tune parameters unprompted.**

## Key conventions

- Strategy modules expose `generate_signals(df, **params) -> DataFrame` with a `signal` column.
- The backtester is the single source of truth for fills, sizing, and PnL accounting.
- Any new feature must be lookahead-safe at the column level (use `.shift(1)` on derived series).
- All plotting happens inline in notebooks. No PNG dumps to disk by default.
- No external data providers. Current dataset and any future OOS pulls are explicit and documented.

## Research discipline

- **Pre-register hypotheses before coding.**
  - For pure measurement screens (conditional-behaviour existence checks, no strategy yet): use a screen-shaped pre-reg like `research/hypotheses/overnight_effect.md`.
  - For strategy hypotheses: use `research/hypotheses/TEMPLATE_twogate.md` (two-gate pipeline, see below).
  - For paper-replication-first hypotheses: `research/hypotheses/TEMPLATE.md` still applies.
- **Replicate first, adapt second.** For paper-derived hypotheses, reproduce the paper's headline number on its own instrument before porting to NQ.
- **Current dataset is contaminated** through researcher degrees of freedom. Do not re-tune on it; treat it as training data only.

## Two-gate pipeline (strategy hypotheses)

Every strategy pre-registration runs through two independent gates on the training slice before earning the right to touch the sealed test slice.

- **Gate 1 — Edge existence.** Does the proposed conditional behaviour measure positive and stable on training data? Pure edge detection, no prop-firm mechanics. Criteria: monotonicity / significance / stability across halves, pre-registered per hypothesis.
- **Gate 2 — MFFU Phase 1 viability.** Only reached if Gate 1 passes. Does a strategy implementing the edge survive MFFU Phase 1 mechanics (EOD trailing DD, 50% consistency rule, $6k target on $100k account) under Monte Carlo trade-order shuffling? Use `engine.prop_firm.simulate_prop_firm` with `MFFU_PHASE1_100K`.
- **Parameter optimisation inside Gate 2 must be pre-registered.** Tunable parameters, search grid, and mechanical selection rule are locked in the hypothesis file before any code runs. Post-hoc "this one looked better" is overfitting.
- **Sealed test slice (2023–2024)** is touched exactly once per hypothesis, and only if both gates passed. No re-runs, no re-tuning.

Malleability: the pipeline itself can evolve between hypotheses. Within a single hypothesis, everything above the Test log section is frozen at commit time.

- **Target prop firm: MFFU (My Funded Futures).** Phase 1 mechanics are the sole prop-firm model in `engine/prop_firm.py`. FTMO-style rules were removed deliberately to prevent cross-contamination between rulesets.
- **Locked vault (2005–2015, not yet acquired):** reserved for one-shot validation of frozen strategies. Do not look at results on it until making a deployment decision.

## End-of-Session Protocol

Before the session ends, **always** overwrite `memory/session-state.md` with a fresh snapshot of:
- What was changed this session (specific files / line-level detail where material)
- Current state of research direction
- Hypotheses in flight (with reference to `research/hypotheses/` files)
- Next-session resume point
- Hard constraints or open questions

This is the primary way context is preserved between sessions. Do not skip this.

## Git Workflow

- Remote: `AjaxTTR/AlgoTrade` on GitHub
- Branch: `main`
- Commit after each meaningful change; push at end of session
- Use clear, descriptive commit messages summarizing what changed and why
- `nbstripout` is installed as a git filter — notebook outputs are stripped automatically on commit
