# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

Python backtesting framework for NQ (Nasdaq 100) futures intraday strategies. Built for systematic day-trading research — no live execution, no external APIs.

## Repository Structure

```
main.py                              # Entry point: load data, run strategy, export results
run_strategy.py                      # CLI wrapper to run any strategy by name
engine/
  data_loader.py                     # CSV ingestion and validation for OHLCV data
  backtester.py                      # Event-loop backtest engine with multi-position tracking
  metrics.py                         # Performance metrics (Sharpe, CAGR, etc.) and plotting
  feature_engineering.py             # Feature computation utilities
  prop_firm.py                       # Prop firm challenge simulation (Monte Carlo)
  external_data_guard.py             # Blocks accidental external data imports
strategies/
  first_hour_momentum.py             # First-Hour Momentum signal generator (the active strategy)
research/
  edge_analysis.py                   # Statistical edge discovery and conditional analysis
  edge_to_strategy.py               # Edge-to-strategy conversion utilities
  threshold_comparison.py            # Pullback re-entry and tier comparison runner
  optimizer.py                       # Parallel grid search over parameter space
  prop_firm_optimizer.py             # Prop firm pass-rate optimizer (risk/TP/SL grid search)
  tier_validation.py                 # Train/test tier-by-tier robustness validation
  walk_forward.py                    # Walk-forward analysis for overfitting detection
data/
  nq_15m_data.csv                    # 7 years of NQ 15-min OHLCV (not in git)
output/                              # All backtest artifacts land here (not in git)
```

## Running

```bash
# Full backtest (outputs to output/)
python main.py

# Grid search optimizer
python -m research.optimizer

# Walk-forward validation
python -m research.walk_forward

# Tier-by-tier robustness validation (train/test split)
python -m research.tier_validation

# Prop firm pass-rate optimizer (risk/TP/SL/frequency grid search)
python -m research.prop_firm_optimizer
```

Requires `data/nq_15m_data.csv` with columns: `timestamp, open, high, low, close, volume`

## Current Configuration

**Backtest:**
- Initial capital: $100,000
- Risk per trade: 0.5% of equity
- Point value: $20 (NQ futures)
- Commission: $2/side, slippage: 0.25 pts
- Daily drawdown limit: 2%
- Max daily risk: 2% (blocks new entries if cumulative risk exceeds cap)
- Max concurrent trades: 1 (single position at a time)
- Min bars between entries: 2
- Consecutive loss scaling: after 2 losses, halve position size (resets on next win)
- Trailing stop: disabled

**Strategy (First-Hour Momentum):**
- Edge: FH_Up -> Rest_Up (t=30.38, stability=0.946), long only
- Early entry window: 09:30-10:00 ET (if 30-min return already exceeds threshold)
- Full first-hour window: 09:30-10:30 ET (standard entry fallback)
- Bias filter: top 20% first-hour moves (80th percentile, expanding window)
- Session: 09:30-16:00 ET, entry cutoff 15:45
- ATR period: 14, stop at 1.5x ATR, TP at 2.0x ATR
- Holding period: 8 bars (2 hours on 15-min bars)
- Pullback re-entries: two-stage trigger (1 ATR dip + rejection bar, then strong bullish entry bar), max 1/day, only after Tier 2 entry
- Max 4 trades/day (1 initial + pullback re-entries + 1 midday continuation)

## Strategy Pipeline

1. **Early Bias Detection (Tier 1)** — Measure return from 09:30-10:00 (first 30 min); if it already exceeds the threshold, enter early at 10:00
2. **Full First-Hour Check (Tier 2)** — If no early entry, measure return from 09:30-10:30 and enter at 10:30 if threshold exceeded
3. **Bias Filter** — Threshold = 80th percentile of |fh_return| from all prior days (expanding window, no lookahead)
4. **Pullback Re-entries (Tier 3)** — Two-stage trigger, only after a confirmed Tier 2 entry. Stage 1: pullback dip >= 1 ATR from session high with rejection (close in top 25% of bar). Stage 2: next strong bullish bar (close > open, close in top 25%, bar range >= 0.25 ATR). Max 1 pullback re-entry per day.
5. **Midday Continuation (Tier 4)** — After 11:00, enter on consolidation breakout (4-bar range < 1x ATR, close above range) or higher-high after pullback (dip >= 0.5 ATR from session high, new high with upper-half close). Same bias direction only. Max 1/day.
6. **Risk Sizing** — ATR-based volatility sizing: equity * risk_per_trade / (ATR * point_value), supports size_factor column. Dynamic scaling: after 2 consecutive losses, position size halved until next win.
7. **Trade Management** — Stop at 1.5x ATR below entry, TP at 2.0x ATR above; 8-bar holding period exit via max_bars_in_trade
8. **Daily Limits** — Max 4 trades/day, 2% daily drawdown limit, pre-entry risk gate, max 1 concurrent position

## Key Conventions

- All output files go to `output/` (trade_log.csv, equity_curve.csv, drawdown_series.csv, performance_metrics.json, backtest_results.png)
- No external API calls — data is loaded from local CSV only
- Strategy modules expose `generate_signals(df, **params) -> DataFrame`
- Backtester config and strategy config are separate dicts in main.py
- Research tools (optimizer, walk_forward) share synced config with main.py

## End-of-Session Protocol

Before the session ends, **always** overwrite `memory/session-state.md` with a fresh snapshot of:
- Current strategy config (exact parameters)
- Latest backtest results (key numbers)
- What's been tested and rejected (with reasons)
- Current focus and next steps
- Hard constraints

This is the primary way context is preserved between sessions. Do not skip this.

## Git Workflow

- Remote: `AjaxTTR/AlgoTrade` on GitHub
- Branch: `main`
- Commit and push changes after each meaningful modification
- Use clear, descriptive commit messages summarizing what changed and why
