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
  backtester.py                      # Event-loop backtest engine with position/equity tracking
  metrics.py                         # Performance metrics (Sharpe, CAGR, etc.) and plotting
  feature_engineering.py             # Feature computation utilities
  prop_firm.py                       # Prop firm risk rules
  external_data_guard.py             # Blocks accidental external data imports
strategies/
  first_hour_momentum.py             # First-Hour Momentum signal generator (the active strategy)
research/
  edge_analysis.py                   # Statistical edge discovery and conditional analysis
  edge_to_strategy.py               # Edge-to-strategy conversion utilities
  threshold_comparison.py            # Pullback re-entry and tier comparison runner
  optimizer.py                       # Parallel grid search over parameter space
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
- Trailing stop: disabled

**Strategy (First-Hour Momentum):**
- Edge: FH_Up -> Rest_Up (t=30.38, stability=0.946), long only
- First-hour window: 09:30-10:30 ET
- Bias filter: top 20% first-hour moves (80th percentile, expanding window)
- Session: 09:30-16:00 ET, entry cutoff 15:45
- ATR period: 14, stop at 1.5x ATR, TP at 2.0x ATR
- Holding period: 8 bars (2 hours on 15-min bars)
- Pullback re-entries: 0.5 ATR dip from session high + upper-half close
- Max 3 trades/day (1 initial + up to 2 pullback re-entries)

## Strategy Pipeline

1. **First-Hour Observation** — Measure return from 09:30-10:30 (open to close)
2. **Bias Filter** — Only trade if first-hour return exceeds 80th percentile of prior days (expanding window, no lookahead)
3. **Initial Entry** — Signal on first bar at/after 10:30, fill at 10:45 open (long only)
4. **Pullback Re-entries** — After each trade's holding-period exit, re-enter on pullback bars (dip >= 0.5 ATR from session high, bar closes in upper half)
5. **Risk Sizing** — ATR-based volatility sizing: equity * risk_per_trade / (ATR * point_value), supports size_factor column
6. **Trade Management** — Stop at 1.5x ATR below entry, TP at 2.0x ATR above; 8-bar holding period exit via session_close flag
7. **Daily Limits** — Max 3 trades/day, 5% daily drawdown limit enforced

## Key Conventions

- All output files go to `output/` (trade_log.csv, equity_curve.csv, drawdown_series.csv, performance_metrics.json, backtest_results.png)
- No external API calls — data is loaded from local CSV only
- Strategy modules expose `generate_signals(df, **params) -> DataFrame`
- Backtester config and strategy config are separate dicts in main.py
- Research tools (optimizer, walk_forward) share synced config with main.py

## Git Workflow

- Remote: `AjaxTTR/AlgoTrade` on GitHub
- Branch: `main`
- Commit and push changes after each meaningful modification
- Use clear, descriptive commit messages summarizing what changed and why
