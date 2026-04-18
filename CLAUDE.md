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
  gap_momentum.py                    # Gap-Up Momentum signal generator (the active strategy)
research/
  edge_analysis.py                   # Statistical edge discovery and conditional analysis
  edge_to_strategy.py                # Edge-to-strategy conversion utilities
data/
  nq_15m_data.csv                    # 7 years of NQ 15-min OHLCV (not in git)
output/                              # All backtest artifacts land here (not in git)
```

## Running

```bash
# Full backtest (outputs to output/)
python main.py

# Run an arbitrary strategy module by name
python run_strategy.py gap_momentum
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

**Strategy (Gap-Up Momentum):**
- Edge: Gap-up × FH_Up -> Rest_Up (t=30.39, stability=0.946), long only
- Gap filter: session open > prior close by >= 0.10%
- First-hour window: 09:30-10:30 ET; entry at 10:30
- Bias filter: FH return >= 75th percentile of prior FH returns (expanding, no lookahead)
- Session: 09:30-16:00 ET, entry cutoff 15:45
- ATR period: 14, stop at 1.5x ATR, TP at 2.0x ATR
- Holding period: 8 bars (2 hours on 15-min bars)
- Max 1 trade/day, long only

## Strategy Pipeline

1. **Gap Detection** — Compute gap_pct = (session_open - prior_close) / prior_close; require gap_pct >= gap_threshold_pct
2. **First-Hour Check** — Measure return from 09:30-10:30; require fh_return >= dynamic percentile threshold (expanding window, no lookahead)
3. **Entry** — On qualifying days, enter long at 10:30 close
4. **Risk Sizing** — ATR-based volatility sizing: equity * risk_per_trade / (ATR * point_value), supports size_factor column. Dynamic scaling: after 2 consecutive losses, position size halved until next win.
5. **Trade Management** — Stop at 1.5x ATR below entry, TP at 2.0x ATR above; 8-bar holding period exit via max_bars_in_trade
6. **Daily Limits** — Max 1 trade/day, 2% daily drawdown limit, pre-entry risk gate, max 1 concurrent position

## Key Conventions

- All output files go to `output/` (trade_log.csv, equity_curve.csv, drawdown_series.csv, performance_metrics.json, backtest_results.png)
- No external API calls — data is loaded from local CSV only
- Strategy modules expose `generate_signals(df, **params) -> DataFrame`
- Backtester config and strategy config are separate dicts in main.py
- Research tools share synced config with main.py where applicable

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
