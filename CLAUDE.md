# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Systematic trading research environment for NQ (Nasdaq 100) futures. Two components:

1. **Pine Script strategy** — TradingView implementation for live chart analysis
2. **Python backtesting framework** — Offline research engine for strategy development and evaluation

## Repository Structure

- `NQ_Compression_Breakout.pine` — Pine Script v5 strategy for TradingView
- `data_loader.py` — CSV ingestion and validation for OHLCV data
- `strategy.py` — Strategy functions that return signal DataFrames
- `backtester.py` — Event-loop backtest engine with position/equity tracking
- `metrics.py` — Performance metrics (Sharpe, CAGR, etc.) and plotting
- `main.py` — Entry point: loads data, runs strategy, prints results, generates plots

## Languages & Platforms

- **Pine Script v5** — TradingView Strategy Tester (paste into Pine Editor)
- **Python 3.10+** — pandas, numpy, matplotlib
- **Instrument:** NQ1! continuous futures, 15-minute chart

## Running the Backtest

```
python main.py
```

Requires `nq_15m_data.csv` in the project root with columns: `timestamp, open, high, low, close, volume`

## Strategy Architecture (numbered pipeline)

The strategy executes as a sequential pipeline each bar:

1. **HTF Trend Filter** — Requests 4H EMA(200) and 4H close via `request.security()` to determine bullish/bearish bias (no repainting: `lookahead_off`)
2. **Compression Detection** — ATR(14) < SMA(ATR,100) × 0.85 triggers compression; a static high/low range is locked on entry
3. **Session Filter** — Entries restricted to US cash session 09:30–16:00 New York
4. **Risk Controls** — Daily loss count, daily P&L %, and weekly drawdown % gates evaluated before any entry
5. **Breakout Entry** — Close above/below compression range with max 2 attempts per phase; position sized by `equity × riskPct / stopDistance`
6. **Asymmetric Sizing** — With-trend trades use full base risk (0.5%), counter-trend trades use half
7. **Trade Management** — Partial TP at +2R (50% close, stop to breakeven), then ATR trailing stop ratchets remainder
8. **State Reset** — All `var` trade-state variables reset when position goes flat

## Key Conventions

- All per-trade state uses Pine Script `var` variables (persist across bars, reset on flat)
- `pendingStop` captures the compression boundary on the signal bar; `entryConfirmed` flag gates recalculation using the actual fill price from `strategy.position_avg_price`
- Stop distances and R-values are recalculated from the real fill price, not the signal bar close
- Position sizing floors to whole contracts (`math.floor`); trades with < 1 contract are skipped

## Git Workflow

- Remote: `AjaxTTR/AlgoTrade` on GitHub
- Branch: `main`
- Commit and push changes after each meaningful modification
- Use clear, descriptive commit messages summarizing what changed and why
