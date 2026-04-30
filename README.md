# Strategy Backtesting Engine

**Live Demo:** _[link coming soon]_

A Python quantitative finance project that backtests five systematic trading strategies against real historical market data and presents the results on an interactive Streamlit dashboard.

---

## Overview

This project builds a complete backtesting pipeline from raw market data to interactive charts — no paid data feeds, no external databases, no live trading infrastructure. Everything runs from pre-computed results that are committed to the repo, so the dashboard loads instantly.

The five strategies are **fully locked**: their parameters and rules are identical across all tested assets and timeframes, making results directly comparable.

---

## Live App

> **Try it:** _[Streamlit Community Cloud link — coming soon]_

The app lets you select any strategy, asset, and timeframe combination and see the full equity curve, drawdown chart, entry/exit signals, and performance metrics — all in the browser with no setup.

---

## Strategies

| # | Strategy | Type | Entry | Exit |
|---|----------|------|-------|------|
| 1 | Moving Average Crossover | Trend Following | SMA 50 crosses above SMA 200 | SMA 50 crosses below SMA 200 |
| 2 | RSI Mean Reversion | Mean Reversion | RSI(14) < 30 | RSI(14) > 55 |
| 3 | Breakout A — Simple | Breakout / Momentum | Close > prev 20-day high | Close < prev 10-day low |
| 4 | Breakout B — Filtered | Breakout / Momentum (Filtered) | Same as A + volume filter + SMA 200 + RSI in [55,75] | Close < prev 10-day low |
| 5 | Hybrid Allocation | Multi-Strategy Portfolio | 50% Breakout B + 30% MA Crossover + 20% RSI MR | — |

**Anchor assets:** SPY · QQQ · BTC-USD · IWM  
**Additional:** GLD  
**Timeframes:** 1 · 5 · 10 · 20 years of daily bars

---

## Features

- **Single / Dual / Quad layout** — view 1, 2, or 4 strategy panels simultaneously
- **Per-panel controls** — independently set strategy, asset, and timeframe for each panel
- **Equity curve tab** — portfolio value over time vs. buy-and-hold benchmark
- **Signals chart tab** — price chart with entry/exit markers overlaid
- **Sleeves chart** (Hybrid only) — visualize the three sub-strategy allocations independently
- **Stats sidebar** — 11 metrics per panel: total return, annualized return, win rate, avg win/loss, max drawdown, avg drawdown, buy-and-hold benchmark, outperformance
- **Instant load** — pre-computed results committed to repo; no network calls at runtime

---

## Screenshots

_[screenshots coming soon]_

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data | [yfinance](https://github.com/ranaroussi/yfinance) — daily OHLCV, adjusted prices |
| Computation | pandas · numpy |
| Charts | [Plotly](https://plotly.com/python/) |
| App | [Streamlit](https://streamlit.io/) (primary) · Flask (local dev alternative) |
| Tests | pytest |

---

## Local Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd strategy-backtesting-engine
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Pre-computed results for all 100 strategy × asset × timeframe combinations are already in `reports/results/`, so the dashboard works immediately — no internet required.

### 3. (Optional) Run the Flask version locally

```bash
python run_website.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

### 4. Run the test suite

```bash
# Fast tests only (~5 seconds, no network)
pytest tests/ -m "not slow" -q

# Full test suite (~15 seconds)
pytest tests/ -q
```

---

## How to Use

1. Open the app (Streamlit link or local).
2. Choose a **layout**: Single, Dual, or Quad.
3. For each panel, pick a **strategy**, **asset**, and **timeframe** from the dropdowns.
4. The chart and stats update immediately.
5. Switch between the **Equity**, **Signals**, and **Sleeves** (Hybrid only) tabs to explore different chart views.
6. Compare panels side-by-side to see how strategies perform across different markets.

---

## Project Structure

```
app.py                  # Streamlit entrypoint
run_website.py          # Flask entrypoint (local dev alternative)
requirements.txt
.streamlit/config.toml  # Streamlit theme config

src/
  config/               # Locked strategy params, assets, timeframes
  data/                 # OHLCV fetching and normalization (yfinance)
  indicators/           # SMA 50/200, RSI 14, rolling high/low, avg volume
  strategies/           # One module per strategy — produces entry/exit signals
  backtester/           # Signals → trades → portfolio history + trade log
  analytics/            # Metrics: Sharpe, drawdown, win rate, returns, etc.
  visualization/        # Plotly chart generation and JSON serialization
  research/             # Research runner and result export utilities
  website/              # Flask app, routes, Jinja2 templates, CSS, JS

reports/
  results/              # Pre-computed JSON results (100 files, committed)
  exports/              # Flat CSV and JSON exports for external use

tests/                  # Test suite mirroring src/ structure
```

---

## Re-generating Results

Results are pre-committed. To regenerate from scratch (requires internet access):

```python
from src.research.runner import run_all_backtests
run_all_backtests(verbose=True)   # writes 100 JSON files to reports/results/
```

To regenerate the export files:

```python
from src.research.exporter import run_all_exports
run_all_exports(verbose=True)     # writes reports/exports/
```

---

## Notes

- Long-only, daily bar execution. Signals generated and filled at end-of-day close.
- Starting capital: $10,000 per strategy per backtest.
- Transaction cost: 0.1% per side (0.2% round-trip).
- No look-ahead bias: all indicator values use only data available at bar close.
- BTC-USD 20-year backtest uses all available history (~11.6 years).
- This is a demo/research project — not financial advice, not a live trading system.
