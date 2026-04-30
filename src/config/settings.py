"""
Central configuration for the Strategy Backtesting Engine demo.

All locked strategy parameters, asset definitions, timeframes, and engine
defaults live here. Nothing in this file should be modified without updating
the corresponding entry in sync-markdowns/idea.md.
"""

# ---------------------------------------------------------------------------
# Capital and execution defaults
# ---------------------------------------------------------------------------

DEFAULT_STARTING_CAPITAL: float = 10_000.0
TRANSACTION_COST_PCT: float = 0.001  # 0.1% per trade (entry + exit = 0.2% round-trip)

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

# Anchor markets: results for these must be generated and displayed on the site
ANCHOR_ASSETS: list[str] = ["SPY", "QQQ", "BTC-USD", "IWM"]

# Additional asset for broader internal testing (commodities proxy)
EXTRA_ASSETS: list[str] = ["GLD"]

ALL_ASSETS: list[str] = ANCHOR_ASSETS + EXTRA_ASSETS

# ---------------------------------------------------------------------------
# Backtest timeframes (in years)
# ---------------------------------------------------------------------------

TIMEFRAMES_YEARS: list[int] = [1, 5, 10, 20]

# ---------------------------------------------------------------------------
# Locked strategy definitions
# Each entry is consumed by the indicator engine and strategy modules.
# Keys map directly to strategy class names / identifiers used throughout.
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, dict] = {
    "ma_crossover": {
        "name": "Moving Average Crossover",
        "type": "Trend Following",
        "description": (
            "Buys when the 50-day SMA crosses above the 200-day SMA (golden cross) "
            "and sells when the 50-day SMA crosses below the 200-day SMA (death cross)."
        ),
        "params": {
            "fast_sma": 50,
            "slow_sma": 200,
        },
    },
    "rsi_mean_reversion": {
        "name": "RSI Mean Reversion",
        "type": "Mean Reversion",
        "description": (
            "Buys when RSI(14) falls below 30 (oversold) "
            "and sells when RSI(14) rises above 55."
        ),
        "params": {
            "rsi_period": 14,
            "entry_threshold": 30,
            "exit_threshold": 55,
        },
    },
    "breakout_a": {
        "name": "Breakout Strategy A — Simple Breakout",
        "type": "Breakout / Momentum",
        "description": (
            "Buys when the close exceeds the previous 20-day high. "
            "Exits when the close falls below the previous 10-day low."
        ),
        "params": {
            "entry_lookback": 20,  # rolling high window
            "exit_lookback": 10,   # rolling low window
        },
    },
    "breakout_b": {
        "name": "Breakout Strategy B — Filtered Breakout Swing",
        "type": "Breakout / Momentum (Filtered)",
        "description": (
            "Same breakout entry as Strategy A with four confirmation filters: "
            "volume > 1.5× 20-day average volume, close > SMA 200, RSI(14) between 55 and 75. "
            "Exit: close < previous 10-day low."
        ),
        "params": {
            "entry_lookback": 20,
            "exit_lookback": 10,
            "volume_lookback": 20,
            "volume_multiplier": 1.5,
            "trend_sma": 200,
            "rsi_period": 14,
            "rsi_min": 55,
            "rsi_max": 75,
        },
    },
    "hybrid": {
        "name": "Hybrid Allocation Strategy",
        "type": "Multi-Strategy Portfolio",
        "description": (
            "Allocates capital across three sub-strategies running independently: "
            "50% to Breakout B, 30% to MA Crossover, 20% to RSI Mean Reversion. "
            "Tracks a single combined portfolio value."
        ),
        "params": {
            "allocations": {
                "breakout_b": 0.50,
                "ma_crossover": 0.30,
                "rsi_mean_reversion": 0.20,
            }
        },
    },
}

# Ordered list of strategy keys — used for UI dropdowns and report ordering
STRATEGY_ORDER: list[str] = [
    "ma_crossover",
    "rsi_mean_reversion",
    "breakout_a",
    "breakout_b",
    "hybrid",
]

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
