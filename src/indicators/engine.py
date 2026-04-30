"""
Indicator engine — computes all technical indicators required by the five
locked strategies.

Public interface
----------------
add_indicators(df)              -> pd.DataFrame   (preferred entry point)

Individual calculators (also importable for testing / inspection):
    compute_sma(series, window) -> pd.Series
    compute_rsi(series, period) -> pd.Series
    compute_rolling_high(series, window) -> pd.Series   [look-ahead safe]
    compute_rolling_low(series, window)  -> pd.Series   [look-ahead safe]
    compute_avg_volume(series, window)   -> pd.Series   [look-ahead safe]

No look-ahead bias
------------------
Indicators that feed breakout entry/exit signals (rolling_high_20,
rolling_low_10, avg_volume_20) are computed on the *previous* completed
window via .shift(1).  On day N, the value reflects the window [N-window, N-1],
so it is fully known at the close of day N without using future data.

SMA and RSI use the current day's close price — this is correct for daily
backtesting where the signal is generated at end-of-day and executed the
following day.

Column names added by add_indicators()
---------------------------------------
  sma_50          50-day simple moving average of close
  sma_200         200-day simple moving average of close
  rsi_14          14-period Wilder RSI of close
  rolling_high_20 max(high) over the previous 20 trading days  [shifted]
  rolling_low_10  min(low)  over the previous 10 trading days  [shifted]
  avg_volume_20   mean(volume) over the previous 20 trading days [shifted]
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Individual indicator calculators
# ---------------------------------------------------------------------------


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average of *series* over *window* periods.

    Returns NaN for the first (window - 1) rows (warmup).
    """
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI of *series* over *period* periods.

    Uses exponential smoothing with alpha = 1/period (Wilder's method),
    which matches the RSI produced by most charting platforms.
    Returns values in [0, 100]. NaN for the first *period* rows.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder's smoothing: alpha = 1 / period, adjust=False matches the
    # recursive definition.  min_periods ensures no output until the warm-up
    # window is complete.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 (no down moves in window), RSI is 100 by definition
    rsi[avg_loss == 0] = 100.0
    return rsi


def compute_rolling_high(series: pd.Series, window: int) -> pd.Series:
    """Maximum of *series* over the previous *window* periods.

    Shifted by 1 so that day-N value = max(series[N-window : N]).
    The current bar's value is NOT included — safe for entry signals.
    """
    return series.rolling(window=window, min_periods=window).max().shift(1)


def compute_rolling_low(series: pd.Series, window: int) -> pd.Series:
    """Minimum of *series* over the previous *window* periods.

    Shifted by 1 so that day-N value = min(series[N-window : N]).
    The current bar's value is NOT included — safe for exit signals.
    """
    return series.rolling(window=window, min_periods=window).min().shift(1)


def compute_avg_volume(series: pd.Series, window: int) -> pd.Series:
    """Mean of *series* over the previous *window* periods.

    Shifted by 1 so that day-N value = mean(series[N-window : N]).
    Used for volume-filter confirmation in Breakout B.
    """
    return series.rolling(window=window, min_periods=window).mean().shift(1)


# ---------------------------------------------------------------------------
# Composite entry point
# ---------------------------------------------------------------------------


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df* with all strategy indicator columns appended.

    The input DataFrame must be a clean OHLCV frame produced by
    src.data.loader (lowercase columns, DatetimeIndex named 'date').

    The returned DataFrame preserves all original columns and adds:
        sma_50, sma_200, rsi_14,
        rolling_high_20, rolling_low_10, avg_volume_20
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_indicators: input DataFrame missing columns: {missing}")

    out = df.copy()

    out["sma_50"] = compute_sma(out["close"], 50)
    out["sma_200"] = compute_sma(out["close"], 200)
    out["rsi_14"] = compute_rsi(out["close"], period=14)

    # Lagged rolling indicators — window applied to high/low/volume then shifted
    out["rolling_high_20"] = compute_rolling_high(out["high"], window=20)
    out["rolling_low_10"] = compute_rolling_low(out["low"], window=10)
    out["avg_volume_20"] = compute_avg_volume(out["volume"], window=20)

    return out
