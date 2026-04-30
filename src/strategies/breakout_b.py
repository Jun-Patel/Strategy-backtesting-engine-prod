"""
Strategy 4 — Breakout Strategy B: Filtered Breakout Swing (locked, do not modify).

Entry : ALL of the following must be true on the same bar —
          1. close > rolling_high_20       (price breaks previous 20-day high)
          2. volume > 1.5 × avg_volume_20  (volume confirmation)
          3. close > sma_200               (long-term uptrend filter)
          4. 55 ≤ RSI(14) ≤ 75            (momentum zone; inclusive on both ends)

Exit  : close < rolling_low_10            (price breaks previous 10-day low)

Signal contract
---------------
generate_signals(df) accepts a DataFrame that already contains indicator
columns produced by src.indicators.engine.add_indicators().  It returns a
copy of the frame with two boolean columns appended:

    signal_entry  True when all four entry conditions are simultaneously met
    signal_exit   True when close < rolling_low_10

No-look-ahead guarantee
-----------------------
All indicator columns consumed here (rolling_high_20, avg_volume_20,
rolling_low_10) are produced by add_indicators() with a .shift(1), so their
values on day N reflect windows ending at day N-1.  sma_200 and rsi_14 use
the current day's close, which is appropriate for end-of-day signal generation.

Consecutive bars
----------------
Like Breakout A, every bar where all conditions hold fires signal_entry.
Single-position enforcement is the backtesting engine's responsibility.

RSI bounds
----------
The locked spec states RSI ∈ [55, 75] — both endpoints are inclusive.
"""

import pandas as pd

_REQUIRED_COLS = {
    "close", "volume",
    "sma_200", "rsi_14",
    "rolling_high_20", "rolling_low_10", "avg_volume_20",
}

# Locked parameters (from settings.py / idea.md — do not change)
_VOLUME_MULTIPLIER = 1.5
_RSI_MIN = 55   # inclusive
_RSI_MAX = 75   # inclusive


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with ``signal_entry`` and ``signal_exit`` columns appended.

    Parameters
    ----------
    df:
        A DataFrame produced by ``src.indicators.engine.add_indicators()``.
        Must contain: ``close``, ``volume``, ``sma_200``, ``rsi_14``,
        ``rolling_high_20``, ``rolling_low_10``, ``avg_volume_20``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two new boolean columns:

        * ``signal_entry`` — True when all four entry filters are met
        * ``signal_exit``  — True when close < rolling_low_10

    Raises
    ------
    ValueError
        If any required indicator columns are missing from *df*.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"generate_signals (breakout_b): missing required columns: {missing}. "
            "Run add_indicators() first."
        )

    out = df.copy()

    # --- Entry filters (all must be True) ---

    # 1. Price breakout: close exceeds the previous 20-day high
    price_breakout = out["close"] > out["rolling_high_20"]

    # 2. Volume confirmation: today's volume exceeds 1.5× the previous 20-day avg
    volume_confirm = out["volume"] > (_VOLUME_MULTIPLIER * out["avg_volume_20"])

    # 3. Trend filter: price is above the 200-day SMA
    trend_filter = out["close"] > out["sma_200"]

    # 4. Momentum zone: RSI(14) is between 55 and 75, inclusive
    rsi_filter = (out["rsi_14"] >= _RSI_MIN) & (out["rsi_14"] <= _RSI_MAX)

    out["signal_entry"] = price_breakout & volume_confirm & trend_filter & rsi_filter

    # --- Exit ---
    out["signal_exit"] = out["close"] < out["rolling_low_10"]

    return out
