"""
Strategy 3 — Breakout Strategy A: Simple Breakout (locked, do not modify).

Entry : close > previous 20-day high  (price breaks to a new 20-day high)
Exit  : close < previous 10-day low   (price breaks below the 10-day low)

Signal contract
---------------
generate_signals(df) accepts a DataFrame that already contains indicator
columns produced by src.indicators.engine.add_indicators().  It returns a
copy of the frame with two boolean columns appended:

    signal_entry  True on bars where close exceeds the previous 20-day high
    signal_exit   True on bars where close falls below the previous 10-day low

No-look-ahead guarantee
-----------------------
``rolling_high_20`` and ``rolling_low_10`` are produced by add_indicators()
with a .shift(1), so on day N each value reflects the window ending at day
N-1.  Comparing today's close against those values introduces no future data.

Multiple consecutive breakout bars
-----------------------------------
Unlike the RSI strategy, this strategy does NOT suppress repeated signals on
consecutive bars above the rolling high.  Each bar where the condition holds
is a valid signal — position management (single-position enforcement) is the
backtesting engine's responsibility, not the signal generator's.

Signal independence
-------------------
Entry and exit conditions are evaluated independently.  On the same bar both
can technically be True (e.g. open gap that simultaneously breaches high and
low).  The backtesting engine resolves priority.
"""

import pandas as pd

_REQUIRED_COLS = {"close", "rolling_high_20", "rolling_low_10"}


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with ``signal_entry`` and ``signal_exit`` columns appended.

    Parameters
    ----------
    df:
        A DataFrame produced by ``src.indicators.engine.add_indicators()``.
        Must contain ``close``, ``rolling_high_20``, and ``rolling_low_10``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two new boolean columns:

        * ``signal_entry`` — True when close > previous 20-day high
        * ``signal_exit``  — True when close < previous 10-day low

    Raises
    ------
    ValueError
        If any required columns are missing from *df*.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"generate_signals (breakout_a): missing required columns: {missing}. "
            "Run add_indicators() first."
        )

    out = df.copy()

    # rolling_high_20 and rolling_low_10 are already shifted in add_indicators()
    # so these comparisons are look-ahead safe.
    out["signal_entry"] = out["close"] > out["rolling_high_20"]
    out["signal_exit"] = out["close"] < out["rolling_low_10"]

    return out
