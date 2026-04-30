"""
Strategy 2 — RSI Mean Reversion (locked, do not modify).

Entry : RSI(14) crosses *below* 30  (oversold threshold)
Exit  : RSI(14) crosses *above* 55  (recovery threshold)

Signal contract
---------------
generate_signals(df) accepts a DataFrame that already contains indicator
columns produced by src.indicators.engine.add_indicators().  It returns a
copy of the frame with two boolean columns appended:

    signal_entry  True on the first bar where RSI(14) drops below 30
    signal_exit   True on the first bar where RSI(14) rises above 55

No-look-ahead guarantee
-----------------------
Both thresholds are evaluated using the current bar's RSI value, which is
derived from end-of-day closes already in the historical record.  The signal
is generated at end-of-day and would be acted on the following day's open —
no future data is consumed.

Edge cases
----------
- No signal is emitted during the RSI warmup period (first 14 bars are NaN).
- Entry and exit cannot both be True on the same bar.
- Multiple consecutive bars below 30 do NOT re-trigger an entry — only the
  first crossing bar fires.  Symmetrically for exit above 55.
"""

import pandas as pd

_REQUIRED_COLS = {"rsi_14"}

# Locked thresholds from settings — hardcoded here as the strategy is locked
_ENTRY_THRESHOLD = 30  # RSI below this → oversold → entry
_EXIT_THRESHOLD = 55   # RSI above this → recovered → exit


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with ``signal_entry`` and ``signal_exit`` columns appended.

    Parameters
    ----------
    df:
        A DataFrame produced by ``src.indicators.engine.add_indicators()``.
        Must contain column ``rsi_14``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two new boolean columns:

        * ``signal_entry`` — True on the first bar RSI(14) crosses below 30
        * ``signal_exit``  — True on the first bar RSI(14) crosses above 55

    Raises
    ------
    ValueError
        If ``rsi_14`` is missing from *df*.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"generate_signals (rsi_mean_reversion): missing required columns: {missing}. "
            "Run add_indicators() first."
        )

    out = df.copy()

    rsi = out["rsi_14"]

    # Boolean series: is RSI in the oversold zone / recovery zone?
    below_entry = rsi < _ENTRY_THRESHOLD   # True when RSI < 30
    above_exit = rsi > _EXIT_THRESHOLD     # True when RSI > 55

    # Cross-below 30: RSI is below 30 today AND was NOT below 30 yesterday
    prev_below_entry = below_entry.shift(1, fill_value=False)
    out["signal_entry"] = below_entry & ~prev_below_entry

    # Cross-above 55: RSI is above 55 today AND was NOT above 55 yesterday
    prev_above_exit = above_exit.shift(1, fill_value=False)
    out["signal_exit"] = above_exit & ~prev_above_exit

    return out
