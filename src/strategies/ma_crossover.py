"""
Strategy 1 — Moving Average Crossover (locked, do not modify).

Entry : SMA 50 crosses *above* SMA 200  (golden cross)
Exit  : SMA 50 crosses *below* SMA 200  (death cross)

Signal contract
---------------
generate_signals(df) accepts a DataFrame that already contains indicator
columns produced by src.indicators.engine.add_indicators().  It returns a
copy of the frame with two boolean columns appended:

    signal_entry  True on the bar where SMA 50 first rises above SMA 200
    signal_exit   True on the bar where SMA 50 first falls below SMA 200

No-look-ahead guarantee
-----------------------
The crossover is detected by comparing today's (sma_50 > sma_200) to
yesterday's (sma_50 > sma_200).  Both values are derived from end-of-day
closes already in the historical record — no future data is used.
"""

import pandas as pd

# The indicator columns this strategy reads
_REQUIRED_COLS = {"sma_50", "sma_200"}


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with ``signal_entry`` and ``signal_exit`` columns appended.

    Parameters
    ----------
    df:
        A DataFrame produced by ``src.indicators.engine.add_indicators()``.
        Must contain columns ``sma_50`` and ``sma_200``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two new boolean columns:

        * ``signal_entry`` — True on the first day SMA 50 crosses above SMA 200
        * ``signal_exit``  — True on the first day SMA 50 crosses below SMA 200

    Raises
    ------
    ValueError
        If required indicator columns are missing from *df*.
    """
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"generate_signals (ma_crossover): missing required columns: {missing}. "
            "Run add_indicators() first."
        )

    out = df.copy()

    # Boolean series: is SMA 50 above SMA 200 on each day?
    # NaN rows (warmup period) evaluate to False via comparison — correct
    # behaviour because we never generate a signal before both SMAs are valid.
    sma_above = out["sma_50"] > out["sma_200"]

    # Crossover on day N = condition is True today AND was False yesterday
    # Crossunder on day N = condition is False today AND was True yesterday
    prev_sma_above = sma_above.shift(1, fill_value=False)

    out["signal_entry"] = sma_above & ~prev_sma_above
    out["signal_exit"] = ~sma_above & prev_sma_above

    return out
