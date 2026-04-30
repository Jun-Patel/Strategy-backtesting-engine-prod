"""
Strategy 5 — Hybrid Allocation Strategy (locked, do not modify).

Capital allocation (fixed):
    50%  Breakout Strategy B
    30%  Moving Average Crossover
    20%  RSI Mean Reversion

Each sub-strategy runs independently within its own capital bucket.
The combined portfolio value is the weighted sum of the three equity curves.

Public interface
----------------
generate_signals(df)
    Runs all three sub-strategies on *df* and returns a dict mapping each
    sub-strategy key to its signals DataFrame (same schema as the standalone
    strategies: boolean ``signal_entry`` and ``signal_exit`` columns).

combine_portfolio_equity(sub_equities, allocations=None)
    Accepts a dict of {sub_strategy_key: equity_series} and returns a single
    combined equity Series weighted by ALLOCATIONS.

ALLOCATIONS
    Module-level dict of fixed weights — the single source of truth.

Design rationale
----------------
generate_signals() returns a *dict* rather than a single DataFrame because the
three sub-strategies are independent and the backtesting engine (Chunk 8) needs
to simulate each bucket separately before combining the resulting equity curves.

combine_portfolio_equity() is a pure function: it knows nothing about trade
simulation.  It simply computes the weighted portfolio value after the engine
has produced per-bucket equity series.
"""

import pandas as pd

from src.strategies.breakout_b import generate_signals as _bb_signals
from src.strategies.ma_crossover import generate_signals as _ma_signals
from src.strategies.rsi_mean_reversion import generate_signals as _rsi_signals

# ---------------------------------------------------------------------------
# Fixed allocation weights (locked — do not change)
# ---------------------------------------------------------------------------

ALLOCATIONS: dict[str, float] = {
    "breakout_b": 0.50,
    "ma_crossover": 0.30,
    "rsi_mean_reversion": 0.20,
}


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def generate_signals(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run all three sub-strategies and return their signals independently.

    Parameters
    ----------
    df:
        A DataFrame produced by ``src.indicators.engine.add_indicators()``.
        Must contain all columns required by Breakout B, MA Crossover, and
        RSI Mean Reversion (i.e., the full output of add_indicators()).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys match ``ALLOCATIONS``:

        * ``"breakout_b"``        — signals from Strategy 4
        * ``"ma_crossover"``      — signals from Strategy 1
        * ``"rsi_mean_reversion"`` — signals from Strategy 2

        Each value is a DataFrame with the full indicator set plus
        ``signal_entry`` and ``signal_exit`` boolean columns.

    Notes
    -----
    Each sub-strategy receives the same *df* but is called independently.
    Mutations in one sub-strategy's output cannot affect another because each
    generate_signals() call returns a copy.
    """
    return {
        "breakout_b": _bb_signals(df),
        "ma_crossover": _ma_signals(df),
        "rsi_mean_reversion": _rsi_signals(df),
    }


# ---------------------------------------------------------------------------
# Portfolio equity combination
# ---------------------------------------------------------------------------

def combine_portfolio_equity(
    sub_equities: dict[str, pd.Series],
    allocations: dict[str, float] | None = None,
) -> pd.Series:
    """Combine per-sub-strategy equity curves into one weighted portfolio.

    Parameters
    ----------
    sub_equities:
        Dict mapping each sub-strategy key to its equity curve (a pd.Series
        of portfolio value over time, typically produced by the backtesting
        engine from the corresponding signals).  All Series must share the
        same DatetimeIndex.

    allocations:
        Weight dict to use.  Defaults to the module-level ``ALLOCATIONS``.
        Weights must sum to 1.0.

    Returns
    -------
    pd.Series
        Combined portfolio equity, computed as:
            Σ (equity_i × weight_i)  for each sub-strategy i

    Raises
    ------
    ValueError
        If *sub_equities* is missing a key required by *allocations*, or
        if the weights do not sum to 1.0 (within floating-point tolerance).
    """
    if allocations is None:
        allocations = ALLOCATIONS

    # Validate weights sum to 1
    weight_sum = sum(allocations.values())
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(
            f"combine_portfolio_equity: allocation weights must sum to 1.0, "
            f"got {weight_sum:.10f}."
        )

    # Validate all required keys are present
    missing = set(allocations.keys()) - set(sub_equities.keys())
    if missing:
        raise ValueError(
            f"combine_portfolio_equity: sub_equities missing keys: {missing}."
        )

    combined = sum(
        sub_equities[key] * weight
        for key, weight in allocations.items()
    )
    combined.name = "hybrid_equity"
    return combined
