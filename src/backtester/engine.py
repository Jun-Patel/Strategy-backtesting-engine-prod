"""
Core backtesting engine — converts strategy signals into trades, portfolio
history, and a buy-and-hold benchmark.

Public interface
----------------
run_backtest(signals_df, ...)          -> BacktestResult   (Strategies 1–4)
run_hybrid_backtest(signals_dict, ...) -> BacktestResult   (Strategy 5)
BacktestResult                          dataclass holding all outputs

Execution model
---------------
Signals are assumed to be generated at the end-of-day close.  Trades execute
at the *same* day's close (close-price execution model), which is standard for
daily research backtests and consistent with a market-on-close order.

Position rules
--------------
- Long-only, single position per standalone strategy.
- Entry fires on the first bar where signal_entry is True and no position is
  held.  Consecutive signal_entry bars while in a position are ignored.
- Exit fires on the first bar where signal_exit is True and a position is held.
- If both signals are True on the same bar:
    * If in position  → exit takes priority; no re-entry on the same bar.
    * If not in position → entry fires; no exit on the same bar.
- Any position still open at the end of the series is force-closed at the
  last bar's close.

Transaction costs
-----------------
Applied on both legs of every round-trip.

Entry:  shares = cash / (close × (1 + cost_pct))   → all cash invested
Exit:   proceeds = shares × close × (1 − cost_pct)  → net cash received

Benchmark
---------
Buy-and-hold: invest all starting_capital at the first bar's close (with
entry transaction cost applied).  Portfolio value each day = shares × close.
No exit cost is applied to the daily series (it would only apply at
liquidation).

Hybrid strategy
---------------
run_hybrid_backtest() accepts the dict returned by hybrid.generate_signals().
It runs run_backtest() independently for each sub-strategy with a proportional
share of capital, then sums the resulting equity curves.  The benchmark is a
standard buy-and-hold using the full starting_capital.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config.settings import DEFAULT_STARTING_CAPITAL, TRANSACTION_COST_PCT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns present in every non-empty trade log
TRADE_LOG_COLUMNS: list[str] = [
    "entry_date",
    "entry_price",
    "exit_date",
    "exit_price",
    "shares",
    "pnl",
    "return_pct",
    "holding_days",
]

_REQUIRED_SIGNAL_COLS: set[str] = {"close", "signal_entry", "signal_exit"}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """All outputs from a single backtest run.

    Attributes
    ----------
    portfolio_history : pd.Series
        Daily portfolio value indexed by date.  Name: ``"portfolio"``.
    trade_log : pd.DataFrame
        One row per completed trade (open trades are force-closed at the last
        bar).  Empty DataFrame (with correct columns) when no trades occur.
        Columns: entry_date, entry_price, exit_date, exit_price, shares,
                 pnl, return_pct, holding_days.
    benchmark : pd.Series
        Daily buy-and-hold portfolio value, same index as portfolio_history.
        Name: ``"benchmark"``.
    strategy_name : str
    ticker : str
    starting_capital : float
    transaction_cost_pct : float
    """

    portfolio_history: pd.Series
    trade_log: pd.DataFrame
    benchmark: pd.Series
    strategy_name: str = ""
    ticker: str = ""
    starting_capital: float = DEFAULT_STARTING_CAPITAL
    transaction_cost_pct: float = TRANSACTION_COST_PCT


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_trade_log() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical trade log column schema."""
    return pd.DataFrame(columns=TRADE_LOG_COLUMNS)


def _make_benchmark(
    df: pd.DataFrame,
    starting_capital: float,
    cost_pct: float,
) -> pd.Series:
    """Buy-and-hold equity curve starting from the first bar's close."""
    first_close = df["close"].iloc[0]
    bh_shares = starting_capital / (first_close * (1 + cost_pct))
    benchmark = df["close"] * bh_shares
    benchmark.name = "benchmark"
    return benchmark


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def run_backtest(
    signals_df: pd.DataFrame,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
    transaction_cost_pct: float = TRANSACTION_COST_PCT,
    strategy_name: str = "",
    ticker: str = "",
) -> BacktestResult:
    """Simulate a standalone strategy (Strategies 1–4) from its signal frame.

    Parameters
    ----------
    signals_df:
        DataFrame produced by a strategy's ``generate_signals()`` call.
        Must contain ``close``, ``signal_entry``, and ``signal_exit`` columns.
        The index must be a DatetimeIndex (as produced by the data loader).
    starting_capital:
        Initial cash available.  Defaults to ``DEFAULT_STARTING_CAPITAL``.
    transaction_cost_pct:
        Round-trip cost rate applied on each trade leg.
        Defaults to ``TRANSACTION_COST_PCT``.
    strategy_name:
        Label stored in the result (used for display / reporting).
    ticker:
        Asset label stored in the result.

    Returns
    -------
    BacktestResult
        Contains portfolio_history, trade_log, benchmark, and metadata.

    Raises
    ------
    ValueError
        If required columns are missing from *signals_df*.
    """
    missing = _REQUIRED_SIGNAL_COLS - set(signals_df.columns)
    if missing:
        raise ValueError(
            f"run_backtest: signals_df is missing required columns: {missing}."
        )

    # ---- simulation state ------------------------------------------------
    cash: float = float(starting_capital)
    shares: float = 0.0
    in_position: bool = False
    entry_date = None
    entry_price: float = 0.0
    entry_value: float = 0.0   # total cash committed at entry (for P&L calc)

    portfolio_values: list[float] = []
    trades: list[dict] = []

    # ---- main loop -------------------------------------------------------
    for date, row in signals_df.iterrows():
        close: float = float(row["close"])
        sig_entry: bool = bool(row["signal_entry"])
        sig_exit: bool = bool(row["signal_exit"])

        if in_position:
            if sig_exit:
                # ---- EXIT at today's close --------------------------------
                exit_price = close
                gross = shares * exit_price
                proceeds = gross * (1.0 - transaction_cost_pct)
                pnl = proceeds - entry_value
                return_pct = pnl / entry_value if entry_value > 0 else 0.0
                holding_days = (date - entry_date).days

                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": date,
                    "exit_price": exit_price,
                    "shares": shares,
                    "pnl": pnl,
                    "return_pct": return_pct,
                    "holding_days": holding_days,
                })

                cash = proceeds
                shares = 0.0
                in_position = False

        else:  # not in position
            if sig_entry:
                # ---- ENTRY at today's close --------------------------------
                entry_price = close
                entry_date = date
                entry_value = cash  # total cash committed
                shares = cash / (close * (1.0 + transaction_cost_pct))
                cash = 0.0
                in_position = True

        # ---- record portfolio value at close ------------------------------
        portfolio_values.append(cash + shares * close)

    # ---- force-close any open position at the last bar -------------------
    if in_position:
        last_date = signals_df.index[-1]
        last_close = float(signals_df["close"].iloc[-1])
        proceeds = shares * last_close * (1.0 - transaction_cost_pct)
        pnl = proceeds - entry_value
        return_pct = pnl / entry_value if entry_value > 0 else 0.0
        holding_days = (last_date - entry_date).days

        trades.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": last_date,
            "exit_price": last_close,
            "shares": shares,
            "pnl": pnl,
            "return_pct": return_pct,
            "holding_days": holding_days,
        })
        # portfolio_history already recorded the closing value for the last bar

    # ---- assemble outputs ------------------------------------------------
    portfolio_history = pd.Series(
        portfolio_values,
        index=signals_df.index,
        name="portfolio",
        dtype=float,
    )

    trade_log = (
        pd.DataFrame(trades, columns=TRADE_LOG_COLUMNS)
        if trades
        else _empty_trade_log()
    )

    benchmark = _make_benchmark(signals_df, starting_capital, transaction_cost_pct)

    return BacktestResult(
        portfolio_history=portfolio_history,
        trade_log=trade_log,
        benchmark=benchmark,
        strategy_name=strategy_name,
        ticker=ticker,
        starting_capital=starting_capital,
        transaction_cost_pct=transaction_cost_pct,
    )


# ---------------------------------------------------------------------------
# Hybrid strategy runner
# ---------------------------------------------------------------------------

def run_hybrid_backtest(
    signals_dict: dict[str, pd.DataFrame],
    allocations: dict[str, float],
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
    transaction_cost_pct: float = TRANSACTION_COST_PCT,
    ticker: str = "",
) -> BacktestResult:
    """Simulate the Hybrid Allocation Strategy (Strategy 5).

    Runs each sub-strategy independently in its own capital bucket, then sums
    the resulting equity curves to form the combined portfolio.

    Parameters
    ----------
    signals_dict:
        Dict returned by ``hybrid.generate_signals()``.
        Keys: ``"breakout_b"``, ``"ma_crossover"``, ``"rsi_mean_reversion"``.
        Values: signal DataFrames (each with ``close``, ``signal_entry``,
        ``signal_exit``).
    allocations:
        Weight dict (e.g. ``hybrid.ALLOCATIONS``).  Must sum to 1.0.
    starting_capital:
        Total portfolio capital to allocate across sub-strategies.
    transaction_cost_pct:
        Applied on each sub-strategy's trades.
    ticker:
        Asset label stored in the result.

    Returns
    -------
    BacktestResult
        portfolio_history — sum of sub-strategy equity curves.
        trade_log — concatenated trade logs with a ``sub_strategy`` column.
        benchmark — buy-and-hold using full *starting_capital*.
    """
    weight_sum = sum(allocations.values())
    if abs(weight_sum - 1.0) > 1e-9:
        raise ValueError(
            f"run_hybrid_backtest: allocation weights must sum to 1.0, "
            f"got {weight_sum:.10f}."
        )

    missing_keys = set(allocations.keys()) - set(signals_dict.keys())
    if missing_keys:
        raise ValueError(
            f"run_hybrid_backtest: signals_dict missing keys: {missing_keys}."
        )

    sub_results: dict[str, BacktestResult] = {}
    for key, weight in allocations.items():
        sub_capital = starting_capital * weight
        sub_results[key] = run_backtest(
            signals_df=signals_dict[key],
            starting_capital=sub_capital,
            transaction_cost_pct=transaction_cost_pct,
            strategy_name=key,
            ticker=ticker,
        )

    # ---- combine equity curves ------------------------------------------
    # Sum the sub-strategy portfolio histories directly (each is already
    # scaled by its proportional capital, so the sum = total portfolio value).
    combined_portfolio = sum(r.portfolio_history for r in sub_results.values())
    combined_portfolio.name = "portfolio"

    # ---- aggregate trade logs -------------------------------------------
    log_frames = []
    for key, result in sub_results.items():
        if not result.trade_log.empty:
            sub_log = result.trade_log.copy()
            sub_log.insert(0, "sub_strategy", key)
            log_frames.append(sub_log)

    if log_frames:
        combined_trade_log = pd.concat(log_frames, ignore_index=True)
        combined_trade_log.sort_values("entry_date", inplace=True)
        combined_trade_log.reset_index(drop=True, inplace=True)
    else:
        combined_trade_log = _empty_trade_log()
        combined_trade_log.insert(0, "sub_strategy", pd.Series(dtype=str))

    # ---- benchmark: buy-and-hold with full starting_capital --------------
    # Use the first sub-strategy's signals_df (all share the same price data)
    first_key = next(iter(allocations))
    benchmark = _make_benchmark(
        signals_dict[first_key], starting_capital, transaction_cost_pct
    )

    return BacktestResult(
        portfolio_history=combined_portfolio,
        trade_log=combined_trade_log,
        benchmark=benchmark,
        strategy_name="Hybrid Allocation Strategy",
        ticker=ticker,
        starting_capital=starting_capital,
        transaction_cost_pct=transaction_cost_pct,
    )
