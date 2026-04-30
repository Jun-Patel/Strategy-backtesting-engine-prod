"""
Analytics layer — compute performance metrics from a completed BacktestResult.

Public interface
----------------
compute_metrics(result)  ->  dict          primary entry point
metrics_to_series(m)     ->  pd.Series     display-ready (percentages where appropriate)

Metric definitions
------------------
total_return          (final_portfolio - starting_capital) / starting_capital
annualized_return     geometric annualization over the full period (calendar days)
volatility            annualized std of daily portfolio returns (252 trading days/year)
sharpe_ratio          annualized_mean_daily_return / annualized_std_daily_return
                      (risk-free rate = 0; industry-standard arithmetic method)
max_drawdown          minimum of (portfolio - running_peak) / running_peak  [≤ 0]
win_rate              fraction of trades with return_pct > 0
trade_count           number of completed trades (force-closed trades included)
avg_trade_return      mean of trade_log["return_pct"] across all trades
avg_holding_period    mean of trade_log["holding_days"] across all trades (calendar days)
benchmark_return      (benchmark_final - starting_capital) / starting_capital
outperformance        total_return - benchmark_return

NaN conventions
---------------
Metrics that cannot be computed are returned as float('nan'):
  - annualized_return if the series spans < 1 day
  - volatility / sharpe_ratio if the series has < 2 bars
  - sharpe_ratio if volatility == 0 (flat portfolio)
  - win_rate / avg_trade_return / avg_holding_period if trade_count == 0
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.backtester.engine import BacktestResult

# 252 trading days is the standard annualization factor for daily equity data
TRADING_DAYS_PER_YEAR: int = 252
CALENDAR_DAYS_PER_YEAR: int = 365


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def compute_metrics(result: BacktestResult) -> dict:
    """Return a dict of all performance metrics for *result*.

    Parameters
    ----------
    result:
        A ``BacktestResult`` produced by ``run_backtest()`` or
        ``run_hybrid_backtest()``.

    Returns
    -------
    dict with keys:
        total_return, annualized_return, volatility, sharpe_ratio,
        max_drawdown, win_rate, trade_count, avg_trade_return,
        avg_holding_period, benchmark_return, outperformance.

    All return / ratio values are in decimal form (e.g. 0.15 = 15 %).
    max_drawdown is ≤ 0 (e.g. -0.20 = -20 % peak-to-trough decline).
    """
    portfolio = result.portfolio_history
    trade_log = result.trade_log
    benchmark = result.benchmark
    starting_capital = float(result.starting_capital)

    # ------------------------------------------------------------------ #
    # Return metrics                                                        #
    # ------------------------------------------------------------------ #

    final_value = float(portfolio.iloc[-1])
    total_return = (final_value - starting_capital) / starting_capital

    # Annualized return — geometric, using calendar days
    n_calendar_days = (portfolio.index[-1] - portfolio.index[0]).days
    if n_calendar_days > 0:
        annualized_return = (
            (1.0 + total_return) ** (CALENDAR_DAYS_PER_YEAR / n_calendar_days) - 1.0
        )
    else:
        annualized_return = float("nan")

    # ------------------------------------------------------------------ #
    # Risk metrics                                                          #
    # ------------------------------------------------------------------ #

    # Daily portfolio returns (percentage change)
    daily_returns = portfolio.pct_change().dropna()

    if len(daily_returns) >= 2:
        std_daily = float(daily_returns.std())
        mean_daily = float(daily_returns.mean())
        volatility = std_daily * math.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        std_daily = float("nan")
        mean_daily = float("nan")
        volatility = float("nan")

    # Sharpe ratio (risk-free rate = 0)
    if not math.isnan(volatility) and volatility > 0:
        annualized_mean = mean_daily * TRADING_DAYS_PER_YEAR
        sharpe_ratio = annualized_mean / volatility
    else:
        sharpe_ratio = float("nan")

    # Max drawdown: minimum of rolling (value - peak) / peak
    running_peak = portfolio.cummax()
    drawdown_series = (portfolio - running_peak) / running_peak
    max_drawdown = float(drawdown_series.min())   # ≤ 0

    # ------------------------------------------------------------------ #
    # Trade-based metrics                                                   #
    # ------------------------------------------------------------------ #

    trade_count = len(trade_log)

    if trade_count > 0:
        returns = trade_log["return_pct"]
        wins_mask = returns > 0
        losses_mask = returns <= 0

        wins = int(wins_mask.sum())
        win_rate = wins / trade_count
        avg_trade_return = float(returns.mean())
        avg_holding_period = float(trade_log["holding_days"].mean())

        win_returns = returns[wins_mask]
        loss_returns = returns[losses_mask]
        avg_win  = float(win_returns.mean())  if len(win_returns)  > 0 else float("nan")
        avg_loss = float(loss_returns.mean()) if len(loss_returns) > 0 else float("nan")
        max_win  = float(returns.max())
        max_loss = float(returns.min())
    else:
        win_rate = float("nan")
        avg_trade_return = float("nan")
        avg_holding_period = float("nan")
        avg_win  = float("nan")
        avg_loss = float("nan")
        max_win  = float("nan")
        max_loss = float("nan")

    # Avg drawdown — mean of the rolling drawdown series (≤ 0)
    avg_drawdown = float(drawdown_series.mean())

    # ------------------------------------------------------------------ #
    # Benchmark and outperformance                                          #
    # ------------------------------------------------------------------ #

    # Benchmark total return is measured from starting_capital (same base as strategy)
    benchmark_final = float(benchmark.iloc[-1])
    benchmark_return = (benchmark_final - starting_capital) / starting_capital

    outperformance = total_return - benchmark_return

    # ------------------------------------------------------------------ #
    # Assemble result                                                       #
    # ------------------------------------------------------------------ #

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "avg_trade_return": avg_trade_return,
        "avg_holding_period": avg_holding_period,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_win": max_win,
        "max_loss": max_loss,
        "benchmark_return": benchmark_return,
        "outperformance": outperformance,
    }


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def metrics_to_series(metrics: dict) -> pd.Series:
    """Format a metrics dict as a display-ready pd.Series.

    Percentage values are multiplied by 100 and labelled with a ``%`` suffix
    in the index.  Counts and ratios are shown as-is.

    Parameters
    ----------
    metrics:
        Dict returned by ``compute_metrics()``.

    Returns
    -------
    pd.Series with string index and float values, rounded for display.
    """
    pct_keys = {
        "total_return",
        "annualized_return",
        "volatility",
        "max_drawdown",
        "win_rate",
        "avg_trade_return",
        "benchmark_return",
        "outperformance",
    }

    labels = {
        "total_return": "Total Return (%)",
        "annualized_return": "Annualized Return (%)",
        "volatility": "Volatility (%)",
        "sharpe_ratio": "Sharpe Ratio",
        "max_drawdown": "Max Drawdown (%)",
        "win_rate": "Win Rate (%)",
        "trade_count": "Trade Count",
        "avg_trade_return": "Avg Trade Return (%)",
        "avg_holding_period": "Avg Holding Period (days)",
        "benchmark_return": "Benchmark Return (%)",
        "outperformance": "Outperformance (%)",
    }

    data = {}
    for key, value in metrics.items():
        label = labels.get(key, key)
        if key in pct_keys and not (isinstance(value, float) and math.isnan(value)):
            data[label] = round(float(value) * 100, 2)
        elif key == "trade_count":
            data[label] = int(value)
        elif isinstance(value, float) and math.isnan(value):
            data[label] = float("nan")
        else:
            data[label] = round(float(value), 4)

    return pd.Series(data)
