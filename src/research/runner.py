"""
Internal research runner — generates and persists backtest results for all
strategy × asset × timeframe combinations.

Public interface
----------------
run_all_backtests(...)   Run the full matrix and save JSON result files.
save_result(...)         Serialize one BacktestResult + metrics to JSON.
load_result(...)         Deserialize a previously saved result from JSON.
results_summary(...)     DataFrame showing which result files exist.

Output layout
-------------
Each result is stored as a self-contained JSON file at:

    <output_dir>/results/<ticker>/<strategy>/<timeframe>y.json

File contents
-------------
{
  "metadata": {strategy, ticker, timeframe_years, starting_capital,
               transaction_cost_pct, generated_at, data_start, data_end,
               n_bars},
  "metrics":  {total_return, annualized_return, ...},   // 11 keys
  "equity":   [{"date": "YYYY-MM-DD", "portfolio": float, "benchmark": float}, ...],
  "trade_log":[{"entry_date": "...", "entry_price": float, ...}, ...]
}

NaN values are serialised as JSON null.

Usage
-----
    from src.research.runner import run_all_backtests
    run_all_backtests(verbose=True)
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.analytics.metrics import compute_metrics
from src.backtester.engine import BacktestResult, run_backtest, run_hybrid_backtest
from src.config.settings import (
    ALL_ASSETS,
    ANCHOR_ASSETS,
    DEFAULT_STARTING_CAPITAL,
    REPORTS_DIR,
    STRATEGY_ORDER,
    TIMEFRAMES_YEARS,
    TRANSACTION_COST_PCT,
)
from src.data.loader import load_ohlcv
from src.indicators.engine import add_indicators
from src.strategies.breakout_a import generate_signals as ba_signals
from src.strategies.breakout_b import generate_signals as bb_signals
from src.strategies.hybrid import ALLOCATIONS
from src.strategies.hybrid import generate_signals as hybrid_signals
from src.strategies.ma_crossover import generate_signals as ma_signals
from src.strategies.rsi_mean_reversion import generate_signals as rsi_signals

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STRATEGY_RUNNERS = {
    "ma_crossover": ma_signals,
    "rsi_mean_reversion": rsi_signals,
    "breakout_a": ba_signals,
    "breakout_b": bb_signals,
    # hybrid handled separately — returns dict, not DataFrame
}

# Indicator columns to save per strategy for strategy-specific chart rendering
_CHART_COLUMNS: dict[str, list[str]] = {
    "ma_crossover":       ["close", "sma_50", "sma_200"],
    "rsi_mean_reversion": ["close", "rsi_14"],
    "breakout_a":         ["close", "rolling_high_20", "rolling_low_10"],
    "breakout_b":         ["close", "rolling_high_20", "rolling_low_10",
                           "sma_200", "volume", "avg_volume_20", "rsi_14"],
    "hybrid":             [],  # handled separately with sleeve equity
}


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _nan_to_none(value: Any) -> Any:
    """Convert float NaN to None so json.dumps() can serialise it as null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _serialise_chart_timeseries(signals_df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """Extract indicator columns from signals_df as date-keyed records for chart rendering."""
    records = []
    for date, row in signals_df[cols].iterrows():
        record: dict = {"date": str(date.date())}
        for col in cols:
            val = row[col]
            if isinstance(val, float) and math.isnan(val):
                record[col] = None
            elif hasattr(val, "item"):
                v = val.item()
                record[col] = int(v) if col == "volume" else round(float(v), 4)
            else:
                record[col] = int(val) if col == "volume" else round(float(val), 4)
        records.append(record)
    return records


def _serialise_sleeve_equity(portfolio: pd.Series) -> list[dict]:
    return [
        {"date": str(d.date()), "value": round(float(v), 6)}
        for d, v in zip(portfolio.index, portfolio.values)
    ]


def _serialise_metrics(metrics: dict) -> dict:
    return {k: _nan_to_none(v) for k, v in metrics.items()}


def _serialise_equity(
    portfolio: pd.Series,
    benchmark: pd.Series,
) -> list[dict]:
    return [
        {
            "date": str(date.date()),
            "portfolio": round(float(pv), 6),
            "benchmark": round(float(bv), 6),
        }
        for date, pv, bv in zip(portfolio.index, portfolio.values, benchmark.values)
    ]


def _serialise_trade_log(trade_log: pd.DataFrame) -> list[dict]:
    if trade_log.empty:
        return []
    records = []
    for _, row in trade_log.iterrows():
        record: dict = {}
        for col in trade_log.columns:
            val = row[col]
            if isinstance(val, pd.Timestamp):
                record[col] = str(val.date())
            elif isinstance(val, float) and math.isnan(val):
                record[col] = None
            elif hasattr(val, "item"):        # numpy scalar
                record[col] = val.item()
            else:
                record[col] = val
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def _result_path(output_dir: str, strategy_key: str, ticker: str, timeframe_years: int) -> str:
    safe_ticker = ticker.replace("-", "_")   # BTC-USD → BTC_USD for filesystem
    return os.path.join(output_dir, "results", safe_ticker, strategy_key,
                        f"{timeframe_years}y.json")


def save_result(
    result: BacktestResult,
    metrics: dict,
    output_dir: str,
    strategy_key: str,
    ticker: str,
    timeframe_years: int,
    chart_data: "dict | None" = None,
) -> str:
    """Serialise *result* and *metrics* to a JSON file.

    Returns the absolute path of the written file.
    """
    path = _result_path(output_dir, strategy_key, ticker, timeframe_years)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    portfolio = result.portfolio_history
    payload = {
        "metadata": {
            "strategy": strategy_key,
            "ticker": ticker,
            "timeframe_years": timeframe_years,
            "starting_capital": result.starting_capital,
            "transaction_cost_pct": result.transaction_cost_pct,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_start": str(portfolio.index[0].date()),
            "data_end": str(portfolio.index[-1].date()),
            "n_bars": len(portfolio),
        },
        "metrics": _serialise_metrics(metrics),
        "equity": _serialise_equity(portfolio, result.benchmark),
        "trade_log": _serialise_trade_log(result.trade_log),
    }

    if chart_data is not None:
        payload["chart_data"] = chart_data

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path


def load_result(
    strategy_key: str,
    ticker: str,
    timeframe_years: int,
    output_dir: str = REPORTS_DIR,
) -> dict:
    """Load a previously saved result JSON file.

    Returns the raw payload dict with keys:
      ``metadata``, ``metrics``, ``equity``, ``trade_log``.

    Raises
    ------
    FileNotFoundError
        If the result file does not exist yet.
    """
    path = _result_path(output_dir, strategy_key, ticker, timeframe_years)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_equity_series(
    strategy_key: str,
    ticker: str,
    timeframe_years: int,
    output_dir: str = REPORTS_DIR,
) -> pd.DataFrame:
    """Return the equity curve as a DataFrame with columns date, portfolio, benchmark.

    The ``date`` column is parsed to a DatetimeIndex.
    """
    payload = load_result(strategy_key, ticker, timeframe_years, output_dir)
    df = pd.DataFrame(payload["equity"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def load_metrics(
    strategy_key: str,
    ticker: str,
    timeframe_years: int,
    output_dir: str = REPORTS_DIR,
) -> dict:
    """Return the metrics dict for a saved result."""
    return load_result(strategy_key, ticker, timeframe_years, output_dir)["metrics"]


# ---------------------------------------------------------------------------
# Existence / summary helpers
# ---------------------------------------------------------------------------

def result_exists(
    strategy_key: str,
    ticker: str,
    timeframe_years: int,
    output_dir: str = REPORTS_DIR,
) -> bool:
    """Return True if the result file for this combination exists."""
    return os.path.isfile(_result_path(output_dir, strategy_key, ticker, timeframe_years))


def results_summary(
    assets: list[str] | None = None,
    strategies: list[str] | None = None,
    timeframes: list[int] | None = None,
    output_dir: str = REPORTS_DIR,
) -> pd.DataFrame:
    """Return a DataFrame showing which result files exist.

    Columns: strategy, ticker, timeframe_years, exists, path.
    """
    if assets is None:
        assets = ALL_ASSETS
    if strategies is None:
        strategies = STRATEGY_ORDER
    if timeframes is None:
        timeframes = TIMEFRAMES_YEARS

    rows = []
    for strategy in strategies:
        for ticker in assets:
            for tf in timeframes:
                path = _result_path(output_dir, strategy, ticker, tf)
                rows.append({
                    "strategy": strategy,
                    "ticker": ticker,
                    "timeframe_years": tf,
                    "exists": os.path.isfile(path),
                    "path": path,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_all_backtests(
    assets: list[str] | None = None,
    timeframes: list[int] | None = None,
    starting_capital: float = DEFAULT_STARTING_CAPITAL,
    transaction_cost_pct: float = TRANSACTION_COST_PCT,
    output_dir: str = REPORTS_DIR,
    verbose: bool = True,
) -> dict[str, int]:
    """Run all strategy × asset × timeframe combinations and save results.

    Parameters
    ----------
    assets:
        List of ticker symbols to run.  Defaults to ``ALL_ASSETS``
        (anchor assets + GLD).
    timeframes:
        List of lookback windows in years.  Defaults to ``TIMEFRAMES_YEARS``
        (1, 5, 10, 20).
    starting_capital:
        Capital used for every backtest run.
    transaction_cost_pct:
        Round-trip transaction cost applied on each leg.
    output_dir:
        Root directory for saved result files.  Defaults to ``REPORTS_DIR``.
    verbose:
        If True, print progress to stdout.

    Returns
    -------
    dict with keys ``"saved"`` (int), ``"skipped"`` (int), ``"errors"`` (int).
    """
    if assets is None:
        assets = ALL_ASSETS
    if timeframes is None:
        timeframes = TIMEFRAMES_YEARS

    saved = skipped = errors = 0

    for ticker in assets:
        for tf in timeframes:
            # ---- Load and prepare data once per (ticker, timeframe) ------
            if verbose:
                print(f"\n[{ticker} / {tf}y]  loading data ...", flush=True)
            try:
                raw = load_ohlcv(ticker, tf)
                ind = add_indicators(raw)
            except Exception as exc:
                print(f"  ERROR loading {ticker} {tf}y: {exc}")
                errors += len(STRATEGY_ORDER)
                continue

            # ---- Pre-generate signals for all strategies -----------------
            try:
                strategy_signals = {
                    "ma_crossover": ma_signals(ind),
                    "rsi_mean_reversion": rsi_signals(ind),
                    "breakout_a": ba_signals(ind),
                    "breakout_b": bb_signals(ind),
                    "hybrid": hybrid_signals(ind),   # returns a dict
                }
            except Exception as exc:
                print(f"  ERROR generating signals for {ticker} {tf}y: {exc}")
                errors += len(STRATEGY_ORDER)
                continue

            # ---- Run each strategy and save result -----------------------
            for strategy_key in STRATEGY_ORDER:
                if verbose:
                    print(f"  {strategy_key} ...", end=" ", flush=True)

                try:
                    if strategy_key == "hybrid":
                        result = run_hybrid_backtest(
                            strategy_signals["hybrid"],
                            ALLOCATIONS,
                            starting_capital=starting_capital,
                            transaction_cost_pct=transaction_cost_pct,
                            ticker=ticker,
                        )
                        # Per-sleeve equity for the signals/sleeve-breakdown chart
                        sleeve_equity_data: dict = {}
                        for sub_key, weight in ALLOCATIONS.items():
                            sub_res = run_backtest(
                                strategy_signals["hybrid"][sub_key],
                                starting_capital=starting_capital * weight,
                                transaction_cost_pct=transaction_cost_pct,
                            )
                            sleeve_equity_data[sub_key] = _serialise_sleeve_equity(
                                sub_res.portfolio_history
                            )
                        chart_data: "dict | None" = {
                            "type": "hybrid",
                            "sleeve_equity": sleeve_equity_data,
                            "allocations": {k: float(v) for k, v in ALLOCATIONS.items()},
                        }
                    else:
                        result = run_backtest(
                            strategy_signals[strategy_key],
                            starting_capital=starting_capital,
                            transaction_cost_pct=transaction_cost_pct,
                            strategy_name=strategy_key,
                            ticker=ticker,
                        )
                        cols = _CHART_COLUMNS.get(strategy_key, [])
                        if cols:
                            chart_data = {
                                "type": strategy_key,
                                "timeseries": _serialise_chart_timeseries(
                                    strategy_signals[strategy_key], cols
                                ),
                            }
                        else:
                            chart_data = None

                    metrics = compute_metrics(result)
                    path = save_result(
                        result, metrics, output_dir, strategy_key, ticker, tf,
                        chart_data=chart_data,
                    )
                    saved += 1
                    if verbose:
                        total_ret = metrics.get("total_return", float("nan"))
                        ret_str = (
                            f"{total_ret*100:+.1f}%"
                            if not math.isnan(total_ret)
                            else "n/a"
                        )
                        print(f"saved  [return {ret_str}]")

                except Exception as exc:
                    errors += 1
                    if verbose:
                        print(f"ERROR: {exc}")

    # ---- Summary ---------------------------------------------------------
    if verbose:
        print(f"\n{'='*50}")
        print(f"Done.  saved={saved}  skipped={skipped}  errors={errors}")

    return {"saved": saved, "skipped": skipped, "errors": errors}
