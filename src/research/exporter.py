"""
Report data export layer for the Strategy Backtesting Engine.

Reads saved JSON result files (written by runner.py) and produces
flat, human-readable export artefacts in ``<output_dir>/exports/``.

Public interface
----------------
export_metrics_table(...)
    Flat CSV + JSON of every available result; one row per
    (strategy, asset, timeframe) with all 11 metrics.

export_strategy_comparisons(...)
    One CSV per (asset × timeframe); columns = strategies, rows = metrics.
    Easy to open in Excel for a side-by-side strategy comparison.

export_summary_json(...)
    High-level JSON summary: coverage, top performers, generated timestamp.

run_all_exports(...)
    Convenience wrapper that calls all three functions and returns a dict
    of written paths/counts.

Output layout
-------------
    <output_dir>/exports/
        metrics_all.csv          — flat table, one row per result
        metrics_all.json         — same data as JSON array
        comparison_<TICKER>_<TF>y.csv  — strategy comparison per asset/timeframe
        summary.json             — coverage + top-performer summary
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config.settings import (
    ALL_ASSETS,
    REPORTS_DIR,
    STRATEGIES,
    STRATEGY_ORDER,
    TIMEFRAMES_YEARS,
)
from src.research.runner import load_metrics, load_result, result_exists

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXPORTS_SUBDIR = "exports"

# Metrics that are stored as decimals but displayed as percentages
_PCT_METRICS = {
    "total_return", "annualized_return", "volatility",
    "max_drawdown", "win_rate", "avg_trade_return",
    "benchmark_return", "outperformance",
}

# Column order in the flat metrics CSV
_FLAT_COLUMNS = [
    "strategy", "strategy_name", "ticker", "timeframe_years",
    "total_return_pct", "annualized_return_pct", "volatility_pct",
    "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "trade_count",
    "avg_trade_return_pct", "avg_holding_period_days",
    "benchmark_return_pct", "outperformance_pct",
    "data_start", "data_end", "n_bars",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _exports_dir(output_dir: str) -> str:
    path = os.path.join(output_dir, _EXPORTS_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def _safe_ticker(ticker: str) -> str:
    """BTC-USD → BTC_USD for use in filenames."""
    return ticker.replace("-", "_")


def _scale_pct(value: Any) -> Any:
    """Multiply a decimal metric by 100 for human-readable output.
    Preserves None and NaN.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value) * 100, 4)


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _build_flat_row(
    strategy_key: str,
    ticker: str,
    timeframe_years: int,
    output_dir: str,
) -> dict | None:
    """Load one result and return a flat row dict, or None if missing."""
    if not result_exists(strategy_key, ticker, timeframe_years, output_dir):
        return None

    payload = load_result(strategy_key, ticker, timeframe_years, output_dir)
    m = payload["metrics"]
    meta = payload["metadata"]

    return {
        "strategy":               strategy_key,
        "strategy_name":          STRATEGIES[strategy_key]["name"],
        "ticker":                 ticker,
        "timeframe_years":        timeframe_years,
        "total_return_pct":       _scale_pct(m.get("total_return")),
        "annualized_return_pct":  _scale_pct(m.get("annualized_return")),
        "volatility_pct":         _scale_pct(m.get("volatility")),
        "sharpe_ratio":           _nan_to_none(m.get("sharpe_ratio")),
        "max_drawdown_pct":       _scale_pct(m.get("max_drawdown")),
        "win_rate_pct":           _scale_pct(m.get("win_rate")),
        "trade_count":            m.get("trade_count"),
        "avg_trade_return_pct":   _scale_pct(m.get("avg_trade_return")),
        "avg_holding_period_days": _nan_to_none(m.get("avg_holding_period")),
        "benchmark_return_pct":   _scale_pct(m.get("benchmark_return")),
        "outperformance_pct":     _scale_pct(m.get("outperformance")),
        "data_start":             meta.get("data_start"),
        "data_end":               meta.get("data_end"),
        "n_bars":                 meta.get("n_bars"),
    }


# ---------------------------------------------------------------------------
# export_metrics_table
# ---------------------------------------------------------------------------

def export_metrics_table(
    output_dir: str = REPORTS_DIR,
    assets: list[str] | None = None,
    strategies: list[str] | None = None,
    timeframes: list[int] | None = None,
    skip_missing: bool = True,
) -> dict[str, str]:
    """Export all available metrics to a flat CSV and JSON file.

    Each row corresponds to one (strategy, asset, timeframe) result.
    Percentage-type metrics are scaled ×100 for readability.

    Parameters
    ----------
    output_dir:
        Root results directory.  Defaults to ``REPORTS_DIR``.
    assets, strategies, timeframes:
        Subsets to export.  Default to all assets/strategies/timeframes.
    skip_missing:
        If True (default), silently skip combinations with no result file.
        If False, raise ``FileNotFoundError`` on the first missing file.

    Returns
    -------
    dict with keys ``"csv"`` and ``"json"`` — absolute paths of written files.
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
                row = _build_flat_row(strategy, ticker, tf, output_dir)
                if row is None:
                    if not skip_missing:
                        raise FileNotFoundError(
                            f"Result missing: {strategy} / {ticker} / {tf}y"
                        )
                    continue
                rows.append(row)

    df = pd.DataFrame(rows, columns=_FLAT_COLUMNS)

    exports = _exports_dir(output_dir)
    csv_path  = os.path.join(exports, "metrics_all.csv")
    json_path = os.path.join(exports, "metrics_all.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    return {"csv": csv_path, "json": json_path}


# ---------------------------------------------------------------------------
# export_strategy_comparisons
# ---------------------------------------------------------------------------

def export_strategy_comparisons(
    output_dir: str = REPORTS_DIR,
    assets: list[str] | None = None,
    timeframes: list[int] | None = None,
    skip_missing: bool = True,
) -> list[str]:
    """Export one comparison CSV per (asset × timeframe).

    Each CSV has strategies as rows and metrics as columns, making
    side-by-side comparison straightforward in a spreadsheet tool.

    Parameters
    ----------
    output_dir, assets, timeframes, skip_missing:
        See ``export_metrics_table``.

    Returns
    -------
    List of absolute paths of written comparison CSV files.
    """
    if assets is None:
        assets = ALL_ASSETS
    if timeframes is None:
        timeframes = TIMEFRAMES_YEARS

    exports = _exports_dir(output_dir)
    written: list[str] = []

    for ticker in assets:
        for tf in timeframes:
            rows = []
            for strategy in STRATEGY_ORDER:
                row = _build_flat_row(strategy, ticker, tf, output_dir)
                if row is None:
                    if not skip_missing:
                        raise FileNotFoundError(
                            f"Result missing: {strategy} / {ticker} / {tf}y"
                        )
                    continue
                rows.append(row)

            if not rows:
                continue

            df = pd.DataFrame(rows, columns=_FLAT_COLUMNS)

            safe = _safe_ticker(ticker)
            fname = f"comparison_{safe}_{tf}y.csv"
            path = os.path.join(exports, fname)
            df.to_csv(path, index=False)
            written.append(path)

    return written


# ---------------------------------------------------------------------------
# export_summary_json
# ---------------------------------------------------------------------------

def export_summary_json(
    output_dir: str = REPORTS_DIR,
    assets: list[str] | None = None,
    strategies: list[str] | None = None,
    timeframes: list[int] | None = None,
) -> str:
    """Export a high-level JSON summary of the result set.

    Includes coverage stats (available vs total), the best-performing
    strategy/asset/timeframe by total return, Sharpe ratio, and lowest
    max drawdown.

    Returns
    -------
    Absolute path of the written ``summary.json`` file.
    """
    if assets is None:
        assets = ALL_ASSETS
    if strategies is None:
        strategies = STRATEGY_ORDER
    if timeframes is None:
        timeframes = TIMEFRAMES_YEARS

    total_possible = len(strategies) * len(assets) * len(timeframes)
    rows = []

    for strategy in strategies:
        for ticker in assets:
            for tf in timeframes:
                row = _build_flat_row(strategy, ticker, tf, output_dir)
                if row is not None:
                    rows.append(row)

    available = len(rows)

    # Top performers — ignore None values
    def _best(key: str, highest: bool = True) -> dict | None:
        valid = [r for r in rows if r.get(key) is not None]
        if not valid:
            return None
        fn = max if highest else min
        winner = fn(valid, key=lambda r: r[key])
        return {
            "strategy":     winner["strategy"],
            "strategy_name": winner["strategy_name"],
            "ticker":       winner["ticker"],
            "timeframe_years": winner["timeframe_years"],
            "value":        winner[key],
        }

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage": {
            "available":     available,
            "missing":       total_possible - available,
            "total_possible": total_possible,
        },
        "assets":     assets,
        "strategies": strategies,
        "timeframes": timeframes,
        "top_performers": {
            "highest_total_return":     _best("total_return_pct",       highest=True),
            "highest_sharpe_ratio":     _best("sharpe_ratio",           highest=True),
            "lowest_max_drawdown":      _best("max_drawdown_pct",       highest=False),
            "highest_outperformance":   _best("outperformance_pct",     highest=True),
        },
    }

    exports = _exports_dir(output_dir)
    path = os.path.join(exports, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return path


# ---------------------------------------------------------------------------
# run_all_exports
# ---------------------------------------------------------------------------

def run_all_exports(
    output_dir: str = REPORTS_DIR,
    assets: list[str] | None = None,
    strategies: list[str] | None = None,
    timeframes: list[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all three export functions and return a dict of results.

    Returns
    -------
    dict with keys:
        ``"metrics_csv"``      — path to metrics_all.csv
        ``"metrics_json"``     — path to metrics_all.json
        ``"comparisons"``      — list of comparison CSV paths
        ``"summary_json"``     — path to summary.json
        ``"n_rows"``           — total rows in the flat metrics table
        ``"n_comparisons"``    — number of comparison files written
    """
    if verbose:
        print("Exporting metrics table …", flush=True)
    metrics_paths = export_metrics_table(output_dir, assets, strategies, timeframes)

    if verbose:
        print("Exporting strategy comparisons …", flush=True)
    comparison_paths = export_strategy_comparisons(output_dir, assets, timeframes)

    if verbose:
        print("Exporting summary JSON …", flush=True)
    summary_path = export_summary_json(output_dir, assets, strategies, timeframes)

    # Row count from the CSV
    df = pd.read_csv(metrics_paths["csv"])

    result = {
        "metrics_csv":    metrics_paths["csv"],
        "metrics_json":   metrics_paths["json"],
        "comparisons":    comparison_paths,
        "summary_json":   summary_path,
        "n_rows":         len(df),
        "n_comparisons":  len(comparison_paths),
    }

    if verbose:
        print(f"\nExports written to: {_exports_dir(output_dir)}")
        print(f"  metrics_all.csv     — {result['n_rows']} rows")
        print(f"  metrics_all.json    — {result['n_rows']} rows")
        print(f"  comparison CSVs     — {result['n_comparisons']} files")
        print(f"  summary.json        — coverage {result['n_rows']}/{len(strategies or STRATEGY_ORDER) * len(assets or ALL_ASSETS) * len(timeframes or TIMEFRAMES_YEARS)}")

    return result
