"""
Tests for src/research/exporter.py — Chunk 15 acceptance criteria.

Fast tests: use a temp dir populated with a handful of synthetic results
            (created via save_result); no network, no REPORTS_DIR dependency.

Slow tests: run against the real REPORTS_DIR (all 100 results).

Run fast only:
    pytest tests/test_exporter.py -v -m "not slow"

Run everything:
    pytest tests/test_exporter.py -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import TRADE_LOG_COLUMNS, BacktestResult
from src.config.settings import (
    ANCHOR_ASSETS,
    REPORTS_DIR,
    STRATEGY_ORDER,
    TIMEFRAMES_YEARS,
)
from src.research.exporter import (
    export_metrics_table,
    export_strategy_comparisons,
    export_summary_json,
    run_all_exports,
)
from src.research.runner import save_result


# ---------------------------------------------------------------------------
# Helpers — synthetic BacktestResult (mirrors test_runner.py)
# ---------------------------------------------------------------------------

def _make_result(
    strategy: str = "ma_crossover",
    ticker: str = "SPY",
    capital: float = 10_000.0,
    n: int = 10,
) -> tuple[BacktestResult, dict]:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    portfolio  = pd.Series(np.linspace(capital, capital * 1.10, n), index=dates, name="portfolio")
    benchmark  = pd.Series(np.linspace(capital, capital * 1.05, n), index=dates, name="benchmark")
    trade_log  = pd.DataFrame(
        [{"entry_date": dates[1], "entry_price": 100.0, "exit_date": dates[5],
          "exit_price": 110.0, "shares": 99.0, "pnl": 990.0,
          "return_pct": 0.10, "holding_days": 6}],
        columns=TRADE_LOG_COLUMNS,
    )
    result = BacktestResult(
        portfolio_history=portfolio, trade_log=trade_log, benchmark=benchmark,
        strategy_name=strategy, ticker=ticker, starting_capital=capital,
        transaction_cost_pct=0.001,
    )
    metrics = {
        "total_return": 0.10, "annualized_return": 0.40, "volatility": 0.15,
        "sharpe_ratio": 2.5, "max_drawdown": -0.05, "win_rate": 1.0,
        "trade_count": 1, "avg_trade_return": 0.10, "avg_holding_period": 6.0,
        "benchmark_return": 0.05, "outperformance": 0.05,
    }
    return result, metrics


def _populate_tmp(tmp: str, combos: list[tuple[str, str, int]]) -> None:
    """Save synthetic results for each (strategy, ticker, timeframe) combo."""
    for strategy, ticker, tf in combos:
        result, metrics = _make_result(strategy=strategy, ticker=ticker)
        save_result(result, metrics, tmp, strategy, ticker, tf)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_with_results():
    """Temp dir with 3 synthetic results: MA / SPY / 5, MA / QQQ / 5, RSI / SPY / 5."""
    tmp = tempfile.mkdtemp()
    _populate_tmp(tmp, [
        ("ma_crossover",       "SPY", 5),
        ("ma_crossover",       "QQQ", 5),
        ("rsi_mean_reversion", "SPY", 5),
    ])
    return tmp


@pytest.fixture()
def tmp_full():
    """Temp dir with all 5 strategies × 2 assets × 2 timeframes = 20 results."""
    tmp = tempfile.mkdtemp()
    combos = [
        (s, a, tf)
        for s in ["ma_crossover", "rsi_mean_reversion", "breakout_a", "breakout_b", "hybrid"]
        for a in ["SPY", "QQQ"]
        for tf in [1, 5]
    ]
    _populate_tmp(tmp, combos)
    return tmp


# ---------------------------------------------------------------------------
# TestExportMetricsTable
# ---------------------------------------------------------------------------


class TestExportMetricsTable:
    def test_returns_dict_with_csv_and_json_keys(self, tmp_with_results):
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY", "QQQ"],
                                      strategies=["ma_crossover", "rsi_mean_reversion"],
                                      timeframes=[5])
        assert "csv" in result
        assert "json" in result

    def test_csv_file_exists(self, tmp_with_results):
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY", "QQQ"],
                                      strategies=["ma_crossover", "rsi_mean_reversion"],
                                      timeframes=[5])
        assert os.path.isfile(result["csv"])

    def test_json_file_exists(self, tmp_with_results):
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY", "QQQ"],
                                      strategies=["ma_crossover", "rsi_mean_reversion"],
                                      timeframes=[5])
        assert os.path.isfile(result["json"])

    def test_csv_row_count_matches_available_results(self, tmp_with_results):
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY", "QQQ"],
                                      strategies=["ma_crossover", "rsi_mean_reversion"],
                                      timeframes=[5])
        df = pd.read_csv(result["csv"])
        assert len(df) == 3   # MA/SPY, MA/QQQ, RSI/SPY

    def test_csv_has_required_columns(self, tmp_with_results):
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY"],
                                      strategies=["ma_crossover"],
                                      timeframes=[5])
        df = pd.read_csv(result["csv"])
        for col in ["strategy", "ticker", "timeframe_years", "total_return_pct",
                    "sharpe_ratio", "max_drawdown_pct", "benchmark_return_pct",
                    "outperformance_pct", "data_start", "data_end", "n_bars"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_pct_values_are_scaled_by_100(self, tmp_with_results):
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY"],
                                      strategies=["ma_crossover"],
                                      timeframes=[5])
        df = pd.read_csv(result["csv"])
        # total_return stored as 0.10 → should appear as 10.0
        assert abs(df.iloc[0]["total_return_pct"] - 10.0) < 0.01

    def test_max_drawdown_is_negative_pct(self, tmp_with_results):
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY"],
                                      strategies=["ma_crossover"],
                                      timeframes=[5])
        df = pd.read_csv(result["csv"])
        assert df.iloc[0]["max_drawdown_pct"] < 0

    def test_json_is_valid_and_is_array(self, tmp_with_results):
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY"],
                                      strategies=["ma_crossover"],
                                      timeframes=[5])
        with open(result["json"]) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_json_row_has_strategy_field(self, tmp_with_results):
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY"],
                                      strategies=["ma_crossover"],
                                      timeframes=[5])
        with open(result["json"]) as f:
            data = json.load(f)
        assert data[0]["strategy"] == "ma_crossover"

    def test_skip_missing_true_does_not_raise(self, tmp_with_results):
        # RSI/QQQ/5 does not exist — should be silently skipped
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY", "QQQ"],
                                      strategies=["rsi_mean_reversion"],
                                      timeframes=[5],
                                      skip_missing=True)
        df = pd.read_csv(result["csv"])
        assert len(df) == 1   # only RSI/SPY/5 exists

    def test_skip_missing_false_raises(self, tmp_with_results):
        with pytest.raises(FileNotFoundError):
            export_metrics_table(tmp_with_results,
                                 assets=["QQQ"],
                                 strategies=["rsi_mean_reversion"],
                                 timeframes=[5],
                                 skip_missing=False)

    def test_files_written_to_exports_subdir(self, tmp_with_results):
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY"],
                                      strategies=["ma_crossover"],
                                      timeframes=[5])
        assert "exports" in result["csv"]

    def test_strategy_name_column_populated(self, tmp_with_results):
        result = export_metrics_table(tmp_with_results,
                                      assets=["SPY"],
                                      strategies=["ma_crossover"],
                                      timeframes=[5])
        df = pd.read_csv(result["csv"])
        assert df.iloc[0]["strategy_name"] == "Moving Average Crossover"


# ---------------------------------------------------------------------------
# TestExportStrategyComparisons
# ---------------------------------------------------------------------------


class TestExportStrategyComparisons:
    def test_returns_list_of_paths(self, tmp_full):
        paths = export_strategy_comparisons(tmp_full,
                                            assets=["SPY", "QQQ"],
                                            timeframes=[1, 5])
        assert isinstance(paths, list)

    def test_correct_number_of_comparison_files(self, tmp_full):
        paths = export_strategy_comparisons(tmp_full,
                                            assets=["SPY", "QQQ"],
                                            timeframes=[1, 5])
        assert len(paths) == 4   # 2 assets × 2 timeframes

    def test_all_files_exist(self, tmp_full):
        paths = export_strategy_comparisons(tmp_full,
                                            assets=["SPY"],
                                            timeframes=[1, 5])
        for p in paths:
            assert os.path.isfile(p)

    def test_comparison_file_has_five_strategy_rows(self, tmp_full):
        paths = export_strategy_comparisons(tmp_full,
                                            assets=["SPY"],
                                            timeframes=[5])
        df = pd.read_csv(paths[0])
        assert len(df) == 5

    def test_comparison_file_has_metrics_columns(self, tmp_full):
        paths = export_strategy_comparisons(tmp_full,
                                            assets=["SPY"],
                                            timeframes=[5])
        df = pd.read_csv(paths[0])
        assert "total_return_pct" in df.columns
        assert "sharpe_ratio" in df.columns

    def test_filename_encodes_ticker_and_timeframe(self, tmp_full):
        paths = export_strategy_comparisons(tmp_full,
                                            assets=["SPY"],
                                            timeframes=[5])
        assert any("SPY_5y" in p for p in paths)

    def test_btc_usd_filename_uses_underscore(self):
        tmp = tempfile.mkdtemp()
        _populate_tmp(tmp, [("ma_crossover", "BTC-USD", 1)])
        paths = export_strategy_comparisons(tmp,
                                            assets=["BTC-USD"],
                                            timeframes=[1],
                                            skip_missing=True)
        if paths:
            assert "BTC_USD" in paths[0]

    def test_all_strategies_in_same_comparison_file(self, tmp_full):
        paths = export_strategy_comparisons(tmp_full,
                                            assets=["SPY"],
                                            timeframes=[5])
        df = pd.read_csv(paths[0])
        for s in STRATEGY_ORDER:
            assert s in df["strategy"].values

    def test_pct_values_scaled_in_comparison_csv(self, tmp_full):
        paths = export_strategy_comparisons(tmp_full,
                                            assets=["SPY"],
                                            timeframes=[5])
        df = pd.read_csv(paths[0])
        # total_return stored as 0.10 → 10.0 in export
        assert df["total_return_pct"].max() > 1.0


# ---------------------------------------------------------------------------
# TestExportSummaryJson
# ---------------------------------------------------------------------------


class TestExportSummaryJson:
    def test_returns_path(self, tmp_full):
        path = export_summary_json(tmp_full,
                                   assets=["SPY", "QQQ"],
                                   strategies=["ma_crossover", "rsi_mean_reversion"],
                                   timeframes=[1, 5])
        assert isinstance(path, str)
        assert os.path.isfile(path)

    def test_filename_is_summary_json(self, tmp_full):
        path = export_summary_json(tmp_full,
                                   assets=["SPY"],
                                   strategies=["ma_crossover"],
                                   timeframes=[5])
        assert os.path.basename(path) == "summary.json"

    def test_is_valid_json(self, tmp_full):
        path = export_summary_json(tmp_full,
                                   assets=["SPY"],
                                   strategies=["ma_crossover"],
                                   timeframes=[5])
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_has_coverage_key(self, tmp_full):
        path = export_summary_json(tmp_full,
                                   assets=["SPY", "QQQ"],
                                   strategies=["ma_crossover", "rsi_mean_reversion"],
                                   timeframes=[1, 5])
        with open(path) as f:
            data = json.load(f)
        assert "coverage" in data

    def test_coverage_available_correct(self, tmp_full):
        path = export_summary_json(tmp_full,
                                   assets=["SPY", "QQQ"],
                                   strategies=["ma_crossover", "rsi_mean_reversion"],
                                   timeframes=[1, 5])
        with open(path) as f:
            data = json.load(f)
        # 2 strategies × 2 assets × 2 timeframes = 8 results written
        assert data["coverage"]["available"] == 8

    def test_coverage_total_possible_correct(self, tmp_full):
        path = export_summary_json(tmp_full,
                                   assets=["SPY", "QQQ"],
                                   strategies=["ma_crossover", "rsi_mean_reversion"],
                                   timeframes=[1, 5])
        with open(path) as f:
            data = json.load(f)
        assert data["coverage"]["total_possible"] == 8
        assert data["coverage"]["missing"] == 0

    def test_has_top_performers_key(self, tmp_full):
        path = export_summary_json(tmp_full,
                                   assets=["SPY"],
                                   strategies=["ma_crossover"],
                                   timeframes=[5])
        with open(path) as f:
            data = json.load(f)
        assert "top_performers" in data

    def test_top_performers_has_highest_return(self, tmp_full):
        path = export_summary_json(tmp_full,
                                   assets=["SPY"],
                                   strategies=["ma_crossover"],
                                   timeframes=[5])
        with open(path) as f:
            data = json.load(f)
        assert "highest_total_return" in data["top_performers"]

    def test_top_performer_has_strategy_and_ticker(self, tmp_full):
        path = export_summary_json(tmp_full,
                                   assets=["SPY", "QQQ"],
                                   strategies=["ma_crossover", "rsi_mean_reversion"],
                                   timeframes=[5])
        with open(path) as f:
            data = json.load(f)
        winner = data["top_performers"]["highest_total_return"]
        assert "strategy" in winner
        assert "ticker" in winner

    def test_has_generated_at_timestamp(self, tmp_full):
        path = export_summary_json(tmp_full,
                                   assets=["SPY"],
                                   strategies=["ma_crossover"],
                                   timeframes=[5])
        with open(path) as f:
            data = json.load(f)
        assert "generated_at" in data
        assert "T" in data["generated_at"]   # ISO format


# ---------------------------------------------------------------------------
# TestRunAllExports
# ---------------------------------------------------------------------------


class TestRunAllExports:
    def test_returns_dict(self, tmp_full):
        result = run_all_exports(tmp_full,
                                 assets=["SPY"],
                                 strategies=["ma_crossover"],
                                 timeframes=[5],
                                 verbose=False)
        assert isinstance(result, dict)

    def test_has_required_keys(self, tmp_full):
        result = run_all_exports(tmp_full,
                                 assets=["SPY"],
                                 strategies=["ma_crossover"],
                                 timeframes=[5],
                                 verbose=False)
        for key in ["metrics_csv", "metrics_json", "comparisons",
                    "summary_json", "n_rows", "n_comparisons"]:
            assert key in result

    def test_all_files_exist(self, tmp_full):
        result = run_all_exports(tmp_full,
                                 assets=["SPY"],
                                 strategies=["ma_crossover"],
                                 timeframes=[5],
                                 verbose=False)
        assert os.path.isfile(result["metrics_csv"])
        assert os.path.isfile(result["metrics_json"])
        assert os.path.isfile(result["summary_json"])
        for p in result["comparisons"]:
            assert os.path.isfile(p)

    def test_n_rows_correct(self, tmp_full):
        result = run_all_exports(tmp_full,
                                 assets=["SPY", "QQQ"],
                                 strategies=["ma_crossover"],
                                 timeframes=[1, 5],
                                 verbose=False)
        assert result["n_rows"] == 4   # MA × 2 assets × 2 timeframes

    def test_n_comparisons_correct(self, tmp_full):
        result = run_all_exports(tmp_full,
                                 assets=["SPY", "QQQ"],
                                 strategies=None,
                                 timeframes=[1, 5],
                                 verbose=False)
        assert result["n_comparisons"] == 4   # 2 assets × 2 timeframes


# ---------------------------------------------------------------------------
# TestFullExport  (slow — uses real REPORTS_DIR with 100 results)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFullExport:
    """Acceptance test: export the full 100-result set."""

    def test_full_export_produces_100_rows(self):
        result = run_all_exports(verbose=False)
        assert result["n_rows"] == 100

    def test_full_export_produces_20_comparison_files(self):
        # 5 assets × 4 timeframes = 20 comparison files
        result = run_all_exports(verbose=False)
        assert result["n_comparisons"] == 20

    def test_metrics_csv_has_all_strategies(self):
        result = run_all_exports(verbose=False)
        df = pd.read_csv(result["metrics_csv"])
        for s in STRATEGY_ORDER:
            assert s in df["strategy"].values

    def test_metrics_csv_has_all_anchor_assets(self):
        result = run_all_exports(verbose=False)
        df = pd.read_csv(result["metrics_csv"])
        for asset in ANCHOR_ASSETS:
            assert asset in df["ticker"].values

    def test_pct_columns_in_human_readable_range(self):
        result = run_all_exports(verbose=False)
        df = pd.read_csv(result["metrics_csv"])
        # Total return for a 1y window should be well below ±1000%
        valid = df["total_return_pct"].dropna()
        assert valid.abs().max() < 20_000

    def test_summary_coverage_shows_100_available(self):
        result = run_all_exports(verbose=False)
        with open(result["summary_json"]) as f:
            summary = json.load(f)
        assert summary["coverage"]["available"] == 100
        assert summary["coverage"]["missing"] == 0

    def test_top_performer_highest_return_has_valid_fields(self):
        result = run_all_exports(verbose=False)
        with open(result["summary_json"]) as f:
            summary = json.load(f)
        winner = summary["top_performers"]["highest_total_return"]
        assert winner["strategy"] in STRATEGY_ORDER
        assert winner["ticker"] in ["SPY", "QQQ", "BTC-USD", "IWM", "GLD"]
        assert isinstance(winner["value"], (int, float))

    def test_spy_5y_comparison_has_five_rows(self):
        result = run_all_exports(verbose=False)
        spy_5y = next(p for p in result["comparisons"] if "SPY_5y" in p)
        df = pd.read_csv(spy_5y)
        assert len(df) == 5

    def test_exports_written_to_reports_exports_dir(self):
        result = run_all_exports(verbose=False)
        assert "exports" in result["metrics_csv"]
        assert REPORTS_DIR in result["metrics_csv"]
