"""
Tests for src/research/runner.py — Chunk 10 acceptance criteria.

Fast tests:  save/load round-trip, results_summary, path structure.
Slow tests:  run a small subset (SPY 1y × 2 strategies) to verify end-to-end
             file generation.
Acceptance:  verify all 20 anchor result files exist after run_all_backtests().

Run fast tests only:
    pytest tests/test_runner.py -v -m "not slow"

Run everything:
    pytest tests/test_runner.py -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.config.settings import (
    ANCHOR_ASSETS,
    DEFAULT_STARTING_CAPITAL,
    STRATEGY_ORDER,
    TIMEFRAMES_YEARS,
    TRANSACTION_COST_PCT,
)
from src.research.runner import (
    load_equity_series,
    load_metrics,
    load_result,
    result_exists,
    results_summary,
    run_all_backtests,
    save_result,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic BacktestResult
# ---------------------------------------------------------------------------

from src.backtester.engine import TRADE_LOG_COLUMNS, BacktestResult


def _synthetic_result(n: int = 10, capital: float = 10_000.0) -> tuple[BacktestResult, dict]:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    portfolio = pd.Series(
        np.linspace(capital, capital * 1.10, n),
        index=dates,
        name="portfolio",
    )
    benchmark = pd.Series(
        np.linspace(capital, capital * 1.05, n),
        index=dates,
        name="benchmark",
    )
    trade_log = pd.DataFrame(
        [
            {
                "entry_date": dates[1],
                "entry_price": 100.0,
                "exit_date": dates[5],
                "exit_price": 110.0,
                "shares": 99.0,
                "pnl": 990.0,
                "return_pct": 0.10,
                "holding_days": 6,
            }
        ],
        columns=TRADE_LOG_COLUMNS,
    )
    result = BacktestResult(
        portfolio_history=portfolio,
        trade_log=trade_log,
        benchmark=benchmark,
        strategy_name="ma_crossover",
        ticker="SPY",
        starting_capital=capital,
        transaction_cost_pct=TRANSACTION_COST_PCT,
    )
    metrics = {
        "total_return": 0.10,
        "annualized_return": 0.40,
        "volatility": 0.15,
        "sharpe_ratio": 2.5,
        "max_drawdown": -0.05,
        "win_rate": 1.0,
        "trade_count": 1,
        "avg_trade_return": 0.10,
        "avg_holding_period": 6.0,
        "benchmark_return": 0.05,
        "outperformance": 0.05,
    }
    return result, metrics


# ---------------------------------------------------------------------------
# TestSaveLoad
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """save_result() and load_result() round-trip correctly."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.result, self.metrics = _synthetic_result()

    def test_save_returns_path(self):
        path = save_result(self.result, self.metrics, self.tmp, "ma_crossover", "SPY", 5)
        assert isinstance(path, str)
        assert os.path.isfile(path)

    def test_saved_file_is_valid_json(self):
        save_result(self.result, self.metrics, self.tmp, "ma_crossover", "SPY", 5)
        path = os.path.join(self.tmp, "results", "SPY", "ma_crossover", "5y.json")
        with open(path) as f:
            payload = json.load(f)
        assert isinstance(payload, dict)

    def test_payload_has_required_top_level_keys(self):
        save_result(self.result, self.metrics, self.tmp, "ma_crossover", "SPY", 5)
        payload = load_result("ma_crossover", "SPY", 5, self.tmp)
        assert set(payload.keys()) == {"metadata", "metrics", "equity", "trade_log"}

    def test_metadata_fields(self):
        save_result(self.result, self.metrics, self.tmp, "ma_crossover", "SPY", 5)
        meta = load_result("ma_crossover", "SPY", 5, self.tmp)["metadata"]
        assert meta["strategy"] == "ma_crossover"
        assert meta["ticker"] == "SPY"
        assert meta["timeframe_years"] == 5
        assert meta["starting_capital"] == DEFAULT_STARTING_CAPITAL
        assert meta["n_bars"] == 10

    def test_metrics_round_trip(self):
        save_result(self.result, self.metrics, self.tmp, "ma_crossover", "SPY", 5)
        loaded = load_result("ma_crossover", "SPY", 5, self.tmp)["metrics"]
        assert loaded["total_return"] == pytest.approx(0.10)
        assert loaded["max_drawdown"] == pytest.approx(-0.05)
        assert loaded["trade_count"] == 1

    def test_nan_serialised_as_null(self):
        metrics_with_nan = dict(self.metrics)
        metrics_with_nan["win_rate"] = float("nan")
        save_result(self.result, metrics_with_nan, self.tmp, "ma_crossover", "SPY", 5)
        loaded = load_result("ma_crossover", "SPY", 5, self.tmp)["metrics"]
        assert loaded["win_rate"] is None   # JSON null deserialises to None

    def test_equity_length_matches_input(self):
        save_result(self.result, self.metrics, self.tmp, "ma_crossover", "SPY", 5)
        equity = load_result("ma_crossover", "SPY", 5, self.tmp)["equity"]
        assert len(equity) == 10

    def test_equity_has_date_portfolio_benchmark(self):
        save_result(self.result, self.metrics, self.tmp, "ma_crossover", "SPY", 5)
        equity = load_result("ma_crossover", "SPY", 5, self.tmp)["equity"]
        assert set(equity[0].keys()) == {"date", "portfolio", "benchmark"}

    def test_equity_values_correct(self):
        save_result(self.result, self.metrics, self.tmp, "ma_crossover", "SPY", 5)
        equity = load_result("ma_crossover", "SPY", 5, self.tmp)["equity"]
        assert equity[0]["portfolio"] == pytest.approx(10_000.0, rel=1e-4)

    def test_trade_log_round_trip(self):
        save_result(self.result, self.metrics, self.tmp, "ma_crossover", "SPY", 5)
        trades = load_result("ma_crossover", "SPY", 5, self.tmp)["trade_log"]
        assert len(trades) == 1
        assert trades[0]["return_pct"] == pytest.approx(0.10)
        assert trades[0]["holding_days"] == 6

    def test_btc_usd_ticker_safe_path(self):
        """BTC-USD must not create a path with a hyphen — uses underscore."""
        path = save_result(self.result, self.metrics, self.tmp, "breakout_a", "BTC-USD", 1)
        assert "BTC_USD" in path
        assert os.path.isfile(path)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_result("ma_crossover", "NOTREAL", 99, self.tmp)


# ---------------------------------------------------------------------------
# TestLoadHelpers
# ---------------------------------------------------------------------------


class TestLoadHelpers:
    """load_equity_series() and load_metrics() return correct types."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        result, metrics = _synthetic_result()
        save_result(result, metrics, self.tmp, "ma_crossover", "SPY", 5)

    def test_load_equity_series_returns_dataframe(self):
        df = load_equity_series("ma_crossover", "SPY", 5, self.tmp)
        assert isinstance(df, pd.DataFrame)

    def test_load_equity_series_has_correct_columns(self):
        df = load_equity_series("ma_crossover", "SPY", 5, self.tmp)
        assert "portfolio" in df.columns
        assert "benchmark" in df.columns

    def test_load_equity_series_index_is_datetime(self):
        df = load_equity_series("ma_crossover", "SPY", 5, self.tmp)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_load_metrics_returns_dict(self):
        m = load_metrics("ma_crossover", "SPY", 5, self.tmp)
        assert isinstance(m, dict)

    def test_load_metrics_has_total_return(self):
        m = load_metrics("ma_crossover", "SPY", 5, self.tmp)
        assert "total_return" in m


# ---------------------------------------------------------------------------
# TestResultExists
# ---------------------------------------------------------------------------


class TestResultExists:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def test_returns_false_when_missing(self):
        assert result_exists("ma_crossover", "SPY", 5, self.tmp) is False

    def test_returns_true_after_save(self):
        result, metrics = _synthetic_result()
        save_result(result, metrics, self.tmp, "ma_crossover", "SPY", 5)
        assert result_exists("ma_crossover", "SPY", 5, self.tmp) is True


# ---------------------------------------------------------------------------
# TestResultsSummary
# ---------------------------------------------------------------------------


class TestResultsSummary:
    def setup_method(self):
        self.tmp = tempfile.mkdtemp()

    def test_returns_dataframe(self):
        df = results_summary(output_dir=self.tmp)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        df = results_summary(output_dir=self.tmp)
        for col in ["strategy", "ticker", "timeframe_years", "exists", "path"]:
            assert col in df.columns

    def test_correct_row_count_all_assets(self):
        from src.config.settings import ALL_ASSETS
        df = results_summary(output_dir=self.tmp)
        expected = len(STRATEGY_ORDER) * len(ALL_ASSETS) * len(TIMEFRAMES_YEARS)
        assert len(df) == expected

    def test_all_missing_before_run(self):
        df = results_summary(output_dir=self.tmp)
        assert not df["exists"].any()

    def test_saved_result_shows_as_existing(self):
        result, metrics = _synthetic_result()
        save_result(result, metrics, self.tmp, "ma_crossover", "SPY", 5)
        df = results_summary(assets=["SPY"], strategies=["ma_crossover"],
                             timeframes=[5], output_dir=self.tmp)
        assert df.iloc[0]["exists"] == True  # noqa: E712  (np.True_ is not True)


# ---------------------------------------------------------------------------
# TestRunSubset (slow — hits network)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRunSubset:
    """Run a small subset (SPY, 1y, MA Crossover + hybrid) end-to-end."""

    def test_run_small_subset_saves_files(self, tmp_path):
        counts = run_all_backtests(
            assets=["SPY"],
            timeframes=[1],
            output_dir=str(tmp_path),
            verbose=False,
        )
        # All 5 strategies × 1 asset × 1 timeframe = 5 files
        assert counts["saved"] == 5
        assert counts["errors"] == 0

    def test_saved_files_have_correct_structure(self, tmp_path):
        run_all_backtests(
            assets=["SPY"],
            timeframes=[1],
            output_dir=str(tmp_path),
            verbose=False,
        )
        for strategy in STRATEGY_ORDER:
            payload = load_result(strategy, "SPY", 1, str(tmp_path))
            assert "metadata" in payload
            assert "metrics" in payload
            assert "equity" in payload
            assert "trade_log" in payload
            assert len(payload["equity"]) > 200   # ~252 trading days in 1y

    def test_equity_values_are_positive(self, tmp_path):
        run_all_backtests(
            assets=["SPY"],
            timeframes=[1],
            output_dir=str(tmp_path),
            verbose=False,
        )
        payload = load_result("ma_crossover", "SPY", 1, str(tmp_path))
        for row in payload["equity"]:
            assert row["portfolio"] > 0
            assert row["benchmark"] > 0

    def test_metrics_outperformance_equals_difference(self, tmp_path):
        run_all_backtests(
            assets=["SPY"],
            timeframes=[1],
            output_dir=str(tmp_path),
            verbose=False,
        )
        for strategy in STRATEGY_ORDER:
            m = load_metrics(strategy, "SPY", 1, str(tmp_path))
            if m["outperformance"] is not None:
                expected = (m["total_return"] or 0) - (m["benchmark_return"] or 0)
                assert m["outperformance"] == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# TestAnchorResultsExist  (slow — acceptance test)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAnchorResultsExist:
    """
    Acceptance test: verify all 20 anchor result files (5 strategies × 4 anchor
    assets × default timeframe) exist in REPORTS_DIR after run_all_backtests().

    This test will run the full research suite if files are missing, so it may
    be slow on first run (~2-3 minutes for all downloads).
    """

    def test_20_anchor_result_sets_exist(self):
        from src.config.settings import REPORTS_DIR

        # Run if any anchor file is missing
        summary = results_summary(
            assets=ANCHOR_ASSETS,
            strategies=STRATEGY_ORDER,
            timeframes=TIMEFRAMES_YEARS,
            output_dir=REPORTS_DIR,
        )
        if not summary["exists"].all():
            run_all_backtests(verbose=True)

        # Re-check after potential run
        summary = results_summary(
            assets=ANCHOR_ASSETS,
            strategies=STRATEGY_ORDER,
            timeframes=TIMEFRAMES_YEARS,
            output_dir=REPORTS_DIR,
        )
        missing = summary[~summary["exists"]]
        assert missing.empty, (
            f"{len(missing)} anchor result files are still missing:\n"
            + missing[["strategy", "ticker", "timeframe_years"]].to_string()
        )

    def test_gld_results_exist(self):
        """GLD validation results must also be present."""
        from src.config.settings import REPORTS_DIR

        summary = results_summary(
            assets=["GLD"],
            strategies=STRATEGY_ORDER,
            timeframes=TIMEFRAMES_YEARS,
            output_dir=REPORTS_DIR,
        )
        missing = summary[~summary["exists"]]
        assert missing.empty, (
            f"{len(missing)} GLD result files are missing."
        )

    def test_anchor_metrics_are_well_formed(self):
        """Spot-check: every anchor result file has valid metrics."""
        from src.config.settings import REPORTS_DIR

        REQUIRED_METRIC_KEYS = {
            "total_return", "annualized_return", "volatility", "sharpe_ratio",
            "max_drawdown", "win_rate", "trade_count", "avg_trade_return",
            "avg_holding_period", "benchmark_return", "outperformance",
        }
        for ticker in ANCHOR_ASSETS:
            for strategy in STRATEGY_ORDER:
                for tf in TIMEFRAMES_YEARS:
                    if result_exists(strategy, ticker, tf, REPORTS_DIR):
                        m = load_metrics(strategy, ticker, tf, REPORTS_DIR)
                        assert set(m.keys()) == REQUIRED_METRIC_KEYS
                        # Max drawdown must be ≤ 0 (or null)
                        if m["max_drawdown"] is not None:
                            assert m["max_drawdown"] <= 0.0
