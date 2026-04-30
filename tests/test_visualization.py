"""
Tests for src/visualization/charts.py — Chunk 11 acceptance criteria.

Fast tests (no network, no saved results):
    equity_curve_chart, drawdown_chart, price_with_signals_chart,
    metrics_summary_figure, figure_to_dict, figure_to_json.

Slow tests (load from REPORTS_DIR):
    build_chart_bundle round-trip.

Run fast only:
    pytest tests/test_visualization.py -v -m "not slow"

Run everything:
    pytest tests/test_visualization.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization.charts import (
    build_chart_bundle,
    drawdown_chart,
    equity_curve_chart,
    figure_to_dict,
    figure_to_json,
    metrics_summary_figure,
    price_with_signals_chart,
)


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

def _equity_df(n: int = 50, capital: float = 10_000.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "portfolio": np.linspace(capital, capital * 1.15, n),
            "benchmark": np.linspace(capital, capital * 1.08, n),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def _price_signals_df(n: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.linspace(100.0, 120.0, n)
    price_df = pd.DataFrame(
        {"open": close - 1, "high": close + 1, "low": close - 1, "close": close,
         "volume": np.full(n, 1_000_000.0)},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    entry = np.zeros(n, dtype=bool)
    exit_ = np.zeros(n, dtype=bool)
    entry[5] = True
    exit_[20] = True
    signals_df = pd.DataFrame(
        {"signal_entry": entry, "signal_exit": exit_},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return price_df, signals_df


def _sample_metrics() -> dict:
    return {
        "total_return": 0.15,
        "annualized_return": 0.40,
        "volatility": 0.18,
        "sharpe_ratio": 2.1,
        "max_drawdown": -0.08,
        "win_rate": 0.70,
        "trade_count": 5,
        "avg_trade_return": 0.03,
        "avg_holding_period": 22.5,
        "benchmark_return": 0.08,
        "outperformance": 0.07,
    }


# ---------------------------------------------------------------------------
# TestEquityCurveChart
# ---------------------------------------------------------------------------


class TestEquityCurveChart:
    def setup_method(self):
        self.df = _equity_df()
        self.fig = equity_curve_chart(self.df, "MA Crossover", "SPY", 5)

    def test_returns_figure(self):
        assert isinstance(self.fig, go.Figure)

    def test_has_two_traces(self):
        assert len(self.fig.data) == 2

    def test_first_trace_is_portfolio(self):
        assert "MA Crossover" in self.fig.data[0].name

    def test_second_trace_is_benchmark(self):
        assert "Buy" in self.fig.data[1].name

    def test_title_contains_ticker(self):
        assert "SPY" in self.fig.layout.title.text

    def test_title_contains_timeframe(self):
        assert "5y" in self.fig.layout.title.text

    def test_portfolio_trace_length(self):
        assert len(self.fig.data[0].y) == 50

    def test_benchmark_trace_length(self):
        assert len(self.fig.data[1].y) == 50

    def test_portfolio_first_value(self):
        assert abs(self.fig.data[0].y[0] - 10_000.0) < 1.0

    def test_benchmark_last_value(self):
        assert abs(self.fig.data[1].y[-1] - 10_800.0) < 1.0


# ---------------------------------------------------------------------------
# TestDrawdownChart
# ---------------------------------------------------------------------------


class TestDrawdownChart:
    def setup_method(self):
        self.df = _equity_df()
        self.fig = drawdown_chart(self.df, "ma_crossover", "SPY", 5)

    def test_returns_figure(self):
        assert isinstance(self.fig, go.Figure)

    def test_has_one_trace(self):
        assert len(self.fig.data) == 1

    def test_trace_is_scatter(self):
        assert isinstance(self.fig.data[0], go.Scatter)

    def test_title_contains_ticker(self):
        assert "SPY" in self.fig.layout.title.text

    def test_all_values_nonpositive(self):
        """Drawdown must be ≤ 0 for a monotonically increasing equity curve."""
        assert all(v <= 0.01 for v in self.fig.data[0].y)

    def test_first_bar_is_zero(self):
        """At the start there is no drawdown from peak."""
        assert abs(self.fig.data[0].y[0]) < 1e-6

    def test_trace_length(self):
        assert len(self.fig.data[0].y) == 50

    def test_non_monotonic_equity_has_negative_drawdown(self):
        """A peak-then-drop equity should show a negative drawdown."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        vals = [10000, 10500, 11000, 10000, 9500, 10200, 10100, 10300, 10400, 10600]
        df = pd.DataFrame({"portfolio": vals, "benchmark": vals}, index=dates)
        fig = drawdown_chart(df, "test", "SPY", 1)
        assert min(fig.data[0].y) < 0


# ---------------------------------------------------------------------------
# TestPriceWithSignalsChart
# ---------------------------------------------------------------------------


class TestPriceWithSignalsChart:
    def setup_method(self):
        self.price_df, self.signals_df = _price_signals_df()
        self.fig = price_with_signals_chart(self.price_df, self.signals_df, "SPY")

    def test_returns_figure(self):
        assert isinstance(self.fig, go.Figure)

    def test_has_three_traces(self):
        """Price line + entry markers + exit markers = 3 traces."""
        assert len(self.fig.data) == 3

    def test_first_trace_is_close(self):
        assert "Close" in self.fig.data[0].name

    def test_entry_trace_is_markers(self):
        entry_trace = self.fig.data[1]
        assert entry_trace.mode == "markers"
        assert "Entry" in entry_trace.name

    def test_exit_trace_is_markers(self):
        exit_trace = self.fig.data[2]
        assert exit_trace.mode == "markers"
        assert "Exit" in exit_trace.name

    def test_entry_has_one_point(self):
        assert len(self.fig.data[1].x) == 1

    def test_exit_has_one_point(self):
        assert len(self.fig.data[2].x) == 1

    def test_title_contains_ticker(self):
        assert "SPY" in self.fig.layout.title.text

    def test_no_signals_produces_one_trace(self):
        """When there are no signals, only the price line is added."""
        _, signals = _price_signals_df()
        signals["signal_entry"] = False
        signals["signal_exit"] = False
        fig = price_with_signals_chart(self.price_df, signals, "SPY")
        assert len(fig.data) == 1


# ---------------------------------------------------------------------------
# TestMetricsSummaryFigure
# ---------------------------------------------------------------------------


class TestMetricsSummaryFigure:
    def setup_method(self):
        self.metrics = _sample_metrics()
        self.fig = metrics_summary_figure(self.metrics, "ma_crossover", "SPY", 5)

    def test_returns_figure(self):
        assert isinstance(self.fig, go.Figure)

    def test_has_one_table_trace(self):
        assert len(self.fig.data) == 1
        assert isinstance(self.fig.data[0], go.Table)

    def test_table_has_eleven_metric_rows(self):
        # cells.values[0] is the label column
        assert len(self.fig.data[0].cells.values[0]) == 11

    def test_total_return_formatted_as_percent(self):
        labels = list(self.fig.data[0].cells.values[0])
        vals   = list(self.fig.data[0].cells.values[1])
        idx = labels.index("Total Return")
        assert "%" in vals[idx]

    def test_trade_count_is_integer_string(self):
        labels = list(self.fig.data[0].cells.values[0])
        vals   = list(self.fig.data[0].cells.values[1])
        idx = labels.index("Trade Count")
        assert vals[idx] == "5"

    def test_none_metric_displays_dash(self):
        metrics = dict(self.metrics)
        metrics["win_rate"] = None
        fig = metrics_summary_figure(metrics, "ma_crossover", "SPY", 5)
        labels = list(fig.data[0].cells.values[0])
        vals   = list(fig.data[0].cells.values[1])
        idx = labels.index("Win Rate")
        assert vals[idx] == "—"

    def test_header_contains_ticker(self):
        header_text = self.fig.data[0].header.values[0]
        assert "SPY" in header_text


# ---------------------------------------------------------------------------
# TestSerialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def setup_method(self):
        self.fig = equity_curve_chart(_equity_df(), "MA Crossover", "SPY", 5)

    def test_figure_to_json_returns_string(self):
        result = figure_to_json(self.fig)
        assert isinstance(result, str)

    def test_figure_to_json_is_valid_json(self):
        result = figure_to_json(self.fig)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_figure_to_dict_returns_dict(self):
        result = figure_to_dict(self.fig)
        assert isinstance(result, dict)

    def test_figure_to_dict_has_data_key(self):
        result = figure_to_dict(self.fig)
        assert "data" in result

    def test_figure_to_dict_has_layout_key(self):
        result = figure_to_dict(self.fig)
        assert "layout" in result

    def test_figure_to_dict_data_has_two_traces(self):
        result = figure_to_dict(self.fig)
        assert len(result["data"]) == 2

    def test_dict_is_json_serialisable(self):
        """figure_to_dict output must survive a round-trip through json.dumps."""
        result = figure_to_dict(self.fig)
        dumped = json.dumps(result)
        assert isinstance(dumped, str)

    def test_drawdown_chart_to_dict(self):
        fig = drawdown_chart(_equity_df(), "ma_crossover", "SPY", 5)
        d = figure_to_dict(fig)
        assert "data" in d and "layout" in d


# ---------------------------------------------------------------------------
# TestBuildChartBundle  (slow — reads from REPORTS_DIR)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBuildChartBundle:
    """Verifies build_chart_bundle() against real saved result files."""

    def test_returns_dict(self):
        bundle = build_chart_bundle("ma_crossover", "SPY", 5)
        assert isinstance(bundle, dict)

    def test_has_required_keys(self):
        bundle = build_chart_bundle("ma_crossover", "SPY", 5)
        assert set(bundle.keys()) == {"equity", "drawdown", "metrics_table",
                                       "metrics", "metadata"}

    def test_equity_chart_has_data(self):
        bundle = build_chart_bundle("ma_crossover", "SPY", 5)
        assert "data" in bundle["equity"]
        assert len(bundle["equity"]["data"]) == 2

    def test_drawdown_chart_has_data(self):
        bundle = build_chart_bundle("ma_crossover", "SPY", 5)
        assert "data" in bundle["drawdown"]
        assert len(bundle["drawdown"]["data"]) == 1

    def test_metrics_table_has_data(self):
        bundle = build_chart_bundle("ma_crossover", "SPY", 5)
        assert "data" in bundle["metrics_table"]

    def test_metrics_has_eleven_keys(self):
        bundle = build_chart_bundle("ma_crossover", "SPY", 5)
        assert len(bundle["metrics"]) == 11

    def test_metadata_has_strategy_field(self):
        bundle = build_chart_bundle("ma_crossover", "SPY", 5)
        assert bundle["metadata"]["strategy"] == "ma_crossover"

    def test_bundle_is_json_serialisable(self):
        """Full bundle (minus metrics/metadata which contain None) must serialise."""
        bundle = build_chart_bundle("ma_crossover", "SPY", 5)
        dumped = json.dumps({
            "equity": bundle["equity"],
            "drawdown": bundle["drawdown"],
            "metrics_table": bundle["metrics_table"],
        })
        assert isinstance(dumped, str)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            build_chart_bundle("ma_crossover", "NOTREAL", 99)

    def test_btc_usd_bundle(self):
        """BTC-USD path (stored as BTC_USD) must work end-to-end."""
        bundle = build_chart_bundle("ma_crossover", "BTC-USD", 5)
        assert bundle["metadata"]["ticker"] == "BTC-USD"

    def test_all_five_strategies_bundleable(self):
        from src.config.settings import STRATEGY_ORDER
        for strategy in STRATEGY_ORDER:
            bundle = build_chart_bundle(strategy, "SPY", 1)
            assert "equity" in bundle
