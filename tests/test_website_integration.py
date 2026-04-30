"""
Tests for Chunk 14 — Graph and Stats Integration.

Fast tests (no saved files): HTML structure changes added in Chunk 14.
Slow tests (read REPORTS_DIR): API correctness — complete metrics, benchmark
    return present, drawdown in response, and unavailable data handling.

Run fast only:
    pytest tests/test_website_integration.py -v -m "not slow"

Run everything:
    pytest tests/test_website_integration.py -v
"""

from __future__ import annotations

import pytest

from src.config.settings import ANCHOR_ASSETS, STRATEGY_ORDER, TIMEFRAMES_YEARS
from src.website.app import create_app


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# TestPanelLabelStructure — panel label div present for all 4 panels
# ---------------------------------------------------------------------------


class TestPanelLabelStructure:
    def test_panel_label_div_panel_1(self, client):
        html = client.get("/").data.decode()
        assert 'id="panel-label-1"' in html

    def test_panel_label_div_panel_2(self, client):
        html = client.get("/").data.decode()
        assert 'id="panel-label-2"' in html

    def test_panel_label_div_panel_3(self, client):
        html = client.get("/").data.decode()
        assert 'id="panel-label-3"' in html

    def test_panel_label_div_panel_4(self, client):
        html = client.get("/").data.decode()
        assert 'id="panel-label-4"' in html

    def test_panel_label_starts_empty(self, client):
        """Label div must start empty — JS fills it after load."""
        html = client.get("/").data.decode()
        # panel-label divs are empty on initial render
        assert 'id="panel-label-1" aria-live="polite"></div>' in html


# ---------------------------------------------------------------------------
# TestChartTypeTabs — Equity / Drawdown tab buttons present for all panels
# ---------------------------------------------------------------------------


class TestChartTypeTabs:
    def test_chart_tabs_container_panel_1(self, client):
        html = client.get("/").data.decode()
        assert 'id="chart-tabs-1"' in html

    def test_chart_tabs_container_panel_2(self, client):
        html = client.get("/").data.decode()
        assert 'id="chart-tabs-2"' in html

    def test_chart_tabs_container_panel_3(self, client):
        html = client.get("/").data.decode()
        assert 'id="chart-tabs-3"' in html

    def test_chart_tabs_container_panel_4(self, client):
        html = client.get("/").data.decode()
        assert 'id="chart-tabs-4"' in html

    def test_equity_tab_all_panels(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="tab-equity-{i}"' in html

    def test_drawdown_tab_all_panels(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="tab-drawdown-{i}"' in html

    def test_equity_tab_has_correct_data_type(self, client):
        html = client.get("/").data.decode()
        assert 'data-type="equity"' in html

    def test_drawdown_tab_has_correct_data_type(self, client):
        html = client.get("/").data.decode()
        assert 'data-type="drawdown"' in html

    def test_chart_tabs_hidden_initially(self, client):
        """Tabs start hidden — JS shows them after a successful load."""
        html = client.get("/").data.decode()
        # The style="display:none;" is on the tabs container
        assert 'id="chart-tabs-1" style="display:none;"' in html

    def test_drawdown_plotly_divs_present(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="plotly-drawdown-{i}"' in html


# ---------------------------------------------------------------------------
# TestBenchmarkStatRow — B&H return row present alongside outperformance
# ---------------------------------------------------------------------------


class TestBenchmarkStatRow:
    def test_benchmark_stat_in_panel_1(self, client):
        html = client.get("/").data.decode()
        assert 'id="stat-benchmark-1"' in html

    def test_benchmark_stat_in_all_panels(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="stat-benchmark-{i}"' in html

    def test_buy_and_hold_label_present(self, client):
        html = client.get("/").data.decode()
        assert "Buy" in html and "Hold" in html

    def test_outperformance_stat_still_present(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="stat-outperformance-{i}"' in html

    def test_benchmark_row_is_divider(self, client):
        """Benchmark row has stat-row-divider class to visually separate it."""
        html = client.get("/").data.decode()
        assert "stat-row-divider" in html

    def test_outperformance_row_is_highlighted(self, client):
        html = client.get("/").data.decode()
        assert "stat-row-highlight" in html


# ---------------------------------------------------------------------------
# TestApiIntegration  (slow — reads REPORTS_DIR)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestApiIntegration:
    """Verify the API response contains everything needed for Chunk 14 rendering."""

    def test_response_has_drawdown_key(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert "drawdown" in data

    def test_drawdown_has_data_and_layout(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert "data" in data["drawdown"]
        assert "layout" in data["drawdown"]

    def test_drawdown_has_one_trace(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert len(data["drawdown"]["data"]) == 1

    def test_metrics_has_benchmark_return(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert "benchmark_return" in data["metrics"]

    def test_metrics_has_outperformance(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert "outperformance" in data["metrics"]

    def test_benchmark_return_is_numeric_or_null(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        bh = data["metrics"]["benchmark_return"]
        assert bh is None or isinstance(bh, (int, float))

    def test_outperformance_equals_return_minus_benchmark(self, client):
        """Outperformance must equal total_return − benchmark_return."""
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        m = data["metrics"]
        if m["outperformance"] is not None and m["total_return"] is not None and m["benchmark_return"] is not None:
            expected = m["total_return"] - m["benchmark_return"]
            assert abs(m["outperformance"] - expected) < 1e-9

    def test_metadata_has_data_start_and_end(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert "data_start" in data["metadata"]
        assert "data_end"   in data["metadata"]

    def test_metadata_has_n_bars(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert "n_bars" in data["metadata"]
        assert data["metadata"]["n_bars"] > 200

    def test_equity_benchmark_trace_visible(self, client):
        """The equity chart must include a Buy & Hold trace."""
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        trace_names = [t.get("name", "") for t in data["equity"]["data"]]
        # One trace contains "Buy" (Buy & Hold benchmark)
        assert any("Buy" in name for name in trace_names)

    def test_unavailable_result_returns_404(self, client):
        resp = client.get("/api/chart?strategy=ma_crossover&asset=AAPL&timeframe=5")
        assert resp.status_code == 404

    def test_404_response_has_error_message(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=AAPL&timeframe=5").json
        assert "error" in data
        assert isinstance(data["error"], str)
        assert len(data["error"]) > 0

    def test_all_strategies_have_benchmark_return(self, client):
        for strategy in STRATEGY_ORDER:
            data = client.get(f"/api/chart?strategy={strategy}&asset=SPY&timeframe=1").json
            assert "benchmark_return" in data["metrics"], f"Missing for {strategy}"

    def test_all_anchor_assets_return_drawdown(self, client):
        for asset in ANCHOR_ASSETS:
            data = client.get(f"/api/chart?strategy=ma_crossover&asset={asset}&timeframe=1").json
            assert "drawdown" in data, f"Missing drawdown for {asset}"
            assert len(data["drawdown"]["data"]) == 1
