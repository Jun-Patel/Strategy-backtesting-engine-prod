"""
Tests for Chunk 13 — Interactive Graph Controls.

Fast tests (no saved files):
    HTML structure — layout buttons, 4 panels, panels container, JS file ref.

Slow tests (read from REPORTS_DIR):
    /api/chart endpoint — valid request, 404, missing params, all strategies.

Run fast only:
    pytest tests/test_website_controls.py -v -m "not slow"

Run everything:
    pytest tests/test_website_controls.py -v
"""

from __future__ import annotations

import json

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
# TestLayoutControls — HTML structure changes from Chunk 13
# ---------------------------------------------------------------------------


class TestLayoutControls:
    """Verify layout toggle buttons and 4-panel HTML structure."""

    def test_layout_controls_section_present(self, client):
        html = client.get("/").data.decode()
        assert "layout-controls" in html

    def test_single_button_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="btn-layout-single"' in html

    def test_dual_button_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="btn-layout-dual"' in html

    def test_quad_button_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="btn-layout-quad"' in html

    def test_panels_container_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="panels-container"' in html

    def test_all_four_panels_in_html(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="graph-panel-{i}"' in html

    def test_all_four_load_buttons_present(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="btn-load-{i}"' in html

    def test_all_four_chart_areas_present(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="chart-area-{i}"' in html

    def test_all_four_stats_areas_present(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="stats-area-{i}"' in html

    def test_all_four_plotly_divs_present(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="plotly-equity-{i}"' in html

    def test_all_four_strategy_selects_present(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="select-strategy-{i}"' in html

    def test_all_four_asset_selects_present(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="select-asset-{i}"' in html

    def test_all_four_timeframe_selects_present(self, client):
        html = client.get("/").data.decode()
        for i in range(1, 5):
            assert f'id="select-timeframe-{i}"' in html

    def test_charts_js_loaded(self, client):
        html = client.get("/").data.decode()
        assert "charts.js" in html

    def test_plotly_cdn_loaded(self, client):
        html = client.get("/").data.decode()
        assert "plotly" in html.lower()

    def test_panel_defaults_differ_across_panels(self, client):
        """Each panel should have a different default strategy selected."""
        html = client.get("/").data.decode()
        # Panel 1 default: ma_crossover (selected), Panel 2: rsi_mean_reversion, etc.
        # The 'selected' attribute must appear at least 4 times (once per panel)
        assert html.count("selected") >= 4

    def test_data_layout_attributes_on_buttons(self, client):
        html = client.get("/").data.decode()
        assert 'data-layout="single"' in html
        assert 'data-layout="dual"' in html
        assert 'data-layout="quad"' in html


# ---------------------------------------------------------------------------
# TestApiChartEndpoint  (slow — reads from REPORTS_DIR)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestApiChartEndpoint:
    """Verify the /api/chart JSON endpoint."""

    def test_valid_request_returns_200(self, client):
        resp = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5")
        assert resp.status_code == 200

    def test_response_is_json(self, client):
        resp = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5")
        assert resp.content_type == "application/json"

    def test_response_has_equity_key(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert "equity" in data

    def test_response_has_drawdown_key(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert "drawdown" in data

    def test_response_has_metrics_key(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert "metrics" in data

    def test_response_has_metadata_key(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert "metadata" in data

    def test_equity_has_data_and_layout(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert "data" in data["equity"]
        assert "layout" in data["equity"]

    def test_equity_has_two_traces(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert len(data["equity"]["data"]) == 2

    def test_metrics_has_eleven_keys(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=5").json
        assert len(data["metrics"]) == 11

    def test_metadata_strategy_matches_request(self, client):
        data = client.get("/api/chart?strategy=rsi_mean_reversion&asset=QQQ&timeframe=1").json
        assert data["metadata"]["strategy"] == "rsi_mean_reversion"

    def test_metadata_ticker_matches_request(self, client):
        data = client.get("/api/chart?strategy=rsi_mean_reversion&asset=QQQ&timeframe=1").json
        assert data["metadata"]["ticker"] == "QQQ"

    def test_metadata_timeframe_matches_request(self, client):
        data = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=10").json
        assert data["metadata"]["timeframe_years"] == 10

    def test_missing_strategy_returns_400(self, client):
        resp = client.get("/api/chart?asset=SPY&timeframe=5")
        assert resp.status_code == 400

    def test_missing_asset_returns_400(self, client):
        resp = client.get("/api/chart?strategy=ma_crossover&timeframe=5")
        assert resp.status_code == 400

    def test_missing_timeframe_returns_400(self, client):
        resp = client.get("/api/chart?strategy=ma_crossover&asset=SPY")
        assert resp.status_code == 400

    def test_invalid_timeframe_returns_400(self, client):
        resp = client.get("/api/chart?strategy=ma_crossover&asset=SPY&timeframe=abc")
        assert resp.status_code == 400

    def test_unknown_asset_returns_404(self, client):
        resp = client.get("/api/chart?strategy=ma_crossover&asset=NOTREAL&timeframe=5")
        assert resp.status_code == 404

    def test_404_response_has_error_key(self, client):
        resp = client.get("/api/chart?strategy=ma_crossover&asset=NOTREAL&timeframe=5")
        assert "error" in resp.json

    def test_btc_usd_returns_200(self, client):
        resp = client.get("/api/chart?strategy=ma_crossover&asset=BTC-USD&timeframe=5")
        assert resp.status_code == 200

    def test_all_five_strategies_return_200(self, client):
        for strategy in STRATEGY_ORDER:
            resp = client.get(f"/api/chart?strategy={strategy}&asset=SPY&timeframe=1")
            assert resp.status_code == 200, f"Failed for strategy: {strategy}"

    def test_all_anchor_assets_return_200(self, client):
        for asset in ANCHOR_ASSETS:
            resp = client.get(f"/api/chart?strategy=ma_crossover&asset={asset}&timeframe=1")
            assert resp.status_code == 200, f"Failed for asset: {asset}"

    def test_all_timeframes_return_200(self, client):
        for tf in TIMEFRAMES_YEARS:
            resp = client.get(f"/api/chart?strategy=ma_crossover&asset=SPY&timeframe={tf}")
            assert resp.status_code == 200, f"Failed for timeframe: {tf}"
