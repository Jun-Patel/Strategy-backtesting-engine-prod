"""
Tests for Chunk 12 — Website Skeleton and Content Sections.

All tests use Flask's built-in test client; no network or saved files needed.

Run:
    pytest tests/test_website.py -v
"""

from __future__ import annotations

import pytest

from src.config.settings import (
    ANCHOR_ASSETS,
    STRATEGIES,
    STRATEGY_ORDER,
    TIMEFRAMES_YEARS,
)
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
# TestHomepageResponse
# ---------------------------------------------------------------------------


class TestHomepageResponse:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_content_type_is_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.content_type

    def test_response_has_content(self, client):
        resp = client.get("/")
        assert len(resp.data) > 500


# ---------------------------------------------------------------------------
# TestPageStructure
# ---------------------------------------------------------------------------


class TestPageStructure:
    """Verify major structural sections are present in the rendered HTML."""

    def test_nav_is_present(self, client):
        html = client.get("/").data.decode()
        assert "<nav" in html

    def test_footer_is_present(self, client):
        html = client.get("/").data.decode()
        assert "<footer" in html

    def test_hero_section_present(self, client):
        html = client.get("/").data.decode()
        assert "hero" in html

    def test_summary_section_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="summary"' in html

    def test_strategies_section_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="strategies"' in html

    def test_charts_section_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="charts"' in html

    def test_project_title_in_page(self, client):
        html = client.get("/").data.decode()
        assert "Strategy Backtesting Engine" in html

    def test_nav_links_to_summary(self, client):
        html = client.get("/").data.decode()
        assert 'href="#summary"' in html

    def test_nav_links_to_strategies(self, client):
        html = client.get("/").data.decode()
        assert 'href="#strategies"' in html

    def test_nav_links_to_charts(self, client):
        html = client.get("/").data.decode()
        assert 'href="#charts"' in html


# ---------------------------------------------------------------------------
# TestProjectSummary
# ---------------------------------------------------------------------------


class TestProjectSummary:
    """Verify the Project Summary section contains expected content."""

    def test_summary_heading_present(self, client):
        html = client.get("/").data.decode()
        assert "Project Summary" in html

    def test_mentions_yfinance(self, client):
        html = client.get("/").data.decode()
        assert "yfinance" in html

    def test_mentions_transaction_cost(self, client):
        html = client.get("/").data.decode()
        assert "0.1%" in html

    def test_mentions_starting_capital(self, client):
        html = client.get("/").data.decode()
        assert "10,000" in html

    def test_mentions_anchor_assets(self, client):
        html = client.get("/").data.decode()
        for asset in ANCHOR_ASSETS:
            # BTC-USD is rendered as BTC-USD but HTML may escape the hyphen
            assert asset.replace("-", "") in html.replace("-", "")

    def test_long_only_mentioned(self, client):
        html = client.get("/").data.decode()
        assert "Long-only" in html or "long-only" in html or "Long-Only" in html


# ---------------------------------------------------------------------------
# TestStrategySection
# ---------------------------------------------------------------------------


class TestStrategySection:
    """Verify strategy cards match the locked strategy definitions exactly."""

    def test_all_five_strategy_names_present(self, client):
        html = client.get("/").data.decode()
        for key in STRATEGY_ORDER:
            assert STRATEGIES[key]["name"] in html

    def test_all_five_strategy_descriptions_present(self, client):
        html = client.get("/").data.decode()
        for key in STRATEGY_ORDER:
            # Check first ~60 chars of description to avoid encoding edge cases
            snippet = STRATEGIES[key]["description"][:60]
            assert snippet in html

    def test_all_five_strategy_types_present(self, client):
        html = client.get("/").data.decode()
        for key in STRATEGY_ORDER:
            assert STRATEGIES[key]["type"] in html

    def test_strategy_keys_as_data_attributes(self, client):
        html = client.get("/").data.decode()
        for key in STRATEGY_ORDER:
            assert f'data-strategy-key="{key}"' in html

    def test_ma_crossover_mentions_sma(self, client):
        html = client.get("/").data.decode()
        assert "SMA" in html or "sma" in html.lower()

    def test_rsi_strategy_mentions_threshold_30(self, client):
        html = client.get("/").data.decode()
        assert "30" in html

    def test_rsi_strategy_mentions_threshold_55(self, client):
        html = client.get("/").data.decode()
        assert "55" in html

    def test_hybrid_mentions_allocation_percentages(self, client):
        html = client.get("/").data.decode()
        assert "50%" in html
        assert "30%" in html
        assert "20%" in html

    def test_strategy_order_in_html(self, client):
        html = client.get("/").data.decode()
        positions = [html.find(STRATEGIES[k]["name"]) for k in STRATEGY_ORDER]
        assert positions == sorted(positions), "Strategies must appear in STRATEGY_ORDER order"


# ---------------------------------------------------------------------------
# TestGraphSectionShell
# ---------------------------------------------------------------------------


class TestGraphSectionShell:
    """Verify the graph section shell is ready for interactive controls."""

    def test_graph_panel_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="graph-panel-1"' in html

    def test_strategy_select_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="select-strategy-1"' in html

    def test_asset_select_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="select-asset-1"' in html

    def test_timeframe_select_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="select-timeframe-1"' in html

    def test_load_button_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="btn-load-1"' in html

    def test_chart_area_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="chart-area-1"' in html

    def test_stats_area_present(self, client):
        html = client.get("/").data.decode()
        assert 'id="stats-area-1"' in html

    def test_all_anchor_assets_in_dropdown(self, client):
        html = client.get("/").data.decode()
        for asset in ANCHOR_ASSETS:
            assert f'value="{asset}"' in html

    def test_all_timeframes_in_dropdown(self, client):
        html = client.get("/").data.decode()
        for tf in TIMEFRAMES_YEARS:
            assert f'value="{tf}"' in html

    def test_all_strategies_in_dropdown(self, client):
        html = client.get("/").data.decode()
        for key in STRATEGY_ORDER:
            assert f'value="{key}"' in html

    def test_stats_placeholder_rows_present(self, client):
        html = client.get("/").data.decode()
        # Stat value spans use placeholder IDs
        for stat_id in [
            "stat-total-return-1", "stat-sharpe-1",
            "stat-drawdown-1", "stat-trades-1",
        ]:
            assert f'id="{stat_id}"' in html

    def test_data_panel_attributes_set(self, client):
        html = client.get("/").data.decode()
        assert 'data-panel="1"' in html
