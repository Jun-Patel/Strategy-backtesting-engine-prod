"""
Tests for src/data/loader.py — Chunk 1 acceptance criteria.

These are integration tests: they call yfinance and require a network connection.
They are intentionally kept lightweight (1-year window, one ticker per case)
to keep runtime reasonable.

Run with:
    pytest tests/test_data_loader.py -v
"""

import warnings

import pandas as pd
import pytest

from src.config.settings import ANCHOR_ASSETS, ALL_ASSETS, EXTRA_ASSETS, TIMEFRAMES_YEARS
from src.data.loader import load_all_assets, load_ohlcv

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def assert_clean_schema(df: pd.DataFrame, ticker: str = "") -> None:
    """Assert that a DataFrame matches the standard OHLCV contract."""
    label = f"[{ticker}] " if ticker else ""

    assert isinstance(df, pd.DataFrame), f"{label}result is not a DataFrame"
    assert isinstance(df.index, pd.DatetimeIndex), f"{label}index is not DatetimeIndex"
    assert df.index.name == "date", f"{label}index.name should be 'date'"
    assert not df.empty, f"{label}DataFrame is empty"
    assert REQUIRED_COLUMNS.issubset(set(df.columns)), (
        f"{label}missing columns: {REQUIRED_COLUMNS - set(df.columns)}"
    )
    assert df["close"].isna().sum() == 0, f"{label}close column has NaN values"
    assert df.index.is_monotonic_increasing, f"{label}index is not sorted ascending"
    assert df.index.tz is None, f"{label}index should be tz-naive"


# ---------------------------------------------------------------------------
# Chunk 1 acceptance criterion 1: clean OHLCV DataFrame returned
# ---------------------------------------------------------------------------


class TestOutputSchema:
    def test_spy_1y_schema(self):
        df = load_ohlcv("SPY", 1)
        assert_clean_schema(df, "SPY")

    def test_columns_are_lowercase(self):
        df = load_ohlcv("SPY", 1)
        for col in df.columns:
            assert col == col.lower(), f"Column '{col}' is not lowercase"

    def test_no_close_nans(self):
        df = load_ohlcv("SPY", 1)
        assert df["close"].notna().all()

    def test_volume_no_nans(self):
        df = load_ohlcv("SPY", 1)
        assert df["volume"].notna().all()

    def test_rows_sorted_ascending(self):
        df = load_ohlcv("SPY", 1)
        assert df.index.is_monotonic_increasing

    def test_numeric_dtypes(self):
        df = load_ohlcv("SPY", 1)
        for col in REQUIRED_COLUMNS:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"Column '{col}' is not numeric: {df[col].dtype}"
            )


# ---------------------------------------------------------------------------
# Chunk 1 acceptance criterion 2: all required assets can be loaded
# ---------------------------------------------------------------------------


class TestAssetCoverage:
    @pytest.mark.parametrize("ticker", ANCHOR_ASSETS)
    def test_anchor_asset_loads(self, ticker):
        df = load_ohlcv(ticker, 1)
        assert_clean_schema(df, ticker)

    @pytest.mark.parametrize("ticker", EXTRA_ASSETS)
    def test_extra_asset_loads(self, ticker):
        df = load_ohlcv(ticker, 1)
        assert_clean_schema(df, ticker)

    def test_load_all_assets_returns_all(self):
        result = load_all_assets(1)
        for ticker in ALL_ASSETS:
            assert ticker in result, f"'{ticker}' missing from load_all_assets result"
            assert_clean_schema(result[ticker], ticker)

    def test_btc_usd_schema(self):
        """BTC-USD has no traditional volume gaps — verify it loads and is clean."""
        df = load_ohlcv("BTC-USD", 1)
        assert_clean_schema(df, "BTC-USD")
        assert len(df) > 300, "BTC-USD 1y should have 300+ daily rows"


# ---------------------------------------------------------------------------
# Chunk 1 acceptance criterion 3: timeframe selection works
# ---------------------------------------------------------------------------


class TestTimeframeSelection:
    @pytest.mark.parametrize("years", TIMEFRAMES_YEARS)
    def test_spy_all_timeframes(self, years):
        df = load_ohlcv("SPY", years)
        assert_clean_schema(df, f"SPY-{years}y")
        # Row count should grow with more years (allowing for data availability)
        min_rows = max(200, years * 200)  # ~200 trading days/year, conservative
        assert len(df) >= min_rows, (
            f"SPY {years}y: expected >={min_rows} rows, got {len(df)}"
        )

    def test_unsupported_timeframe_raises(self):
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            load_ohlcv("SPY", 3)

    def test_longer_window_has_more_rows(self):
        df_1y = load_ohlcv("SPY", 1)
        df_5y = load_ohlcv("SPY", 5)
        assert len(df_5y) > len(df_1y), "5y window should return more rows than 1y"


# ---------------------------------------------------------------------------
# Chunk 1 acceptance criterion 4: invalid / missing data handled cleanly
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_invalid_ticker_raises(self):
        with pytest.raises(ValueError, match="No data returned"):
            load_ohlcv("INVALID_TICKER_XYZ_999", 1)

    def test_load_all_assets_skips_bad_ticker_with_warning(self, monkeypatch):
        """load_all_assets should warn and skip, not raise, on a bad ticker."""
        original_assets = __import__(
            "src.config.settings", fromlist=["ALL_ASSETS"]
        ).ALL_ASSETS

        # Temporarily inject a bad ticker
        import src.config.settings as cfg
        monkeypatch.setattr(cfg, "ALL_ASSETS", ["SPY", "INVALID_XYZ_999"])

        import src.data.loader as loader_mod
        monkeypatch.setattr(loader_mod, "ALL_ASSETS", ["SPY", "INVALID_XYZ_999"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = load_all_assets(1)

        assert "SPY" in result, "Valid ticker should still be present"
        assert "INVALID_XYZ_999" not in result, "Bad ticker should be absent"
        warning_messages = [str(w.message) for w in caught]
        assert any("INVALID_XYZ_999" in m for m in warning_messages), (
            "Expected a warning about the bad ticker"
        )

    def test_short_history_emits_warning(self):
        """BTC-USD requesting 20y of data should warn (BTC is ~10y old)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = load_ohlcv("BTC-USD", 20)
        assert_clean_schema(df, "BTC-USD-20y")
        warning_messages = [str(w.message) for w in caught]
        assert any("BTC-USD" in m for m in warning_messages), (
            "Expected a short-history warning for BTC-USD with 20y request"
        )
