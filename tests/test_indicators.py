"""
Tests for src/indicators/engine.py — Chunk 2 acceptance criteria.

Uses a synthetic deterministic price series so tests are fast, offline,
and reproducible. A small real-data smoke test is included at the end.

Run with:
    pytest tests/test_indicators.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.engine import (
    add_indicators,
    compute_avg_volume,
    compute_rsi,
    compute_rolling_high,
    compute_rolling_low,
    compute_sma,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 300  # enough rows to warm up all indicators (200-day SMA needs ~200)


def make_ohlcv(n: int = N, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame matching the loader contract."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    spread = rng.uniform(0.5, 2.0, n)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    dates = pd.date_range("2020-01-01", periods=n, freq="B")  # business days
    return pd.DataFrame(
        {
            "open": close - spread / 2,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture
def with_indicators(ohlcv) -> pd.DataFrame:
    return add_indicators(ohlcv)


# ---------------------------------------------------------------------------
# Acceptance criterion 1: reusable module — all columns present
# ---------------------------------------------------------------------------


class TestOutputSchema:
    EXPECTED_INDICATOR_COLS = {
        "sma_50", "sma_200", "rsi_14",
        "rolling_high_20", "rolling_low_10", "avg_volume_20",
    }
    OHLCV_COLS = {"open", "high", "low", "close", "volume"}

    def test_all_indicator_columns_present(self, with_indicators):
        assert self.EXPECTED_INDICATOR_COLS.issubset(set(with_indicators.columns))

    def test_original_columns_preserved(self, with_indicators):
        assert self.OHLCV_COLS.issubset(set(with_indicators.columns))

    def test_returns_new_dataframe(self, ohlcv):
        result = add_indicators(ohlcv)
        assert result is not ohlcv, "add_indicators must return a copy, not mutate input"

    def test_row_count_unchanged(self, ohlcv, with_indicators):
        assert len(with_indicators) == len(ohlcv)

    def test_index_unchanged(self, ohlcv, with_indicators):
        pd.testing.assert_index_equal(with_indicators.index, ohlcv.index)

    def test_missing_columns_raises(self):
        bad = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing columns"):
            add_indicators(bad)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: outputs align correctly with source data
# ---------------------------------------------------------------------------


class TestSMA:
    def test_sma_50_value(self, ohlcv):
        result = compute_sma(ohlcv["close"], 50)
        # Day 49 (0-indexed) is the first valid SMA-50 value
        idx = 49
        expected = ohlcv["close"].iloc[:50].mean()
        assert abs(result.iloc[idx] - expected) < 1e-10

    def test_sma_200_value(self, ohlcv):
        result = compute_sma(ohlcv["close"], 200)
        idx = 199
        expected = ohlcv["close"].iloc[:200].mean()
        assert abs(result.iloc[idx] - expected) < 1e-10

    def test_sma_warmup_is_nan(self, ohlcv):
        result = compute_sma(ohlcv["close"], 50)
        assert result.iloc[:49].isna().all(), "First 49 rows should be NaN"

    def test_sma_valid_after_warmup(self, ohlcv):
        result = compute_sma(ohlcv["close"], 50)
        assert result.iloc[49:].notna().all()

    def test_sma_rolls_forward(self, ohlcv):
        result = compute_sma(ohlcv["close"], 50)
        # SMA on day 50 = mean of rows 1..50 (0-indexed)
        expected = ohlcv["close"].iloc[1:51].mean()
        assert abs(result.iloc[50] - expected) < 1e-10


class TestRSI:
    def test_rsi_range(self, ohlcv):
        result = compute_rsi(ohlcv["close"], period=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all(), "RSI must be in [0, 100]"

    def test_rsi_warmup_is_nan(self, ohlcv):
        result = compute_rsi(ohlcv["close"], period=14)
        # First 14 rows should be NaN (min_periods=period)
        assert result.iloc[:14].isna().all()

    def test_rsi_valid_after_warmup(self, ohlcv):
        result = compute_rsi(ohlcv["close"], period=14)
        assert result.iloc[14:].notna().all()

    def test_rsi_rising_series_high(self):
        """A strictly rising series should produce RSI close to 100."""
        prices = pd.Series([float(i) for i in range(1, 101)])
        rsi = compute_rsi(prices, period=14)
        assert rsi.iloc[-1] > 90, f"Expected RSI near 100 for rising series, got {rsi.iloc[-1]:.1f}"

    def test_rsi_falling_series_low(self):
        """A strictly falling series should produce RSI close to 0."""
        prices = pd.Series([float(100 - i) for i in range(100)])
        rsi = compute_rsi(prices, period=14)
        assert rsi.iloc[-1] < 10, f"Expected RSI near 0 for falling series, got {rsi.iloc[-1]:.1f}"


class TestRollingHighLow:
    def test_rolling_high_value(self, ohlcv):
        result = compute_rolling_high(ohlcv["high"], window=20)
        # Day 20 (0-indexed): shifted value = max of rows 0..19
        idx = 20
        expected = ohlcv["high"].iloc[0:20].max()
        assert abs(result.iloc[idx] - expected) < 1e-10

    def test_rolling_low_value(self, ohlcv):
        result = compute_rolling_low(ohlcv["low"], window=10)
        # Day 10 (0-indexed): shifted value = min of rows 0..9
        idx = 10
        expected = ohlcv["low"].iloc[0:10].min()
        assert abs(result.iloc[idx] - expected) < 1e-10

    def test_rolling_high_warmup(self, ohlcv):
        result = compute_rolling_high(ohlcv["high"], window=20)
        # rolling(20) first produces a value at index 19; shift(1) pushes it to
        # index 20, so indices 0..19 (slice [:20]) must all be NaN.
        assert result.iloc[:20].isna().all()

    def test_rolling_low_warmup(self, ohlcv):
        result = compute_rolling_low(ohlcv["low"], window=10)
        # rolling(10) first valid at index 9; shift(1) → first valid at index 10,
        # so indices 0..9 (slice [:10]) must all be NaN.
        assert result.iloc[:10].isna().all()

    def test_rolling_high_valid_after_warmup(self, ohlcv):
        result = compute_rolling_high(ohlcv["high"], window=20)
        assert result.iloc[21:].notna().all()

    def test_rolling_low_valid_after_warmup(self, ohlcv):
        result = compute_rolling_low(ohlcv["low"], window=10)
        assert result.iloc[11:].notna().all()


class TestAvgVolume:
    def test_avg_volume_value(self, ohlcv):
        result = compute_avg_volume(ohlcv["volume"], window=20)
        idx = 20
        expected = ohlcv["volume"].iloc[0:20].mean()
        assert abs(result.iloc[idx] - expected) < 1e-6

    def test_avg_volume_warmup(self, ohlcv):
        result = compute_avg_volume(ohlcv["volume"], window=20)
        # Same shift logic as rolling_high: first valid at index 20, so [:20] are NaN.
        assert result.iloc[:20].isna().all()

    def test_avg_volume_positive(self, ohlcv):
        result = compute_avg_volume(ohlcv["volume"], window=20)
        assert (result.dropna() > 0).all()


# ---------------------------------------------------------------------------
# Acceptance criterion 3: no look-ahead bias
# ---------------------------------------------------------------------------


class TestNoLookAheadBias:
    """
    Verify that lagged indicators on day N do not depend on data from day N+.

    Strategy: compute indicators on a full series, then recompute on a
    truncated series (dropping the last row). The value at the last valid
    shared index must be identical — i.e., knowing future rows changes nothing.
    """

    def _check_no_lookahead(self, series: pd.Series, fn, **kwargs) -> None:
        full = fn(series, **kwargs)
        truncated = fn(series.iloc[:-1], **kwargs)
        # Last shared index
        shared_idx = truncated.index[-1]
        full_val = full.loc[shared_idx]
        trunc_val = truncated.loc[shared_idx]
        if pd.isna(full_val) and pd.isna(trunc_val):
            return  # both NaN — consistent
        assert abs(full_val - trunc_val) < 1e-10, (
            f"Look-ahead bias detected: full={full_val}, truncated={trunc_val}"
        )

    def test_sma_50_no_lookahead(self, ohlcv):
        self._check_no_lookahead(ohlcv["close"], compute_sma, window=50)

    def test_sma_200_no_lookahead(self, ohlcv):
        self._check_no_lookahead(ohlcv["close"], compute_sma, window=200)

    def test_rsi_no_lookahead(self, ohlcv):
        self._check_no_lookahead(ohlcv["close"], compute_rsi, period=14)

    def test_rolling_high_no_lookahead(self, ohlcv):
        self._check_no_lookahead(ohlcv["high"], compute_rolling_high, window=20)

    def test_rolling_low_no_lookahead(self, ohlcv):
        self._check_no_lookahead(ohlcv["low"], compute_rolling_low, window=10)

    def test_avg_volume_no_lookahead(self, ohlcv):
        self._check_no_lookahead(ohlcv["volume"], compute_avg_volume, window=20)

    def test_rolling_high_strictly_before_current(self, ohlcv):
        """rolling_high_20[N] must equal max(high[N-20:N]), NOT max(high[N-20:N+1])."""
        result = compute_rolling_high(ohlcv["high"], window=20)
        for i in range(21, min(50, len(ohlcv))):
            expected = ohlcv["high"].iloc[i - 20 : i].max()
            assert abs(result.iloc[i] - expected) < 1e-10, (
                f"rolling_high_20 at row {i}: expected {expected}, got {result.iloc[i]}"
            )

    def test_rolling_low_strictly_before_current(self, ohlcv):
        """rolling_low_10[N] must equal min(low[N-10:N]), NOT min(low[N-10:N+1])."""
        result = compute_rolling_low(ohlcv["low"], window=10)
        for i in range(11, min(30, len(ohlcv))):
            expected = ohlcv["low"].iloc[i - 10 : i].min()
            assert abs(result.iloc[i] - expected) < 1e-10, (
                f"rolling_low_10 at row {i}: expected {expected}, got {result.iloc[i]}"
            )


# ---------------------------------------------------------------------------
# Smoke test on real data (requires network)
# ---------------------------------------------------------------------------


class TestRealDataSmoke:
    def test_spy_indicators_load_and_align(self):
        """End-to-end: load SPY, add indicators, verify alignment and ranges."""
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 5)
        out = add_indicators(df)

        assert len(out) == len(df)
        # After 200-day warmup, SMA-200 should be present
        valid_sma200 = out["sma_200"].dropna()
        assert len(valid_sma200) > 0
        assert (valid_sma200 > 0).all()

        # RSI in valid range
        valid_rsi = out["rsi_14"].dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()

        # rolling_high_20 should always be >= rolling_low_10 where both are valid
        both_valid = out[["rolling_high_20", "rolling_low_10"]].dropna()
        assert (both_valid["rolling_high_20"] >= both_valid["rolling_low_10"]).all()
