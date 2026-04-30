"""
Tests for src/strategies/breakout_b.py — Chunk 6 acceptance criteria.

Uses synthetic price series and hand-crafted indicator frames to verify that
all four entry filters are evaluated correctly and that the strategy produces
fewer signals than Breakout A on representative data.

Run with:
    pytest tests/test_breakout_b.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.engine import add_indicators
from src.strategies.breakout_b import generate_signals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VOLUME_MULTIPLIER = 1.5
_RSI_MIN = 55
_RSI_MAX = 75


def make_ohlcv(closes: list[float], seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    spread = rng.uniform(0.2, 0.8, n)
    volume = rng.integers(1_000_000, 3_000_000, n).astype(float)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": closes_arr - spread / 2,
            "high": closes_arr + spread,
            "low": closes_arr - spread,
            "close": closes_arr,
            "volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


def make_full_frame(n: int = 10, **overrides) -> pd.DataFrame:
    """
    Build a DataFrame where all four entry conditions are met by default.
    Any individual field can be overridden via keyword arguments to test
    that removing one condition suppresses the entry signal.

    Defaults (all conditions satisfied):
      close=120, rolling_high_20=110, volume=3_000_000, avg_volume_20=1_000_000,
      sma_200=100, rsi_14=65, rolling_low_10=90
    """
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    defaults = dict(
        open=119.0,
        high=121.0,
        low=119.0,
        close=120.0,
        volume=3_000_000.0,       # > 1.5 × 1_000_000 = 1_500_000 ✓
        sma_50=115.0,
        sma_200=100.0,            # close(120) > sma_200(100) ✓
        rsi_14=65.0,              # 55 ≤ 65 ≤ 75 ✓
        rolling_high_20=110.0,    # close(120) > rolling_high_20(110) ✓
        rolling_low_10=90.0,      # close(120) > rolling_low_10(90) — no exit ✓
        avg_volume_20=1_000_000.0,
    )
    defaults.update(overrides)

    data = {k: np.full(n, float(v)) for k, v in defaults.items()}
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="date"))


# ---------------------------------------------------------------------------
# TestOutputSchema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """Signal output shape and column contract."""

    def setup_method(self):
        df = make_full_frame()
        self.result = generate_signals(df)

    def test_returns_dataframe(self):
        assert isinstance(self.result, pd.DataFrame)

    def test_signal_entry_column_present(self):
        assert "signal_entry" in self.result.columns

    def test_signal_exit_column_present(self):
        assert "signal_exit" in self.result.columns

    def test_signal_entry_is_boolean(self):
        assert self.result["signal_entry"].dtype == bool

    def test_signal_exit_is_boolean(self):
        assert self.result["signal_exit"].dtype == bool

    def test_original_columns_preserved(self):
        df = make_full_frame()
        for col in df.columns:
            assert col in self.result.columns

    def test_row_count_unchanged(self):
        df = make_full_frame()
        assert len(self.result) == len(df)

    def test_index_unchanged(self):
        df = make_full_frame()
        pd.testing.assert_index_equal(self.result.index, df.index)

    def test_returns_copy_not_mutation(self):
        df = make_full_frame()
        cols_before = list(df.columns)
        generate_signals(df)
        assert list(df.columns) == cols_before


# ---------------------------------------------------------------------------
# TestAllFiltersRequired
# ---------------------------------------------------------------------------


class TestAllFiltersRequired:
    """
    When all conditions are met the entry fires.
    Removing any single condition must suppress it.
    """

    def test_entry_fires_when_all_conditions_met(self):
        result = generate_signals(make_full_frame())
        assert result["signal_entry"].all()

    def test_no_entry_when_price_breakout_fails(self):
        # close == rolling_high_20: not a breakout (strict >)
        result = generate_signals(make_full_frame(close=110.0, rolling_high_20=110.0))
        assert not result["signal_entry"].any()

    def test_no_entry_when_price_below_rolling_high(self):
        result = generate_signals(make_full_frame(close=105.0, rolling_high_20=110.0))
        assert not result["signal_entry"].any()

    def test_no_entry_when_volume_too_low(self):
        # volume exactly at the threshold (not strictly above)
        result = generate_signals(make_full_frame(volume=1_500_000.0, avg_volume_20=1_000_000.0))
        assert not result["signal_entry"].any()

    def test_no_entry_when_volume_below_threshold(self):
        result = generate_signals(make_full_frame(volume=1_000_000.0, avg_volume_20=1_000_000.0))
        assert not result["signal_entry"].any()

    def test_no_entry_when_close_below_sma200(self):
        result = generate_signals(make_full_frame(close=95.0, sma_200=100.0,
                                                   rolling_high_20=90.0))
        assert not result["signal_entry"].any()

    def test_no_entry_when_close_equals_sma200(self):
        # Must be strictly above SMA 200
        result = generate_signals(make_full_frame(close=100.0, sma_200=100.0,
                                                   rolling_high_20=90.0))
        assert not result["signal_entry"].any()

    def test_no_entry_when_rsi_below_55(self):
        result = generate_signals(make_full_frame(rsi_14=54.9))
        assert not result["signal_entry"].any()

    def test_no_entry_when_rsi_above_75(self):
        result = generate_signals(make_full_frame(rsi_14=75.1))
        assert not result["signal_entry"].any()


# ---------------------------------------------------------------------------
# TestRSIBounds
# ---------------------------------------------------------------------------


class TestRSIBounds:
    """RSI ∈ [55, 75] — both endpoints are inclusive."""

    def test_entry_fires_when_rsi_exactly_55(self):
        result = generate_signals(make_full_frame(rsi_14=55.0))
        assert result["signal_entry"].all()

    def test_entry_fires_when_rsi_exactly_75(self):
        result = generate_signals(make_full_frame(rsi_14=75.0))
        assert result["signal_entry"].all()

    def test_no_entry_when_rsi_just_below_55(self):
        result = generate_signals(make_full_frame(rsi_14=54.99))
        assert not result["signal_entry"].any()

    def test_no_entry_when_rsi_just_above_75(self):
        result = generate_signals(make_full_frame(rsi_14=75.01))
        assert not result["signal_entry"].any()


# ---------------------------------------------------------------------------
# TestVolumeThreshold
# ---------------------------------------------------------------------------


class TestVolumeThreshold:
    """Volume must be strictly greater than 1.5 × avg_volume_20."""

    def test_entry_fires_when_volume_just_above_threshold(self):
        # volume = 1_500_001 > 1.5 × 1_000_000 = 1_500_000
        result = generate_signals(make_full_frame(volume=1_500_001.0, avg_volume_20=1_000_000.0))
        assert result["signal_entry"].all()

    def test_no_entry_when_volume_exactly_at_threshold(self):
        # volume == 1.5 × avg: not strictly greater
        result = generate_signals(make_full_frame(volume=1_500_000.0, avg_volume_20=1_000_000.0))
        assert not result["signal_entry"].any()


# ---------------------------------------------------------------------------
# TestExitLogic
# ---------------------------------------------------------------------------


class TestExitLogic:
    """Exit mirrors Breakout A: close < rolling_low_10, strict inequality."""

    def test_exit_fires_when_close_below_rolling_low(self):
        result = generate_signals(make_full_frame(close=85.0, rolling_low_10=90.0,
                                                   rolling_high_20=110.0))
        assert result["signal_exit"].all()

    def test_no_exit_when_close_equals_rolling_low(self):
        result = generate_signals(make_full_frame(close=90.0, rolling_low_10=90.0))
        assert not result["signal_exit"].any()

    def test_no_exit_when_close_above_rolling_low(self):
        result = generate_signals(make_full_frame(close=95.0, rolling_low_10=90.0))
        assert not result["signal_exit"].any()


# ---------------------------------------------------------------------------
# TestFewerSignalsThanBreakoutA
# ---------------------------------------------------------------------------


class TestFewerSignalsThanBreakoutA:
    """
    Breakout B applies three additional filters on top of the Breakout A price
    condition, so it must always produce ≤ Breakout A entry signals on any
    given dataset.
    """

    def _run_both(self, ticker: str, years: int):
        from src.data.loader import load_ohlcv
        from src.strategies.breakout_a import generate_signals as ba_signals

        df = load_ohlcv(ticker, years)
        ind = add_indicators(df)
        ba_count = ba_signals(ind)["signal_entry"].sum()
        bb_count = generate_signals(ind)["signal_entry"].sum()
        return ba_count, bb_count

    def test_spy_5y_breakout_b_fewer_entries_than_a(self):
        ba, bb = self._run_both("SPY", 5)
        assert bb < ba, f"Expected B({bb}) < A({ba})"

    def test_qqq_5y_breakout_b_fewer_entries_than_a(self):
        ba, bb = self._run_both("QQQ", 5)
        assert bb < ba, f"Expected B({bb}) < A({ba})"


# ---------------------------------------------------------------------------
# TestNoLookAheadBias
# ---------------------------------------------------------------------------


class TestNoLookAheadBias:
    """Truncating the series must not change signals on surviving rows."""

    def test_truncation_does_not_change_past_signals(self):
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 5)
        full = generate_signals(add_indicators(df))
        short = generate_signals(add_indicators(df.iloc[:-30]))

        common = short.index
        pd.testing.assert_series_equal(
            full.loc[common, "signal_entry"], short["signal_entry"], check_names=False
        )
        pd.testing.assert_series_equal(
            full.loc[common, "signal_exit"], short["signal_exit"], check_names=False
        )


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Missing columns raise clearly."""

    @pytest.mark.parametrize("col", [
        "close", "volume", "sma_200", "rsi_14",
        "rolling_high_20", "rolling_low_10", "avg_volume_20",
    ])
    def test_missing_column_raises(self, col):
        df = make_full_frame().drop(columns=[col])
        with pytest.raises(ValueError, match=col):
            generate_signals(df)


# ---------------------------------------------------------------------------
# TestRealDataSmoke
# ---------------------------------------------------------------------------


class TestRealDataSmoke:
    """End-to-end: load SPY, add indicators, generate signals."""

    def test_spy_signals_generate_without_error(self):
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 10)
        result = generate_signals(add_indicators(df))

        assert "signal_entry" in result.columns
        assert "signal_exit" in result.columns
        assert result["signal_entry"].dtype == bool
        assert result["signal_exit"].dtype == bool

    def test_spy_entry_rows_satisfy_all_conditions(self):
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 10)
        result = generate_signals(add_indicators(df))
        valid = result.dropna(subset=["rolling_high_20", "avg_volume_20", "sma_200", "rsi_14"])
        entries = valid[valid["signal_entry"]]

        assert (entries["close"] > entries["rolling_high_20"]).all()
        assert (entries["volume"] > _VOLUME_MULTIPLIER * entries["avg_volume_20"]).all()
        assert (entries["close"] > entries["sma_200"]).all()
        assert (entries["rsi_14"] >= _RSI_MIN).all()
        assert (entries["rsi_14"] <= _RSI_MAX).all()

    def test_spy_exit_rows_close_below_rolling_low(self):
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 10)
        result = generate_signals(add_indicators(df))
        valid = result.dropna(subset=["rolling_low_10"])
        exits = valid[valid["signal_exit"]]
        assert (exits["close"] < exits["rolling_low_10"]).all()
