"""
Tests for src/strategies/rsi_mean_reversion.py — Chunk 4 acceptance criteria.

Uses synthetic price series engineered to produce known RSI crossings so
every assertion is deterministic and offline.  A real-data smoke test on SPY
verifies end-to-end integration with the data loader and indicator engine.

Run with:
    pytest tests/test_rsi_mean_reversion.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.engine import add_indicators, compute_rsi
from src.strategies.rsi_mean_reversion import generate_signals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTRY_THRESHOLD = 30
_EXIT_THRESHOLD = 55


def make_ohlcv(closes: list[float], seed: int = 0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    rng = np.random.default_rng(seed)
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    spread = rng.uniform(0.3, 1.0, n)
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


def make_rsi_frame() -> pd.DataFrame:
    """
    Build a price series engineered to cross RSI thresholds at known points.

    Phase 1 (warmup + stable): gradual uptrend — RSI stays 50–70
    Phase 2 (crash):           sharp drop — RSI falls below 30  → entry signal
    Phase 3 (recovery):        sharp rebound — RSI rises above 55 → exit signal
    Phase 4 (second crash):    another drop below 30 → second entry signal
    Phase 5 (tail):            gradual recovery — RSI rises above 55 again → second exit
    """
    rng = np.random.default_rng(7)

    # Phase 1: 250 bars of slow uptrend (ensures warmup is complete)
    phase1 = np.linspace(100, 130, 250)
    # Phase 2: 30 bars of sharp decline
    phase2 = np.linspace(130, 70, 30)
    # Phase 3: 60 bars of strong recovery
    phase3 = np.linspace(70, 140, 60)
    # Phase 4: 30 bars of another decline
    phase4 = np.linspace(140, 75, 30)
    # Phase 5: 60 bars of recovery
    phase5 = np.linspace(75, 145, 60)
    # Phase 6: flat tail
    phase6 = np.full(20, 145.0)

    closes = np.concatenate([phase1, phase2, phase3, phase4, phase5, phase6])
    return make_ohlcv(closes.tolist())


def make_minimal_rsi_frame(rsi_values: list[float]) -> pd.DataFrame:
    """
    Build a DataFrame with a manually patched rsi_14 column.

    This lets us place RSI crossings at exact positions without relying on
    price construction to produce specific RSI values.
    """
    n = len(rsi_values)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
            "sma_50": np.full(n, 100.0),
            "sma_200": np.full(n, 100.0),
            "rolling_high_20": np.full(n, 110.0),
            "rolling_low_10": np.full(n, 90.0),
            "avg_volume_20": np.full(n, 1_000_000.0),
            "rsi_14": np.array(rsi_values, dtype=float),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return df


# ---------------------------------------------------------------------------
# TestOutputSchema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """Signal output shape and column contract."""

    def setup_method(self):
        df = make_rsi_frame()
        df_ind = add_indicators(df)
        self.result = generate_signals(df_ind)

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
        df = make_rsi_frame()
        df_ind = add_indicators(df)
        for col in df_ind.columns:
            assert col in self.result.columns

    def test_row_count_unchanged(self):
        df = make_rsi_frame()
        df_ind = add_indicators(df)
        assert len(self.result) == len(df_ind)

    def test_index_unchanged(self):
        df = make_rsi_frame()
        df_ind = add_indicators(df)
        pd.testing.assert_index_equal(self.result.index, df_ind.index)

    def test_returns_copy_not_mutation(self):
        df = make_rsi_frame()
        df_ind = add_indicators(df)
        cols_before = list(df_ind.columns)
        generate_signals(df_ind)
        assert list(df_ind.columns) == cols_before


# ---------------------------------------------------------------------------
# TestSignalLogic
# ---------------------------------------------------------------------------


class TestSignalLogic:
    """Signals fire at the correct RSI crossing bars."""

    def setup_method(self):
        df = make_rsi_frame()
        df_ind = add_indicators(df)
        self.result = generate_signals(df_ind)

    def test_entry_signals_exist(self):
        """At least one RSI-below-30 crossing must be detected."""
        assert self.result["signal_entry"].any()

    def test_exit_signals_exist(self):
        """At least one RSI-above-55 crossing must be detected."""
        assert self.result["signal_exit"].any()

    def test_entry_fires_only_when_rsi_below_30(self):
        """Every signal_entry row must have rsi_14 < 30."""
        entry_rows = self.result[self.result["signal_entry"]]
        assert (entry_rows["rsi_14"] < _ENTRY_THRESHOLD).all()

    def test_exit_fires_only_when_rsi_above_55(self):
        """Every signal_exit row must have rsi_14 > 55."""
        exit_rows = self.result[self.result["signal_exit"]]
        assert (exit_rows["rsi_14"] > _EXIT_THRESHOLD).all()

    def test_entry_fires_on_first_crossing_bar(self):
        """On each entry bar, the previous bar must have had RSI >= 30."""
        entry_rows = self.result[self.result["signal_entry"]]
        for idx in entry_rows.index:
            loc = self.result.index.get_loc(idx)
            if loc > 0:
                prev_rsi = self.result.iloc[loc - 1]["rsi_14"]
                if pd.notna(prev_rsi):
                    assert prev_rsi >= _ENTRY_THRESHOLD

    def test_exit_fires_on_first_crossing_bar(self):
        """On each exit bar, the previous bar must have had RSI <= 55."""
        exit_rows = self.result[self.result["signal_exit"]]
        for idx in exit_rows.index:
            loc = self.result.index.get_loc(idx)
            if loc > 0:
                prev_rsi = self.result.iloc[loc - 1]["rsi_14"]
                if pd.notna(prev_rsi):
                    assert prev_rsi <= _EXIT_THRESHOLD

    def test_entry_and_exit_never_both_true_same_day(self):
        """Entry below 30 and exit above 55 cannot occur on the same bar."""
        both = self.result["signal_entry"] & self.result["signal_exit"]
        assert not both.any()

    def test_no_signal_during_warmup(self):
        """No signals before RSI 14 is valid."""
        warmup_mask = self.result["rsi_14"].isna()
        assert not self.result.loc[warmup_mask, "signal_entry"].any()
        assert not self.result.loc[warmup_mask, "signal_exit"].any()


# ---------------------------------------------------------------------------
# TestNoLookAheadBias
# ---------------------------------------------------------------------------


class TestNoLookAheadBias:
    """Truncating the series must not change signals on surviving rows."""

    def test_truncation_does_not_change_past_signals(self):
        df = make_rsi_frame()
        df_ind = add_indicators(df)
        full = generate_signals(df_ind)

        short_ind = add_indicators(df.iloc[:-40])
        short = generate_signals(short_ind)

        common_idx = short.index
        pd.testing.assert_series_equal(
            full.loc[common_idx, "signal_entry"],
            short["signal_entry"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            full.loc[common_idx, "signal_exit"],
            short["signal_exit"],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# TestMinimalCrossing
# ---------------------------------------------------------------------------


class TestMinimalCrossing:
    """Hand-crafted RSI sequence to verify crossing detection exactly."""

    def _build(self) -> pd.DataFrame:
        """
        RSI sequence:
          idx 0-3: RSI around 50 (neutral)
          idx 4:   RSI = 28 → first crossing below 30  → entry signal
          idx 5-6: RSI = 25 (still below 30, NO new entry)
          idx 7:   RSI = 58 → first crossing above 55  → exit signal
          idx 8-9: RSI = 60 (still above 55, NO new exit)
        """
        rsi_values = [50.0, 52.0, 48.0, 31.0, 28.0, 25.0, 26.0, 58.0, 60.0, 61.0]
        return make_minimal_rsi_frame(rsi_values)

    def test_entry_fires_exactly_at_crossing(self):
        df = self._build()
        result = generate_signals(df)
        # Entry only at idx 4
        assert result["signal_entry"].iloc[4] is np.bool_(True)
        for i in [0, 1, 2, 3, 5, 6, 7, 8, 9]:
            assert result["signal_entry"].iloc[i] is np.bool_(False)

    def test_exit_fires_exactly_at_crossing(self):
        df = self._build()
        result = generate_signals(df)
        # Exit only at idx 7
        assert result["signal_exit"].iloc[7] is np.bool_(True)
        for i in [0, 1, 2, 3, 4, 5, 6, 8, 9]:
            assert result["signal_exit"].iloc[i] is np.bool_(False)

    def test_no_repeated_entry_while_below_threshold(self):
        """Consecutive bars below 30 must not re-fire the entry signal."""
        rsi_values = [50.0, 50.0, 50.0, 28.0, 25.0, 22.0, 24.0, 26.0, 29.0, 50.0]
        df = make_minimal_rsi_frame(rsi_values)
        result = generate_signals(df)
        # Entry only fires once, on the first bar that crosses below 30
        assert result["signal_entry"].sum() == 1
        assert result["signal_entry"].iloc[3] is np.bool_(True)

    def test_no_repeated_exit_while_above_threshold(self):
        """Consecutive bars above 55 must not re-fire the exit signal."""
        rsi_values = [30.0, 30.0, 30.0, 60.0, 65.0, 70.0, 68.0, 62.0, 60.0, 58.0]
        df = make_minimal_rsi_frame(rsi_values)
        result = generate_signals(df)
        # Exit only fires once, on the first bar that crosses above 55
        assert result["signal_exit"].sum() == 1
        assert result["signal_exit"].iloc[3] is np.bool_(True)

    def test_boundary_exact_threshold_not_triggered(self):
        """RSI exactly at threshold (== 30 or == 55) must NOT trigger a signal."""
        # Boundary: exactly 30 is NOT below 30; exactly 55 is NOT above 55
        rsi_values = [50.0, 50.0, 30.0, 30.0, 55.0, 55.0, 50.0, 50.0, 50.0, 50.0]
        df = make_minimal_rsi_frame(rsi_values)
        result = generate_signals(df)
        assert not result["signal_entry"].any()
        assert not result["signal_exit"].any()


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Missing indicator columns raise clearly."""

    def test_missing_rsi_14_raises(self):
        df = make_rsi_frame()
        df_ind = add_indicators(df)
        df_ind = df_ind.drop(columns=["rsi_14"])
        with pytest.raises(ValueError, match="rsi_14"):
            generate_signals(df_ind)


# ---------------------------------------------------------------------------
# TestRealDataSmoke
# ---------------------------------------------------------------------------


class TestRealDataSmoke:
    """End-to-end: load SPY, add indicators, generate signals — must not raise."""

    def test_spy_signals_generate_without_error(self):
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 10)
        df_ind = add_indicators(df)
        result = generate_signals(df_ind)

        assert "signal_entry" in result.columns
        assert "signal_exit" in result.columns
        assert result["signal_entry"].dtype == bool
        assert result["signal_exit"].dtype == bool
        # SPY over 10 years should have at least a few oversold bounces
        assert result["signal_entry"].sum() >= 1
        assert result["signal_exit"].sum() >= 1

    def test_spy_entry_rows_have_rsi_below_30(self):
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 10)
        df_ind = add_indicators(df)
        result = generate_signals(df_ind)

        entry_rows = result[result["signal_entry"]]
        assert (entry_rows["rsi_14"] < _ENTRY_THRESHOLD).all()

    def test_spy_exit_rows_have_rsi_above_55(self):
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 10)
        df_ind = add_indicators(df)
        result = generate_signals(df_ind)

        exit_rows = result[result["signal_exit"]]
        assert (exit_rows["rsi_14"] > _EXIT_THRESHOLD).all()
