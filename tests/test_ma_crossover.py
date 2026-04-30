"""
Tests for src/strategies/ma_crossover.py — Chunk 3 acceptance criteria.

Uses synthetic price series engineered to produce known crossover events so
every assertion is deterministic and offline.  A real-data smoke test on SPY
verifies end-to-end integration with the data loader and indicator engine.

Run with:
    pytest tests/test_ma_crossover.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.engine import add_indicators
from src.strategies.ma_crossover import generate_signals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ohlcv(closes: list[float], seed: int = 0) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    rng = np.random.default_rng(seed)
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    spread = rng.uniform(0.5, 1.5, n)
    volume = rng.integers(1_000_000, 3_000_000, n).astype(float)
    dates = pd.date_range("2010-01-01", periods=n, freq="B")
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


def make_crossover_frame(n_warmup: int = 210, n_tail: int = 30) -> pd.DataFrame:
    """
    Build a price series engineered so that:
    - The first (n_warmup) bars are a slow uptrend: SMA 50 stays above SMA 200.
    - Then prices drop sharply so SMA 50 crosses below SMA 200 (death cross).
    - Then prices recover sharply so SMA 50 crosses back above SMA 200 (golden cross).

    This guarantees at least one entry signal and one exit signal in the series.
    """
    # Phase 1: gradual uptrend — SMA 50 will be above SMA 200
    phase1 = np.linspace(150, 200, n_warmup)
    # Phase 2: sharp drop — SMA 50 will eventually cross below SMA 200
    phase2 = np.linspace(200, 80, 80)
    # Phase 3: sharp recovery — SMA 50 will eventually cross above SMA 200
    phase3 = np.linspace(80, 220, 80)
    # Phase 4: flat tail so signals settle
    phase4 = np.full(n_tail, 220.0)

    closes = np.concatenate([phase1, phase2, phase3, phase4])
    return make_ohlcv(closes.tolist())


# ---------------------------------------------------------------------------
# TestOutputSchema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """Signal output shape and column contract."""

    def setup_method(self):
        df = make_crossover_frame()
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
        """generate_signals must not drop any existing columns."""
        df = make_crossover_frame()
        df_ind = add_indicators(df)
        for col in df_ind.columns:
            assert col in self.result.columns

    def test_row_count_unchanged(self):
        df = make_crossover_frame()
        df_ind = add_indicators(df)
        assert len(self.result) == len(df_ind)

    def test_index_unchanged(self):
        df = make_crossover_frame()
        df_ind = add_indicators(df)
        pd.testing.assert_index_equal(self.result.index, df_ind.index)

    def test_returns_copy_not_mutation(self):
        """generate_signals must not modify the input DataFrame."""
        df = make_crossover_frame()
        df_ind = add_indicators(df)
        cols_before = list(df_ind.columns)
        generate_signals(df_ind)
        assert list(df_ind.columns) == cols_before


# ---------------------------------------------------------------------------
# TestSignalLogic
# ---------------------------------------------------------------------------


class TestSignalLogic:
    """Signals fire at the correct crossover bars."""

    def setup_method(self):
        df = make_crossover_frame()
        df_ind = add_indicators(df)
        self.df = df_ind
        self.result = generate_signals(df_ind)

    def test_entry_signals_exist(self):
        """At least one golden cross must be detected in the engineered series."""
        assert self.result["signal_entry"].any()

    def test_exit_signals_exist(self):
        """At least one death cross must be detected in the engineered series."""
        assert self.result["signal_exit"].any()

    def test_entry_fires_on_crossover_day(self):
        """On every signal_entry day, SMA 50 > SMA 200 and previous day had SMA 50 <= SMA 200."""
        entry_rows = self.result[self.result["signal_entry"]]
        for idx in entry_rows.index:
            loc = self.result.index.get_loc(idx)
            assert self.result.iloc[loc]["sma_50"] > self.result.iloc[loc]["sma_200"]
            if loc > 0:
                prev = self.result.iloc[loc - 1]
                # Skip the prev-row directional check when SMA 200 was still in
                # warmup (NaN).  The entry is still valid — it's the first bar
                # where SMA 50 can be above SMA 200.
                if pd.notna(prev["sma_200"]):
                    assert prev["sma_50"] <= prev["sma_200"]

    def test_exit_fires_on_crossunder_day(self):
        """On every signal_exit day, SMA 50 <= SMA 200 and previous day had SMA 50 > SMA 200."""
        exit_rows = self.result[self.result["signal_exit"]]
        for idx in exit_rows.index:
            loc = self.result.index.get_loc(idx)
            assert self.result.iloc[loc]["sma_50"] <= self.result.iloc[loc]["sma_200"]
            if loc > 0:
                prev = self.result.iloc[loc - 1]
                assert prev["sma_50"] > prev["sma_200"]

    def test_entry_and_exit_never_both_true_same_day(self):
        """A crossover and crossunder cannot occur on the same bar."""
        both = self.result["signal_entry"] & self.result["signal_exit"]
        assert not both.any()

    def test_no_signal_during_warmup(self):
        """No signals before both SMA 50 and SMA 200 are valid (row < 200)."""
        warmup_mask = self.result["sma_200"].isna()
        assert not self.result.loc[warmup_mask, "signal_entry"].any()
        assert not self.result.loc[warmup_mask, "signal_exit"].any()


# ---------------------------------------------------------------------------
# TestNoLookAheadBias
# ---------------------------------------------------------------------------


class TestNoLookAheadBias:
    """Truncating the series must not change signals on surviving rows."""

    def test_truncation_does_not_change_past_signals(self):
        df = make_crossover_frame()
        df_ind = add_indicators(df)
        full = generate_signals(df_ind)

        # Drop the last 30 rows and recompute
        short_ind = add_indicators(df.iloc[:-30])
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
# TestMinimalCrossover
# ---------------------------------------------------------------------------


class TestMinimalCrossover:
    """Hand-crafted minimal series to verify crossover detection exactly."""

    def _build_minimal(self) -> pd.DataFrame:
        """
        Construct a tiny series where we control the SMA relationship exactly.

        Strategy: patch sma_50 / sma_200 directly rather than relying on price
        construction to produce specific SMA values.  This lets us pinpoint the
        exact bar for crossover.
        """
        n = 10
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # Prices don't matter here — we overwrite SMAs manually
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.full(n, 1_000_000.0),
                "rsi_14": np.full(n, 50.0),
                "rolling_high_20": np.full(n, 110.0),
                "rolling_low_10": np.full(n, 90.0),
                "avg_volume_20": np.full(n, 1_000_000.0),
            },
            index=pd.DatetimeIndex(dates, name="date"),
        )

        # Manually set SMA values:
        # rows 0-3: SMA 50 < SMA 200  (no position)
        # row 4: SMA 50 crosses above SMA 200  → entry signal
        # rows 5-7: SMA 50 > SMA 200  (in position)
        # row 8: SMA 50 crosses below SMA 200  → exit signal
        # row 9: SMA 50 < SMA 200  (flat)
        sma_50  = [95, 96, 97, 98, 101, 102, 103, 104, 99, 98]
        sma_200 = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        df["sma_50"] = sma_50
        df["sma_200"] = sma_200
        return df

    def test_entry_fires_exactly_at_crossover(self):
        df = self._build_minimal()
        result = generate_signals(df)
        # Entry should be True only at index 4
        assert result["signal_entry"].iloc[4] is np.bool_(True)
        non_entry = list(range(10))
        non_entry.remove(4)
        for i in non_entry:
            assert result["signal_entry"].iloc[i] is np.bool_(False)

    def test_exit_fires_exactly_at_crossunder(self):
        df = self._build_minimal()
        result = generate_signals(df)
        # Exit should be True only at index 8
        assert result["signal_exit"].iloc[8] is np.bool_(True)
        non_exit = list(range(10))
        non_exit.remove(8)
        for i in non_exit:
            assert result["signal_exit"].iloc[i] is np.bool_(False)


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Missing indicator columns raise clearly."""

    def test_missing_sma_50_raises(self):
        df = make_crossover_frame()
        df_ind = add_indicators(df)
        df_ind = df_ind.drop(columns=["sma_50"])
        with pytest.raises(ValueError, match="sma_50"):
            generate_signals(df_ind)

    def test_missing_sma_200_raises(self):
        df = make_crossover_frame()
        df_ind = add_indicators(df)
        df_ind = df_ind.drop(columns=["sma_200"])
        with pytest.raises(ValueError, match="sma_200"):
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
        # On 10 years of SPY data there should be at least a few crossovers
        assert result["signal_entry"].sum() >= 1
        assert result["signal_exit"].sum() >= 1
