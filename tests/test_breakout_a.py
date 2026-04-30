"""
Tests for src/strategies/breakout_a.py — Chunk 5 acceptance criteria.

Uses synthetic price series with known breakout events for deterministic,
offline testing.  A real-data smoke test on SPY verifies end-to-end
integration with the data loader and indicator engine.

Run with:
    pytest tests/test_breakout_a.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.engine import add_indicators, compute_rolling_high, compute_rolling_low
from src.strategies.breakout_a import generate_signals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ohlcv(closes: list[float], seed: int = 1) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices."""
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


def make_breakout_frame(n_warmup: int = 250) -> pd.DataFrame:
    """
    Build a price series engineered so that:
    - Warmup phase (n_warmup bars): slow drift, no major breakout
    - Spike up: a single bar that clearly exceeds the prior 20-day high  → entry
    - Pullback: a sharp drop that clearly falls below the prior 10-day low → exit
    - Second spike: another breakout above the new 20-day high             → entry
    """
    rng = np.random.default_rng(42)

    # Phase 1: slow upward drift with small variance (warmup)
    phase1 = np.linspace(100, 110, n_warmup) + rng.normal(0, 0.3, n_warmup)

    # Phase 2: sharp spike clearly above recent 20-day high
    # After phase1, rolling_high_20 is ~ max of last 20 phase1 bars ≈ 110
    # We jump to 125, which is clearly above that.
    phase2 = np.array([125.0, 124.0, 123.0, 122.0])

    # Phase 3: sharp drop clearly below rolling_low_10
    # rolling_low_10 after phase2 ≈ min of last 10 bars from phase1 / phase2
    # We drop to 85, clearly below any recent low.
    phase3 = np.array([85.0, 84.0, 83.0])

    # Phase 4: recovery and second breakout
    phase4 = np.linspace(83, 115, 30)
    phase5 = np.array([140.0, 139.0, 138.0])  # another clear breakout

    closes = np.concatenate([phase1, phase2, phase3, phase4, phase5])
    return make_ohlcv(closes.tolist())


def make_minimal_frame(closes: list[float], rolling_high: list[float],
                        rolling_low: list[float]) -> pd.DataFrame:
    """
    Build a DataFrame with manually patched rolling_high_20 and rolling_low_10.
    Lets us test signal logic without relying on price construction.
    """
    n = len(closes)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    closes_arr = np.array(closes, dtype=float)
    df = pd.DataFrame(
        {
            "open": closes_arr - 0.5,
            "high": closes_arr + 1.0,
            "low": closes_arr - 1.0,
            "close": closes_arr,
            "volume": np.full(n, 1_000_000.0),
            "sma_50": closes_arr,
            "sma_200": closes_arr,
            "rsi_14": np.full(n, 50.0),
            "rolling_high_20": np.array(rolling_high, dtype=float),
            "rolling_low_10": np.array(rolling_low, dtype=float),
            "avg_volume_20": np.full(n, 1_000_000.0),
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
        df = make_breakout_frame()
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
        df = make_breakout_frame()
        df_ind = add_indicators(df)
        for col in df_ind.columns:
            assert col in self.result.columns

    def test_row_count_unchanged(self):
        df = make_breakout_frame()
        df_ind = add_indicators(df)
        assert len(self.result) == len(df_ind)

    def test_index_unchanged(self):
        df = make_breakout_frame()
        df_ind = add_indicators(df)
        pd.testing.assert_index_equal(self.result.index, df_ind.index)

    def test_returns_copy_not_mutation(self):
        df = make_breakout_frame()
        df_ind = add_indicators(df)
        cols_before = list(df_ind.columns)
        generate_signals(df_ind)
        assert list(df_ind.columns) == cols_before


# ---------------------------------------------------------------------------
# TestSignalLogic
# ---------------------------------------------------------------------------


class TestSignalLogic:
    """Signals match entry and exit conditions exactly."""

    def setup_method(self):
        df = make_breakout_frame()
        df_ind = add_indicators(df)
        self.result = generate_signals(df_ind)

    def test_entry_signals_exist(self):
        assert self.result["signal_entry"].any()

    def test_exit_signals_exist(self):
        assert self.result["signal_exit"].any()

    def test_entry_only_when_close_above_rolling_high(self):
        """Every entry bar must have close > rolling_high_20."""
        valid = self.result.dropna(subset=["rolling_high_20"])
        entry_rows = valid[valid["signal_entry"]]
        assert (entry_rows["close"] > entry_rows["rolling_high_20"]).all()

    def test_exit_only_when_close_below_rolling_low(self):
        """Every exit bar must have close < rolling_low_10."""
        valid = self.result.dropna(subset=["rolling_low_10"])
        exit_rows = valid[valid["signal_exit"]]
        assert (exit_rows["close"] < exit_rows["rolling_low_10"]).all()

    def test_no_entry_when_close_at_or_below_rolling_high(self):
        """Bars where close <= rolling_high_20 must never have signal_entry."""
        valid = self.result.dropna(subset=["rolling_high_20"])
        no_breakout = valid[valid["close"] <= valid["rolling_high_20"]]
        assert not no_breakout["signal_entry"].any()

    def test_no_exit_when_close_at_or_above_rolling_low(self):
        """Bars where close >= rolling_low_10 must never have signal_exit."""
        valid = self.result.dropna(subset=["rolling_low_10"])
        no_breakdown = valid[valid["close"] >= valid["rolling_low_10"]]
        assert not no_breakdown["signal_exit"].any()

    def test_no_signal_during_indicator_warmup(self):
        """No signals before rolling_high_20 and rolling_low_10 are valid."""
        warmup = self.result[
            self.result["rolling_high_20"].isna() | self.result["rolling_low_10"].isna()
        ]
        assert not warmup["signal_entry"].any()
        assert not warmup["signal_exit"].any()


# ---------------------------------------------------------------------------
# TestMinimalBreakout
# ---------------------------------------------------------------------------


class TestMinimalBreakout:
    """Hand-crafted minimal series to verify exact signal positions."""

    def test_entry_fires_exactly_when_close_exceeds_high(self):
        """
        close:         [100, 100, 100, 101, 99, 100]
        rolling_high:  [105, 105, 105, 100, 105, 105]  ← manually set
        rolling_low:   [ 90,  90,  90,  90,  90,  90]

        Entry should fire only at idx 3 where close(101) > rolling_high(100).
        """
        closes       = [100, 100, 100, 101, 99, 100]
        rolling_high = [105, 105, 105, 100, 105, 105]
        rolling_low  = [ 90,  90,  90,  90,  90,  90]

        df = make_minimal_frame(closes, rolling_high, rolling_low)
        result = generate_signals(df)

        assert result["signal_entry"].iloc[3] is np.bool_(True)
        for i in [0, 1, 2, 4, 5]:
            assert result["signal_entry"].iloc[i] is np.bool_(False)

    def test_exit_fires_exactly_when_close_below_low(self):
        """
        close:        [100, 100, 100, 89,  100, 100]
        rolling_high: [105, 105, 105, 105, 105, 105]
        rolling_low:  [ 90,  90,  90,  90,  90,  90]

        Exit should fire only at idx 3 where close(89) < rolling_low(90).
        """
        closes       = [100, 100, 100,  89, 100, 100]
        rolling_high = [105, 105, 105, 105, 105, 105]
        rolling_low  = [ 90,  90,  90,  90,  90,  90]

        df = make_minimal_frame(closes, rolling_high, rolling_low)
        result = generate_signals(df)

        assert result["signal_exit"].iloc[3] is np.bool_(True)
        for i in [0, 1, 2, 4, 5]:
            assert result["signal_exit"].iloc[i] is np.bool_(False)

    def test_boundary_exact_equality_not_triggered(self):
        """
        close == rolling_high → NOT an entry (strict >)
        close == rolling_low  → NOT an exit  (strict <)
        """
        closes       = [100, 100, 100, 100, 100, 100]
        rolling_high = [100, 100, 100, 100, 100, 100]
        rolling_low  = [100, 100, 100, 100, 100, 100]

        df = make_minimal_frame(closes, rolling_high, rolling_low)
        result = generate_signals(df)

        assert not result["signal_entry"].any()
        assert not result["signal_exit"].any()

    def test_consecutive_breakout_bars_all_signal(self):
        """
        Unlike the RSI strategy, every bar above the rolling high fires an entry.
        The engine — not the signal generator — enforces single-position logic.
        """
        closes       = [110, 110, 110, 110, 110, 90]
        rolling_high = [105, 105, 105, 105, 105, 105]
        rolling_low  = [ 80,  80,  80,  80,  80,  95]  # 90 < 95 → exit on last bar

        df = make_minimal_frame(closes, rolling_high, rolling_low)
        result = generate_signals(df)

        # First 5 bars: close(110) > rolling_high(105) → all entry
        assert result["signal_entry"].iloc[:5].all()
        # Last bar: close(90) < rolling_low(95) → exit
        assert result["signal_exit"].iloc[5] is np.bool_(True)

    def test_entry_and_exit_can_be_true_same_bar(self):
        """
        If close simultaneously exceeds rolling_high AND falls below rolling_low
        (theoretically possible with a gap), both signals are True.
        Engine resolves priority — signal generator does not suppress.
        """
        # close=50 > rolling_high=40 AND close=50 < rolling_low=60
        closes       = [50]
        rolling_high = [40]
        rolling_low  = [60]

        df = make_minimal_frame(closes, rolling_high, rolling_low)
        result = generate_signals(df)

        assert result["signal_entry"].iloc[0] is np.bool_(True)
        assert result["signal_exit"].iloc[0] is np.bool_(True)


# ---------------------------------------------------------------------------
# TestShiftCorrectness
# ---------------------------------------------------------------------------


class TestShiftCorrectness:
    """
    Verify that signals correctly use the *previous* window's high/low,
    not the current bar's.  This is the core look-ahead bias check for
    breakout strategies.
    """

    def test_rolling_high_uses_previous_window(self):
        """
        The rolling_high_20 from add_indicators() is shift(1), meaning
        day-N entry compares close to the max of days [N-20, N-1] — not
        including today's bar.

        Verify directly: truncating the series by 1 day must not change
        the rolling_high value on any surviving row.
        """
        df = make_breakout_frame()
        full_ind = add_indicators(df)
        short_ind = add_indicators(df.iloc[:-1])

        common = short_ind.index
        pd.testing.assert_series_equal(
            full_ind.loc[common, "rolling_high_20"],
            short_ind["rolling_high_20"],
        )

    def test_rolling_low_uses_previous_window(self):
        df = make_breakout_frame()
        full_ind = add_indicators(df)
        short_ind = add_indicators(df.iloc[:-1])

        common = short_ind.index
        pd.testing.assert_series_equal(
            full_ind.loc[common, "rolling_low_10"],
            short_ind["rolling_low_10"],
        )

    def test_truncation_does_not_change_past_signals(self):
        """Removing trailing rows must not alter signals on surviving rows."""
        df = make_breakout_frame()
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

    def _base(self):
        return add_indicators(make_breakout_frame())

    def test_missing_rolling_high_raises(self):
        df = self._base().drop(columns=["rolling_high_20"])
        with pytest.raises(ValueError, match="rolling_high_20"):
            generate_signals(df)

    def test_missing_rolling_low_raises(self):
        df = self._base().drop(columns=["rolling_low_10"])
        with pytest.raises(ValueError, match="rolling_low_10"):
            generate_signals(df)

    def test_missing_close_raises(self):
        df = self._base().drop(columns=["close"])
        with pytest.raises(ValueError, match="close"):
            generate_signals(df)


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
        # SPY over 10 years will have multiple breakouts and breakdowns
        assert result["signal_entry"].sum() >= 1
        assert result["signal_exit"].sum() >= 1

    def test_spy_entry_rows_close_above_rolling_high(self):
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 10)
        result = generate_signals(add_indicators(df))
        valid = result.dropna(subset=["rolling_high_20"])
        entry_rows = valid[valid["signal_entry"]]
        assert (entry_rows["close"] > entry_rows["rolling_high_20"]).all()

    def test_spy_exit_rows_close_below_rolling_low(self):
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 10)
        result = generate_signals(add_indicators(df))
        valid = result.dropna(subset=["rolling_low_10"])
        exit_rows = valid[valid["signal_exit"]]
        assert (exit_rows["close"] < exit_rows["rolling_low_10"]).all()

    def test_spy_breakout_a_produces_more_signals_than_ma_crossover(self):
        """
        Breakout A is a raw momentum strategy — it should fire far more often
        than the long-term MA Crossover on equity data over 10 years.
        """
        from src.data.loader import load_ohlcv
        from src.strategies.ma_crossover import generate_signals as ma_signals

        df = load_ohlcv("SPY", 10)
        ind = add_indicators(df)

        breakout_entries = generate_signals(ind)["signal_entry"].sum()
        ma_entries = ma_signals(ind)["signal_entry"].sum()

        assert breakout_entries > ma_entries
