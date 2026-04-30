"""
Tests for src/strategies/hybrid.py — Chunk 7 acceptance criteria.

Verifies:
- generate_signals() returns correctly keyed dict of independent signal frames
- Each sub-strategy's signals are correct (spot-checks, not re-testing each strategy)
- ALLOCATIONS weights are exactly right and sum to 1.0
- Sub-strategies are independent (different signals, not coupled)
- combine_portfolio_equity() applies weights correctly
- Combined equity matches manual weighted calculation
- Error handling for bad weights and missing keys
- Real-data smoke test on SPY 5y

Run with:
    pytest tests/test_hybrid.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators.engine import add_indicators
from src.strategies.hybrid import ALLOCATIONS, combine_portfolio_equity, generate_signals

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ohlcv(n: int = 300, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    spread = rng.uniform(0.3, 1.0, n)
    volume = rng.integers(1_000_000, 3_000_000, n).astype(float)
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
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


def make_equity(n: int, start: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = start + np.cumsum(rng.normal(0, 10, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(values, index=pd.DatetimeIndex(dates, name="date"))


# ---------------------------------------------------------------------------
# TestAllocations
# ---------------------------------------------------------------------------


class TestAllocations:
    """ALLOCATIONS constant is correct and locked."""

    def test_allocations_has_three_keys(self):
        assert len(ALLOCATIONS) == 3

    def test_allocations_has_breakout_b(self):
        assert "breakout_b" in ALLOCATIONS

    def test_allocations_has_ma_crossover(self):
        assert "ma_crossover" in ALLOCATIONS

    def test_allocations_has_rsi_mean_reversion(self):
        assert "rsi_mean_reversion" in ALLOCATIONS

    def test_breakout_b_weight_is_50_pct(self):
        assert ALLOCATIONS["breakout_b"] == pytest.approx(0.50)

    def test_ma_crossover_weight_is_30_pct(self):
        assert ALLOCATIONS["ma_crossover"] == pytest.approx(0.30)

    def test_rsi_mean_reversion_weight_is_20_pct(self):
        assert ALLOCATIONS["rsi_mean_reversion"] == pytest.approx(0.20)

    def test_weights_sum_to_one(self):
        assert sum(ALLOCATIONS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestGenerateSignalsOutputSchema
# ---------------------------------------------------------------------------


class TestGenerateSignalsOutputSchema:
    """generate_signals() returns a properly structured dict."""

    def setup_method(self):
        df = make_ohlcv()
        df_ind = add_indicators(df)
        self.signals = generate_signals(df_ind)

    def test_returns_dict(self):
        assert isinstance(self.signals, dict)

    def test_dict_has_breakout_b_key(self):
        assert "breakout_b" in self.signals

    def test_dict_has_ma_crossover_key(self):
        assert "ma_crossover" in self.signals

    def test_dict_has_rsi_mean_reversion_key(self):
        assert "rsi_mean_reversion" in self.signals

    def test_dict_keys_match_allocations(self):
        assert set(self.signals.keys()) == set(ALLOCATIONS.keys())

    def test_each_value_is_dataframe(self):
        for key, df in self.signals.items():
            assert isinstance(df, pd.DataFrame), f"{key} is not a DataFrame"

    def test_each_df_has_signal_entry(self):
        for key, df in self.signals.items():
            assert "signal_entry" in df.columns, f"{key} missing signal_entry"

    def test_each_df_has_signal_exit(self):
        for key, df in self.signals.items():
            assert "signal_exit" in df.columns, f"{key} missing signal_exit"

    def test_each_signal_entry_is_boolean(self):
        for key, df in self.signals.items():
            assert df["signal_entry"].dtype == bool, f"{key} signal_entry not bool"

    def test_each_signal_exit_is_boolean(self):
        for key, df in self.signals.items():
            assert df["signal_exit"].dtype == bool, f"{key} signal_exit not bool"

    def test_all_dfs_have_same_index(self):
        dfs = list(self.signals.values())
        for df in dfs[1:]:
            pd.testing.assert_index_equal(dfs[0].index, df.index)

    def test_all_dfs_have_same_row_count(self):
        lengths = [len(df) for df in self.signals.values()]
        assert len(set(lengths)) == 1


# ---------------------------------------------------------------------------
# TestSubStrategiesAreIndependent
# ---------------------------------------------------------------------------


class TestSubStrategiesAreIndependent:
    """
    Each sub-strategy produces its own signals — they must not be identical
    copies of each other.  Breakout B and MA Crossover have fundamentally
    different signal patterns; RSI Mean Reversion differs from both.
    """

    def setup_method(self):
        from src.data.loader import load_ohlcv
        df = load_ohlcv("SPY", 5)
        self.signals = generate_signals(add_indicators(df))

    def test_breakout_b_and_ma_crossover_differ(self):
        bb = self.signals["breakout_b"]["signal_entry"]
        ma = self.signals["ma_crossover"]["signal_entry"]
        assert not bb.equals(ma)

    def test_breakout_b_and_rsi_differ(self):
        bb = self.signals["breakout_b"]["signal_entry"]
        rsi = self.signals["rsi_mean_reversion"]["signal_entry"]
        assert not bb.equals(rsi)

    def test_ma_crossover_and_rsi_differ(self):
        ma = self.signals["ma_crossover"]["signal_entry"]
        rsi = self.signals["rsi_mean_reversion"]["signal_entry"]
        assert not ma.equals(rsi)

    def test_mutating_one_output_does_not_affect_another(self):
        """Each sub-strategy DataFrame is an independent copy."""
        df = make_ohlcv()
        df_ind = add_indicators(df)
        signals = generate_signals(df_ind)

        # Mutate breakout_b output
        original_ma_entry = signals["ma_crossover"]["signal_entry"].copy()
        signals["breakout_b"]["signal_entry"] = False

        # MA Crossover output must be unchanged
        pd.testing.assert_series_equal(
            signals["ma_crossover"]["signal_entry"], original_ma_entry
        )


# ---------------------------------------------------------------------------
# TestSubStrategySignalCorrectness
# ---------------------------------------------------------------------------


class TestSubStrategySignalCorrectness:
    """
    Spot-check that each sub-strategy inside generate_signals() matches
    the output of calling that strategy's generate_signals() directly.
    (Full correctness is covered by the individual strategy test suites.)
    """

    def test_breakout_b_signals_match_direct_call(self):
        from src.strategies.breakout_b import generate_signals as bb_direct

        df = make_ohlcv()
        df_ind = add_indicators(df)

        hybrid_bb = generate_signals(df_ind)["breakout_b"]
        direct_bb = bb_direct(df_ind)

        pd.testing.assert_series_equal(hybrid_bb["signal_entry"], direct_bb["signal_entry"])
        pd.testing.assert_series_equal(hybrid_bb["signal_exit"], direct_bb["signal_exit"])

    def test_ma_crossover_signals_match_direct_call(self):
        from src.strategies.ma_crossover import generate_signals as ma_direct

        df = make_ohlcv()
        df_ind = add_indicators(df)

        hybrid_ma = generate_signals(df_ind)["ma_crossover"]
        direct_ma = ma_direct(df_ind)

        pd.testing.assert_series_equal(hybrid_ma["signal_entry"], direct_ma["signal_entry"])
        pd.testing.assert_series_equal(hybrid_ma["signal_exit"], direct_ma["signal_exit"])

    def test_rsi_signals_match_direct_call(self):
        from src.strategies.rsi_mean_reversion import generate_signals as rsi_direct

        df = make_ohlcv()
        df_ind = add_indicators(df)

        hybrid_rsi = generate_signals(df_ind)["rsi_mean_reversion"]
        direct_rsi = rsi_direct(df_ind)

        pd.testing.assert_series_equal(hybrid_rsi["signal_entry"], direct_rsi["signal_entry"])
        pd.testing.assert_series_equal(hybrid_rsi["signal_exit"], direct_rsi["signal_exit"])


# ---------------------------------------------------------------------------
# TestCombinePortfolioEquity
# ---------------------------------------------------------------------------


class TestCombinePortfolioEquity:
    """combine_portfolio_equity() correctly weights and sums equity curves."""

    def _make_equities(self, n: int = 50) -> dict[str, pd.Series]:
        return {
            "breakout_b": make_equity(n, 10_000, seed=1),
            "ma_crossover": make_equity(n, 10_000, seed=2),
            "rsi_mean_reversion": make_equity(n, 10_000, seed=3),
        }

    def test_returns_series(self):
        result = combine_portfolio_equity(self._make_equities())
        assert isinstance(result, pd.Series)

    def test_combined_name_is_hybrid_equity(self):
        result = combine_portfolio_equity(self._make_equities())
        assert result.name == "hybrid_equity"

    def test_index_preserved(self):
        equities = self._make_equities()
        result = combine_portfolio_equity(equities)
        pd.testing.assert_index_equal(result.index, equities["breakout_b"].index)

    def test_row_count_preserved(self):
        equities = self._make_equities(n=60)
        result = combine_portfolio_equity(equities)
        assert len(result) == 60

    def test_weighted_sum_is_correct(self):
        """Manual calculation must match combine_portfolio_equity output."""
        n = 10
        bb = make_equity(n, 10_000, seed=10)
        ma = make_equity(n, 10_000, seed=11)
        rsi = make_equity(n, 10_000, seed=12)

        expected = bb * 0.50 + ma * 0.30 + rsi * 0.20
        result = combine_portfolio_equity(
            {"breakout_b": bb, "ma_crossover": ma, "rsi_mean_reversion": rsi}
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_custom_allocations_applied_correctly(self):
        """Custom weights override ALLOCATIONS."""
        n = 10
        bb = pd.Series(np.full(n, 1000.0))
        ma = pd.Series(np.full(n, 2000.0))
        rsi = pd.Series(np.full(n, 3000.0))

        custom = {"breakout_b": 0.25, "ma_crossover": 0.50, "rsi_mean_reversion": 0.25}
        result = combine_portfolio_equity(
            {"breakout_b": bb, "ma_crossover": ma, "rsi_mean_reversion": rsi},
            allocations=custom,
        )
        expected_val = 1000 * 0.25 + 2000 * 0.50 + 3000 * 0.25
        np.testing.assert_allclose(result.values, expected_val)

    def test_flat_equities_produce_flat_combined(self):
        """Equal flat equity curves yield the same flat combined curve."""
        n = 20
        flat = pd.Series(np.full(n, 10_000.0))
        equities = {k: flat.copy() for k in ALLOCATIONS}
        result = combine_portfolio_equity(equities)
        np.testing.assert_allclose(result.values, 10_000.0)

    def test_uses_module_allocations_by_default(self):
        """Default call (no allocations arg) must apply ALLOCATIONS exactly."""
        n = 10
        bb = pd.Series(np.full(n, 100.0))
        ma = pd.Series(np.full(n, 200.0))
        rsi = pd.Series(np.full(n, 300.0))

        result = combine_portfolio_equity(
            {"breakout_b": bb, "ma_crossover": ma, "rsi_mean_reversion": rsi}
        )
        expected = 100 * 0.50 + 200 * 0.30 + 300 * 0.20
        np.testing.assert_allclose(result.values, expected)


# ---------------------------------------------------------------------------
# TestCombinePortfolioEquityErrors
# ---------------------------------------------------------------------------


class TestCombinePortfolioEquityErrors:
    """combine_portfolio_equity() raises on invalid inputs."""

    def test_weights_not_summing_to_one_raises(self):
        n = 5
        equities = {k: pd.Series(np.ones(n)) for k in ALLOCATIONS}
        bad_alloc = {"breakout_b": 0.50, "ma_crossover": 0.30, "rsi_mean_reversion": 0.30}
        with pytest.raises(ValueError, match="sum to 1.0"):
            combine_portfolio_equity(equities, allocations=bad_alloc)

    def test_missing_sub_equity_key_raises(self):
        n = 5
        equities = {
            "breakout_b": pd.Series(np.ones(n)),
            "ma_crossover": pd.Series(np.ones(n)),
            # rsi_mean_reversion missing
        }
        with pytest.raises(ValueError, match="rsi_mean_reversion"):
            combine_portfolio_equity(equities)


# ---------------------------------------------------------------------------
# TestRealDataSmoke
# ---------------------------------------------------------------------------


class TestRealDataSmoke:
    """End-to-end: load SPY, generate hybrid signals, combine mock equity."""

    def test_spy_hybrid_signals_generate_without_error(self):
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 5)
        signals = generate_signals(add_indicators(df))

        assert set(signals.keys()) == set(ALLOCATIONS.keys())
        for key, df_sig in signals.items():
            assert "signal_entry" in df_sig.columns
            assert "signal_exit" in df_sig.columns
            assert df_sig["signal_entry"].dtype == bool
            assert df_sig["signal_exit"].dtype == bool

    def test_combine_equity_with_spy_length(self):
        """combine_portfolio_equity works correctly at real data lengths."""
        from src.data.loader import load_ohlcv

        df = load_ohlcv("SPY", 5)
        n = len(df)
        # Simulate three equity curves at the real data length
        equities = {
            "breakout_b": pd.Series(np.linspace(10_000, 15_000, n), index=df.index),
            "ma_crossover": pd.Series(np.linspace(10_000, 13_000, n), index=df.index),
            "rsi_mean_reversion": pd.Series(np.linspace(10_000, 12_000, n), index=df.index),
        }
        combined = combine_portfolio_equity(equities)

        assert len(combined) == n
        # First bar: all start at 10_000, so weighted sum = 10_000
        assert combined.iloc[0] == pytest.approx(10_000.0)
        # Last bar: 15000*0.5 + 13000*0.3 + 12000*0.2 = 7500+3900+2400 = 13800
        assert combined.iloc[-1] == pytest.approx(13_800.0)
