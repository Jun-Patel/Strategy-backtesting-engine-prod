"""
Tests for src/backtester/engine.py — Chunk 8 acceptance criteria.

Uses hand-crafted synthetic signal frames so every P&L number can be verified
by manual arithmetic.  Real-data integration tests confirm the engine runs
cleanly on all five strategies.

Run with:
    pytest tests/test_engine.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester.engine import (
    TRADE_LOG_COLUMNS,
    BacktestResult,
    _empty_trade_log,
    run_backtest,
    run_hybrid_backtest,
)
from src.config.settings import DEFAULT_STARTING_CAPITAL, TRANSACTION_COST_PCT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

COST = TRANSACTION_COST_PCT  # 0.001


def make_signals(
    closes: list[float],
    entries: list[int],   # bar indices where signal_entry is True
    exits: list[int],     # bar indices where signal_exit is True
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """Build a minimal signals DataFrame for engine testing."""
    n = len(closes)
    dates = pd.date_range(start, periods=n, freq="B")
    closes_arr = np.array(closes, dtype=float)
    entry_arr = np.zeros(n, dtype=bool)
    exit_arr = np.zeros(n, dtype=bool)
    for i in entries:
        entry_arr[i] = True
    for i in exits:
        exit_arr[i] = True
    return pd.DataFrame(
        {
            "open": closes_arr * 0.99,  # included but not used by engine
            "close": closes_arr,
            "signal_entry": entry_arr,
            "signal_exit": exit_arr,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


# ---------------------------------------------------------------------------
# TestBacktestResultSchema
# ---------------------------------------------------------------------------


class TestBacktestResultSchema:
    """run_backtest() returns a well-formed BacktestResult."""

    def setup_method(self):
        df = make_signals([100, 101, 102, 103, 104], entries=[0], exits=[3])
        self.result = run_backtest(df, starting_capital=10_000)

    def test_returns_backtest_result(self):
        assert isinstance(self.result, BacktestResult)

    def test_portfolio_history_is_series(self):
        assert isinstance(self.result.portfolio_history, pd.Series)

    def test_portfolio_history_name(self):
        assert self.result.portfolio_history.name == "portfolio"

    def test_trade_log_is_dataframe(self):
        assert isinstance(self.result.trade_log, pd.DataFrame)

    def test_trade_log_has_correct_columns(self):
        for col in TRADE_LOG_COLUMNS:
            assert col in self.result.trade_log.columns

    def test_benchmark_is_series(self):
        assert isinstance(self.result.benchmark, pd.Series)

    def test_benchmark_name(self):
        assert self.result.benchmark.name == "benchmark"

    def test_portfolio_history_index_matches_input(self):
        df = make_signals([100, 101, 102], entries=[], exits=[])
        result = run_backtest(df)
        pd.testing.assert_index_equal(result.portfolio_history.index, df.index)

    def test_benchmark_index_matches_portfolio(self):
        pd.testing.assert_index_equal(
            self.result.portfolio_history.index,
            self.result.benchmark.index,
        )

    def test_row_count_matches_input(self):
        df = make_signals([100, 101, 102, 103], entries=[0], exits=[2])
        result = run_backtest(df, starting_capital=10_000)
        assert len(result.portfolio_history) == 4

    def test_starting_capital_stored(self):
        assert self.result.starting_capital == 10_000

    def test_cost_stored(self):
        assert self.result.transaction_cost_pct == COST

    def test_strategy_name_stored(self):
        df = make_signals([100], entries=[], exits=[])
        r = run_backtest(df, strategy_name="test_strategy")
        assert r.strategy_name == "test_strategy"

    def test_ticker_stored(self):
        df = make_signals([100], entries=[], exits=[])
        r = run_backtest(df, ticker="SPY")
        assert r.ticker == "SPY"


# ---------------------------------------------------------------------------
# TestNoSignals
# ---------------------------------------------------------------------------


class TestNoSignals:
    """When there are no signals the portfolio stays as cash."""

    def test_portfolio_equals_starting_capital_throughout(self):
        df = make_signals([100, 102, 98, 105], entries=[], exits=[])
        result = run_backtest(df, starting_capital=10_000)
        assert (result.portfolio_history == 10_000.0).all()

    def test_trade_log_is_empty(self):
        df = make_signals([100, 102], entries=[], exits=[])
        result = run_backtest(df)
        assert result.trade_log.empty

    def test_empty_trade_log_has_correct_columns(self):
        df = make_signals([100], entries=[], exits=[])
        result = run_backtest(df)
        for col in TRADE_LOG_COLUMNS:
            assert col in result.trade_log.columns


# ---------------------------------------------------------------------------
# TestSingleTrade
# ---------------------------------------------------------------------------


class TestSingleTrade:
    """One entry → one exit: verify exact P&L arithmetic."""

    # closes:  [100, 110, 120, 130]
    # entry bar 0 → execute at close=100
    # exit  bar 2 → execute at close=120
    # starting_capital = 10_000

    CAPITAL = 10_000.0
    ENTRY_CLOSE = 100.0
    EXIT_CLOSE = 120.0

    def _result(self):
        df = make_signals(
            [100, 110, 120, 130],
            entries=[0],
            exits=[2],
        )
        return run_backtest(df, starting_capital=self.CAPITAL)

    def test_one_trade_recorded(self):
        assert len(self._result().trade_log) == 1

    def test_entry_price_correct(self):
        r = self._result()
        assert r.trade_log.iloc[0]["entry_price"] == pytest.approx(self.ENTRY_CLOSE)

    def test_exit_price_correct(self):
        r = self._result()
        assert r.trade_log.iloc[0]["exit_price"] == pytest.approx(self.EXIT_CLOSE)

    def test_shares_correct(self):
        r = self._result()
        expected_shares = self.CAPITAL / (self.ENTRY_CLOSE * (1 + COST))
        assert r.trade_log.iloc[0]["shares"] == pytest.approx(expected_shares)

    def test_pnl_correct(self):
        r = self._result()
        shares = self.CAPITAL / (self.ENTRY_CLOSE * (1 + COST))
        proceeds = shares * self.EXIT_CLOSE * (1 - COST)
        expected_pnl = proceeds - self.CAPITAL
        assert r.trade_log.iloc[0]["pnl"] == pytest.approx(expected_pnl)

    def test_return_pct_correct(self):
        r = self._result()
        shares = self.CAPITAL / (self.ENTRY_CLOSE * (1 + COST))
        proceeds = shares * self.EXIT_CLOSE * (1 - COST)
        expected_return = (proceeds - self.CAPITAL) / self.CAPITAL
        assert r.trade_log.iloc[0]["return_pct"] == pytest.approx(expected_return)

    def test_portfolio_value_while_in_position(self):
        """Portfolio = shares × close while holding."""
        r = self._result()
        shares = self.CAPITAL / (self.ENTRY_CLOSE * (1 + COST))
        # Bar 0: entered at 100, value = shares * 100
        assert r.portfolio_history.iloc[0] == pytest.approx(shares * 100)
        # Bar 1: still in, value = shares * 110
        assert r.portfolio_history.iloc[1] == pytest.approx(shares * 110)
        # Bar 2: exit, cash = shares * 120 * (1 - COST)
        expected_cash = shares * 120 * (1 - COST)
        assert r.portfolio_history.iloc[2] == pytest.approx(expected_cash)

    def test_portfolio_value_after_exit_is_cash(self):
        """After exit, portfolio stays at cash (no more price exposure)."""
        r = self._result()
        shares = self.CAPITAL / (self.ENTRY_CLOSE * (1 + COST))
        expected_cash = shares * self.EXIT_CLOSE * (1 - COST)
        # Bar 3: cash after exit, price moves to 130 but we're out
        assert r.portfolio_history.iloc[3] == pytest.approx(expected_cash)

    def test_holding_days_correct(self):
        r = self._result()
        # Entry: 2020-01-01 (Wed), Exit: 2020-01-07 (Tue) = 6 calendar days
        # Business days: 0→2 is 2 steps, but calendar days is what matters
        entry_date = r.trade_log.iloc[0]["entry_date"]
        exit_date = r.trade_log.iloc[0]["exit_date"]
        expected_days = (exit_date - entry_date).days
        assert r.trade_log.iloc[0]["holding_days"] == expected_days


# ---------------------------------------------------------------------------
# TestMultipleTrades
# ---------------------------------------------------------------------------


class TestMultipleTrades:
    """Two complete round-trips in sequence."""

    def _result(self):
        # closes: [100, 110, 90, 80, 100, 120]
        # entry at bar 0 (100), exit at bar 1 (110)
        # entry at bar 4 (100), exit at bar 5 (120)
        df = make_signals(
            [100, 110, 90, 80, 100, 120],
            entries=[0, 4],
            exits=[1, 5],
        )
        return run_backtest(df, starting_capital=10_000)

    def test_two_trades_recorded(self):
        assert len(self._result().trade_log) == 2

    def test_second_trade_uses_updated_capital(self):
        """After trade 1, cash flows into trade 2 — P&L compounds."""
        r = self._result()
        # Trade 1: enter 100, exit 110
        capital = 10_000.0
        shares1 = capital / (100 * (1 + COST))
        cash_after_t1 = shares1 * 110 * (1 - COST)

        # Trade 2: enter 100 with cash_after_t1, exit 120
        shares2 = cash_after_t1 / (100 * (1 + COST))
        cash_after_t2 = shares2 * 120 * (1 - COST)

        assert r.portfolio_history.iloc[-1] == pytest.approx(cash_after_t2)

    def test_no_reentry_while_in_position(self):
        """Entry signal while already in position must be ignored."""
        # Entry signals on bars 0 AND 1, exit on bar 3
        df = make_signals(
            [100, 105, 102, 115],
            entries=[0, 1],  # bar 1 entry ignored — still in position
            exits=[3],
        )
        result = run_backtest(df, starting_capital=10_000)
        # Only one trade: entered at bar 0 (100), exited at bar 3 (115)
        assert len(result.trade_log) == 1
        assert result.trade_log.iloc[0]["entry_price"] == pytest.approx(100.0)
        assert result.trade_log.iloc[0]["exit_price"] == pytest.approx(115.0)


# ---------------------------------------------------------------------------
# TestForceClose
# ---------------------------------------------------------------------------


class TestForceClose:
    """Open position at end of series is force-closed at the last close."""

    def _result(self):
        # Enter at bar 1 (close=110), no exit signal → force-close at bar 3 (close=130)
        df = make_signals([100, 110, 120, 130], entries=[1], exits=[])
        return run_backtest(df, starting_capital=10_000)

    def test_one_trade_recorded(self):
        assert len(self._result().trade_log) == 1

    def test_exit_price_is_last_close(self):
        r = self._result()
        assert r.trade_log.iloc[0]["exit_price"] == pytest.approx(130.0)

    def test_exit_date_is_last_date(self):
        r = self._result()
        assert r.trade_log.iloc[0]["exit_date"] == r.portfolio_history.index[-1]

    def test_portfolio_last_bar_reflects_open_position(self):
        """Last bar's portfolio value = shares × last_close (unrealized)."""
        r = self._result()
        shares = 10_000 / (110 * (1 + COST))
        assert r.portfolio_history.iloc[-1] == pytest.approx(shares * 130)


# ---------------------------------------------------------------------------
# TestExitWithoutPosition
# ---------------------------------------------------------------------------


class TestExitWithoutPosition:
    """Exit signal when not in position must be ignored."""

    def test_exit_signal_before_entry_is_ignored(self):
        # exit at bar 0, entry at bar 2, exit at bar 3
        df = make_signals([100, 105, 110, 120], entries=[2], exits=[0, 3])
        result = run_backtest(df, starting_capital=10_000)
        # Should have 1 trade, entered at bar 2 (110), exited at bar 3 (120)
        assert len(result.trade_log) == 1
        assert result.trade_log.iloc[0]["entry_price"] == pytest.approx(110.0)

    def test_portfolio_stays_at_cash_before_entry(self):
        df = make_signals([100, 105, 110, 120], entries=[2], exits=[0, 3])
        result = run_backtest(df, starting_capital=10_000)
        assert result.portfolio_history.iloc[0] == pytest.approx(10_000.0)
        assert result.portfolio_history.iloc[1] == pytest.approx(10_000.0)


# ---------------------------------------------------------------------------
# TestBenchmark
# ---------------------------------------------------------------------------


class TestBenchmark:
    """Buy-and-hold benchmark is computed correctly."""

    CAPITAL = 10_000.0
    CLOSES = [100.0, 110.0, 90.0, 130.0]

    def _result(self):
        df = make_signals(self.CLOSES, entries=[], exits=[])
        return run_backtest(df, starting_capital=self.CAPITAL)

    def test_benchmark_first_bar(self):
        """At bar 0, benchmark = capital (buy at 100, value at 100)."""
        r = self._result()
        bh_shares = self.CAPITAL / (100 * (1 + COST))
        assert r.benchmark.iloc[0] == pytest.approx(bh_shares * 100)

    def test_benchmark_tracks_price(self):
        """Benchmark scales linearly with price movement."""
        r = self._result()
        bh_shares = self.CAPITAL / (100 * (1 + COST))
        for i, close in enumerate(self.CLOSES):
            assert r.benchmark.iloc[i] == pytest.approx(bh_shares * close)

    def test_benchmark_independent_of_strategy_signals(self):
        """Benchmark is the same regardless of what signals are present."""
        df1 = make_signals(self.CLOSES, entries=[], exits=[])
        df2 = make_signals(self.CLOSES, entries=[0, 2], exits=[1, 3])
        r1 = run_backtest(df1, starting_capital=self.CAPITAL)
        r2 = run_backtest(df2, starting_capital=self.CAPITAL)
        pd.testing.assert_series_equal(r1.benchmark, r2.benchmark)


# ---------------------------------------------------------------------------
# TestTransactionCosts
# ---------------------------------------------------------------------------


class TestTransactionCosts:
    """Transaction costs reduce P&L vs a zero-cost simulation."""

    def test_profitable_trade_is_less_profitable_with_costs(self):
        df = make_signals([100, 120], entries=[0], exits=[1])
        with_cost = run_backtest(df, starting_capital=10_000, transaction_cost_pct=COST)
        zero_cost = run_backtest(df, starting_capital=10_000, transaction_cost_pct=0.0)
        assert with_cost.trade_log.iloc[0]["pnl"] < zero_cost.trade_log.iloc[0]["pnl"]

    def test_zero_cost_round_trip_pnl_correct(self):
        """With zero cost: pnl = shares × (exit − entry); shares = capital / entry."""
        df = make_signals([100, 150], entries=[0], exits=[1])
        result = run_backtest(df, starting_capital=10_000, transaction_cost_pct=0.0)
        shares = 10_000 / 100
        expected_pnl = shares * (150 - 100)
        assert result.trade_log.iloc[0]["pnl"] == pytest.approx(expected_pnl)


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Missing required columns raise ValueError."""

    @pytest.mark.parametrize("col", ["close", "signal_entry", "signal_exit"])
    def test_missing_column_raises(self, col):
        df = make_signals([100, 101], entries=[0], exits=[1])
        df = df.drop(columns=[col])
        with pytest.raises(ValueError, match=col):
            run_backtest(df)


# ---------------------------------------------------------------------------
# TestHybridEngine
# ---------------------------------------------------------------------------


class TestHybridEngine:
    """run_hybrid_backtest() combines sub-strategies correctly."""

    def _make_hybrid_signals(self, n: int = 6) -> tuple[dict, dict]:
        """Return (signals_dict, allocations) for a controlled hybrid test."""
        allocations = {
            "breakout_b": 0.50,
            "ma_crossover": 0.30,
            "rsi_mean_reversion": 0.20,
        }
        closes = [100.0] * n
        # Each sub-strategy has its own independent signals
        signals = {}
        for key in allocations:
            signals[key] = make_signals(
                closes,
                entries=[0],  # all enter at bar 0
                exits=[n - 1],  # all exit at last bar
            )
        return signals, allocations

    def test_returns_backtest_result(self):
        signals, alloc = self._make_hybrid_signals()
        result = run_hybrid_backtest(signals, alloc, starting_capital=10_000)
        assert isinstance(result, BacktestResult)

    def test_portfolio_history_length(self):
        signals, alloc = self._make_hybrid_signals(n=8)
        result = run_hybrid_backtest(signals, alloc, starting_capital=10_000)
        assert len(result.portfolio_history) == 8

    def test_portfolio_name_is_portfolio(self):
        signals, alloc = self._make_hybrid_signals()
        result = run_hybrid_backtest(signals, alloc, starting_capital=10_000)
        assert result.portfolio_history.name == "portfolio"

    def test_strategy_name_is_hybrid(self):
        signals, alloc = self._make_hybrid_signals()
        result = run_hybrid_backtest(signals, alloc, starting_capital=10_000)
        assert result.strategy_name == "Hybrid Allocation Strategy"

    def test_starting_capital_split_correctly(self):
        """
        With flat closes (100 throughout) and identical entry/exit for each
        sub-strategy, the combined portfolio should start at starting_capital.
        """
        signals, alloc = self._make_hybrid_signals(n=6)
        result = run_hybrid_backtest(signals, alloc, starting_capital=10_000)
        # On bar 0, all sub-strategies just entered at close=100.
        # Each portfolio_history[0] = sub_capital / (100 * (1+cost)) * 100
        # ≈ sub_capital (minus tiny cost).  Sum ≈ 10_000.
        assert result.portfolio_history.iloc[0] == pytest.approx(
            10_000 / (1 + COST), rel=1e-4
        )

    def test_capital_allocation_weights_applied(self):
        """
        Sub-strategy A gets 50% of capital, B 30%, C 20%.
        Verify by checking the trade logs' entry_value proportions.
        """
        signals, alloc = self._make_hybrid_signals()
        # Run sub-strategies independently to compare
        sub_a = run_backtest(signals["breakout_b"], starting_capital=5_000)
        sub_b = run_backtest(signals["ma_crossover"], starting_capital=3_000)
        sub_c = run_backtest(signals["rsi_mean_reversion"], starting_capital=2_000)

        combined = sub_a.portfolio_history + sub_b.portfolio_history + sub_c.portfolio_history
        result = run_hybrid_backtest(signals, alloc, starting_capital=10_000)

        pd.testing.assert_series_equal(
            result.portfolio_history, combined, check_names=False
        )

    def test_trade_log_has_sub_strategy_column(self):
        signals, alloc = self._make_hybrid_signals()
        result = run_hybrid_backtest(signals, alloc, starting_capital=10_000)
        assert "sub_strategy" in result.trade_log.columns

    def test_trade_log_contains_all_sub_strategy_keys(self):
        signals, alloc = self._make_hybrid_signals()
        result = run_hybrid_backtest(signals, alloc, starting_capital=10_000)
        assert set(result.trade_log["sub_strategy"].unique()) == set(alloc.keys())

    def test_benchmark_uses_full_capital(self):
        """Benchmark is buy-and-hold with the full 10_000, not 5_000."""
        signals, alloc = self._make_hybrid_signals()
        result = run_hybrid_backtest(signals, alloc, starting_capital=10_000)
        bh_shares = 10_000 / (100 * (1 + COST))
        assert result.benchmark.iloc[0] == pytest.approx(bh_shares * 100)

    def test_hybrid_bad_weights_raise(self):
        signals, _ = self._make_hybrid_signals()
        bad_alloc = {"breakout_b": 0.60, "ma_crossover": 0.30, "rsi_mean_reversion": 0.20}
        with pytest.raises(ValueError, match="sum to 1.0"):
            run_hybrid_backtest(signals, bad_alloc, starting_capital=10_000)

    def test_hybrid_missing_key_raises(self):
        signals, alloc = self._make_hybrid_signals()
        del signals["breakout_b"]
        with pytest.raises(ValueError, match="breakout_b"):
            run_hybrid_backtest(signals, alloc, starting_capital=10_000)


# ---------------------------------------------------------------------------
# TestAllFiveStrategiesIntegration
# ---------------------------------------------------------------------------


class TestAllFiveStrategiesIntegration:
    """Engine runs cleanly on all five strategies with real SPY data."""

    @pytest.fixture(scope="class")
    def spy_ind(self):
        from src.data.loader import load_ohlcv
        from src.indicators.engine import add_indicators
        df = load_ohlcv("SPY", 5)
        return add_indicators(df)

    def test_strategy_1_ma_crossover(self, spy_ind):
        from src.strategies.ma_crossover import generate_signals
        signals = generate_signals(spy_ind)
        result = run_backtest(signals, strategy_name="ma_crossover", ticker="SPY")
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_history) == len(spy_ind)
        assert result.portfolio_history.notna().all()

    def test_strategy_2_rsi_mean_reversion(self, spy_ind):
        from src.strategies.rsi_mean_reversion import generate_signals
        signals = generate_signals(spy_ind)
        result = run_backtest(signals, strategy_name="rsi_mean_reversion", ticker="SPY")
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_history) == len(spy_ind)

    def test_strategy_3_breakout_a(self, spy_ind):
        from src.strategies.breakout_a import generate_signals
        signals = generate_signals(spy_ind)
        result = run_backtest(signals, strategy_name="breakout_a", ticker="SPY")
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_history) == len(spy_ind)

    def test_strategy_4_breakout_b(self, spy_ind):
        from src.strategies.breakout_b import generate_signals
        signals = generate_signals(spy_ind)
        result = run_backtest(signals, strategy_name="breakout_b", ticker="SPY")
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_history) == len(spy_ind)

    def test_strategy_5_hybrid(self, spy_ind):
        from src.strategies.hybrid import ALLOCATIONS, generate_signals
        signals_dict = generate_signals(spy_ind)
        result = run_hybrid_backtest(signals_dict, ALLOCATIONS, ticker="SPY")
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_history) == len(spy_ind)
        assert "sub_strategy" in result.trade_log.columns

    def test_portfolio_history_always_positive(self, spy_ind):
        """Portfolio value must always be positive (no margin, no short)."""
        from src.strategies.breakout_a import generate_signals
        signals = generate_signals(spy_ind)
        result = run_backtest(signals, ticker="SPY")
        assert (result.portfolio_history > 0).all()

    def test_benchmark_always_positive(self, spy_ind):
        from src.strategies.ma_crossover import generate_signals
        signals = generate_signals(spy_ind)
        result = run_backtest(signals, ticker="SPY")
        assert (result.benchmark > 0).all()

    def test_trade_log_columns_present_for_all_strategies(self, spy_ind):
        from src.strategies.ma_crossover import generate_signals as s1
        from src.strategies.rsi_mean_reversion import generate_signals as s2
        from src.strategies.breakout_a import generate_signals as s3
        from src.strategies.breakout_b import generate_signals as s4
        for gen in [s1, s2, s3, s4]:
            result = run_backtest(gen(spy_ind))
            for col in TRADE_LOG_COLUMNS:
                assert col in result.trade_log.columns

    def test_breakout_a_produces_more_trades_than_ma_crossover(self, spy_ind):
        """Breakout A fires far more entry signals → more trades than MA Crossover."""
        from src.strategies.breakout_a import generate_signals as ba
        from src.strategies.ma_crossover import generate_signals as ma
        r_ba = run_backtest(ba(spy_ind))
        r_ma = run_backtest(ma(spy_ind))
        assert len(r_ba.trade_log) > len(r_ma.trade_log)
