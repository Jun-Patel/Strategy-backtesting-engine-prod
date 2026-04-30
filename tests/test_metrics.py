"""
Tests for src/analytics/metrics.py — Chunk 9 acceptance criteria.

Uses synthetic BacktestResults with hand-crafted portfolio/benchmark/trade
data so every metric value can be verified by manual arithmetic.  A real-data
integration test at the end confirms the pipeline works end-to-end.

Run with:
    pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.metrics import TRADING_DAYS_PER_YEAR, compute_metrics, metrics_to_series
from src.backtester.engine import TRADE_LOG_COLUMNS, BacktestResult
from src.config.settings import DEFAULT_STARTING_CAPITAL, TRANSACTION_COST_PCT

# ---------------------------------------------------------------------------
# Helpers — synthetic BacktestResult factories
# ---------------------------------------------------------------------------

CAPITAL = 10_000.0
COST = TRANSACTION_COST_PCT


def _make_result(
    portfolio_values: list[float],
    benchmark_values: list[float],
    trades: list[dict] | None = None,
    starting_capital: float = CAPITAL,
    n_days_offset: int = 0,
) -> BacktestResult:
    """Build a synthetic BacktestResult from raw value lists."""
    n = len(portfolio_values)
    # Use a date range that spans roughly 1 year of business days
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    # Optionally offset start date so n_calendar_days is predictable
    if n_days_offset:
        dates = dates + pd.Timedelta(days=n_days_offset)

    portfolio = pd.Series(portfolio_values, index=dates, name="portfolio", dtype=float)
    benchmark = pd.Series(benchmark_values, index=dates, name="benchmark", dtype=float)

    if trades:
        trade_log = pd.DataFrame(trades, columns=TRADE_LOG_COLUMNS)
    else:
        trade_log = pd.DataFrame(columns=TRADE_LOG_COLUMNS)

    return BacktestResult(
        portfolio_history=portfolio,
        trade_log=trade_log,
        benchmark=benchmark,
        strategy_name="test",
        ticker="TEST",
        starting_capital=starting_capital,
        transaction_cost_pct=COST,
    )


def _trade(return_pct: float, holding_days: int = 10) -> dict:
    """Minimal trade row for trade_log."""
    return {
        "entry_date": pd.Timestamp("2020-01-02"),
        "entry_price": 100.0,
        "exit_date": pd.Timestamp("2020-01-12"),
        "exit_price": 100.0 * (1 + return_pct),
        "shares": CAPITAL / 100.0,
        "pnl": CAPITAL * return_pct,
        "return_pct": return_pct,
        "holding_days": holding_days,
    }


# ---------------------------------------------------------------------------
# TestOutputSchema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """compute_metrics() returns a dict with all required keys."""

    REQUIRED_KEYS = {
        "total_return", "annualized_return", "volatility", "sharpe_ratio",
        "max_drawdown", "win_rate", "trade_count", "avg_trade_return",
        "avg_holding_period", "benchmark_return", "outperformance",
    }

    def setup_method(self):
        result = _make_result(
            portfolio_values=[10_000, 10_100, 10_200],
            benchmark_values=[10_000, 10_050, 10_100],
            trades=[_trade(0.05)],
        )
        self.metrics = compute_metrics(result)

    def test_returns_dict(self):
        assert isinstance(self.metrics, dict)

    def test_all_required_keys_present(self):
        assert self.REQUIRED_KEYS == set(self.metrics.keys())

    def test_no_extra_keys(self):
        assert set(self.metrics.keys()) == self.REQUIRED_KEYS


# ---------------------------------------------------------------------------
# TestTotalReturn
# ---------------------------------------------------------------------------


class TestTotalReturn:
    """(final - capital) / capital"""

    def test_positive_return(self):
        r = _make_result([10_000, 10_500, 11_000], [10_000, 10_000, 10_000])
        m = compute_metrics(r)
        assert m["total_return"] == pytest.approx(0.10)

    def test_negative_return(self):
        r = _make_result([10_000, 9_500, 9_000], [10_000, 10_000, 10_000])
        m = compute_metrics(r)
        assert m["total_return"] == pytest.approx(-0.10)

    def test_flat_return(self):
        r = _make_result([10_000, 10_000, 10_000], [10_000, 10_000, 10_000])
        m = compute_metrics(r)
        assert m["total_return"] == pytest.approx(0.0)

    def test_uses_starting_capital_not_first_bar(self):
        """total_return denominator is starting_capital, not portfolio.iloc[0]."""
        # If portfolio.iloc[0] < starting_capital (entry cost already deducted),
        # total_return should still use starting_capital as the base.
        r = _make_result([9_990, 10_500], [9_990, 10_500], starting_capital=10_000)
        m = compute_metrics(r)
        assert m["total_return"] == pytest.approx((10_500 - 10_000) / 10_000)


# ---------------------------------------------------------------------------
# TestAnnualizedReturn
# ---------------------------------------------------------------------------


class TestAnnualizedReturn:
    """Geometric annualization: (1 + R)^(365/days) - 1."""

    def test_one_year_return_unchanged(self):
        """A 10% return over exactly 365 days annualizes to 10%."""
        n = 252  # ~1 year of business days
        vals = np.linspace(10_000, 11_000, n).tolist()
        bvals = [10_000.0] * n
        r = _make_result(vals, bvals)
        m = compute_metrics(r)
        # calendar days span: ~364 business days * 7/5 ≈ 364 cal days
        cal_days = (r.portfolio_history.index[-1] - r.portfolio_history.index[0]).days
        expected = (1.10) ** (365 / cal_days) - 1
        assert m["annualized_return"] == pytest.approx(expected, rel=1e-4)

    def test_positive_return_over_multiple_years(self):
        """Annualized return < total return when period > 1 year."""
        n = 504  # ~2 years
        vals = [10_000.0] * n
        vals[-1] = 12_000.0  # 20% total return over ~2 years
        bvals = [10_000.0] * n
        r = _make_result(vals, bvals)
        m = compute_metrics(r)
        # Annualized ≈ (1.20)^(1/2) - 1 ≈ 9.5%
        assert m["annualized_return"] < m["total_return"]
        assert m["annualized_return"] > 0


# ---------------------------------------------------------------------------
# TestVolatility
# ---------------------------------------------------------------------------


class TestVolatility:
    """Annualized std of daily portfolio returns."""

    def test_flat_portfolio_zero_volatility(self):
        r = _make_result([10_000] * 50, [10_000] * 50)
        m = compute_metrics(r)
        assert m["volatility"] == pytest.approx(0.0, abs=1e-10)

    def test_known_series_volatility(self):
        """Verify formula: vol = std(daily_pct_change) * sqrt(252)."""
        # Build a simple alternating series: 10000, 10100, 10000, 10100, ...
        n = 20
        vals = [10_000.0 if i % 2 == 0 else 10_100.0 for i in range(n)]
        r = _make_result(vals, [10_000.0] * n)
        daily_rets = pd.Series(vals).pct_change().dropna()
        expected_vol = daily_rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        m = compute_metrics(r)
        assert m["volatility"] == pytest.approx(expected_vol)

    def test_higher_variance_higher_volatility(self):
        """More volatile portfolio → higher volatility metric."""
        n = 30
        rng = np.random.default_rng(0)
        low_var = (10_000 + np.cumsum(rng.normal(0, 1, n))).tolist()
        high_var = (10_000 + np.cumsum(rng.normal(0, 50, n))).tolist()
        bvals = [10_000.0] * n
        r_low = _make_result(low_var, bvals)
        r_high = _make_result(high_var, bvals)
        assert compute_metrics(r_high)["volatility"] > compute_metrics(r_low)["volatility"]


# ---------------------------------------------------------------------------
# TestSharpeRatio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    """Sharpe = annualized_mean_daily_return / annualized_std (rf=0)."""

    def test_flat_portfolio_sharpe_is_nan(self):
        r = _make_result([10_000] * 30, [10_000] * 30)
        m = compute_metrics(r)
        assert math.isnan(m["sharpe_ratio"])

    def test_positive_return_positive_sharpe(self):
        n = 50
        vals = np.linspace(10_000, 12_000, n).tolist()
        r = _make_result(vals, [10_000.0] * n)
        m = compute_metrics(r)
        assert m["sharpe_ratio"] > 0

    def test_negative_return_negative_sharpe(self):
        n = 50
        vals = np.linspace(10_000, 8_000, n).tolist()
        r = _make_result(vals, [10_000.0] * n)
        m = compute_metrics(r)
        assert m["sharpe_ratio"] < 0

    def test_higher_return_same_vol_higher_sharpe(self):
        """Same volatility pattern, higher return → higher Sharpe."""
        n = 30
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 10, n)
        low_ret = (10_000 + np.cumsum(noise + 1)).tolist()
        high_ret = (10_000 + np.cumsum(noise + 5)).tolist()
        bvals = [10_000.0] * n
        r_low = _make_result(low_ret, bvals)
        r_high = _make_result(high_ret, bvals)
        assert compute_metrics(r_high)["sharpe_ratio"] > compute_metrics(r_low)["sharpe_ratio"]

    def test_sharpe_formula(self):
        """Verify exact formula: mean(daily_ret) * 252 / (std(daily_ret) * sqrt(252))."""
        n = 40
        vals = np.linspace(10_000, 11_500, n).tolist()
        r = _make_result(vals, [10_000.0] * n)
        daily = pd.Series(vals).pct_change().dropna()
        expected = (daily.mean() * TRADING_DAYS_PER_YEAR) / (
            daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        m = compute_metrics(r)
        assert m["sharpe_ratio"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestMaxDrawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    """Max drawdown ≤ 0; measures largest peak-to-trough decline."""

    def test_no_drawdown_flat_is_zero(self):
        r = _make_result([10_000] * 10, [10_000] * 10)
        m = compute_metrics(r)
        assert m["max_drawdown"] == pytest.approx(0.0)

    def test_strictly_increasing_no_drawdown(self):
        vals = np.linspace(10_000, 12_000, 20).tolist()
        r = _make_result(vals, vals)
        m = compute_metrics(r)
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown_value(self):
        """Peak 12000, trough 9000 → drawdown = (9000-12000)/12000 = -25%."""
        vals = [10_000, 11_000, 12_000, 10_000, 9_000, 10_000, 11_000]
        r = _make_result(vals, vals)
        m = compute_metrics(r)
        assert m["max_drawdown"] == pytest.approx(-0.25)

    def test_drawdown_is_non_positive(self):
        rng = np.random.default_rng(1)
        vals = (10_000 + np.cumsum(rng.normal(0, 100, 50))).tolist()
        r = _make_result(vals, vals)
        assert compute_metrics(r)["max_drawdown"] <= 0.0

    def test_drawdown_at_start_counted(self):
        """If the series immediately declines, that drawdown is captured."""
        vals = [10_000, 8_000, 9_000, 10_000]
        r = _make_result(vals, vals)
        m = compute_metrics(r)
        assert m["max_drawdown"] == pytest.approx(-0.20)


# ---------------------------------------------------------------------------
# TestWinRate
# ---------------------------------------------------------------------------


class TestWinRate:
    """win_rate = wins / trade_count where win = return_pct > 0."""

    def test_three_wins_two_losses(self):
        trades = [
            _trade(0.10), _trade(0.05), _trade(0.02),  # wins
            _trade(-0.03), _trade(-0.08),               # losses
        ]
        r = _make_result([10_000] * 10, [10_000] * 10, trades=trades)
        assert compute_metrics(r)["win_rate"] == pytest.approx(0.60)

    def test_all_wins(self):
        trades = [_trade(0.05), _trade(0.10), _trade(0.03)]
        r = _make_result([10_000] * 10, [10_000] * 10, trades=trades)
        assert compute_metrics(r)["win_rate"] == pytest.approx(1.0)

    def test_all_losses(self):
        trades = [_trade(-0.05), _trade(-0.02)]
        r = _make_result([10_000] * 10, [10_000] * 10, trades=trades)
        assert compute_metrics(r)["win_rate"] == pytest.approx(0.0)

    def test_breakeven_trade_is_not_a_win(self):
        """return_pct == 0 is NOT a win (condition is strictly > 0)."""
        trades = [_trade(0.0), _trade(0.05)]
        r = _make_result([10_000] * 10, [10_000] * 10, trades=trades)
        assert compute_metrics(r)["win_rate"] == pytest.approx(0.50)

    def test_no_trades_win_rate_is_nan(self):
        r = _make_result([10_000] * 5, [10_000] * 5)
        m = compute_metrics(r)
        assert math.isnan(m["win_rate"])


# ---------------------------------------------------------------------------
# TestTradeCount
# ---------------------------------------------------------------------------


class TestTradeCount:
    def test_zero_trades(self):
        r = _make_result([10_000] * 5, [10_000] * 5)
        assert compute_metrics(r)["trade_count"] == 0

    def test_one_trade(self):
        r = _make_result([10_000] * 5, [10_000] * 5, trades=[_trade(0.05)])
        assert compute_metrics(r)["trade_count"] == 1

    def test_five_trades(self):
        trades = [_trade(0.01 * i) for i in range(1, 6)]
        r = _make_result([10_000] * 5, [10_000] * 5, trades=trades)
        assert compute_metrics(r)["trade_count"] == 5

    def test_trade_count_is_int(self):
        r = _make_result([10_000] * 5, [10_000] * 5, trades=[_trade(0.05)])
        assert isinstance(compute_metrics(r)["trade_count"], int)


# ---------------------------------------------------------------------------
# TestAvgTradeReturn
# ---------------------------------------------------------------------------


class TestAvgTradeReturn:
    def test_known_average(self):
        trades = [_trade(0.10), _trade(0.20), _trade(-0.05)]
        r = _make_result([10_000] * 5, [10_000] * 5, trades=trades)
        expected = (0.10 + 0.20 - 0.05) / 3
        assert compute_metrics(r)["avg_trade_return"] == pytest.approx(expected)

    def test_no_trades_is_nan(self):
        r = _make_result([10_000] * 5, [10_000] * 5)
        assert math.isnan(compute_metrics(r)["avg_trade_return"])


# ---------------------------------------------------------------------------
# TestAvgHoldingPeriod
# ---------------------------------------------------------------------------


class TestAvgHoldingPeriod:
    def test_known_average(self):
        t1 = _trade(0.05, holding_days=10)
        t2 = _trade(0.05, holding_days=20)
        t3 = _trade(0.05, holding_days=30)
        r = _make_result([10_000] * 5, [10_000] * 5, trades=[t1, t2, t3])
        assert compute_metrics(r)["avg_holding_period"] == pytest.approx(20.0)

    def test_no_trades_is_nan(self):
        r = _make_result([10_000] * 5, [10_000] * 5)
        assert math.isnan(compute_metrics(r)["avg_holding_period"])


# ---------------------------------------------------------------------------
# TestBenchmarkReturn
# ---------------------------------------------------------------------------


class TestBenchmarkReturn:
    """benchmark_return = (benchmark_final - starting_capital) / starting_capital."""

    def test_10_pct_benchmark_gain(self):
        # benchmark goes from 9990 (≈ 10000 minus entry cost) to 10990
        # benchmark_return = (10990 - 10000) / 10000 = 9.9%
        r = _make_result(
            [10_000, 10_000],
            [9_990, 10_990],
            starting_capital=10_000,
        )
        m = compute_metrics(r)
        assert m["benchmark_return"] == pytest.approx((10_990 - 10_000) / 10_000)

    def test_benchmark_loss(self):
        r = _make_result([10_000, 10_000], [10_000, 8_000])
        m = compute_metrics(r)
        assert m["benchmark_return"] == pytest.approx(-0.20)

    def test_benchmark_return_independent_of_portfolio(self):
        """Changing portfolio values must not change benchmark_return."""
        bvals = [10_000, 11_000]
        r1 = _make_result([10_000, 10_000], bvals)
        r2 = _make_result([10_000, 12_000], bvals)
        assert compute_metrics(r1)["benchmark_return"] == pytest.approx(
            compute_metrics(r2)["benchmark_return"]
        )


# ---------------------------------------------------------------------------
# TestOutperformance
# ---------------------------------------------------------------------------


class TestOutperformance:
    """outperformance = total_return - benchmark_return."""

    def test_strategy_beats_benchmark(self):
        r = _make_result([10_000, 12_000], [10_000, 10_500])
        m = compute_metrics(r)
        assert m["outperformance"] == pytest.approx(0.20 - 0.05)

    def test_strategy_lags_benchmark(self):
        r = _make_result([10_000, 10_200], [10_000, 11_000])
        m = compute_metrics(r)
        assert m["outperformance"] == pytest.approx(0.02 - 0.10)
        assert m["outperformance"] < 0

    def test_outperformance_equals_difference(self):
        r = _make_result([10_000, 11_000], [10_000, 10_500])
        m = compute_metrics(r)
        assert m["outperformance"] == pytest.approx(
            m["total_return"] - m["benchmark_return"]
        )


# ---------------------------------------------------------------------------
# TestMetricsToSeries
# ---------------------------------------------------------------------------


class TestMetricsToSeries:
    """metrics_to_series() formats metrics as a display-ready pd.Series."""

    def setup_method(self):
        trades = [_trade(0.10), _trade(-0.05)]
        r = _make_result([10_000, 10_500, 11_000], [10_000, 10_200, 10_400], trades=trades)
        self.m = compute_metrics(r)
        self.s = metrics_to_series(self.m)

    def test_returns_series(self):
        assert isinstance(self.s, pd.Series)

    def test_total_return_scaled_to_pct(self):
        assert self.s["Total Return (%)"] == pytest.approx(self.m["total_return"] * 100, rel=1e-4)

    def test_sharpe_ratio_not_scaled(self):
        if not math.isnan(self.m["sharpe_ratio"]):
            assert self.s["Sharpe Ratio"] == pytest.approx(self.m["sharpe_ratio"], rel=1e-4)

    def test_trade_count_is_whole_number(self):
        # pd.Series coerces to np.float64; check the value is an integer quantity
        assert self.s["Trade Count"] == int(self.s["Trade Count"])

    def test_all_eleven_metrics_present(self):
        assert len(self.s) == 11


# ---------------------------------------------------------------------------
# TestEndToEndIntegration
# ---------------------------------------------------------------------------


class TestEndToEndIntegration:
    """Full pipeline: data → indicators → strategy → engine → metrics."""

    @pytest.fixture(scope="class")
    def spy_metrics(self):
        from src.backtester.engine import run_backtest
        from src.data.loader import load_ohlcv
        from src.indicators.engine import add_indicators
        from src.strategies.ma_crossover import generate_signals

        df = load_ohlcv("SPY", 5)
        signals = generate_signals(add_indicators(df))
        result = run_backtest(signals, ticker="SPY")
        return compute_metrics(result)

    def test_all_keys_present(self, spy_metrics):
        required = {
            "total_return", "annualized_return", "volatility", "sharpe_ratio",
            "max_drawdown", "win_rate", "trade_count", "avg_trade_return",
            "avg_holding_period", "benchmark_return", "outperformance",
        }
        assert required == set(spy_metrics.keys())

    def test_max_drawdown_is_non_positive(self, spy_metrics):
        assert spy_metrics["max_drawdown"] <= 0.0

    def test_volatility_is_non_negative(self, spy_metrics):
        assert spy_metrics["volatility"] >= 0.0

    def test_trade_count_is_int(self, spy_metrics):
        assert isinstance(spy_metrics["trade_count"], int)

    def test_outperformance_equals_difference(self, spy_metrics):
        assert spy_metrics["outperformance"] == pytest.approx(
            spy_metrics["total_return"] - spy_metrics["benchmark_return"],
            abs=1e-10,
        )

    def test_all_five_strategies_produce_metrics(self):
        """All five strategies produce a complete metrics dict without error."""
        from src.backtester.engine import run_backtest, run_hybrid_backtest
        from src.data.loader import load_ohlcv
        from src.indicators.engine import add_indicators
        from src.strategies.breakout_a import generate_signals as s3
        from src.strategies.breakout_b import generate_signals as s4
        from src.strategies.hybrid import ALLOCATIONS
        from src.strategies.hybrid import generate_signals as s5
        from src.strategies.ma_crossover import generate_signals as s1
        from src.strategies.rsi_mean_reversion import generate_signals as s2

        df = load_ohlcv("SPY", 5)
        ind = add_indicators(df)

        standalone = [
            run_backtest(s1(ind)),
            run_backtest(s2(ind)),
            run_backtest(s3(ind)),
            run_backtest(s4(ind)),
        ]
        hybrid = run_hybrid_backtest(s5(ind), ALLOCATIONS)

        for result in standalone + [hybrid]:
            m = compute_metrics(result)
            assert len(m) == 11
            assert m["max_drawdown"] <= 0.0
            assert not math.isnan(m["total_return"])
            assert not math.isnan(m["benchmark_return"])

    def test_metrics_to_series_works_end_to_end(self, spy_metrics):
        s = metrics_to_series(spy_metrics)
        assert isinstance(s, pd.Series)
        assert len(s) == 11
        assert s["Trade Count"] >= 0
