"""
Reusable chart generation utilities for the Strategy Backtesting Engine.

All chart functions return ``plotly.graph_objects.Figure`` instances, making
them renderable in notebooks, saved as HTML/PNG, or serialised to JSON for
the website layer.

Public interface
----------------
equity_curve_chart(equity_df, strategy_name, ticker, timeframe_years)
    Portfolio equity vs buy-and-hold benchmark on dual y-axis.

drawdown_chart(equity_df, strategy_name, ticker, timeframe_years)
    Rolling drawdown from peak for the strategy portfolio.

price_with_signals_chart(price_df, signals_df, ticker)
    Closing price with entry (▲) and exit (▼) markers overlaid.

metrics_summary_figure(metrics, strategy_name, ticker, timeframe_years)
    Plotly Table figure showing the 11-key metrics dict.

build_chart_bundle(strategy_key, ticker, timeframe_years, output_dir)
    Convenience wrapper: loads saved results and returns a dict with
    serialisable chart dicts ready for the website layer.

figure_to_dict(fig)
    Serialise a Figure to a plain dict (JSON-safe via plotly's own encoder).

figure_to_json(fig)
    Serialise a Figure to a JSON string.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

from src.config.settings import REPORTS_DIR

# ---------------------------------------------------------------------------
# Colour palette — used consistently across all charts
# ---------------------------------------------------------------------------

_PORTFOLIO_COLOR = "#2563EB"   # blue
_BENCHMARK_COLOR = "#9CA3AF"   # grey
_DRAWDOWN_COLOR  = "#EF4444"   # red
_ENTRY_COLOR     = "#22C55E"   # green
_EXIT_COLOR      = "#EF4444"   # red

# Per-sleeve colours for hybrid chart
_SLEEVE_COLORS = {
    "breakout_b":         "#2563EB",  # blue
    "ma_crossover":       "#F59E0B",  # amber
    "rsi_mean_reversion": "#22C55E",  # green
}
_SLEEVE_LABELS = {
    "breakout_b":         "Breakout B (50%)",
    "ma_crossover":       "MA Crossover (30%)",
    "rsi_mean_reversion": "RSI Mean Rev (20%)",
}

_LAYOUT_DEFAULTS: dict[str, Any] = {
    "template": "plotly_white",
    "font": {"family": "Inter, sans-serif", "size": 12},
    "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
    "hovermode": "x unified",
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "right", "x": 1},
}


def _base_layout(title: str, **overrides: Any) -> dict:
    layout = {**_LAYOUT_DEFAULTS, "title": {"text": title, "x": 0.5}}
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_drawdown(portfolio: pd.Series) -> pd.Series:
    """Return rolling drawdown from peak as a non-positive decimal series."""
    peak = portfolio.cummax()
    return (portfolio - peak) / peak


def _strategy_label(strategy_key: str) -> str:
    return strategy_key.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def equity_curve_chart(
    equity_df: pd.DataFrame,
    strategy_name: str,
    ticker: str,
    timeframe_years: int,
    trade_log: "pd.DataFrame | None" = None,
) -> go.Figure:
    """Return a Plotly Figure: portfolio equity curve vs buy-and-hold benchmark.

    Parameters
    ----------
    equity_df:
        DataFrame with a ``DatetimeIndex`` and columns ``portfolio`` and
        ``benchmark``.  Typically the output of ``load_equity_series()``.
    strategy_name:
        Human-readable strategy label for the legend.
    ticker:
        Asset symbol (e.g. ``"SPY"``).
    timeframe_years:
        Lookback window in years, used in the chart title.
    trade_log:
        Optional DataFrame with ``entry_date`` and ``exit_date`` columns
        (datetime).  When provided, buy/sell markers are overlaid on the
        equity curve at the corresponding portfolio values.
    """
    dates = equity_df.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity_df["portfolio"],
        name=strategy_name,
        line={"color": _PORTFOLIO_COLOR, "width": 2},
        hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates,
        y=equity_df["benchmark"],
        name="Buy & Hold",
        line={"color": _BENCHMARK_COLOR, "width": 1.5, "dash": "dot"},
        hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>",
    ))

    if trade_log is not None and not trade_log.empty:
        portfolio_series = equity_df["portfolio"]

        entry_dates = pd.to_datetime(trade_log["entry_date"])
        entry_vals = portfolio_series.reindex(entry_dates, method="nearest")
        fig.add_trace(go.Scatter(
            x=entry_vals.index,
            y=entry_vals.values,
            mode="markers",
            name="Buy",
            marker={"symbol": "triangle-up", "color": _ENTRY_COLOR,
                    "size": 10, "line": {"color": "white", "width": 1}},
            hovertemplate="%{y:$,.0f}<extra>Buy</extra>",
        ))

        exits = trade_log.dropna(subset=["exit_date"])
        if not exits.empty:
            exit_dates = pd.to_datetime(exits["exit_date"])
            exit_vals = portfolio_series.reindex(exit_dates, method="nearest")
            fig.add_trace(go.Scatter(
                x=exit_vals.index,
                y=exit_vals.values,
                mode="markers",
                name="Sell",
                marker={"symbol": "triangle-down", "color": _EXIT_COLOR,
                        "size": 10, "line": {"color": "white", "width": 1}},
                hovertemplate="%{y:$,.0f}<extra>Sell</extra>",
            ))

    title = f"{ticker} — {_strategy_label(strategy_name)} vs Buy & Hold ({timeframe_years}y)"
    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Date",
    )
    return fig


def drawdown_chart(
    equity_df: pd.DataFrame,
    strategy_name: str,
    ticker: str,
    timeframe_years: int,
) -> go.Figure:
    """Return a Plotly Figure showing rolling drawdown from peak.

    Parameters
    ----------
    equity_df:
        DataFrame with ``DatetimeIndex`` and at least a ``portfolio`` column.
    strategy_name, ticker, timeframe_years:
        Used for the chart title.
    """
    dd = _compute_drawdown(equity_df["portfolio"]) * 100   # convert to %

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df.index,
        y=dd,
        name="Drawdown",
        fill="tozeroy",
        line={"color": _DRAWDOWN_COLOR, "width": 1.5},
        fillcolor="rgba(239,68,68,0.15)",
        hovertemplate="%{y:.2f}%<extra>Drawdown</extra>",
    ))

    title = f"{ticker} — {_strategy_label(strategy_name)} Drawdown ({timeframe_years}y)"
    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Drawdown (%)",
        yaxis_tickformat=".1f",
        xaxis_title="Date",
    )
    return fig


def price_with_signals_chart(
    price_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    ticker: str,
) -> go.Figure:
    """Return a Plotly Figure: closing price with entry/exit signal markers.

    Parameters
    ----------
    price_df:
        OHLCV DataFrame with at least a ``close`` column and a
        ``DatetimeIndex``.
    signals_df:
        DataFrame (same index as ``price_df``) with boolean columns
        ``signal_entry`` and ``signal_exit``.  Typically the output of any
        ``generate_signals()`` function.
    ticker:
        Asset symbol used in the chart title.
    """
    fig = go.Figure()

    # Closing price line
    fig.add_trace(go.Scatter(
        x=price_df.index,
        y=price_df["close"],
        name="Close",
        line={"color": "#374151", "width": 1.5},
        hovertemplate="%{y:$.2f}<extra>Close</extra>",
    ))

    # Entry markers
    entry_mask = signals_df["signal_entry"]
    if entry_mask.any():
        fig.add_trace(go.Scatter(
            x=price_df.index[entry_mask],
            y=price_df["close"][entry_mask],
            mode="markers",
            name="Entry",
            marker={"symbol": "triangle-up", "color": _ENTRY_COLOR,
                    "size": 10, "line": {"color": "white", "width": 1}},
            hovertemplate="%{y:$.2f}<extra>Entry</extra>",
        ))

    # Exit markers
    exit_mask = signals_df["signal_exit"]
    if exit_mask.any():
        fig.add_trace(go.Scatter(
            x=price_df.index[exit_mask],
            y=price_df["close"][exit_mask],
            mode="markers",
            name="Exit",
            marker={"symbol": "triangle-down", "color": _EXIT_COLOR,
                    "size": 10, "line": {"color": "white", "width": 1}},
            hovertemplate="%{y:$.2f}<extra>Exit</extra>",
        ))

    title = f"{ticker} — Price with Entry / Exit Signals"
    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Price ($)",
        xaxis_title="Date",
    )
    return fig


# ---------------------------------------------------------------------------
# Strategy-specific signals charts
# ---------------------------------------------------------------------------

def _add_price_markers(
    fig: go.Figure,
    trade_log: "pd.DataFrame | None",
    row: "int | None" = None,
    col: "int | None" = None,
    show_legend: bool = True,
) -> None:
    """Add buy/sell markers at execution prices from a trade_log onto a figure.

    Pass ``row`` and ``col`` only for figures created with ``make_subplots``.
    """
    if trade_log is None or trade_log.empty:
        return

    subplot_kwargs: dict = {"row": row, "col": col} if row is not None else {}

    entry_dates = pd.to_datetime(trade_log["entry_date"])
    entry_prices = trade_log["entry_price"].astype(float).values
    fig.add_trace(go.Scatter(
        x=entry_dates, y=entry_prices,
        mode="markers", name="Buy",
        marker={"symbol": "triangle-up", "color": _ENTRY_COLOR, "size": 10,
                "line": {"color": "white", "width": 1}},
        hovertemplate="%{y:$.2f}<extra>Buy</extra>",
        showlegend=show_legend,
    ), **subplot_kwargs)

    exits = trade_log.dropna(subset=["exit_date"])
    if not exits.empty:
        exit_dates = pd.to_datetime(exits["exit_date"])
        exit_prices = exits["exit_price"].astype(float).values
        fig.add_trace(go.Scatter(
            x=exit_dates, y=exit_prices,
            mode="markers", name="Sell",
            marker={"symbol": "triangle-down", "color": _EXIT_COLOR, "size": 10,
                    "line": {"color": "white", "width": 1}},
            hovertemplate="%{y:$.2f}<extra>Sell</extra>",
            showlegend=show_legend,
        ), **subplot_kwargs)


def _subplot_layout(title: str, height: int) -> dict:
    return {
        "template": "plotly_white",
        "title": {"text": title, "x": 0.5},
        "font": {"family": "Inter, sans-serif", "size": 12},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02,
                   "xanchor": "right", "x": 1},
        "margin": {"l": 60, "r": 30, "t": 80, "b": 50},
        "height": height,
    }


def ma_crossover_signals_chart(
    chart_df: pd.DataFrame,
    trade_log: "pd.DataFrame | None",
    ticker: str,
    timeframe_years: int,
) -> go.Figure:
    """Price + SMA 50 + SMA 200 + buy/sell markers for MA Crossover strategy."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["close"],
        name="Close", line={"color": "#374151", "width": 1.5},
        hovertemplate="%{y:$.2f}<extra>Close</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["sma_50"],
        name="SMA 50", line={"color": "#F59E0B", "width": 1.5},
        hovertemplate="%{y:$.2f}<extra>SMA 50</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["sma_200"],
        name="SMA 200", line={"color": "#2563EB", "width": 1.5},
        hovertemplate="%{y:$.2f}<extra>SMA 200</extra>",
    ))

    _add_price_markers(fig, trade_log)

    title = f"{ticker} — MA Crossover: SMA 50 / SMA 200 ({timeframe_years}y)"
    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Price ($)",
        xaxis_title="Date",
        height=400,
    )
    return fig


def rsi_signals_chart(
    chart_df: pd.DataFrame,
    trade_log: "pd.DataFrame | None",
    ticker: str,
    timeframe_years: int,
) -> go.Figure:
    """Price with buy/sell markers (top) + RSI 14 with threshold lines (bottom)."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.68, 0.32],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Price", "RSI (14)"),
    )

    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["close"],
        name="Close", line={"color": "#374151", "width": 1.5},
        hovertemplate="%{y:$.2f}<extra>Close</extra>",
    ), row=1, col=1)

    _add_price_markers(fig, trade_log, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["rsi_14"],
        name="RSI 14", line={"color": "#7C3AED", "width": 1.5},
        hovertemplate="%{y:.1f}<extra>RSI 14</extra>",
    ), row=2, col=1)

    # Entry threshold (RSI < 30 triggers buy)
    fig.add_hline(y=30, line_dash="dash", line_color="#EF4444", line_width=1,
                  annotation_text="30 — entry", annotation_position="bottom right",
                  row=2, col=1)
    # Exit threshold (RSI > 55 triggers sell)
    fig.add_hline(y=55, line_dash="dash", line_color="#22C55E", line_width=1,
                  annotation_text="55 — exit", annotation_position="top right",
                  row=2, col=1)

    title = f"{ticker} — RSI Mean Reversion: RSI(14) entry &lt;30 / exit &gt;55 ({timeframe_years}y)"
    fig.update_layout(**_subplot_layout(title, height=480))
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    return fig


def breakout_a_signals_chart(
    chart_df: pd.DataFrame,
    trade_log: "pd.DataFrame | None",
    ticker: str,
    timeframe_years: int,
) -> go.Figure:
    """Price + 20-day rolling high (entry) + 10-day rolling low (exit) + markers."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["close"],
        name="Close", line={"color": "#374151", "width": 1.5},
        hovertemplate="%{y:$.2f}<extra>Close</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["rolling_high_20"],
        name="20-Day High (entry threshold)",
        line={"color": "#22C55E", "width": 1.2, "dash": "dot"},
        hovertemplate="%{y:$.2f}<extra>20d High</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["rolling_low_10"],
        name="10-Day Low (exit threshold)",
        line={"color": "#EF4444", "width": 1.2, "dash": "dot"},
        hovertemplate="%{y:$.2f}<extra>10d Low</extra>",
    ))

    _add_price_markers(fig, trade_log)

    title = f"{ticker} — Breakout A: Price vs Rolling High/Low Levels ({timeframe_years}y)"
    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Price ($)",
        xaxis_title="Date",
        height=400,
    )
    return fig


def breakout_b_signals_chart(
    chart_df: pd.DataFrame,
    trade_log: "pd.DataFrame | None",
    ticker: str,
    timeframe_years: int,
) -> go.Figure:
    """Price + breakout levels + SMA 200 + markers (top); volume + 1.5× avg threshold (bottom)."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Price", "Volume"),
    )

    # Price
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["close"],
        name="Close", line={"color": "#374151", "width": 1.5},
        hovertemplate="%{y:$.2f}<extra>Close</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["rolling_high_20"],
        name="20-Day High (entry)",
        line={"color": "#22C55E", "width": 1.2, "dash": "dot"},
        hovertemplate="%{y:$.2f}<extra>20d High</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["rolling_low_10"],
        name="10-Day Low (exit)",
        line={"color": "#EF4444", "width": 1.2, "dash": "dot"},
        hovertemplate="%{y:$.2f}<extra>10d Low</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["sma_200"],
        name="SMA 200 (trend filter)",
        line={"color": "#2563EB", "width": 1.5, "dash": "dash"},
        hovertemplate="%{y:$.2f}<extra>SMA 200</extra>",
    ), row=1, col=1)

    _add_price_markers(fig, trade_log, row=1, col=1)

    # Volume bars
    fig.add_trace(go.Bar(
        x=chart_df.index, y=chart_df["volume"],
        name="Volume", marker_color="rgba(156,163,175,0.55)",
        hovertemplate="%{y:,.0f}<extra>Volume</extra>",
        showlegend=True,
    ), row=2, col=1)

    # 1.5× avg volume — the confirmation threshold that must be exceeded for entry
    threshold_vol = chart_df["avg_volume_20"] * 1.5
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=threshold_vol,
        name="1.5× Avg Vol (confirm threshold)",
        line={"color": "#F59E0B", "width": 1.5, "dash": "dash"},
        hovertemplate="%{y:,.0f}<extra>1.5× Avg Vol</extra>",
    ), row=2, col=1)

    title = (
        f"{ticker} — Breakout B: Filtered Breakout ({timeframe_years}y)"
        f"<br><sup>Entry requires: Price &gt; 20d High · Vol &gt; 1.5× Avg · "
        f"Close &gt; SMA 200 · RSI ∈ [55,75]</sup>"
    )
    fig.update_layout(**_subplot_layout(title, height=510))
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


_IDLE_CASH_COLOR = "rgba(229,231,235,0.7)"

_SLEEVE_FILL_COLORS = {
    "breakout_b":         "rgba(37,99,235,0.65)",
    "ma_crossover":       "rgba(245,158,11,0.65)",
    "rsi_mean_reversion": "rgba(34,197,94,0.65)",
}


def _compute_allocation_series(
    equity_df: pd.DataFrame,
    trade_log: "pd.DataFrame | None",
    allocations: dict,
) -> pd.DataFrame:
    """Return a DataFrame (same index as equity_df) showing % of total capital
    deployed per sleeve on each day, plus idle_cash to always sum to 100."""
    idx = equity_df.index
    alloc = pd.DataFrame(0.0, index=idx, columns=list(allocations.keys()))

    if trade_log is not None and not trade_log.empty and "sub_strategy" in trade_log.columns:
        for sub_key, weight in allocations.items():
            sub_trades = trade_log[trade_log["sub_strategy"] == sub_key]
            for _, trade in sub_trades.iterrows():
                entry = pd.Timestamp(trade["entry_date"])
                exit_ = pd.Timestamp(trade["exit_date"])
                mask = (idx >= entry) & (idx <= exit_)
                alloc.loc[mask, sub_key] = weight * 100.0

    alloc["idle_cash"] = 100.0 - alloc.sum(axis=1)
    return alloc


def hybrid_signals_chart(
    equity_df: pd.DataFrame,
    trade_log: "pd.DataFrame | None",
    sleeve_equity: "dict[str, pd.Series] | None",
    ticker: str,
    timeframe_years: int,
    allocations: "dict | None" = None,
) -> go.Figure:
    """Hybrid: equity + sleeve curves + markers (top) + capital allocation layer (bottom)."""
    if allocations is None:
        allocations = {"breakout_b": 0.5, "ma_crossover": 0.3, "rsi_mean_reversion": 0.2}

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.62, 0.38],
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("Sleeve Performance", "Capital Deployed (%)"),
    )

    # ---- Row 1: equity + sleeve + markers --------------------------------
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["portfolio"],
        name="Hybrid Portfolio",
        line={"color": _PORTFOLIO_COLOR, "width": 2.5},
        hovertemplate="%{y:$,.0f}<extra>Hybrid Portfolio</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["benchmark"],
        name="Buy & Hold",
        line={"color": _BENCHMARK_COLOR, "width": 1.5, "dash": "dot"},
        hovertemplate="%{y:$,.0f}<extra>Buy &amp; Hold</extra>",
    ), row=1, col=1)

    if sleeve_equity:
        for key, series in sleeve_equity.items():
            color = _SLEEVE_COLORS.get(key, "#9CA3AF")
            label = _SLEEVE_LABELS.get(key, key)
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values,
                name=label,
                line={"color": color, "width": 1, "dash": "dot"},
                opacity=0.65,
                hovertemplate="%{y:$,.0f}<extra>" + label + "</extra>",
            ), row=1, col=1)

    if trade_log is not None and not trade_log.empty and "sub_strategy" in trade_log.columns:
        for sub_key in list(allocations.keys()):
            sub_trades = trade_log[trade_log["sub_strategy"] == sub_key]
            if sub_trades.empty:
                continue
            color = _SLEEVE_COLORS.get(sub_key, "#9CA3AF")
            label = _SLEEVE_LABELS.get(sub_key, sub_key)

            entry_dates = pd.to_datetime(sub_trades["entry_date"])
            entry_vals = equity_df["portfolio"].reindex(entry_dates, method="nearest")
            fig.add_trace(go.Scatter(
                x=entry_vals.index, y=entry_vals.values,
                mode="markers", name=f"{label} Buy",
                marker={"symbol": "triangle-up", "color": color, "size": 9,
                        "line": {"color": "white", "width": 1}},
                hovertemplate="%{y:$,.0f}<extra>" + label + " Buy</extra>",
            ), row=1, col=1)

            exits = sub_trades.dropna(subset=["exit_date"])
            if not exits.empty:
                exit_dates = pd.to_datetime(exits["exit_date"])
                exit_vals = equity_df["portfolio"].reindex(exit_dates, method="nearest")
                fig.add_trace(go.Scatter(
                    x=exit_vals.index, y=exit_vals.values,
                    mode="markers", name=f"{label} Sell",
                    marker={"symbol": "triangle-down", "color": color, "size": 9,
                            "line": {"color": "white", "width": 1}},
                    hovertemplate="%{y:$,.0f}<extra>" + label + " Sell</extra>",
                ), row=1, col=1)

    # ---- Row 2: stacked capital-allocation area --------------------------
    alloc_df = _compute_allocation_series(equity_df, trade_log, allocations)

    # Draw from bottom to top: active sleeves first, idle_cash on top
    sleeve_order = list(allocations.keys())
    for sub_key in sleeve_order:
        label = _SLEEVE_LABELS.get(sub_key, sub_key)
        fill_color = _SLEEVE_FILL_COLORS.get(sub_key, "rgba(156,163,175,0.5)")
        line_color = _SLEEVE_COLORS.get(sub_key, "#9CA3AF")
        fig.add_trace(go.Scatter(
            x=alloc_df.index, y=alloc_df[sub_key],
            name=label + " (deployed)",
            mode="lines",
            line={"width": 0, "color": line_color},
            fillcolor=fill_color,
            stackgroup="alloc",
            hovertemplate="%{y:.0f}%<extra>" + label + "</extra>",
            showlegend=True,
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=alloc_df.index, y=alloc_df["idle_cash"],
        name="Idle Cash",
        mode="lines",
        line={"width": 0},
        fillcolor=_IDLE_CASH_COLOR,
        stackgroup="alloc",
        hovertemplate="%{y:.0f}%<extra>Idle Cash</extra>",
        showlegend=True,
    ), row=2, col=1)

    title = f"{ticker} — Hybrid Allocation: Sleeve Breakdown ({timeframe_years}y)"
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.5},
        font={"family": "Inter, sans-serif", "size": 12},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "right", "x": 1, "font": {"size": 9}},
        margin={"l": 60, "r": 30, "t": 80, "b": 50},
        height=560,
    )
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Capital (%)", range=[0, 100], row=2, col=1)
    return fig


def _build_signals_chart(
    strategy_key: str,
    chart_data: "dict | None",
    trade_log_df: "pd.DataFrame | None",
    equity_df: pd.DataFrame,
    ticker: str,
    timeframe_years: int,
) -> "go.Figure | None":
    """Dispatch to the correct strategy-specific signals chart. Returns None if no data."""
    if chart_data is None:
        return None

    if strategy_key == "hybrid":
        sleeve_equity_raw = chart_data.get("sleeve_equity")
        if not sleeve_equity_raw:
            return None
        sleeve_equity: dict[str, pd.Series] = {}
        for key, records in sleeve_equity_raw.items():
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            sleeve_equity[key] = df["value"].rename(key)
        allocations = chart_data.get(
            "allocations",
            {"breakout_b": 0.5, "ma_crossover": 0.3, "rsi_mean_reversion": 0.2},
        )
        return hybrid_signals_chart(
            equity_df, trade_log_df, sleeve_equity, ticker, timeframe_years,
            allocations=allocations,
        )

    timeseries = chart_data.get("timeseries", [])
    if not timeseries:
        return None

    chart_df = pd.DataFrame(timeseries)
    chart_df["date"] = pd.to_datetime(chart_df["date"])
    chart_df = chart_df.set_index("date")

    if strategy_key == "ma_crossover":
        return ma_crossover_signals_chart(chart_df, trade_log_df, ticker, timeframe_years)
    if strategy_key == "rsi_mean_reversion":
        return rsi_signals_chart(chart_df, trade_log_df, ticker, timeframe_years)
    if strategy_key == "breakout_a":
        return breakout_a_signals_chart(chart_df, trade_log_df, ticker, timeframe_years)
    if strategy_key == "breakout_b":
        return breakout_b_signals_chart(chart_df, trade_log_df, ticker, timeframe_years)

    return None


# ---------------------------------------------------------------------------
# Hybrid sleeve metrics Plotly table
# ---------------------------------------------------------------------------

def sleeve_metrics_table(
    sleeve_data: list,
    ticker: str,
    timeframe_years: int,
) -> go.Figure:
    """Return a Plotly Table summarising per-sleeve contributions and trade metrics.

    Parameters
    ----------
    sleeve_data:
        List of dicts (one per sleeve), each with keys:
        name, allocation, contribution, win_rate, avg_win, avg_loss, trade_count.
    """
    _pct  = lambda v: f"{v*100:.1f}%" if v is not None and not _isnan(v) else "—"
    _spct = lambda v: (f"+{v*100:.2f}%" if v >= 0 else f"{v*100:.2f}%") if v is not None and not _isnan(v) else "—"
    _int  = lambda v: str(int(v)) if v is not None else "—"

    headers = ["Sleeve", "Allocation", "Contrib", "Win Rate", "Avg Win", "Avg Loss", "Trades"]
    col_align = ["left", "right", "right", "right", "right", "right", "right"]

    rows: list[list] = []
    for s in sleeve_data:
        rows.append([
            s["name"],
            _pct(s["allocation"]),
            _spct(s["contribution"]),
            _pct(s["win_rate"]),
            _spct(s["avg_win"]) if s["avg_win"] is not None and not _isnan(s["avg_win"]) else "—",
            _spct(s["avg_loss"]) if s["avg_loss"] is not None and not _isnan(s["avg_loss"]) else "—",
            _int(s["trade_count"]),
        ])

    cell_values = [[r[i] for r in rows] for i in range(len(headers))]

    sleeve_bg = ["#EFF6FF", "#FFFBEB", "#F0FDF4"]
    row_fill = [sleeve_bg[i % len(sleeve_bg)] for i in range(len(rows))]
    cell_fill = [row_fill for _ in headers]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#1E3A5F",
            font={"color": "white", "size": 12},
            align=col_align,
            height=36,
        ),
        cells=dict(
            values=cell_values,
            fill_color=cell_fill,
            align=col_align,
            font={"size": 12},
            height=32,
        ),
    ))
    fig.update_layout(
        title={"text": f"{ticker} — Hybrid Sleeve Breakdown ({timeframe_years}y)", "x": 0.5},
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 10},
        height=260,
    )
    return fig


def _isnan(v: Any) -> bool:
    try:
        return v is None or (isinstance(v, float) and __import__("math").isnan(v))
    except Exception:
        return False


def metrics_summary_figure(
    metrics: dict,
    strategy_name: str,
    ticker: str,
    timeframe_years: int,
) -> go.Figure:
    """Return a Plotly Table Figure summarising the 11 standard metrics.

    Parameters
    ----------
    metrics:
        Dict as returned by ``compute_metrics()`` or ``load_metrics()``.
    strategy_name, ticker, timeframe_years:
        Displayed in the table header.
    """
    _PCT = {
        "total_return", "annualized_return", "volatility",
        "max_drawdown", "win_rate", "avg_trade_return",
        "benchmark_return", "outperformance",
    }

    labels = {
        "total_return": "Total Return",
        "annualized_return": "Annualized Return",
        "volatility": "Volatility",
        "sharpe_ratio": "Sharpe Ratio",
        "max_drawdown": "Max Drawdown",
        "win_rate": "Win Rate",
        "trade_count": "Trade Count",
        "avg_trade_return": "Avg Trade Return",
        "avg_holding_period": "Avg Holding Period (days)",
        "benchmark_return": "Benchmark Return",
        "outperformance": "Outperformance vs B&H",
    }

    rows: list[tuple[str, str]] = []
    for key, label in labels.items():
        raw = metrics.get(key)
        if raw is None:
            formatted = "—"
        elif key == "trade_count":
            formatted = str(int(raw))
        elif key == "avg_holding_period":
            formatted = f"{raw:.1f}"
        elif key == "sharpe_ratio":
            formatted = f"{raw:.2f}"
        elif key in _PCT:
            formatted = f"{raw * 100:+.2f}%" if key in {"outperformance"} else f"{raw * 100:.2f}%"
        else:
            formatted = str(raw)
        rows.append((label, formatted))

    metric_names = [r[0] for r in rows]
    metric_vals  = [r[1] for r in rows]

    header_text = (
        f"{_strategy_label(strategy_name)} · {ticker} · {timeframe_years}y"
    )

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{header_text}</b>", ""],
            fill_color="#1E3A5F",
            font={"color": "white", "size": 13},
            align="left",
            height=36,
        ),
        cells=dict(
            values=[metric_names, metric_vals],
            fill_color=[["#F9FAFB", "#FFFFFF"] * 6, "white"],
            align=["left", "right"],
            font={"size": 12},
            height=28,
        ),
    ))
    fig.update_layout(
        title={"text": "Performance Metrics", "x": 0.5},
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 50, "b": 10},
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def figure_to_dict(fig: go.Figure) -> dict:
    """Return a JSON-safe dict representation of *fig*.

    Uses Plotly's own ``PlotlyJSONEncoder`` so numpy arrays and special types
    are handled correctly.  The result can be passed directly to
    ``json.dumps()`` without further conversion.
    """
    return json.loads(fig.to_json())


def figure_to_json(fig: go.Figure) -> str:
    """Return a compact JSON string representation of *fig*."""
    return fig.to_json()


# ---------------------------------------------------------------------------
# Convenience bundle builder
# ---------------------------------------------------------------------------

def build_chart_bundle(
    strategy_key: str,
    ticker: str,
    timeframe_years: int,
    output_dir: str = REPORTS_DIR,
) -> dict[str, Any]:
    """Load a saved result and return a chart bundle ready for the website layer.

    The bundle is a plain ``dict`` with JSON-serialisable values:

    .. code-block:: python

        {
            "equity":   { ... plotly figure dict ... },
            "drawdown": { ... plotly figure dict ... },
            "metrics":  { ... metrics dict (11 keys) ... },
            "metadata": { ... strategy/ticker/timeframe/capital/etc. ... },
        }

    Parameters
    ----------
    strategy_key:
        One of the five strategy keys (e.g. ``"ma_crossover"``).
    ticker:
        Asset symbol (e.g. ``"SPY"``).
    timeframe_years:
        Lookback window in years.
    output_dir:
        Root directory for result files.  Defaults to ``REPORTS_DIR``.

    Raises
    ------
    FileNotFoundError
        If the result file does not exist.
    """
    from src.research.runner import load_equity_series, load_result

    payload    = load_result(strategy_key, ticker, timeframe_years, output_dir)
    equity_df  = load_equity_series(strategy_key, ticker, timeframe_years, output_dir)
    metrics    = payload["metrics"]
    metadata   = payload["metadata"]

    label = _strategy_label(strategy_key)

    trade_log_records = payload.get("trade_log", [])
    if trade_log_records:
        trade_log_df = pd.DataFrame(trade_log_records)
        trade_log_df["entry_date"] = pd.to_datetime(trade_log_df["entry_date"])
        trade_log_df["exit_date"] = pd.to_datetime(trade_log_df["exit_date"])
    else:
        trade_log_df = None

    equity_fig  = equity_curve_chart(equity_df, label, ticker, timeframe_years, trade_log=trade_log_df)
    metrics_fig = metrics_summary_figure(metrics, strategy_key, ticker, timeframe_years)

    chart_data_raw = payload.get("chart_data")
    signals_fig = _build_signals_chart(
        strategy_key, chart_data_raw, trade_log_df, equity_df, ticker, timeframe_years
    )

    bundle: dict[str, Any] = {
        "equity":        figure_to_dict(equity_fig),
        "metrics_table": figure_to_dict(metrics_fig),
        "metrics":       metrics,
        "metadata":      metadata,
    }
    if signals_fig is not None:
        bundle["signals"] = figure_to_dict(signals_fig)

    # ---- Hybrid sleeve breakdown table -----------------------------------
    if strategy_key == "hybrid" and chart_data_raw:
        allocations = chart_data_raw.get(
            "allocations",
            {"breakout_b": 0.5, "ma_crossover": 0.3, "rsi_mean_reversion": 0.2},
        )
        sleeve_equity_raw = chart_data_raw.get("sleeve_equity", {})
        starting_capital = float(metadata.get("starting_capital", 10000.0))

        sleeve_data = []
        for sub_key, weight in allocations.items():
            sleeve_start = starting_capital * weight

            # Sleeve final equity from saved sleeve_equity series
            se_records = sleeve_equity_raw.get(sub_key, [])
            sleeve_fin = float(se_records[-1]["value"]) if se_records else sleeve_start
            contribution = (sleeve_fin - sleeve_start) / starting_capital

            # Per-sleeve trade stats
            sub_trades = (
                trade_log_df[trade_log_df["sub_strategy"] == sub_key]
                if trade_log_df is not None and "sub_strategy" in trade_log_df.columns
                else pd.DataFrame()
            )
            tc = len(sub_trades)
            if tc > 0:
                rets = sub_trades["return_pct"]
                win_mask = rets > 0
                loss_mask = rets <= 0
                wr = float(win_mask.sum()) / tc
                aw_s = rets[win_mask]
                al_s = rets[loss_mask]
                aw = float(aw_s.mean()) if len(aw_s) > 0 else None
                al = float(al_s.mean()) if len(al_s) > 0 else None
            else:
                wr = aw = al = None

            sleeve_data.append({
                "name":        _SLEEVE_LABELS.get(sub_key, sub_key),
                "allocation":  weight,
                "contribution": contribution,
                "win_rate":    wr,
                "avg_win":     aw,
                "avg_loss":    al,
                "trade_count": tc,
            })

        if sleeve_data:
            bundle["sleeves"] = figure_to_dict(
                sleeve_metrics_table(sleeve_data, ticker, timeframe_years)
            )

    return bundle
