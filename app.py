"""
Strategy Backtesting Engine — Streamlit entrypoint.

Reads pre-computed backtest results from reports/results/ (committed to the
repo, no live data fetch required at startup) and renders interactive Plotly
charts via Streamlit's native plotly support.

Local run:
    streamlit run app.py

Streamlit Community Cloud: point the deployment at this file.
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from src.config.settings import (
    ANCHOR_ASSETS,
    STRATEGIES,
    STRATEGY_ORDER,
    TIMEFRAMES_YEARS,
)
from src.visualization.charts import build_chart_bundle

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Strategy Backtesting Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CSS — replicates the Flask dark premium theme as closely as Streamlit allows
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Hide Streamlit chrome ── */
#MainMenu,
header[data-testid="stHeader"],
footer,
[data-testid="stDecoration"],
[data-testid="stToolbar"]              { display: none !important; }

/* ── Page background & font ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"]          { background: #080C18 !important; }
* { font-family: 'Inter', system-ui, sans-serif !important; }

/* ── Main block: full width, no top gap ── */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 1280px !important;
}

/* ── Reduce default vertical spacing between Streamlit elements ── */
[data-testid="stVerticalBlock"]           { gap: 0.5rem !important; }
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]
                                          { gap: 0.55rem !important; }

/* ── Column padding reset ── */
[data-testid="stColumn"] > div           { padding: 0 !important; }

/* ── Horizontal block gap (columns) ── */
[data-testid="stHorizontalBlock"]        { gap: 1rem !important; align-items: start !important; }

/* ── Element containers: remove bottom margin ── */
[data-testid="stElementContainer"]       { margin-bottom: 0 !important; }

/* ═══════════════════════════════════════════════
   PANEL CARDS  (st.container(border=True))
   Matches Flask .graph-panel
═══════════════════════════════════════════════ */
[data-testid="stVerticalBlockBorderWrapper"] {
    background:    #0D1220 !important;
    border:        1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    overflow:      hidden;
    padding:       0 !important;
    transition:    border-color 0.25s ease !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(59,130,246,0.14) !important;
}
/* Inner padding for panel content */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 20px !important;
}

/* ═══════════════════════════════════════════════
   CONTROLS ROW DIVIDER
   Matches Flask .graph-panel-controls border-bottom
═══════════════════════════════════════════════ */
.controls-divider {
    height: 1px;
    background: rgba(255,255,255,0.07);
    margin: 4px -20px 0 -20px;   /* bleed to card edge */
}

/* ═══════════════════════════════════════════════
   SELECTBOXES  — dark, matches Flask selects
═══════════════════════════════════════════════ */
[data-baseweb="select"] > div {
    background-color: #080C18 !important;
    border:           1px solid rgba(255,255,255,0.09) !important;
    border-radius:    7px !important;
    color:            #E2E8F0 !important;
    font-size:        0.875rem !important;
    min-height:       36px !important;
}
[data-baseweb="select"] > div:hover { border-color: rgba(59,130,246,0.4) !important; }
[data-baseweb="select"] > div:focus-within {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
}
/* Dropdown list */
[data-baseweb="menu"]   { background: #1A2236 !important; border: 1px solid rgba(255,255,255,0.09) !important; }
[data-baseweb="option"] { background: #1A2236 !important; color: #E2E8F0 !important; }
[data-baseweb="option"]:hover,
[data-baseweb="option"][aria-selected="true"] { background: rgba(59,130,246,0.18) !important; }
/* Selected value text */
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p { color: #E2E8F0 !important; }

/* ═══════════════════════════════════════════════
   WIDGET LABELS  — STRATEGY / ASSET / TIMEFRAME
═══════════════════════════════════════════════ */
label[data-testid="stWidgetLabel"] p {
    font-size:      0.67rem !important;
    font-weight:    700 !important;
    color:          #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    margin-bottom:  4px !important;
}

/* ═══════════════════════════════════════════════
   LAYOUT TOGGLE  — pill style, matches Flask
═══════════════════════════════════════════════ */
[data-testid="stRadio"] > label { display: none !important; }
[data-testid="stRadio"] > div[role="radiogroup"] {
    display:       inline-flex !important;
    flex-direction: row !important;
    background:    #111827 !important;
    border:        1px solid rgba(255,255,255,0.07) !important;
    border-radius: 9px !important;
    padding:       4px !important;
    gap:           3px !important;
}
[data-testid="stRadio"] label {
    padding:       5px 18px !important;
    border-radius: 6px !important;
    font-size:     0.82rem !important;
    font-weight:   600 !important;
    color:         #475569 !important;
    cursor:        pointer !important;
    transition:    background 0.18s, color 0.18s !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #3B82F6 !important;
    color:      #ffffff !important;
}

/* ═══════════════════════════════════════════════
   TABS  — matches Flask .btn-chart-type
═══════════════════════════════════════════════ */
[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 0;
    background: transparent;
}
[data-testid="stTabs"] button[role="tab"] {
    font-size:     0.78rem !important;
    font-weight:   600 !important;
    color:         #475569 !important;
    background:    #0D1220 !important;
    border:        1px solid rgba(255,255,255,0.07) !important;
    border-radius: 6px 6px 0 0 !important;
    padding:       6px 14px !important;
    transition:    all 0.18s !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color:        #E2E8F0 !important;
    border-color: rgba(59,130,246,0.28) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color:        #93C5FD !important;
    background:   rgba(59,130,246,0.14) !important;
    border-color: rgba(59,130,246,0.35) !important;
}
/* Tab content area */
[data-testid="stTabs"] > div[data-testid="stVerticalBlock"] {
    padding-top: 4px !important;
}

/* ═══════════════════════════════════════════════
   PLOTLY MODEBAR
═══════════════════════════════════════════════ */
.modebar-container  { background: rgba(13,18,32,0.88) !important; border-radius: 6px !important; }
.modebar-btn path   { fill: #64748B !important; }
.modebar-btn:hover path { fill: #E2E8F0 !important; }

/* ═══════════════════════════════════════════════
   DIVIDER  (st.divider())
═══════════════════════════════════════════════ */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 0 !important; }

/* ═══════════════════════════════════════════════
   ERROR / ALERT
═══════════════════════════════════════════════ */
[data-testid="stAlert"] {
    background: rgba(239,68,68,0.08) !important;
    border:     1px solid rgba(239,68,68,0.3) !important;
    color:      #FCA5A5 !important;
    border-radius: 8px !important;
}

/* ═══════════════════════════════════════════════
   CUSTOM HTML COMPONENTS
═══════════════════════════════════════════════ */

/* Panel label badge  — matches Flask .panel-label */
.panel-label {
    display:       inline-block;
    font-size:     0.775rem;
    font-weight:   600;
    color:         #93C5FD;
    background:    rgba(59,130,246,0.06);
    border:        1px solid rgba(59,130,246,0.1);
    border-radius: 6px;
    padding:       5px 11px;
    letter-spacing: 0.02em;
}

/* Stats sidebar  — matches Flask .stats-area */
.stat-block { margin-top: 0; }
.stat-section-head {
    font-size:      0.67rem;
    font-weight:    700;
    color:          #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom:  12px;
    padding-bottom: 9px;
    border-bottom:  1px solid rgba(255,255,255,0.07);
}
.stat-row {
    display:         flex;
    justify-content: space-between;
    align-items:     center;
    padding:         6px 0;
    border-bottom:   1px solid rgba(255,255,255,0.04);
    font-size:       0.81rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-label  { color: #94A3B8; font-size: 0.77rem; }
.stat-value  { font-weight: 700; font-variant-numeric: tabular-nums; font-size: 0.83rem; color: #E2E8F0; }
.stat-green  { color: #22C55E !important; }
.stat-red    { color: #EF4444 !important; }
.stat-divider { margin-top: 6px; padding-top: 9px; border-top: 1px solid rgba(255,255,255,0.1); }

/* Hero  — matches Flask .hero */
.hero-wrap {
    position:   relative;
    overflow:   hidden;
    text-align: center;
    padding:    64px 0 40px;
}
.hero-glow {
    position:   absolute;
    top:        -60px;
    left:       50%;
    transform:  translateX(-50%);
    width:      700px;
    height:     420px;
    background: radial-gradient(ellipse at center, rgba(59,130,246,0.11) 0%, transparent 68%);
    pointer-events: none;
}
.hero-content { position: relative; z-index: 1; }
.hero-title {
    font-size:       clamp(2rem, 4vw, 3rem);
    font-weight:     800;
    letter-spacing:  -0.04em;
    line-height:     1.12;
    margin-bottom:   14px;
    background:      linear-gradient(135deg, #F1F5F9 0%, #93C5FD 55%, #C4B5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size:     1.0rem;
    color:         #94A3B8;
    max-width:     600px;
    margin:        0 auto 18px;
    line-height:   1.65;
}
.hero-badges { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.badge {
    display:       inline-block;
    background:    rgba(59,130,246,0.08);
    border:        1px solid rgba(59,130,246,0.18);
    border-radius: 20px;
    padding:       4px 13px;
    font-size:     0.75rem;
    font-weight:   500;
    color:         #93C5FD;
}
.hero-rule {
    height:     1px;
    background: linear-gradient(90deg, transparent 0%, rgba(59,130,246,0.3) 50%, transparent 100%);
    margin-top: 40px;
}

/* Section headings  — matches Flask .section-heading */
.section-heading {
    text-align:    center;
    margin-bottom: 28px;
}
.section-heading h2 {
    font-size:      clamp(1.45rem, 3vw, 2rem);
    font-weight:    700;
    letter-spacing: -0.03em;
    color:          #E2E8F0;
    margin-bottom:  8px;
}
.section-heading p {
    font-size: 0.95rem;
    color:     #94A3B8;
    max-width: 540px;
    margin:    0 auto;
}

/* Summary cards  — matches Flask .summary-card */
.summary-card {
    background:    #111827;
    border:        1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding:       22px;
    height:        100%;
    transition:    border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}
.summary-card:hover {
    border-color: rgba(59,130,246,0.28);
    transform:    translateY(-2px);
    box-shadow:   0 0 28px rgba(59,130,246,0.07), 0 4px 16px rgba(0,0,0,0.3);
}
.summary-card h3 {
    font-size:      0.72rem;
    font-weight:    700;
    color:          #3B82F6;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom:  9px;
}
.summary-card p { font-size: 0.875rem; color: #94A3B8; line-height: 1.65; }

/* Strategy cards  — matches Flask .strategy-card */
.strategy-card {
    background:    #111827;
    border:        1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding:       22px;
    height:        100%;
    position:      relative;
    overflow:      hidden;
    transition:    border-color 0.25s, transform 0.25s, box-shadow 0.25s;
}
/* Top accent bar on hover */
.strategy-card::before {
    content:    "";
    position:   absolute;
    top: 0; left: 0; right: 0;
    height:     2px;
    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    opacity:    0;
    transition: opacity 0.25s;
}
.strategy-card:hover {
    border-color: rgba(59,130,246,0.22);
    transform:    translateY(-3px);
    box-shadow:   0 8px 32px rgba(0,0,0,0.35), 0 0 0 1px rgba(59,130,246,0.08);
}
.strategy-card:hover::before { opacity: 1; }
.strategy-badge {
    display:       inline-block;
    font-size:     0.69rem;
    font-weight:   700;
    color:         #93C5FD;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border:        1px solid rgba(59,130,246,0.2);
    background:    rgba(59,130,246,0.08);
    border-radius: 5px;
    padding:       2px 9px;
    margin-bottom: 12px;
}
.strategy-name { font-size: 0.93rem; font-weight: 700; color: #E2E8F0; margin-bottom: 7px; }
.strategy-desc { font-size: 0.83rem; color: #94A3B8; line-height: 1.65; }

/* Interactive charts heading row */
.charts-heading-row {
    display:         flex;
    align-items:     baseline;
    justify-content: space-between;
    margin-bottom:   16px;
}
.charts-heading-row h2 {
    font-size:      clamp(1.45rem, 3vw, 2rem);
    font-weight:    700;
    letter-spacing: -0.03em;
    color:          #E2E8F0;
}
.charts-heading-row p { font-size: 0.9rem; color: #94A3B8; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STRATEGY_LABELS = {key: STRATEGIES[key]["name"] for key in STRATEGY_ORDER}
_TF_LABELS       = {tf: f"{tf} year{'s' if tf != 1 else ''}" for tf in TIMEFRAMES_YEARS}

_PANEL_DEFAULTS = [
    {"strategy": "ma_crossover",       "asset": "SPY",     "timeframe": 5},
    {"strategy": "rsi_mean_reversion", "asset": "QQQ",     "timeframe": 5},
    {"strategy": "breakout_a",         "asset": "BTC-USD", "timeframe": 5},
    {"strategy": "hybrid",             "asset": "IWM",     "timeframe": 5},
]

_DARK_AXIS = dict(
    gridcolor="rgba(255,255,255,0.07)",
    zerolinecolor="rgba(255,255,255,0.15)",
    linecolor="rgba(255,255,255,0.07)",
    tickfont=dict(color="#64748B", size=11),
    title=dict(font=dict(color="#94A3B8")),
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_bundle(strategy: str, asset: str, timeframe: int) -> dict | None:
    try:
        return build_chart_bundle(strategy, asset, timeframe)
    except FileNotFoundError:
        return None


def _make_fig(d: dict, height: int | None = None) -> go.Figure:
    fig = go.Figure(data=d.get("data", []), layout=d.get("layout", {}))
    overrides: dict = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.025)",
        font=dict(color="#94A3B8", family="'Inter', system-ui, sans-serif"),
    )
    if height is not None:
        overrides["height"] = height
    fig.update_layout(**overrides)
    fig.update_xaxes(**_DARK_AXIS)
    fig.update_yaxes(**_DARK_AXIS)
    return fig


# ---------------------------------------------------------------------------
# Stats sidebar — rendered as a single HTML block to match Flask exactly
# ---------------------------------------------------------------------------

def _pct(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v * 100:.2f}%"


def _signed(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{'+'if v >= 0 else ''}{v * 100:.2f}%"


def _cls(value: str, positive_good: bool | None) -> str:
    if value == "—" or positive_good is None:
        return ""
    return ("stat-green" if value.startswith("+") else "stat-red") if positive_good \
        else ("stat-red" if value.startswith("+") else "stat-green")


def render_stats(metrics: dict) -> None:
    if not metrics:
        st.markdown('<p style="color:#475569;font-size:0.85rem;">No data.</p>', unsafe_allow_html=True)
        return

    rows = [
        ("Total Return",  _pct(metrics.get("total_return")),         True),
        ("Ann. Return",   _pct(metrics.get("annualized_return")),     True),
        ("Win Rate",      _pct(metrics.get("win_rate")),              None),
        ("Avg Win",       _signed(metrics.get("avg_win")),            True),
        ("Avg Loss",      _signed(metrics.get("avg_loss")),           False),
        ("Max Win",       _signed(metrics.get("max_win")),            True),
        ("Max Loss",      _signed(metrics.get("max_loss")),           False),
        ("Max Drawdown",  _pct(metrics.get("max_drawdown")),          False),
        ("Avg Drawdown",  _pct(metrics.get("avg_drawdown")),          False),
    ]
    bh  = _pct(metrics.get("benchmark_return"))
    out = _signed(metrics.get("outperformance"))

    def row(label: str, val: str, pg) -> str:
        c = _cls(val, pg)
        return (
            f'<div class="stat-row">'
            f'<span class="stat-label">{label}</span>'
            f'<span class="stat-value {c}">{val}</span>'
            f'</div>'
        )

    html = '<div class="stat-block"><div class="stat-section-head">Performance</div>'
    for label, val, pg in rows:
        html += row(label, val, pg)
    html += '<div class="stat-divider"></div>'
    html += row("Buy &amp; Hold", bh, None)
    oc = _cls(out, True)
    html += (
        f'<div class="stat-row" style="margin-top:2px;">'
        f'<span class="stat-label" style="font-weight:700;color:#E2E8F0;">vs Buy &amp; Hold</span>'
        f'<span class="stat-value {oc}" style="font-size:0.88rem;">{out}</span>'
        f'</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Panel renderer
# ---------------------------------------------------------------------------

def render_panel(panel_num: int) -> None:
    defaults = _PANEL_DEFAULTS[panel_num - 1]

    with st.container(border=True):
        # ── Controls bar ──────────────────────────────────────
        c1, c2, c3 = st.columns([2, 1, 1])
        strategy = c1.selectbox(
            "Strategy",
            options=STRATEGY_ORDER,
            format_func=lambda k: _STRATEGY_LABELS[k],
            index=STRATEGY_ORDER.index(defaults["strategy"]),
            key=f"strategy_{panel_num}",
        )
        asset = c2.selectbox(
            "Asset",
            options=ANCHOR_ASSETS,
            index=ANCHOR_ASSETS.index(defaults["asset"]),
            key=f"asset_{panel_num}",
        )
        timeframe = c3.selectbox(
            "Timeframe",
            options=TIMEFRAMES_YEARS,
            format_func=lambda tf: _TF_LABELS[tf],
            index=TIMEFRAMES_YEARS.index(defaults["timeframe"]),
            key=f"timeframe_{panel_num}",
        )

        # Divider line between controls and chart body (matches Flask border-bottom)
        st.markdown('<div class="controls-divider"></div>', unsafe_allow_html=True)

        # Panel label badge
        st.markdown(
            f'<div class="panel-label">📊 &nbsp;'
            f'{_STRATEGY_LABELS[strategy]} &middot; {asset} &middot; {timeframe}y'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Load data ─────────────────────────────────────────
        bundle = load_bundle(strategy, asset, int(timeframe))

        if bundle is None:
            st.error(f"Result not found: {strategy} / {asset} / {timeframe}y")
            return

        # ── Chart area (left) + Stats sidebar (right) ─────────
        # Matches Flask .graph-panel-body  grid: 1fr 258px
        chart_col, stats_col = st.columns([4, 1])

        with stats_col:
            render_stats(bundle["metrics"])

        with chart_col:
            has_signals = "signals" in bundle
            has_sleeves = "sleeves" in bundle

            tab_names = ["📈 Equity"]
            if has_signals:
                tab_names.append("📊 Signals")
            if has_sleeves:
                tab_names.append("🧩 Sleeves")

            tabs = st.tabs(tab_names)

            with tabs[0]:
                fig = _make_fig(bundle["equity"], height=350)
                fig.update_layout(margin=dict(l=55, r=20, t=50, b=45))
                st.plotly_chart(fig, use_container_width=True, key=f"equity_{panel_num}")

            tab_idx = 1
            if has_signals:
                with tabs[tab_idx]:
                    fig = _make_fig(bundle["signals"])
                    fig.update_layout(margin=dict(l=60, r=30, t=80, b=50))
                    st.plotly_chart(fig, use_container_width=True, key=f"signals_{panel_num}")
                tab_idx += 1

            if has_sleeves:
                with tabs[tab_idx]:
                    fig = _make_fig(bundle["sleeves"])
                    fig.update_layout(margin=dict(l=60, r=30, t=80, b=50))
                    st.plotly_chart(fig, use_container_width=True, key=f"sleeves_{panel_num}")


# ===========================================================================
# PAGE LAYOUT
# ===========================================================================

# ── Hero ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-glow"></div>
  <div class="hero-content">
    <div class="hero-title">Strategy Backtesting Engine</div>
    <div class="hero-sub">
      Five quantitative trading strategies tested across equities and crypto
      over 1, 5, 10, and 20-year horizons. Compare performance, drawdown,
      and key metrics interactively.
    </div>
    <div class="hero-badges">
      <span class="badge">5 Strategies</span>
      <span class="badge">4 Anchor Assets</span>
      <span class="badge">4 Timeframes</span>
      <span class="badge">Daily OHLCV · yfinance</span>
      <span class="badge">Long-Only · No Look-Ahead Bias</span>
    </div>
    <div class="hero-rule"></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Project Summary (matches Flask #summary section) ─────────────────────
st.markdown("""
<div class="section-heading">
  <h2>Project Summary</h2>
  <p>A Python-based quantitative research engine that backtests five
     systematic trading strategies across multiple markets and timeframes.</p>
</div>
""", unsafe_allow_html=True)

sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
_SUMMARY_CARDS = [
    ("What it does",
     "Fetches daily OHLCV data via yfinance, computes technical indicators, "
     "generates entry/exit signals, simulates trades, and computes performance "
     "metrics — all in a reproducible, modular pipeline."),
    ("Assets covered",
     "SPY (S&P 500), QQQ (Nasdaq-100), BTC-USD (Bitcoin), and IWM "
     "(Russell 2000). Four anchor markets spanning equities and crypto "
     "across different risk/return profiles."),
    ("Methodology",
     "Long-only, daily bar execution. Signals generated at end-of-day close, "
     "executed at the same close. Fixed transaction cost of 0.1% per side. "
     "Starting capital $10,000 per strategy."),
    ("Metrics reported",
     "Total return, annualised return, win rate, avg win/loss, max/avg drawdown, "
     "trade count, average holding period, and outperformance vs buy-and-hold."),
]
for col, (title, body) in zip([sc1, sc2, sc3, sc4], _SUMMARY_CARDS):
    with col:
        st.markdown(
            f'<div class="summary-card"><h3>{title}</h3><p>{body}</p></div>',
            unsafe_allow_html=True,
        )


# ── Trading Strategies ────────────────────────────────────────────────────
st.markdown("""
<div class="section-heading" style="margin-top:40px;">
  <h2>Trading Strategies</h2>
  <p>Five locked strategies — parameters and rules are fixed and identical
     across all assets and timeframes.</p>
</div>
""", unsafe_allow_html=True)

strat_cols = st.columns(len(STRATEGY_ORDER), gap="small")
for col, key in zip(strat_cols, STRATEGY_ORDER):
    s = STRATEGIES[key]
    with col:
        st.markdown(
            f'<div class="strategy-card">'
            f'<div class="strategy-badge">{s["type"]}</div>'
            f'<div class="strategy-name">{s["name"]}</div>'
            f'<div class="strategy-desc">{s["description"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Interactive Charts ────────────────────────────────────────────────────
st.markdown("""
<div class="section-heading" style="margin-top:40px;">
  <h2>Interactive Charts</h2>
  <p>Select a strategy, asset, and timeframe. Charts update automatically
     on selection change. Switch layout to compare panels side by side.</p>
</div>
""", unsafe_allow_html=True)

# Layout toggle — pill style matching Flask .layout-controls
layout_mode = st.radio(
    "Layout",
    options=["Single", "Dual", "Quad"],
    horizontal=True,
    label_visibility="collapsed",
    key="layout_mode",
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if layout_mode == "Single":
    render_panel(1)

elif layout_mode == "Dual":
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        render_panel(1)
    with col_b:
        render_panel(2)

else:  # Quad
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        render_panel(1)
    with col_b:
        render_panel(2)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2, gap="medium")
    with col_c:
        render_panel(3)
    with col_d:
        render_panel(4)


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:48px;margin-bottom:0;" />
<div style="text-align:center;color:#475569;font-size:0.82rem;padding:18px 0 6px;">
  <strong style="color:#94A3B8;">Strategy Backtesting Engine</strong>
  &mdash; Demo research project. Not financial advice.
</div>
""", unsafe_allow_html=True)
