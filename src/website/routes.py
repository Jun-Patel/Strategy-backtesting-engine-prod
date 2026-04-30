"""
Route definitions for the Strategy Backtesting Engine website.

All strategy metadata is sourced from settings.STRATEGIES and
settings.STRATEGY_ORDER so that the website always reflects the locked rules.

Routes
------
GET /               Homepage with all content sections and graph shell.
GET /api/chart      JSON chart bundle for a given strategy/asset/timeframe.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from src.config.settings import (
    ANCHOR_ASSETS,
    STRATEGIES,
    STRATEGY_ORDER,
    TIMEFRAMES_YEARS,
)

bp = Blueprint("main", __name__)

# Default selections per panel (shown when user first opens the page)
_PANEL_DEFAULTS = [
    {"strategy": "ma_crossover",       "asset": "SPY",     "timeframe": 5},
    {"strategy": "rsi_mean_reversion", "asset": "QQQ",     "timeframe": 5},
    {"strategy": "breakout_a",         "asset": "BTC-USD",  "timeframe": 5},
    {"strategy": "hybrid",             "asset": "IWM",     "timeframe": 5},
]


@bp.route("/")
def index():
    """Homepage — renders all static content sections and the graph shell."""
    strategies = [
        {"key": key, **STRATEGIES[key]}
        for key in STRATEGY_ORDER
    ]

    return render_template(
        "index.html",
        strategies=strategies,
        assets=ANCHOR_ASSETS,
        timeframes=TIMEFRAMES_YEARS,
        panel_defaults=_PANEL_DEFAULTS,
    )


@bp.route("/api/chart")
def api_chart():
    """Return a JSON chart bundle for the requested strategy/asset/timeframe.

    Query parameters
    ----------------
    strategy : str   Strategy key, e.g. ``ma_crossover``.
    asset    : str   Ticker symbol, e.g. ``SPY``.
    timeframe: int   Lookback in years, e.g. ``5``.

    Returns
    -------
    200  JSON with keys: ``equity``, ``drawdown``, ``metrics``, ``metadata``.
    400  JSON ``{"error": "..."}`` if a required parameter is missing.
    404  JSON ``{"error": "Result not found"}`` if the file does not exist.
    """
    from src.visualization.charts import build_chart_bundle

    strategy  = request.args.get("strategy", "").strip()
    asset     = request.args.get("asset", "").strip()
    tf_raw    = request.args.get("timeframe", "").strip()

    if not strategy or not asset or not tf_raw:
        return jsonify({"error": "strategy, asset, and timeframe are required"}), 400

    try:
        timeframe = int(tf_raw)
    except ValueError:
        return jsonify({"error": "timeframe must be an integer"}), 400

    try:
        bundle = build_chart_bundle(strategy, asset, timeframe)
    except FileNotFoundError:
        return jsonify({"error": "Result not found"}), 404

    response: dict = {
        "equity":   bundle["equity"],
        "metrics":  bundle["metrics"],
        "metadata": bundle["metadata"],
    }
    if "signals" in bundle:
        response["signals"] = bundle["signals"]
    if "sleeves" in bundle:
        response["sleeves"] = bundle["sleeves"]
    return jsonify(response)
