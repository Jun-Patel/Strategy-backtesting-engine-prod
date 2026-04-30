"""
Flask application factory for the Strategy Backtesting Engine website.

Usage
-----
Run locally:
    python run_website.py

Or via Flask CLI:
    FLASK_APP=src.website.app:create_app flask run
"""

from __future__ import annotations

from flask import Flask


def create_app(test_config: dict | None = None) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    test_config:
        Optional dict of config values to override defaults (used in tests).
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = "dev-key-not-for-production"

    if test_config is not None:
        app.config.update(test_config)

    from src.website.routes import bp
    app.register_blueprint(bp)

    return app
