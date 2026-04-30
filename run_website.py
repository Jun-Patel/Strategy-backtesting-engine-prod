"""
Entry point for the Strategy Backtesting Engine website.

Run from the project root:
    python run_website.py
"""

from src.website.app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
