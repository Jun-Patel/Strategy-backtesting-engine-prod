"""
Historical OHLCV data loader.

Public interface
----------------
load_ohlcv(ticker, years)        -> pd.DataFrame
load_all_assets(years)           -> dict[str, pd.DataFrame]

All returned DataFrames share the same contract:
  - DatetimeIndex named "date" (UTC-naive, daily frequency)
  - Columns: open, high, low, close, volume  (all lowercase, float64 / int64)
  - No rows where close is NaN
  - Rows are sorted ascending by date
"""

import warnings
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.config.settings import ALL_ASSETS, TIMEFRAMES_YEARS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_ohlcv(ticker: str, years: int) -> pd.DataFrame:
    """
    Download and return clean daily OHLCV data for *ticker* covering *years*
    of history ending today.

    Parameters
    ----------
    ticker : str
        A valid yfinance ticker symbol, e.g. "SPY", "BTC-USD".
    years : int
        Number of lookback years. Must be one of TIMEFRAMES_YEARS (1/5/10/20).

    Returns
    -------
    pd.DataFrame
        Clean OHLCV DataFrame with the standard schema described in the module
        docstring.

    Raises
    ------
    ValueError
        If *years* is not a supported timeframe, or the ticker returns no data.
    """
    if years not in TIMEFRAMES_YEARS:
        raise ValueError(
            f"Unsupported timeframe: {years}. Must be one of {TIMEFRAMES_YEARS}."
        )

    end = datetime.today()
    # Add a small buffer so weekends / holidays at the boundary don't clip a day
    start = end - timedelta(days=int(years * 365.25) + 5)

    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Verify the symbol is correct and has historical data."
        )

    df = _normalize(raw, ticker)
    _warn_if_short(df, ticker, years)
    return df


def load_all_assets(years: int) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV data for every asset in ALL_ASSETS (anchor markets + GLD).

    Assets that fail to load are skipped with a warning; they will not appear
    in the returned dict. The caller is responsible for checking that all
    expected assets are present if that matters for downstream logic.

    Parameters
    ----------
    years : int
        Lookback years, passed through to load_ohlcv.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → clean OHLCV DataFrame.
    """
    result: dict[str, pd.DataFrame] = {}
    for ticker in ALL_ASSETS:
        try:
            result[ticker] = load_ohlcv(ticker, years)
        except Exception as exc:
            warnings.warn(
                f"Could not load data for '{ticker}': {exc}",
                UserWarning,
                stacklevel=2,
            )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize a raw yfinance DataFrame to the standard OHLCV schema.

    Handles both flat and MultiIndex column layouts that yfinance may return
    depending on version and download options.
    """
    df = raw.copy()

    # yfinance sometimes returns a MultiIndex (Price / Ticker) even for single
    # ticker downloads. Flatten to the first level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).lower().strip() for c in df.columns]

    # Rename any yfinance aliases to canonical names
    _aliases = {
        "adj close": "close",
        "adj_close": "close",
    }
    df.rename(columns=_aliases, inplace=True)

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Ticker '{ticker}': missing required columns after normalization: {missing}. "
            f"Got: {list(df.columns)}"
        )

    df = df[list(_REQUIRED_COLUMNS)].copy()

    # Normalize index to a plain DatetimeIndex named "date"
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "date"
    df.sort_index(inplace=True)

    # Drop rows where close price is missing — these cannot be used for backtesting
    df.dropna(subset=["close"], inplace=True)

    # Volume can be absent for some crypto data sources; treat as 0
    df["volume"] = df["volume"].fillna(0.0)

    # Ensure numeric dtypes
    for col in _REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _warn_if_short(df: pd.DataFrame, ticker: str, requested_years: int) -> None:
    """Emit a UserWarning if the returned data covers significantly less than requested."""
    if df.empty:
        return
    actual_days = (df.index[-1] - df.index[0]).days
    actual_years = actual_days / 365.25
    # Warn if we received less than 90% of the requested window
    if actual_years < requested_years * 0.9:
        warnings.warn(
            f"'{ticker}': requested {requested_years}y of data but only "
            f"{actual_years:.1f}y is available. Using maximum available period.",
            UserWarning,
            stacklevel=3,
        )
