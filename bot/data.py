"""Daily price download with a local CSV cache."""
from pathlib import Path

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

CACHE = Path(__file__).resolve().parent.parent / "data" / "prices.csv"

# Liquid ETFs that all have history back to ~2006. Using ETFs rather than
# today's biggest stocks avoids survivorship bias (we'd otherwise only be
# testing on companies we already know succeeded).
UNIVERSE = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",            # equity indices
    "XLK", "XLF", "XLE", "XLV", "XLY",            # US sectors
    "XLP", "XLI", "XLU", "XLB", "VNQ",
    "TLT", "IEF", "GLD", "DBC",                   # bonds / commodities
    "SHY",                                        # near-cash
]
CASH = "SHY"


def load_prices(tickers=UNIVERSE, start="2005-01-01", refresh=False) -> pd.DataFrame:
    """Split/dividend-adjusted daily closes, one column per ticker."""
    if CACHE.exists() and not refresh:
        prices = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        if set(tickers) <= set(prices.columns):
            return prices[list(tickers)]

    raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("Price download failed; check your internet connection.")
    prices = pd.DataFrame(raw["Close"])[list(tickers)].dropna(how="all")
    CACHE.parent.mkdir(exist_ok=True)
    prices.to_csv(CACHE)
    return prices
