"""Point-in-time S&P 500 membership and prices for every stock ever in it.

Backtesting only on today's S&P 500 members would quietly skip every company
that shrank, got bought or went bust (survivorship bias) and make any
strategy look better than it really was. Instead we use the index's actual
membership on each date, from github.com/fja05680/sp500.
"""
import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf  # type: ignore[import-untyped]

DATA = Path(__file__).resolve().parent.parent / "data"
MEMBERSHIP_CACHE = DATA / "sp500_membership.csv"
PRICES_CACHE = DATA / "stock_prices.pkl"
MEMBERSHIP_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                  "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


def load_membership(refresh=False) -> pd.Series:
    """Index: snapshot date. Value: frozenset of tickers in the S&P 500 then."""
    if refresh or not MEMBERSHIP_CACHE.exists():
        text = requests.get(MEMBERSHIP_URL, timeout=60).text
        DATA.mkdir(exist_ok=True)
        MEMBERSHIP_CACHE.write_text(text)
    df = pd.read_csv(MEMBERSHIP_CACHE, parse_dates=["date"])
    return pd.Series([frozenset(t.split(",")) for t in df["tickers"]],
                     index=pd.DatetimeIndex(df["date"])).sort_index()


def membership_matrix(membership: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """True where a ticker was in the index on that trading day (using the
    latest snapshot on or before the day)."""
    tickers = sorted(set().union(*membership))
    snaps = pd.DataFrame(False, index=membership.index, columns=tickers)
    for date, members in membership.items():
        snaps.loc[date, list(members)] = True
    return snaps.reindex(dates, method="ffill").fillna(False).astype(bool)


def _yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def load_stock_prices(tickers, start="2005-01-01", refresh=False) -> pd.DataFrame:
    """Adjusted daily closes for as many of the tickers as Yahoo still has.
    Companies that were acquired or went bankrupt are often missing; the
    caller should measure how many."""
    if PRICES_CACHE.exists() and not refresh:
        return pd.read_pickle(PRICES_CACHE)
    symbols = {_yahoo_symbol(t): t for t in tickers}
    frames = []
    batch = list(symbols)
    for i in range(0, len(batch), 100):
        raw = yf.download(batch[i:i + 100], start=start, auto_adjust=True,
                          progress=False, threads=True)
        if raw is not None and not raw.empty:
            frames.append(pd.DataFrame(raw["Close"]))
    prices = pd.concat(frames, axis=1).rename(columns=symbols)
    prices = prices.loc[:, prices.notna().any()].sort_index()
    DATA.mkdir(exist_ok=True)
    prices.to_pickle(PRICES_CACHE)
    return prices
