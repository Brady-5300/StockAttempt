"""Strategies turn prices into daily target weights.

Every function only uses data up to and including each row's date; the
backtester adds the execution lag on top. Rows sum to 1 (fully invested,
long-only, no leverage); anything not allocated goes to the cash ETF.
"""
import pandas as pd

from .data import CASH


def _with_cash(weights: pd.DataFrame, cash: str) -> pd.DataFrame:
    weights = weights.fillna(0.0)
    weights[cash] = 1.0 - weights.sum(axis=1)
    return weights


def buy_and_hold(prices, asset="SPY"):
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w[asset] = 1.0
    return w


def trend(prices, asset="SPY", sma=200, cash=CASH):
    """Hold the asset while it's above its moving average, else cash."""
    above = prices[asset] > prices[asset].rolling(sma).mean()
    w = pd.DataFrame({asset: above.astype(float)}, index=prices.index)
    return _with_cash(w, cash)


def momentum(prices, lookback=252, skip=21, top_n=3, cash=CASH):
    """Hold the top_n ETFs by past return (skipping the most recent month),
    but only those that beat cash over the same window."""
    risky = [c for c in prices.columns if c != cash]
    past = prices.shift(skip) / prices.shift(lookback) - 1
    mom = past[risky]
    ranks = mom.rank(axis=1, ascending=False)
    picks = (ranks <= top_n) & mom.gt(past[cash], axis=0)
    return _with_cash(picks.astype(float) / top_n, cash)


def mean_reversion(prices, lookback=5, top_n=3, trend_sma=200, cash=CASH):
    """Buy the ETFs that fell most over the last few days, but only ones
    still in a long-term uptrend (buy the dip, not the crash)."""
    risky = [c for c in prices.columns if c != cash]
    p = prices[risky]
    uptrend = p > p.rolling(trend_sma).mean()
    score = p.pct_change(lookback, fill_method=None).where(uptrend)
    ranks = score.rank(axis=1, ascending=True)
    picks = (ranks <= top_n) & (score < 0)
    return _with_cash(picks.astype(float) / top_n, cash)
