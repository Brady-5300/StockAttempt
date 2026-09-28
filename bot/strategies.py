"""Strategies turn prices into daily target weights.

Every function only uses data up to and including each row's date; the
backtester adds the execution lag on top. Rows sum to 1 (fully invested,
long-only, no leverage); anything not allocated goes to the cash ETF.
"""
import pandas as pd

from .data import CASH, UNIVERSE


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
    risky = [c for c in UNIVERSE if c != cash]
    past = prices.shift(skip) / prices.shift(lookback) - 1
    mom = past[risky]
    ranks = mom.rank(axis=1, ascending=False)
    picks = (ranks <= top_n) & mom.gt(past[cash], axis=0)
    return _with_cash(picks.astype(float) / top_n, cash)


def mean_reversion(prices, lookback=5, top_n=3, trend_sma=200, cash=CASH):
    """Buy the ETFs that fell most over the last few days, but only ones
    still in a long-term uptrend (buy the dip, not the crash)."""
    risky = [c for c in UNIVERSE if c != cash]
    p = prices[risky]
    uptrend = p > p.rolling(trend_sma).mean()
    score = p.pct_change(lookback, fill_method=None).where(uptrend)
    ranks = score.rank(axis=1, ascending=True)
    picks = (ranks <= top_n) & (score < 0)
    return _with_cash(picks.astype(float) / top_n, cash)


def leveraged_trend(prices, sma=150, leverage=1.5, cash=CASH):
    """Trend filter, but with more than 1x S&P exposure while in an uptrend.

    Leverage between 1x and 2x is built from a SPY/SSO mix: e.g. 1.5x is
    half SPY (1x) and half SSO (2x)."""
    if not 1.0 <= leverage <= 2.0:
        raise ValueError("leverage must be between 1 and 2")
    above = (prices["SPY"] > prices["SPY"].rolling(sma).mean()).astype(float)
    w = pd.DataFrame({"SPY": above * (2.0 - leverage),
                      "SSO": above * (leverage - 1.0)}, index=prices.index)
    return _with_cash(w, cash)


def _spread_evenly(picks: pd.DataFrame) -> pd.DataFrame:
    counts = picks.sum(axis=1)
    return picks.div(counts.where(counts > 0), axis=0).fillna(0.0)


def equal_weight_index(prices, members):
    """Every S&P 500 member we have a price for, equal weight. Compared with
    RSP (the real equal-weight ETF) it shows how much missing data flatters us."""
    picks = (members & prices[members.columns].notna()).astype(float)
    return _spread_evenly(picks)


def stock_momentum(prices, members, lookback=252, skip=21, top_n=20,
                   trend_sma=0, cash=CASH):
    """Hold the top_n S&P 500 members by past return (skipping the latest
    month), equal weight. Only stocks in the index on that date qualify.
    With trend_sma > 0, go to cash while SPY is below that moving average."""
    p = prices[members.columns]
    mom = (p.shift(skip) / p.shift(lookback) - 1).where(members & p.notna())
    picks = (mom.rank(axis=1, ascending=False) <= top_n).astype(float)
    w = _spread_evenly(picks)
    if trend_sma:
        above = prices["SPY"] > prices["SPY"].rolling(trend_sma).mean()
        w = w.mul(above.astype(float), axis=0)
    w[cash] = 1.0 - w.sum(axis=1)
    return w


def trend_momentum(prices, sma=150, lookback=252, top_n=5, trend_share=0.5, cash=CASH):
    """Split the money between the trend and momentum strategies. They did
    well in different years, so together the ride should be smoother."""
    t = trend(prices, sma=sma, cash=cash).reindex(columns=prices.columns, fill_value=0.0)
    m = momentum(prices, lookback=lookback, top_n=top_n, cash=cash)
    m = m.reindex(columns=prices.columns, fill_value=0.0)
    return trend_share * t + (1 - trend_share) * m
