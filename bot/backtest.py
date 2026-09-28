"""Daily portfolio simulator with drift, periodic rebalancing and costs."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def run(prices, targets, rebalance_every=1, cost_bps=5.0, lag=1, start=None, end=None):
    """Simulate a strategy and return its daily net returns.

    Timing: a signal computed from the close on day t is traded at the close
    of day t+lag, and those holdings earn returns from the following day.
    lag=1 is deliberately pessimistic (the live bot trades just before the
    close of the same day).

    cost_bps is charged on every dollar traded (one-way), covering spread
    and slippage. Between rebalances, weights drift with prices.
    """
    rets = prices.pct_change(fill_method=None).fillna(0.0)
    targets = targets.reindex(columns=prices.columns).fillna(0.0).shift(lag)

    window = rets.loc[start:end].index
    r = rets.loc[window].to_numpy()
    t = targets.loc[window].to_numpy()

    w = np.zeros(r.shape[1])
    net = np.empty(len(window))
    turnover = np.zeros(len(window))
    for i in range(len(window)):
        gross = w @ r[i]
        w = w * (1 + r[i]) / (1 + gross)
        cost = 0.0
        if i % rebalance_every == 0 and not np.isnan(t[i]).any():
            turnover[i] = np.abs(t[i] - w).sum()
            cost = turnover[i] * cost_bps / 1e4
            w = t[i].copy()
        net[i] = (1 + gross) * (1 - cost) - 1

    out = pd.Series(net, index=window, name="return")
    out.attrs["turnover"] = turnover.sum()
    return out


def metrics(r: pd.Series) -> dict:
    years = len(r) / TRADING_DAYS
    equity = (1 + r).cumprod()
    return {
        "CAGR": equity.iloc[-1] ** (1 / years) - 1,
        "Vol": r.std() * np.sqrt(TRADING_DAYS),
        "Sharpe": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
        "MaxDD": (equity / equity.cummax() - 1).min(),
        "Turnover/yr": r.attrs.get("turnover", np.nan) / years,
    }


def excess_tstat(r: pd.Series, benchmark: pd.Series) -> float:
    """t-stat of the mean daily return difference vs the benchmark.
    Roughly: |t| > 2 means the gap is unlikely to be luck."""
    diff = (r - benchmark).dropna()
    return diff.mean() / diff.std() * np.sqrt(len(diff))
