"""Walk-forward evaluation: pick each strategy's settings on the training
years only, then score those settings once on the untouched test years."""
import argparse
import itertools

import pandas as pd

from bot import backtest, strategies
from bot.data import load_prices

TRAIN = ("2007-01-01", "2016-12-31")
TEST = ("2017-01-01", None)
W = 62  # width of the label column in printed tables

# name -> (strategy fn, parameter grid). "rebalance" is the backtester's
# rebalance interval in trading days; the rest go to the strategy.
GRIDS = {
    "trend": (strategies.trend, {"sma": [100, 150, 200], "rebalance": [1, 5]}),
    "momentum": (strategies.momentum, {
        "lookback": [63, 126, 252], "top_n": [2, 3, 5], "rebalance": [5, 21]}),
    "mean_reversion": (strategies.mean_reversion, {
        "lookback": [3, 5, 10], "top_n": [2, 3, 5], "rebalance": [3, 5]}),
    "trend_momentum": (strategies.trend_momentum, {
        "sma": [150, 200], "lookback": [126, 252], "top_n": [3, 5],
        "rebalance": [5, 21]}),
    "leveraged_trend": (strategies.leveraged_trend, {
        "sma": [100, 150, 200], "leverage": [1.0, 1.25, 1.5, 1.75, 2.0],
        "rebalance": [5]}),
}

# Sharpe barely changes with leverage, so picking by Sharpe can't choose a
# leverage level. For these, pick the highest return whose training-period
# volatility is no higher than SPY's, i.e. "more return for the same risk".
RISK_MATCHED = {"leveraged_trend"}


def evaluate(prices, fn, params, period):
    params = dict(params)
    rebalance = params.pop("rebalance", 1)
    targets = fn(prices, **params)
    return backtest.run(prices, targets, rebalance_every=rebalance,
                        start=period[0], end=period[1])


def fmt(row):
    return (f"{row['CAGR']:7.1%} {row['Vol']:6.1%} {row['Sharpe']:6.2f} "
            f"{row['MaxDD']:7.1%} {row['Turnover/yr']:7.1f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download prices")
    args = parser.parse_args()

    prices = load_prices(refresh=args.refresh)
    dates = pd.DatetimeIndex(prices.index)
    print(f"Data: {dates[0].date()} to {dates[-1].date()}, {prices.shape[1]} ETFs")
    print(f"Train {TRAIN[0]}..{TRAIN[1]}   Test {TEST[0]}..end\n")

    bench_train = evaluate(prices, strategies.buy_and_hold, {}, TRAIN)
    bench_test = evaluate(prices, strategies.buy_and_hold, {}, TEST)

    header = f"{'':{W}}{'CAGR':>7} {'Vol':>6} {'Sharpe':>6} {'MaxDD':>7} {'Turn':>8}   t vs SPY"
    chosen = {}
    print("TRAINING PERIOD (settings chosen here; leveraged_trend by best "
          "return at <= SPY's risk, the rest by best Sharpe)")
    print(header)
    print(f"{'SPY buy & hold':{W}}{fmt(backtest.metrics(bench_train))}")
    for name, (fn, grid) in GRIDS.items():
        results = []
        for combo in itertools.product(*grid.values()):
            params = dict(zip(grid.keys(), combo))
            r = evaluate(prices, fn, params, TRAIN)
            results.append((params, backtest.metrics(r), r))
        if name in RISK_MATCHED:
            spy_vol = backtest.metrics(bench_train)["Vol"]
            allowed = [res for res in results if res[1]["Vol"] <= spy_vol]
            params, m, r = max(allowed, key=lambda res: res[1]["CAGR"])
        else:
            params, m, r = max(results, key=lambda res: res[1]["Sharpe"])
        chosen[name] = params
        t = backtest.excess_tstat(r, bench_train)
        print(f"{name + ' ' + str(params):{W}}{fmt(m)}   {t:5.2f}")

    print("\nTEST PERIOD (never seen during selection)")
    print(header)
    print(f"{'SPY buy & hold':{W}}{fmt(backtest.metrics(bench_test))}")
    test_returns = {"SPY": bench_test}
    for name, params in chosen.items():
        fn = GRIDS[name][0]
        r = evaluate(prices, fn, params, TEST)
        test_returns[name] = r
        t = backtest.excess_tstat(r, bench_test)
        print(f"{name + ' ' + str(params):{W}}{fmt(backtest.metrics(r))}   {t:5.2f}")

    print("\nTEST PERIOD RETURNS BY YEAR")
    years = pd.DatetimeIndex(bench_test.index).year
    yearly = pd.DataFrame({k: (1 + v).groupby(years).prod() - 1
                           for k, v in test_returns.items()})
    print(yearly.map(lambda x: f"{x:6.1%}").to_string())


if __name__ == "__main__":
    main()
