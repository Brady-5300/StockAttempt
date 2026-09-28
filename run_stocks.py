"""Individual-stock backtest on the S&P 500, using the index's real
membership on each date. Same train/test split and scoring as run_backtest."""
import argparse
import itertools

import pandas as pd

from bot import backtest, strategies
from bot.data import load_prices
from bot.universe import load_membership, load_stock_prices, membership_matrix
from run_backtest import TEST, TRAIN, evaluate, fmt

W = 66
GRID = {"lookback": [126, 252], "top_n": [10, 20, 50], "trend_sma": [0, 150],
        "rebalance": [21]}


def cagr(r: pd.Series) -> float:
    return backtest.metrics(r)["CAGR"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-download everything")
    args = parser.parse_args()

    etfs = load_prices(refresh=args.refresh)
    membership = load_membership(refresh=args.refresh)
    membership = membership[membership.index >= "2006-01-01"]
    stocks = load_stock_prices(sorted(set().union(*membership)), refresh=args.refresh)
    prices = etfs.join(stocks.drop(columns=etfs.columns, errors="ignore"), how="left")
    members = membership_matrix(membership, pd.DatetimeIndex(prices.index))
    members = members.reindex(columns=stocks.columns, fill_value=False)
    print(f"{stocks.shape[1]} stocks with prices, of {len(set().union(*membership))} "
          f"that were ever in the S&P 500 since 2006\n")

    # How much does the missing data flatter results? Our equal-weight index
    # vs the real equal-weight ETF. Anything we beat RSP by here is fake.
    print("SURVIVORSHIP BIAS CHECK (our equal-weight S&P 500 vs the real RSP ETF)")
    bias = {}
    for label, period in (("train", TRAIN), ("test", TEST)):
        ours = backtest.run(prices, strategies.equal_weight_index(prices, members),
                            rebalance_every=21, start=period[0], end=period[1])
        rsp = evaluate(prices, strategies.buy_and_hold, {"asset": "RSP"}, period)
        bias[label] = cagr(ours) - cagr(rsp)
        print(f"  {label}: ours {cagr(ours):.1%}/yr, RSP {cagr(rsp):.1%}/yr "
              f"-> bias about {bias[label]:+.1%}/yr")

    header = f"{'':{W}}{'CAGR':>7} {'Vol':>6} {'Sharpe':>6} {'MaxDD':>7} {'Turn':>8}   t vs SPY"
    bench = {p: evaluate(prices, strategies.buy_and_hold, {}, p) for p in (TRAIN, TEST)}
    bot_params = {"sma": 150, "leverage": 1.75, "rebalance": 5}

    def momentum(params, period):
        params = dict(params)
        rebalance = params.pop("rebalance")
        targets = strategies.stock_momentum(prices, members, **params)
        return backtest.run(prices, targets, rebalance_every=rebalance,
                            start=period[0], end=period[1])

    print("\nTRAINING PERIOD (settings chosen by best Sharpe)")
    print(header)
    print(f"{'SPY buy & hold':{W}}{fmt(backtest.metrics(bench[TRAIN]))}")
    results = []
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(GRID.keys(), combo))
        r = momentum(params, TRAIN)
        results.append((params, backtest.metrics(r), r))
    params, m, r = max(results, key=lambda res: res[1]["Sharpe"])
    t = backtest.excess_tstat(r, bench[TRAIN])
    print(f"{'stock_momentum ' + str(params):{W}}{fmt(m)}   {t:5.2f}")

    print("\nTEST PERIOD (never seen during selection)")
    print(header)
    print(f"{'SPY buy & hold':{W}}{fmt(backtest.metrics(bench[TEST]))}")
    test = {"SPY": bench[TEST]}
    test["current bot"] = evaluate(prices, strategies.leveraged_trend, bot_params, TEST)
    test["stock_momentum"] = momentum(params, TEST)
    for name in ("current bot", "stock_momentum"):
        t = backtest.excess_tstat(test[name], bench[TEST])
        label = name + (" " + str(params) if name == "stock_momentum" else " (leveraged trend)")
        print(f"{label:{W}}{fmt(backtest.metrics(test[name]))}   {t:5.2f}")
    adjusted = cagr(test["stock_momentum"]) - bias["test"]
    print(f"\nstock_momentum return after subtracting the test-period bias: {adjusted:.1%}/yr")

    print("\nTEST PERIOD RETURNS BY YEAR")
    years = pd.DatetimeIndex(bench[TEST].index).year
    yearly = pd.DataFrame({k: (1 + v).groupby(years).prod() - 1 for k, v in test.items()})
    print(yearly.map(lambda x: f"{x:6.1%}").to_string())


if __name__ == "__main__":
    main()
