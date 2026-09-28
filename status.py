"""How is the bot doing? Compares the paper account with simply holding SPY
since the bot's first trade.

    python status.py
"""
from datetime import timedelta
from typing import cast

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models import BarSet
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.models import Position, TradeAccount

from bot import live


def spy_price_at(data: StockHistoricalDataClient, when: pd.Timestamp) -> float:
    """SPY's price at the first minute bar on or after `when`."""
    request = StockBarsRequest(
        symbol_or_symbols="SPY", timeframe=TimeFrame(1, TimeFrameUnit("Min")),
        start=when.to_pydatetime(), end=(when + timedelta(days=5)).to_pydatetime(),
        limit=1, feed=DataFeed.IEX)
    bars = cast(BarSet, data.get_stock_bars(request)).df
    return float(bars["open"].iloc[0])


def main():
    key, secret, paper = live.load_keys()
    trading = TradingClient(key, secret, paper=paper)
    data = StockHistoricalDataClient(key, secret)

    if not live.LOG_FILE.exists():
        raise SystemExit("No trades logged yet. Turn the bot on first.")
    log = pd.read_csv(live.LOG_FILE)
    log = log[log["mode"] != "dry-run"]
    if log.empty:
        raise SystemExit("Only dry runs so far. Turn the bot on to start.")
    start = log.iloc[0]
    started = pd.Timestamp(start["time"])
    start_equity = float(start["equity"])

    account = cast(TradeAccount, trading.get_account())
    equity = float(account.equity or 0)
    spy_then = spy_price_at(data, started)
    latest = data.get_stock_latest_trade(
        StockLatestTradeRequest(symbol_or_symbols="SPY", feed=DataFeed.IEX))
    spy_now = float(cast(dict, latest)["SPY"].price)

    bot_ret = equity / start_equity - 1
    spy_ret = spy_now / spy_then - 1
    days = (pd.Timestamp.now(tz=started.tz) - started).days

    print(f"=== Trading bot status ({'paper' if paper else 'LIVE'}) ===")
    print(f"Running since {started:%b %d, %Y} ({days} days)\n")
    print(f"Account value  ${start_equity:>12,.2f} -> ${equity:>12,.2f}   {bot_ret:+.2%}")
    print(f"Holding SPY    ${spy_then:>12,.2f} -> ${spy_now:>12,.2f}   {spy_ret:+.2%}  (per share)")
    gap = bot_ret - spy_ret
    print(f"\nBot vs SPY: {gap:+.2%} "
          + ("ahead" if gap > 0 else "behind" if gap < 0 else "even"))
    print("(Expect bigger swings both ways: the bot holds 1.75x the S&P in uptrends.)")

    print("\nPositions:")
    for p in cast(list[Position], trading.get_all_positions()):
        pl = float(p.unrealized_pl or 0)
        plpc = float(p.unrealized_plpc or 0)
        print(f"  {p.symbol:5} ${float(p.market_value or 0):>12,.2f}   "
              f"gain/loss ${pl:>+10,.2f} ({plpc:+.2%})")
    print(f"  cash  ${float(account.cash or 0):>12,.2f}")

    last = log.iloc[-1]
    print(f"\nLast run {pd.Timestamp(last['time']):%b %d %I:%M %p}: signal {last['signal']} "
          f"(SPY ${last['spy_close']} vs 150-day avg ${last['sma']})")
    trades = log[log["orders"].fillna("").str.len() > 0]
    print(f"Days with trades: {len(trades)} of {len(log)} runs")


if __name__ == "__main__":
    main()
