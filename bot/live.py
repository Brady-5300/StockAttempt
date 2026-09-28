"""Daily trading run for the leveraged trend strategy on Alpaca.

Each run: read the signal from completed daily closes, compare the target
portfolio to what the account holds, and trade the difference. Running it
more than once a day is harmless; it only trades when something changed.
"""
import csv
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
from alpaca.trading.models import Clock, Order, Position, TradeAccount
from alpaca.trading.requests import MarketOrderRequest
from dotenv import load_dotenv

from . import strategies

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
STOP_FILE = ROOT / "STOP"
LOG_FILE = ROOT / "logs" / "trades.csv"
NY = ZoneInfo("America/New_York")

# Settings picked by run_backtest.py on the 2007-2016 training years.
SMA = 150
LEVERAGE = 1.75
SYMBOLS = ["SPY", "SSO", "SHY"]

CASH_BUFFER = 0.02   # leave 2% uninvested so slippage never overdraws the account
DRIFT_BAND = 0.05    # only rebalance when a holding is >5% of equity off target
MIN_ORDER = 5.0      # ignore orders smaller than $5
FINAL_STATUSES = {OrderStatus.FILLED, OrderStatus.CANCELED,
                  OrderStatus.REJECTED, OrderStatus.EXPIRED}


@dataclass
class PlannedOrder:
    symbol: str
    side: str          # "buy" or "sell"
    dollars: float
    close_all: bool = False

    def __str__(self):
        amount = "all" if self.close_all else f"${self.dollars:,.2f}"
        return f"{self.side.upper()} {amount} {self.symbol}"


def load_keys() -> tuple[str, str, bool]:
    load_dotenv(ENV_FILE)
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    paper = os.getenv("ALPACA_PAPER", "true").strip().lower() != "false"
    if not key or not secret or key.startswith("your_"):
        raise SystemExit(
            f"No Alpaca keys found. Copy .env.example to .env in\n  {ROOT}\n"
            "and paste your paper trading keys into it.")
    return key, secret, paper


def fetch_closes(data: StockHistoricalDataClient) -> pd.Series:
    """SPY daily closes up to yesterday. Today's unfinished bar is left out,
    matching the backtest (signal from one close, traded the next day)."""
    today = datetime.now(NY).replace(hour=0, minute=0, second=0, microsecond=0)
    request = StockBarsRequest(
        symbol_or_symbols="SPY", timeframe=TimeFrame(1, TimeFrameUnit("Day")),
        start=today - timedelta(days=400), end=today,
        adjustment=Adjustment.ALL, feed=DataFeed.IEX)
    bars = cast(BarSet, data.get_stock_bars(request)).df
    closes = bars.xs("SPY", level="symbol")["close"]
    dates = pd.DatetimeIndex(closes.index).tz_convert(NY)
    closes.index = dates
    return closes[dates < pd.Timestamp(today)]


def target_weights(closes: pd.Series) -> pd.Series:
    if len(closes) <= SMA:
        raise RuntimeError(f"Need more than {SMA} days of SPY history, got {len(closes)}.")
    weights = strategies.leveraged_trend(closes.to_frame("SPY"), sma=SMA, leverage=LEVERAGE)
    return weights.iloc[-1].reindex(SYMBOLS, fill_value=0.0)


def plan_orders(weights: pd.Series, equity: float,
                holdings: Mapping[str, float]) -> list[PlannedOrder]:
    """Orders that move the account from holdings ($ per symbol) to the
    target weights. Sells come first so their cash can fund the buys.
    Positions in other symbols are left alone and not counted as the bot's money."""
    unmanaged = sum(v for s, v in holdings.items() if s not in SYMBOLS)
    equity -= unmanaged
    investable = equity * (1 - CASH_BUFFER)
    targets = {s: float(weights.get(s, 0.0)) * investable for s in SYMBOLS}
    diffs = {s: targets[s] - holdings.get(s, 0.0) for s in SYMBOLS}
    if max(abs(d) for d in diffs.values()) < DRIFT_BAND * equity:
        return []

    orders = []
    for symbol, diff in diffs.items():
        if abs(diff) < MIN_ORDER:
            continue
        if diff < 0 and targets[symbol] < MIN_ORDER:
            orders.append(PlannedOrder(symbol, "sell", holdings[symbol], close_all=True))
        else:
            orders.append(PlannedOrder(symbol, "buy" if diff > 0 else "sell", round(abs(diff), 2)))
    return sorted(orders, key=lambda o: o.side != "sell")


def submit(trading: TradingClient, order: PlannedOrder) -> Order:
    if order.close_all:
        return cast(Order, trading.close_position(order.symbol))
    request = MarketOrderRequest(
        symbol=order.symbol, notional=order.dollars,
        side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY)
    return cast(Order, trading.submit_order(request))


def wait_for_fill(trading: TradingClient, order: Order, timeout: float = 60) -> Order:
    deadline = time.monotonic() + timeout
    while order.status not in FINAL_STATUSES and time.monotonic() < deadline:
        time.sleep(2)
        order = cast(Order, trading.get_order_by_id(order.id))
    return order


def log_run(row: dict):
    LOG_FILE.parent.mkdir(exist_ok=True)
    new = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def run(dry_run=False, allow_live=False, force=False):
    key, secret, paper = load_keys()
    mode = "paper" if paper else "LIVE"
    if not paper and not allow_live:
        raise SystemExit("ALPACA_PAPER=false points at real money. Refusing unless "
                         "you also pass --live.")
    if STOP_FILE.exists():
        raise SystemExit(f"Kill switch is on ({STOP_FILE} exists). Delete that file to resume.")

    trading = TradingClient(key, secret, paper=paper)
    data = StockHistoricalDataClient(key, secret)

    account = cast(TradeAccount, trading.get_account())
    if account.trading_blocked or account.account_blocked:
        raise SystemExit("Alpaca says this account is blocked from trading.")
    clock = cast(Clock, trading.get_clock())
    if not clock.is_open and not (force or dry_run):
        print(f"Market is closed (next open {clock.next_open:%Y-%m-%d %H:%M} ET). Nothing to do.")
        return

    closes = fetch_closes(data)
    sma = closes.rolling(SMA).mean().iloc[-1]
    last = closes.iloc[-1]
    weights = target_weights(closes)
    equity = float(account.equity or 0)
    positions = cast(list[Position], trading.get_all_positions())
    holdings = {p.symbol: float(p.market_value or 0) for p in positions}

    signal = "UPTREND" if last > sma else "DOWNTREND"
    print(f"[{mode}] SPY close {closes.index[-1]:%Y-%m-%d}: ${last:,.2f}, "
          f"{SMA}-day average ${sma:,.2f} -> {signal}")
    print(f"Target: " + ", ".join(f"{s} {w:.0%}" for s, w in weights.items()))
    print(f"Equity ${equity:,.2f}; holding: "
          + (", ".join(f"{s} ${v:,.2f}" for s, v in holdings.items()) or "nothing"))
    others = sorted(set(holdings) - set(SYMBOLS))
    if others:
        print(f"Note: ignoring positions the bot doesn't manage: {', '.join(others)}")

    orders = plan_orders(weights, equity, holdings)
    print("Orders: " + ("; ".join(map(str, orders)) or "none (already on target)"))

    results = []
    if not dry_run:
        for order in orders:
            try:
                placed = wait_for_fill(trading, submit(trading, order))
                results.append(f"{order} -> {placed.status.value}")
            except APIError as e:
                results.append(f"{order} -> ERROR {e}")
                print(f"Order failed: {order}: {e}")
                if order.side == "sell":
                    break  # buys would be funded by this sell; stop here
        for line in results:
            print("  " + line)

    log_run({
        "time": datetime.now(NY).isoformat(timespec="seconds"),
        "mode": "dry-run" if dry_run else mode,
        "spy_close": round(float(last), 2), "sma": round(float(sma), 2),
        "signal": signal, "equity": round(equity, 2),
        "orders": "; ".join(results if not dry_run else map(str, orders)),
    })
