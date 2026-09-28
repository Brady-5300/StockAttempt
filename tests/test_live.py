import pandas as pd

from bot.live import CASH_BUFFER, plan_orders

UP = pd.Series({"SPY": 0.25, "SSO": 0.75, "SHY": 0.0})
DOWN = pd.Series({"SPY": 0.0, "SSO": 0.0, "SHY": 1.0})


def test_empty_account_buys_target():
    orders = plan_orders(UP, 100_000, {})
    investable = 100_000 * (1 - CASH_BUFFER)
    assert [(o.symbol, o.side) for o in orders] == [("SPY", "buy"), ("SSO", "buy")]
    assert orders[0].dollars == round(0.25 * investable, 2)
    assert orders[1].dollars == round(0.75 * investable, 2)


def test_on_target_does_nothing():
    holdings = {"SPY": 24_600, "SSO": 73_300}
    assert plan_orders(UP, 100_000, holdings) == []


def test_flip_to_downtrend_sells_everything_first():
    holdings = {"SPY": 24_500, "SSO": 73_500}
    orders = plan_orders(DOWN, 100_000, holdings)
    assert [(o.symbol, o.side, o.close_all) for o in orders] == [
        ("SPY", "sell", True), ("SSO", "sell", True), ("SHY", "buy", False)]


def test_ignores_unmanaged_positions():
    orders = plan_orders(DOWN, 100_000, {"AAPL": 5_000})
    assert [o.symbol for o in orders] == ["SHY"]
    assert orders[0].dollars == round(95_000 * (1 - CASH_BUFFER), 2)
