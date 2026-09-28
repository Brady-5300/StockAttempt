"""Run the trading bot once. Meant to be scheduled daily near the close.

    python trade.py --dry-run   # show what it would do, place no orders
    python trade.py             # trade on the paper account
"""
import argparse

from bot import live


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, place no orders")
    parser.add_argument("--force", action="store_true",
                        help="submit orders even while the market is closed (they fill at the next open)")
    parser.add_argument("--live", action="store_true",
                        help="allow real-money trading when ALPACA_PAPER=false")
    args = parser.parse_args()
    live.run(dry_run=args.dry_run, allow_live=args.live, force=args.force)


if __name__ == "__main__":
    main()
