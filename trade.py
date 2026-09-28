"""Run the trading bot.

    python trade.py --loop      # turn the bot ON: trades now, then every trading day
    python trade.py --dry-run   # show what it would do right now, place no orders
    python trade.py             # check and trade once, right now
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
    parser.add_argument("--loop", action="store_true",
                        help="keep running and trade once each trading day before the close")
    args = parser.parse_args()
    if args.loop:
        try:
            live.run_forever(allow_live=args.live, dry_run=args.dry_run)
        except KeyboardInterrupt:
            print("\nBot is OFF.")
    else:
        live.run(dry_run=args.dry_run, allow_live=args.live, force=args.force)


if __name__ == "__main__":
    main()
