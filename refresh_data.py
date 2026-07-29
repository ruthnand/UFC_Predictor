#!/usr/bin/env python3
"""Refresh ufcstats caches, rebuild fighter states, and retrain models.

Intended for a weekly cron (local or Render Cron Job):

  python refresh_data.py
  python refresh_data.py --skip-scrape   # rebuild + retrain from existing caches
  python refresh_data.py --scrape-only   # only fetch new events/details

This keeps Elo/Glicko/form current so live predictions don't go stale.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Refresh UFC prediction data")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip ufcstats scrape; rebuild from existing caches")
    parser.add_argument("--scrape-only", action="store_true",
                        help="Only scrape; do not rebuild/train")
    parser.add_argument("--skip-train", action="store_true",
                        help="Rebuild fighter state/dataset but skip model training")
    args = parser.parse_args()

    if not args.skip_scrape:
        print("=== Scraping ufcstats.com (resumable) ===", flush=True)
        from ufcstats_collect import main as collect_main
        collect_main()
        if args.scrape_only:
            print("DONE (scrape only)")
            return

    print("=== Building point-in-time dataset + fighter states ===", flush=True)
    from ufcstats_dataset import build
    build(log=lambda *a: print(*a, flush=True))

    if args.skip_train:
        print("DONE (skip train)")
        return

    print("=== Retraining models + backtest ===", flush=True)
    from train import main as train_main
    train_main(rebuild=False, scrape=False)
    print("DONE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
