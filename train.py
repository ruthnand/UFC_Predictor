"""Build the point-in-time dataset and train + backtest the fight models.

Usage:
  python train.py                # train + backtest on existing ufc_pit_dataset.csv
  python train.py --rebuild      # rebuild dataset from the ufcstats caches first
  python train.py --scrape       # re-scrape ufcstats (needs playwright), then rebuild

The scrape step requires playwright + chromium (see requirements-train.txt); the
served app only needs the committed models + fighter_state.json. The training CSV
and ufcstats caches are gitignored rebuild artifacts — use --scrape on a fresh clone.
"""

import os
import sys

import pandas as pd

from backtest import Backtester
from models import FightPredictor

CSV_PATH = "ufc_pit_dataset.csv"
UFCSTATS_CACHES = (
    "ufcstats_events.json",
    "ufcstats_fighters.json",
)


def _require_nonempty(df):
    if df is None or len(df) == 0:
        raise SystemExit(
            f"Training dataset is empty (no rows in {CSV_PATH}). "
            "Run `python train.py --scrape` (needs requirements-train.txt + "
            "`python -m playwright install chromium`) to fetch ufcstats data, "
            "or place a non-empty ufc_pit_dataset.csv in the repo root."
        )


def main(rebuild=False, scrape=False):
    if scrape:
        print("Scraping ufcstats.com (playwright challenge solve + bulk fetch)...")
        from ufcstats_collect import main as collect_main
        collect_main()
        rebuild = True

    if rebuild or not os.path.exists(CSV_PATH):
        missing = [path for path in UFCSTATS_CACHES if not os.path.exists(path)]
        if missing:
            raise SystemExit(
                f"Cannot rebuild: missing ufcstats caches ({', '.join(missing)}). "
                "Run `python train.py --scrape` (needs requirements-train.txt + "
                "`python -m playwright install chromium`), or provide a local "
                f"{CSV_PATH} and run without --rebuild."
            )
        print("Building point-in-time dataset from ufcstats caches...")
        from ufcstats_dataset import build
        df = build(log=lambda *a: print(*a, flush=True))
    else:
        print(f"Loading cached dataset {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)

    _require_nonempty(df)

    predictor = FightPredictor()
    metrics = predictor.train(df)
    predictor.save()
    print("Training complete. Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("Models saved to models/")

    print("Running forward-chaining backtest...")
    report = Backtester().run_and_cache(df)
    for name, m in report["models"].items():
        print(f"  {name:10s} acc={m['accuracy']:.3f} auc={m['auc']:.3f} "
              f"logloss={m['log_loss']:.3f} brier={m['brier']:.3f}")
    print(f"  baseline logloss={report['baseline_log_loss']:.3f} "
          f"(evaluated on {report['n_evaluated']} rows)")
    print("Backtest saved to models/backtest.json")


if __name__ == "__main__":
    main(rebuild="--rebuild" in sys.argv, scrape="--scrape" in sys.argv)
