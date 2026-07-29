import json
import os

import pandas as pd

from features import FeatureCache, fighter_features, matchup_vector, feature_names
from fight_history import FightHistoryExtractor
from name_resolver import FighterNameResolver
from top_fighters import UFCRankingScraper


class DatasetBuilder:
    """Builds a fight-outcome training set from the ranked-fighter universe.

    For every bout between two fighters we have stats for, two balanced rows are
    emitted: (winner - loser -> 1) and (loser - winner -> 0). This removes any
    left/right orientation bias.

    NOTE: features are current career-aggregate stats, so this is a demonstration
    pipeline for the modeling approach rather than a leakage-free predictor.
    """

    def __init__(self, history_cache_file="fight_history_cache.json", verbose=True):
        self.feature_cache = FeatureCache()
        self.resolver = FighterNameResolver()
        self.history_cache_file = history_cache_file
        self.verbose = verbose
        self._history_cache = self._load_history_cache()

    def _log(self, *args):
        if self.verbose:
            print(*args, flush=True)

    def _load_history_cache(self):
        if not os.path.exists(self.history_cache_file):
            return {}
        try:
            with open(self.history_cache_file) as handle:
                return json.load(handle)
        except (ValueError, OSError):
            return {}

    def _save_history_cache(self):
        try:
            with open(self.history_cache_file, "w") as handle:
                json.dump(self._history_cache, handle)
        except OSError:
            pass

    def _slug_from_url(self, url):
        if not url:
            return None
        return url.rstrip("/").rsplit("/", 1)[-1].lower()

    def ranked_universe(self):
        names = []
        seen = set()
        for division in UFCRankingScraper().get_all_rankings():
            for name in ([division["champion"]] if division["champion"] else []) + [
                f["name"] for f in division["fighters"]
            ]:
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    names.append(name)
        return names

    def _resolve_slug(self, name):
        parts = name.split()
        if len(parts) == 1:
            first, middle, last = parts[0], "", ""
        elif len(parts) == 2:
            first, middle, last = parts[0], "", parts[1]
        else:
            first, middle, last = parts[0], " ".join(parts[1:-1]), parts[-1]
        resolved = self.resolver.resolve(first, middle, last)
        if not resolved:
            return None
        return resolved

    def _get_history(self, first, middle, last):
        slug = self.resolver.build_slug(first, middle, last)
        if slug in self._history_cache:
            return self._history_cache[slug]
        history = FightHistoryExtractor().get_fight_history(first, middle, last)
        self._history_cache[slug] = history
        self._save_history_cache()
        return history

    def build(self, max_fighters=None):
        universe = self.ranked_universe()
        if max_fighters:
            universe = universe[:max_fighters]
        self._log(f"Universe: {len(universe)} fighters")

        # Resolve everyone to a slug and pull their features.
        features_by_slug = {}
        for i, name in enumerate(universe, 1):
            try:
                resolved = self._resolve_slug(name)
                if not resolved:
                    continue
                slug = self.resolver.build_slug(resolved["first"], resolved["middle"], resolved["last"])
                stats = self.feature_cache.get_stats(resolved["first"], resolved["middle"], resolved["last"])
                feats = fighter_features(stats)
                if feats:
                    features_by_slug[slug] = {"name": resolved["full"], "features": feats, "resolved": resolved}
            except Exception as exc:  # noqa: BLE001 - keep the build going
                self._log(f"  ! skipped {name}: {type(exc).__name__}")
            if i % 20 == 0:
                self._log(f"  features {i}/{len(universe)} ({len(features_by_slug)} usable)")

        self._log(f"Usable fighters with features: {len(features_by_slug)}")

        # Walk each fighter's history and emit rows for bouts between known fighters.
        rows = []
        seen_bouts = set()
        for idx, (slug, info) in enumerate(features_by_slug.items(), 1):
            resolved = info["resolved"]
            try:
                history = self._get_history(resolved["first"], resolved["middle"], resolved["last"])
            except Exception as exc:  # noqa: BLE001
                self._log(f"  ! history failed for {info['name']}: {type(exc).__name__}")
                history = []
            for bout in history:
                result = (bout.get("result") or "").lower()
                if result not in ("win", "loss"):
                    continue
                opp_slug = self._slug_from_url(bout.get("opponent_url"))
                if not opp_slug or opp_slug not in features_by_slug:
                    continue

                bout_key = (frozenset((slug, opp_slug)), bout.get("date"))
                if bout_key in seen_bouts:
                    continue
                seen_bouts.add(bout_key)

                if result == "win":
                    winner, loser = slug, opp_slug
                else:
                    winner, loser = opp_slug, slug

                wf = features_by_slug[winner]["features"]
                lf = features_by_slug[loser]["features"]
                # Balanced pair of rows.
                rows.append(self._row(features_by_slug[winner]["name"],
                                      features_by_slug[loser]["name"],
                                      matchup_vector(wf, lf), 1, bout.get("date")))
                rows.append(self._row(features_by_slug[loser]["name"],
                                      features_by_slug[winner]["name"],
                                      matchup_vector(lf, wf), 0, bout.get("date")))
            if idx % 20 == 0:
                self._log(f"  histories {idx}/{len(features_by_slug)} ({len(rows)} rows)")

        columns = ["fighter_a", "fighter_b", "date"] + feature_names() + ["label"]
        df = pd.DataFrame(rows, columns=columns)
        self._log(f"Built dataset: {len(df)} rows, {df['label'].mean():.3f} positive rate")
        return df

    def _row(self, name_a, name_b, vector, label, date):
        return [name_a, name_b, date] + list(vector) + [label]

    def build_and_cache(self, csv_path="fight_dataset.csv", max_fighters=None):
        df = self.build(max_fighters=max_fighters)
        df.to_csv(csv_path, index=False)
        self._log(f"Saved {csv_path}")
        return df


if __name__ == "__main__":
    DatasetBuilder().build_and_cache()
