import json
import os
import re

from fighter import FighterStatsExtractor
from name_resolver import FighterNameResolver

# Per-fighter numeric features. Matchup features are the differences A - B, so a
# positive value means fighter A has the edge in that category.
FEATURE_KEYS = [
    "age",
    "height",
    "reach",
    "leg_reach",
    "slpm",          # significant strikes landed per minute
    "sapm",          # significant strikes absorbed per minute
    "str_acc",       # striking accuracy (landed / attempted)
    "td_avg",        # takedowns per 15 minutes
    "td_acc",        # takedown accuracy
    "sub_avg",       # submission attempts per 15 minutes
    "win_rate",
    "total_fights",
    "ko_rate",       # KO wins / total wins
    "sub_rate",      # submission wins / total wins
]


def _num(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def _safe_div(numerator, denominator):
    n, d = _num(numerator), _num(denominator)
    if n is None or d is None or d == 0:
        return None
    return n / d


def fighter_features(stats):
    """Turn a raw stats dict from FighterStatsExtractor into numeric features."""
    if not stats:
        return None

    wins = _num(stats.get("wins")) or 0
    losses = _num(stats.get("losses")) or 0
    draws = _num(stats.get("draws")) or 0
    total = wins + losses + draws

    return {
        "age": _num(stats.get("age") or stats.get("Age")),
        "height": _num(stats.get("Height")),
        "reach": _num(stats.get("Reach")),
        "leg_reach": _num(stats.get("Leg reach")),
        "slpm": _num(stats.get("sig_str_landed_per_min")),
        "sapm": _num(stats.get("sig_str_absorbed_per_min")),
        "str_acc": _safe_div(stats.get("strikes_landed"), stats.get("strikes_attemped")),
        "td_avg": _num(stats.get("takedown_avg")),
        "td_acc": _safe_div(stats.get("takedown_landed"), stats.get("takedown_attempted")),
        "sub_avg": _num(stats.get("submission_avg")),
        "win_rate": (wins / total) if total else None,
        "total_fights": total if total else None,
        "ko_rate": _safe_div(stats.get("knockouts"), wins) if wins else None,
        "sub_rate": _safe_div(stats.get("submissions"), wins) if wins else None,
    }


def matchup_vector(features_a, features_b):
    """Differential feature vector (A - B). Missing values count as 0 (even)."""
    vector = []
    for key in FEATURE_KEYS:
        a = features_a.get(key) if features_a else None
        b = features_b.get(key) if features_b else None
        if a is None or b is None:
            vector.append(0.0)
        else:
            vector.append(float(a) - float(b))
    return vector


def feature_names():
    return [f"{key}_diff" for key in FEATURE_KEYS]


class FeatureCache:
    """Caches raw fighter stats to disk so we don't re-scrape the same athlete."""

    def __init__(self, cache_file="fighter_stats_cache.json"):
        self.cache_file = cache_file
        self._cache = self._load()

    def _load(self):
        if not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file) as handle:
                return json.load(handle)
        except (ValueError, OSError):
            return {}

    def _save(self):
        try:
            with open(self.cache_file, "w") as handle:
                json.dump(self._cache, handle)
        except OSError:
            pass

    def get_stats(self, first, middle, last, use_resolver=False):
        resolver = FighterNameResolver()
        if use_resolver:
            resolved = resolver.resolve(first, middle, last)
            if not resolved:
                return None
            first, middle, last = resolved["first"], resolved["middle"], resolved["last"]

        key = resolver.build_slug(first, middle, last)
        if not key:
            return None
        if key in self._cache:
            return self._cache[key]

        stats = FighterStatsExtractor().get_fighter_stats(first, middle, last)
        if stats:
            self._cache[key] = stats
            self._save()
        return stats

    def get_features(self, first, middle, last, use_resolver=False):
        return fighter_features(self.get_stats(first, middle, last, use_resolver=use_resolver))


if __name__ == "__main__":
    cache = FeatureCache()
    a = cache.get_features("islam", "", "makhachev")
    b = cache.get_features("ian", "machado", "garry")
    print("A:", a)
    print("B:", b)
    print("diff:", dict(zip(feature_names(), matchup_vector(a, b))))
