import json
import os
import time
from difflib import SequenceMatcher

import requests

from all_active_fighter import ActiveFighterNameExtractor
from name_normalization import fighter_name_slug, normalize_fighter_name


class FighterNameResolver:
    """Resolves a (possibly misspelled) fighter name to a valid ufc.com athlete.

    Strategy:
      1. Try the name as-is by checking whether its athlete page exists.
      2. If not, fuzzy-match against the active-fighter roster and validate the
         best match's page.

    The roster is scraped from ufc.com/athletes/all (slow), so it is cached to
    a local JSON file and only refreshed when stale or missing.
    """

    BASE_URL = "https://www.ufc.com/athlete"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def __init__(self, cache_file="active_fighters.json", cache_ttl_seconds=86400):
        self.cache_file = cache_file
        self.cache_ttl_seconds = cache_ttl_seconds

    def build_slug(self, first_name, middle_name, last_name):
        middle = None if middle_name in (None, "", "&") else middle_name
        return fighter_name_slug(first_name, middle, last_name)

    def page_exists(self, first_name, middle_name, last_name):
        slug = self.build_slug(first_name, middle_name, last_name)
        if not slug:
            return False
        try:
            response = requests.get(
                f"{self.BASE_URL}/{slug}", headers=self.HEADERS, timeout=15
            )
        except requests.RequestException:
            return False
        return response.status_code == 200 and "hero-profile__name" in response.text

    def _build_records(self, rows):
        records = []
        for first, middle, last in rows:
            full = " ".join(p for p in (first, middle, last) if p)
            records.append(
                {"first": first, "middle": middle, "last": last, "full": full}
            )
        return records

    def _load_cache(self):
        if not os.path.exists(self.cache_file):
            return None
        try:
            with open(self.cache_file) as cache:
                data = json.load(cache)
        except (ValueError, OSError):
            return None
        if time.time() - data.get("fetched_at", 0) > self.cache_ttl_seconds:
            return None
        return data.get("fighters")

    def _save_cache(self, records):
        try:
            with open(self.cache_file, "w") as cache:
                json.dump({"fetched_at": time.time(), "fighters": records}, cache, indent=1)
        except OSError:
            pass

    def get_roster(self, force_refresh=False):
        if not force_refresh:
            cached = self._load_cache()
            if cached:
                return cached
        rows = ActiveFighterNameExtractor().extractName()
        records = self._build_records(rows)
        self._save_cache(records)
        return records

    @staticmethod
    def _ratio(a, b):
        return SequenceMatcher(None, a, b).ratio()

    def fuzzy_match(self, name, cutoff=0.8):
        if not name or not name.strip():
            return None
        query = normalize_fighter_name(name)
        query_sorted = " ".join(sorted(query.split()))

        best_record = None
        best_score = 0.0
        for record in self.get_roster():
            full = normalize_fighter_name(record["full"])
            # Compare both as-is and with tokens sorted, so "strickland sean"
            # still matches "sean strickland".
            score = max(
                self._ratio(query, full),
                self._ratio(query_sorted, " ".join(sorted(full.split()))),
            )
            if score > best_score:
                best_score = score
                best_record = record

        return best_record if best_score >= cutoff else None

    def resolve(self, first_name, middle_name, last_name, cutoff=0.8):
        """Return a dict describing the resolved athlete, or None if not found.

        {first, middle, last, full, exact, query}
        ``exact`` is True when the supplied name resolved directly, False when a
        fuzzy roster match was used instead.
        """
        query = " ".join(p for p in (first_name, middle_name, last_name) if p)

        if self.page_exists(first_name, middle_name, last_name):
            return {
                "first": normalize_fighter_name(first_name),
                "middle": normalize_fighter_name(middle_name),
                "last": normalize_fighter_name(last_name),
                "full": query,
                "exact": True,
                "query": query,
            }

        match = self.fuzzy_match(query, cutoff=cutoff)
        if match and self.page_exists(match["first"], match["middle"], match["last"]):
            return {
                "first": normalize_fighter_name(match["first"]),
                "middle": normalize_fighter_name(match["middle"]),
                "last": normalize_fighter_name(match["last"]),
                "full": match["full"],
                "exact": False,
                "query": query,
            }

        return None


if __name__ == "__main__":
    resolver = FighterNameResolver()
    for test in [("sean", "", "strickland"), ("shawn", "", "stricklan"), ("bogus", "", "person")]:
        print(test, "->", resolver.resolve(*test))
