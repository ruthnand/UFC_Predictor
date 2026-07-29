import os

import requests


def american_to_prob(odds):
    """Convert American moneyline odds to an implied win probability."""
    if odds is None:
        return None
    odds = float(odds)
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def devig(prob_a, prob_b):
    """Normalize two implied probabilities so they sum to 1 (removes the vig)."""
    if prob_a is None or prob_b is None:
        return prob_a, prob_b
    total = prob_a + prob_b
    if total <= 0:
        return prob_a, prob_b
    return prob_a / total, prob_b / total


class OddsProvider:
    """Fetches UFC/MMA moneyline odds from The Odds API (the-odds-api.com).

    Requires an API key supplied via the ``ODDS_API_KEY`` environment variable
    (or passed explicitly). If no key is configured, methods return a payload
    with ``available: False`` so the rest of the app keeps working.
    """

    SPORT = "mma_mixed_martial_arts"
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")

    def is_available(self):
        return bool(self.api_key)

    def _matchup_from_event(self, event):
        name_a = event.get("home_team")
        name_b = event.get("away_team")

        # Average each fighter's implied probability across all bookmakers.
        totals = {name_a: [], name_b: []}
        books = []
        for book in event.get("bookmakers", []):
            prices = {}
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    prices[outcome["name"]] = outcome.get("price")
            if not prices:
                continue
            books.append({"bookmaker": book.get("title"), "prices": prices})
            for name in (name_a, name_b):
                prob = american_to_prob(prices.get(name))
                if prob is not None:
                    totals[name].append(prob)

        avg_a = sum(totals[name_a]) / len(totals[name_a]) if totals[name_a] else None
        avg_b = sum(totals[name_b]) / len(totals[name_b]) if totals[name_b] else None
        fair_a, fair_b = devig(avg_a, avg_b)

        return {
            "fighter_a": name_a,
            "fighter_b": name_b,
            "commence_time": event.get("commence_time"),
            "bookmakers": books,
            "implied_probability": {
                name_a: fair_a,
                name_b: fair_b,
            },
        }

    def get_mma_odds(self, regions="us", markets="h2h", odds_format="american"):
        if not self.is_available():
            return {
                "available": False,
                "reason": "ODDS_API_KEY not set",
                "matchups": [],
            }
        try:
            response = requests.get(
                f"{self.BASE_URL}/sports/{self.SPORT}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": odds_format,
                },
                timeout=20,
            )
        except requests.RequestException as exc:
            return {"available": False, "reason": str(exc), "matchups": []}

        if response.status_code != 200:
            return {
                "available": False,
                "reason": f"Odds API returned {response.status_code}: {response.text[:200]}",
                "matchups": [],
            }

        matchups = [self._matchup_from_event(e) for e in response.json()]
        return {
            "available": True,
            "requests_remaining": response.headers.get("x-requests-remaining"),
            "matchups": matchups,
        }

    @staticmethod
    def _name_key(name):
        return (name or "").strip().lower()

    def get_matchup_odds(self, name_a, name_b):
        """Find odds for a specific matchup by (loose) fighter-name matching."""
        data = self.get_mma_odds()
        if not data.get("available"):
            return data
        a, b = self._name_key(name_a), self._name_key(name_b)
        a_last, b_last = a.split()[-1] if a else "", b.split()[-1] if b else ""
        for matchup in data["matchups"]:
            names = {self._name_key(matchup["fighter_a"]), self._name_key(matchup["fighter_b"])}
            joined = " ".join(names)
            if a_last and b_last and a_last in joined and b_last in joined:
                return {"available": True, "matchup": matchup}
        return {"available": True, "matchup": None, "reason": "no odds found for matchup"}


if __name__ == "__main__":
    import json

    provider = OddsProvider()
    print("available:", provider.is_available())
    print(json.dumps(provider.get_mma_odds(), indent=2)[:1500])
