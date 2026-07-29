import requests
from bs4 import BeautifulSoup


class UFCRankingScraper:
    """Scrapes the official UFC rankings (champion + top 15 per division)."""

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def __init__(self):
        self.url = "https://www.ufc.com/rankings"
        self.soup = None
        self.groupings = None

    def fetch_data(self):
        page = requests.get(self.url, headers=self.HEADERS, timeout=20)
        self.soup = BeautifulSoup(page.text, "lxml")
        self.groupings = self.soup.select(".view-grouping")

    def _parse_grouping(self, grouping):
        title_node = grouping.select_one(".view-grouping-header")
        division = title_node.get_text(strip=True) if title_node else None

        champ_node = grouping.select_one(".rankings--athlete--champion h5")
        champion = champ_node.get_text(strip=True) if champ_node else None

        fighters = []
        for row in grouping.select("tbody tr"):
            rank_node = row.select_one(".views-field-weight-class-rank")
            name_node = row.select_one(".views-field-title")
            if not name_node:
                continue
            fighters.append({
                "rank": rank_node.get_text(strip=True) if rank_node else None,
                "name": name_node.get_text(strip=True),
            })

        return {"division": division, "champion": champion, "fighters": fighters}

    def get_all_rankings(self):
        if self.groupings is None:
            self.fetch_data()
        rankings = []
        for grouping in self.groupings:
            parsed = self._parse_grouping(grouping)
            if parsed["division"] and (parsed["champion"] or parsed["fighters"]):
                rankings.append(parsed)
        return rankings

    def get_division(self, weight_class):
        for ranking in self.get_all_rankings():
            if weight_class.lower() in (ranking["division"] or "").lower():
                return ranking
        return None

    def get_top_fighters(self, weight_class):
        """Backward-compatible: flat list of [champion, ...ranked fighters]."""
        ranking = self.get_division(weight_class)
        if not ranking:
            return []
        names = []
        if ranking["champion"]:
            names.append(ranking["champion"])
        names.extend(f["name"] for f in ranking["fighters"])
        return names


if __name__ == "__main__":
    import json

    print(json.dumps(UFCRankingScraper().get_all_rankings()[:3], indent=2))
