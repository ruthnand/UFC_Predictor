import time

import requests
from bs4 import BeautifulSoup


class EventScraper:
    """Scrapes UFC events and per-event fight cards from ufc.com.

    The /events listing mixes upcoming and recent events; each is classified as
    upcoming or past using its main-card timestamp. Individual bouts are read
    from the event detail pages (/event/<slug>).
    """

    BASE_URL = "https://www.ufc.com"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def __init__(self, max_pages=5):
        self.max_pages = max_pages

    def _text(self, node):
        return node.get_text(" ", strip=True) if node else None

    def _parse_event_card(self, card):
        logo = card.select_one(".c-card-event--result__logo a")
        href = logo["href"] if logo and logo.has_attr("href") else None
        slug = href.rsplit("/", 1)[-1] if href else None
        date_node = card.select_one(".c-card-event--result__date")
        timestamp = None
        if date_node and date_node.get("data-main-card-timestamp"):
            try:
                timestamp = int(date_node["data-main-card-timestamp"])
            except (TypeError, ValueError):
                timestamp = None
        return {
            "name": self._text(card.select_one(".c-card-event--result__headline")),
            "slug": slug,
            "url": f"{self.BASE_URL}{href}" if href else None,
            "date": self._text(date_node),
            "timestamp": timestamp,
            "location": self._text(card.select_one(".c-card-event--result__location")),
        }

    def get_events(self):
        now = time.time()
        seen = set()
        upcoming, past = [], []

        for page in range(self.max_pages):
            try:
                response = requests.get(
                    f"{self.BASE_URL}/events",
                    params={"page": page},
                    headers=self.HEADERS,
                    timeout=20,
                )
            except requests.RequestException:
                break
            if response.status_code != 200:
                break
            soup = BeautifulSoup(response.text, "lxml")
            cards = soup.select(".c-card-event--result")
            if not cards:
                break
            for card in cards:
                event = self._parse_event_card(card)
                if not event["slug"] or event["slug"] in seen:
                    continue
                seen.add(event["slug"])
                if event["timestamp"] and event["timestamp"] < now:
                    event["upcoming"] = False
                    past.append(event)
                else:
                    event["upcoming"] = True
                    upcoming.append(event)

        upcoming.sort(key=lambda e: e["timestamp"] or 0)
        past.sort(key=lambda e: e["timestamp"] or 0, reverse=True)
        return {"upcoming": upcoming, "past": past}

    def _parse_bout(self, bout):
        corners = bout.select(".c-listing-fight__corner-name")
        fighters = []
        for corner in corners[:2]:
            anchor = corner.select_one("a")
            fighters.append({
                "name": corner.get_text(" ", strip=True),
                "url": anchor["href"] if anchor and anchor.has_attr("href") else None,
            })
        while len(fighters) < 2:
            fighters.append({"name": None, "url": None})

        weight_class = self._text(bout.select_one(".c-listing-fight__class-text"))
        return {
            "red": fighters[0],
            "blue": fighters[1],
            "weight_class": weight_class,
            "title_bout": bool(weight_class and "Title" in weight_class),
        }

    def get_event_card(self, slug):
        slug = slug.strip().strip("/")
        url = f"{self.BASE_URL}/event/{slug}"
        try:
            response = requests.get(url, headers=self.HEADERS, timeout=20)
        except requests.RequestException:
            return None
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, "lxml")
        headline = self._text(soup.select_one(".field--name-node-title")) or slug
        bouts = [self._parse_bout(b) for b in soup.select(".c-listing-fight")]
        if not bouts:
            return None
        return {"name": headline, "slug": slug, "url": url, "bouts": bouts}


if __name__ == "__main__":
    import json

    scraper = EventScraper()
    events = scraper.get_events()
    print("upcoming:", len(events["upcoming"]), "past:", len(events["past"]))
    if events["upcoming"]:
        first = events["upcoming"][0]
        print(json.dumps(first, indent=2))
        card = scraper.get_event_card(first["slug"])
        print(json.dumps(card, indent=2)[:1200])
