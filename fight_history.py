from bs4 import BeautifulSoup
import requests

from name_normalization import fighter_name_slug


class FightHistoryExtractor:
    """Scrapes a fighter's full bout history from their ufc.com athlete page.

    The athlete page renders three result cards per page and exposes the rest
    through a ``?page=N`` pager, so we walk pages until one comes back empty.
    """

    BASE_URL = "https://www.ufc.com/athlete"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    def __init__(self, max_pages=30):
        # Safety cap so a markup change can never cause an infinite loop.
        self.max_pages = max_pages

    def build_slug(self, first_name, middle_name, last_name):
        middle = None if middle_name in (None, "", "&") else middle_name
        return fighter_name_slug(first_name, middle, last_name)

    def _text(self, node):
        return node.get_text(strip=True) if node else None

    def _parse_card(self, card, slug):
        headline_node = card.select_one(".c-card-event--athlete-results__headline")
        headline = headline_node.get_text(" ", strip=True) if headline_node else None
        date = self._text(card.select_one(".c-card-event--athlete-results__date"))

        # Round / Time / Method live as label -> value pairs.
        details = {}
        for result in card.select(".c-card-event--athlete-results__result"):
            label = self._text(result.select_one(".c-card-event--athlete-results__result-label"))
            value = self._text(result.select_one(".c-card-event--athlete-results__result-text"))
            if label:
                details[label.lower()] = value

        # Each matchup has two corner images; one links to this fighter and
        # carries the win/loss class, the other is the opponent.
        outcome = None
        opponent_name = None
        opponent_url = None
        for image in card.select(".c-card-event--athlete-results__image"):
            classes = image.get("class", [])
            corner_outcome = next(
                (o for o in ("win", "loss", "draw", "nc") if o in classes), None
            )
            anchor = image.select_one("a")
            href = anchor["href"] if anchor and anchor.has_attr("href") else ""
            if slug in href:
                outcome = corner_outcome
            else:
                img = image.select_one("img")
                if img and img.get("alt"):
                    opponent_name = img.get("alt").strip()
                opponent_url = href or None

        # Fall back to the headline ("Opponent vs Fighter") when the opponent's
        # headshot has no alt text.
        if not opponent_name and headline and " vs " in headline:
            first, _, second = headline.partition(" vs ")
            last = slug.split("-")[-1].lower()
            opponent_name = second.strip() if first.strip().lower() == last else first.strip()

        event_url = None
        for action in card.select(".c-card-event--athlete-results__actions a"):
            if "Fight Card" in action.get_text():
                event_url = action.get("href")

        return {
            "opponent": opponent_name,
            "opponent_url": opponent_url,
            "result": outcome,
            "date": date,
            "round": details.get("round"),
            "time": details.get("time"),
            "method": details.get("method"),
            "matchup": headline,
            "event_url": event_url,
        }

    def get_fight_history(self, first_name, middle_name, last_name):
        slug = self.build_slug(first_name, middle_name, last_name)
        url = f"{self.BASE_URL}/{slug}"
        history = []

        for page in range(self.max_pages):
            response = requests.get(
                url, params={"page": page}, headers=self.HEADERS, timeout=15
            )
            if response.status_code != 200:
                break
            soup = BeautifulSoup(response.text, "lxml")
            cards = soup.select(".c-card-event--athlete-results")
            if not cards:
                break
            for card in cards:
                history.append(self._parse_card(card, slug))

        return history


if __name__ == "__main__":
    import json

    fights = FightHistoryExtractor().get_fight_history("sean", "", "strickland")
    print(f"{len(fights)} fights found")
    print(json.dumps(fights, indent=2))
