"""Read public ESPN MMA JSON feeds without scraping rendered page markup.

ESPN does not publish these endpoints as a supported developer API, so callers
must tolerate schema changes. Responses are normalized here before the Flask
application exposes them.
"""

import re
from difflib import SequenceMatcher

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class ESPNDataError(RuntimeError):
    """Raised when ESPN data is unavailable or has an unexpected shape."""


class ESPNMMAScraper:
    SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
    SEARCH_URL = "https://site.web.api.espn.com/apis/search/v2"
    ATHLETE_URL = (
        "https://site.web.api.espn.com/apis/common/v3/"
        "sports/mma/ufc/athletes/{athlete_id}"
    )
    NEWS_URL = "https://now.core.api.espn.com/v1/sports/news"

    def __init__(self, timeout=15, session=None):
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update({
            # The scoreboard host serves its JSON feed only to web clients.
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.espn.com/mma/",
        })
        retry = Retry(
            total=2,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _get(self, url, params=None):
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise ESPNDataError(f"ESPN request failed: {type(exc).__name__}") from exc

    @staticmethod
    def _web_link(links):
        if isinstance(links, dict):
            web = links.get("web")
            if isinstance(web, dict):
                return web.get("href")
            return web
        for link in links or []:
            if "desktop" in link.get("rel", []):
                return link.get("href")
        return None

    @staticmethod
    def _athlete_id(result):
        uid = result.get("uid") or ""
        match = re.search(r"~a:(\d+)", uid)
        if match:
            return match.group(1)
        url = ((result.get("link") or {}).get("web") or "")
        match = re.search(r"/id/(\d+)", url)
        return match.group(1) if match else None

    @staticmethod
    def _normalized_name(value):
        return re.sub(r"[^a-z0-9]+", "", (value or "").lower())

    def search_fighters(self, query, limit=10):
        query = (query or "").strip()
        if len(query) < 2:
            return []
        data = self._get(self.SEARCH_URL, {
            "query": query,
            "limit": min(max(int(limit), 1), 25),
        })
        fighters = []
        for group in data.get("results", []):
            if group.get("type") != "player":
                continue
            for item in group.get("contents", []):
                if (item.get("sport") or "").lower() != "mma":
                    continue
                athlete_id = self._athlete_id(item)
                if not athlete_id:
                    continue
                fighters.append({
                    "espn_id": athlete_id,
                    "name": item.get("displayName"),
                    "image": (item.get("image") or {}).get("default"),
                    "url": (item.get("link") or {}).get("web"),
                })

        target = self._normalized_name(query)
        fighters.sort(
            key=lambda item: SequenceMatcher(
                None, target, self._normalized_name(item.get("name"))
            ).ratio(),
            reverse=True,
        )
        return fighters[:limit]

    def get_fighter(self, query=None, athlete_id=None):
        match = None
        if not athlete_id:
            matches = self.search_fighters(query, limit=5)
            if not matches:
                return None
            match = matches[0]
            athlete_id = match["espn_id"]

        data = self._get(self.ATHLETE_URL.format(athlete_id=athlete_id))
        athlete = data.get("athlete") or {}
        if not athlete:
            raise ESPNDataError("ESPN athlete response did not include an athlete")

        summary = {
            item.get("abbreviation"): item.get("displayValue")
            for item in (athlete.get("statsSummary") or {}).get("statistics", [])
        }
        history = []
        event_map = data.get("eventsMap") or {}
        for event_key in data.get("events") or []:
            event = event_map.get(event_key) or {}
            opponent = event.get("opponent") or {}
            status = event.get("status") or {}
            result = status.get("result") or {}
            history.append({
                "event_id": event.get("id"),
                "event": event.get("name"),
                "date": event.get("gameDate"),
                "result": event.get("gameResult"),
                "opponent": opponent.get("displayName"),
                "opponent_espn_id": opponent.get("id"),
                "method": result.get("displayName"),
                "round": status.get("period"),
                "time": status.get("displayClock"),
                "title_fight": bool(event.get("titleFight")),
                "url": self._web_link(event.get("links")),
            })

        return {
            "source": "ESPN MMA",
            "source_url": self._web_link(athlete.get("links")),
            "matched_from_search": match,
            "espn_id": str(athlete.get("id") or athlete_id),
            "name": athlete.get("displayName") or athlete.get("fullName"),
            "first_name": athlete.get("firstName"),
            "last_name": athlete.get("lastName"),
            "headshot": (athlete.get("headshot") or {}).get("href"),
            "active": athlete.get("active"),
            "status": (athlete.get("status") or {}).get("name"),
            "record": summary.get("W-L-D"),
            "knockouts": summary.get("(T)KO"),
            "submissions": summary.get("SUB"),
            "height": athlete.get("displayHeight"),
            "weight": athlete.get("displayWeight"),
            "reach": athlete.get("displayReach"),
            "date_of_birth": athlete.get("displayDOB"),
            "age": athlete.get("age"),
            "division": (athlete.get("weightClass") or {}).get("text"),
            "stance": (athlete.get("stance") or {}).get("text"),
            "fighting_style": athlete.get("displayFightingStyle"),
            "camp": (athlete.get("association") or {}).get("name"),
            "country": athlete.get("citizenship"),
            "flag": (athlete.get("flag") or {}).get("href"),
            "fight_history": history,
        }

    def get_scoreboard(self):
        data = self._get(self.SCOREBOARD_URL)
        calendar = []
        leagues = data.get("leagues") or []
        if leagues:
            for item in leagues[0].get("calendar") or []:
                event_ref = ((item.get("event") or {}).get("$ref") or "")
                match = re.search(r"/events/(\d+)", event_ref)
                calendar.append({
                    "event_id": match.group(1) if match else None,
                    "name": item.get("label"),
                    "start_date": item.get("startDate"),
                    "end_date": item.get("endDate"),
                })
        events = []
        for event in data.get("events") or []:
            competitions = event.get("competitions") or []
            venue = (
                (competitions[0].get("venue") or {}) if competitions else {}
            )
            bouts = []
            for bout in competitions:
                competitors = sorted(
                    bout.get("competitors") or [],
                    key=lambda item: item.get("order", 99),
                )
                fighters = []
                for competitor in competitors:
                    athlete = competitor.get("athlete") or {}
                    record = next(iter(competitor.get("records") or []), {})
                    fighters.append({
                        "espn_id": competitor.get("id"),
                        "name": athlete.get("displayName"),
                        "record": record.get("summary"),
                        "country": (athlete.get("flag") or {}).get("alt"),
                        "flag": (athlete.get("flag") or {}).get("href"),
                        "winner": competitor.get("winner"),
                    })
                status = (bout.get("status") or {}).get("type") or {}
                bouts.append({
                    "bout_id": bout.get("id"),
                    "date": bout.get("date"),
                    "weight_class": (bout.get("type") or {}).get("abbreviation"),
                    "fighters": fighters,
                    "status": status.get("description"),
                    "completed": status.get("completed"),
                    "broadcast": bout.get("broadcast"),
                    "scheduled_rounds": (
                        (bout.get("format") or {}).get("regulation") or {}
                    ).get("periods"),
                })
            events.append({
                "event_id": event.get("id"),
                "name": event.get("name"),
                "short_name": event.get("shortName"),
                "date": event.get("date"),
                "status": ((event.get("status") or {}).get("type") or {}).get(
                    "description"
                ),
                "venue": venue.get("fullName"),
                "location": (venue.get("address") or {}),
                "url": self._web_link(event.get("links")),
                "bouts": bouts,
            })
        return {
            "source": "ESPN MMA",
            "source_url": "https://www.espn.com/mma/",
            "date": (data.get("day") or {}).get("date"),
            "calendar": calendar,
            "events": events,
        }

    def get_news(self, limit=10):
        data = self._get(self.NEWS_URL, {
            "region": "us",
            "lang": "en",
            "contentorigin": "espn",
            "sport": "mma",
            "limit": min(max(int(limit), 1), 25),
        })
        articles = []
        for item in data.get("headlines") or []:
            image = next(iter(item.get("images") or []), {})
            articles.append({
                "id": str(item.get("id")),
                "headline": item.get("headline"),
                "description": item.get("description"),
                "byline": item.get("byline"),
                "published": item.get("published"),
                "image": image.get("url"),
                "image_caption": image.get("caption"),
                "url": self._web_link(item.get("links")),
            })
        return {
            "source": "ESPN MMA",
            "source_url": "https://www.espn.com/mma/",
            "articles": articles,
        }
