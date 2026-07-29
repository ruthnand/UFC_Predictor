"""Scraper for ufcstats.com.

ufcstats.com is protected by a JavaScript proof-of-work challenge, so we solve
it once with a headless browser (Playwright), cache the resulting clearance
cookie, and then bulk-scrape with plain `requests` for speed. Playwright is only
needed for this offline data pipeline, not to serve the app.
"""

import json
import os
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"


class UFCStatsClient:
    def __init__(self, cookie_file="ufcstats_cookie.json", request_delay=0.15):
        self.cookie_file = cookie_file
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})
        self._load_cookie()

    # ----- challenge / cookie handling -----
    def _load_cookie(self):
        if os.path.exists(self.cookie_file):
            try:
                with open(self.cookie_file) as handle:
                    for k, v in json.load(handle).items():
                        self.session.cookies.set(k, v)
            except (ValueError, OSError):
                pass

    def _save_cookie(self, jar):
        try:
            with open(self.cookie_file, "w") as handle:
                json.dump(jar, handle)
        except OSError:
            pass

    def _solve_challenge(self):
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(user_agent=UA)
            page.goto(EVENTS_URL, wait_until="domcontentloaded")
            page.wait_for_selector("a.b-link", timeout=30000)
            jar = {c["name"]: c["value"] for c in page.context.cookies()}
            browser.close()
        for k, v in jar.items():
            self.session.cookies.set(k, v)
        self._save_cookie(jar)

    @staticmethod
    def _is_challenge(html):
        return "Checking your browser" in html or "This site requires JavaScript" in html

    def get(self, url, retries=3):
        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=30)
            except requests.RequestException:
                time.sleep(1.0)
                continue
            if resp.status_code == 200 and not self._is_challenge(resp.text):
                if self.request_delay:
                    time.sleep(self.request_delay)
                return resp.text
            # Stub/challenge page -> (re)solve and retry.
            self._solve_challenge()
        return None

    # ----- parsing helpers -----
    @staticmethod
    def _parse_height(text):
        m = re.search(r"(\d+)'\s*(\d+)", text or "")
        return int(m.group(1)) * 12 + int(m.group(2)) if m else None

    @staticmethod
    def _parse_inches(text):
        m = re.search(r"(\d+(?:\.\d+)?)", (text or "").replace('"', ""))
        return float(m.group(1)) if m else None

    @staticmethod
    def _parse_dob(text):
        text = (text or "").strip()
        for fmt in ("%b %d, %Y",):
            try:
                return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
        return None

    @staticmethod
    def _pair(cell):
        ps = [re.sub(r"\s+", " ", p.get_text(" ", strip=True)) for p in cell.select("p")]
        while len(ps) < 2:
            ps.append("")
        return ps[0], ps[1]

    @staticmethod
    def _landed_attempted(text):
        match = re.search(r"(\d+)\s+of\s+(\d+)", text or "")
        if not match:
            return None, None
        return float(match.group(1)), float(match.group(2))

    @staticmethod
    def _clock_seconds(text):
        match = re.fullmatch(r"\s*(\d+):(\d+)\s*", text or "")
        if not match:
            return None
        return float(int(match.group(1)) * 60 + int(match.group(2)))

    # ----- page scrapers -----
    def list_events(self):
        html = self.get(EVENTS_URL)
        if not html:
            return []
        soup = BeautifulSoup(html, "lxml")
        events = []
        for a in soup.select("a.b-link"):
            href = a.get("href")
            if href and "event-details" in href:
                events.append({"name": a.get_text(strip=True), "url": href,
                               "id": href.rsplit("/", 1)[-1]})
        return events

    def parse_event(self, url):
        html = self.get(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "lxml")
        date = location = None
        for item in soup.select(".b-list__box-list-item"):
            text = " ".join(item.get_text(" ", strip=True).split())
            if text.startswith("Date:"):
                date = text.replace("Date:", "").strip()
            elif text.startswith("Location:"):
                location = text.replace("Location:", "").strip()

        iso_date = None
        if date:
            try:
                iso_date = datetime.strptime(date, "%B %d, %Y").strftime("%Y-%m-%d")
            except ValueError:
                iso_date = date

        fights = []
        for row in soup.select("tr.b-fight-details__table-row[data-link]"):
            cells = row.select("td")
            if len(cells) < 10:
                continue
            links = row.select("a.b-link")
            fighter_links = [a for a in links if a.get("href") and "fighter-details" in a.get("href")]
            if len(fighter_links) < 2:
                continue
            f1, f2 = fighter_links[0], fighter_links[1]
            result_text = cells[0].get_text(" ", strip=True).lower()
            kd1, kd2 = self._pair(cells[2])
            str1, str2 = self._pair(cells[3])
            td1, td2 = self._pair(cells[4])
            sub1, sub2 = self._pair(cells[5])

            if "draw" in result_text or result_text.strip() in ("nc",) or "nc" == result_text.strip():
                outcome = "draw"
            else:
                outcome = "f1_win"  # ufcstats lists the winner first

            fights.append({
                "fight_url": row.get("data-link"),
                "fight_id": row.get("data-link").rsplit("/", 1)[-1],
                "f1_id": f1.get("href").rsplit("/", 1)[-1], "f1_name": f1.get_text(strip=True),
                "f2_id": f2.get("href").rsplit("/", 1)[-1], "f2_name": f2.get_text(strip=True),
                "outcome": outcome,
                "kd": [self._num(kd1), self._num(kd2)],
                "sig_str": [self._num(str1), self._num(str2)],
                "td": [self._num(td1), self._num(td2)],
                "sub_att": [self._num(sub1), self._num(sub2)],
                "weight_class": cells[6].get_text(" ", strip=True),
                "method": cells[7].get_text(" ", strip=True),
                "round": self._num(cells[8].get_text(strip=True)),
                "time": cells[9].get_text(strip=True),
            })
        return {"url": url, "date": iso_date, "location": location, "fights": fights}

    def parse_fight_detail(self, url):
        """Parse cumulative control and strike-target totals for one fight.

        Detail pages use red/blue order, which is not always the winner-first
        order used on event pages. Metrics are therefore keyed by fighter ID.
        """
        html = self.get(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "lxml")
        totals_table = target_table = None
        for table in soup.select("table"):
            headers = [" ".join(th.get_text(" ", strip=True).split())
                       for th in table.select("thead th")]
            # Ignore the round-by-round tables; cumulative tables have no
            # synthetic "Round N" headers.
            if any(header.startswith("Round ") for header in headers):
                continue
            if "Ctrl" in headers:
                totals_table = table
            if all(header in headers for header in ("Head", "Body", "Leg")):
                target_table = table

        if totals_table is None or target_table is None:
            return {
                "id": url.rsplit("/", 1)[-1],
                "url": url,
                "fighters": {},
            }

        totals_row = totals_table.select_one("tbody tr")
        target_row = target_table.select_one("tbody tr")
        if totals_row is None or target_row is None:
            return None
        total_cells = totals_row.select("td")
        target_cells = target_row.select("td")
        fighter_links = [
            a for a in total_cells[0].select("a")
            if a.get("href") and "fighter-details" in a.get("href")
        ]
        if len(fighter_links) < 2:
            return None

        fighter_ids = [a.get("href").rsplit("/", 1)[-1] for a in fighter_links[:2]]
        fighter_names = [a.get_text(" ", strip=True) for a in fighter_links[:2]]
        controls = self._pair(total_cells[9])
        heads = self._pair(target_cells[3])
        bodies = self._pair(target_cells[4])
        legs = self._pair(target_cells[5])

        fighters = {}
        for index, fighter_id in enumerate(fighter_ids):
            head_l, head_a = self._landed_attempted(heads[index])
            body_l, body_a = self._landed_attempted(bodies[index])
            leg_l, leg_a = self._landed_attempted(legs[index])
            fighters[fighter_id] = {
                "name": fighter_names[index],
                "control_seconds": self._clock_seconds(controls[index]),
                "head_landed": head_l,
                "head_attempted": head_a,
                "body_landed": body_l,
                "body_attempted": body_a,
                "leg_landed": leg_l,
                "leg_attempted": leg_a,
            }
        return {
            "id": url.rsplit("/", 1)[-1],
            "url": url,
            "fighters": fighters,
        }

    @staticmethod
    def _num(text):
        m = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", "")) if text else None
        return float(m.group()) if m else None

    def parse_fighter(self, url):
        html = self.get(url)
        if not html:
            return None
        soup = BeautifulSoup(html, "lxml")
        name_node = soup.select_one(".b-content__title-highlight")
        info = {}
        for item in soup.select(".b-list__box-list-item"):
            text = " ".join(item.get_text(" ", strip=True).split())
            if ":" in text:
                key, _, val = text.partition(":")
                info[key.strip()] = val.strip()
        return {
            "id": url.rsplit("/", 1)[-1],
            "name": name_node.get_text(strip=True) if name_node else None,
            "height": self._parse_height(info.get("Height")),
            "reach": self._parse_inches(info.get("Reach")),
            "stance": info.get("STANCE") or None,
            "dob": self._parse_dob(info.get("DOB")),
        }


if __name__ == "__main__":
    client = UFCStatsClient()
    events = client.list_events()
    print("events:", len(events))
    sample = client.parse_event(events[60]["url"])
    print("event date:", sample["date"], "fights:", len(sample["fights"]))
    print(json.dumps(sample["fights"][0], indent=2))
    fighter_url = "http://ufcstats.com/fighter-details/" + sample["fights"][0]["f1_id"]
    print(json.dumps(client.parse_fighter(fighter_url), indent=2))
