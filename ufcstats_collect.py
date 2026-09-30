"""Bulk-collect ufcstats.com data into local JSON caches.

Produces:
  ufcstats_events.json   - list of parsed events (date, location, fights[])
  ufcstats_fighters.json - {fighter_id: {name, height, reach, stance, dob}}
  ufcstats_fight_details.json - control + strike-target totals by fight/fighter

All collectors are resumable.
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from ufcstats_scraper import UFCStatsClient

EVENTS_CACHE = "ufcstats_events.json"
FIGHTERS_CACHE = "ufcstats_fighters.json"
FIGHT_DETAILS_CACHE = "ufcstats_fight_details.json"


def _load(path, default):
    if os.path.exists(path):
        try:
            with open(path) as handle:
                return json.load(handle)
        except (ValueError, OSError):
            return default
    return default


def _save(path, data):
    temp_path = path + ".tmp"
    with open(temp_path, "w") as handle:
        json.dump(data, handle)
    os.replace(temp_path, path)


def collect_events(client, log=print):
    events_by_id = {e["id"]: e for e in _load(EVENTS_CACHE, [])}
    listing = client.list_events()
    log(f"Total events listed: {len(listing)}")

    for i, ev in enumerate(listing, 1):
        cached_fights = events_by_id.get(ev["id"], {}).get("fights", [])
        if cached_fights and all(
            fight.get("fight_id")
            and fight.get("method")
            and fight.get("round") is not None
            and fight.get("time")
            for fight in cached_fights
        ):
            continue
        parsed = client.parse_event(ev["url"])
        if parsed:
            parsed["id"] = ev["id"]
            parsed["name"] = ev["name"]
            events_by_id[ev["id"]] = parsed
        if i % 25 == 0:
            _save(EVENTS_CACHE, list(events_by_id.values()))
            log(f"  events {i}/{len(listing)}")
    _save(EVENTS_CACHE, list(events_by_id.values()))
    log(f"Saved {len(events_by_id)} events")
    return list(events_by_id.values())


def collect_fighters(client, events, log=print):
    fighters = _load(FIGHTERS_CACHE, {})
    ids = {}
    for ev in events:
        for f in ev.get("fights", []):
            ids[f["f1_id"]] = f["f1_name"]
            ids[f["f2_id"]] = f["f2_name"]
    log(f"Unique fighters: {len(ids)}")

    todo = [fid for fid in ids if fid not in fighters]
    log(f"Fighters to fetch: {len(todo)}")
    for i, fid in enumerate(todo, 1):
        parsed = client.parse_fighter(f"http://ufcstats.com/fighter-details/{fid}")
        if parsed:
            fighters[fid] = parsed
        if i % 50 == 0:
            _save(FIGHTERS_CACHE, fighters)
            log(f"  fighters {i}/{len(todo)}")
    _save(FIGHTERS_CACHE, fighters)
    log(f"Saved {len(fighters)} fighters")
    return fighters


def collect_fight_details(client, events, log=print, workers=None):
    details = _load(FIGHT_DETAILS_CACHE, {})
    fights = {}
    for event in events:
        for fight in event.get("fights", []):
            fight_id = fight.get("fight_id")
            fight_url = fight.get("fight_url")
            if fight_id and fight_url:
                fights[fight_id] = fight_url

    todo = [(fight_id, url) for fight_id, url in fights.items()
            if fight_id not in details]
    log(f"Unique fight detail pages: {len(fights)}")
    log(f"Fight details to fetch: {len(todo)}")
    workers = workers or int(os.getenv("UFCSTATS_WORKERS", "8"))
    local = threading.local()

    def fetch(item):
        fight_id, url = item
        if not hasattr(local, "client"):
            local.client = UFCStatsClient(
                cookie_file=client.cookie_file,
                request_delay=0.05,
            )
        return fight_id, local.client.parse_fight_detail(url)

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, item) for item in todo]
        for future in as_completed(futures):
            try:
                fight_id, parsed = future.result()
            except Exception as exc:  # retry failed pages on the next resumable run
                completed += 1
                log(f"  fight detail failed: {type(exc).__name__}")
                continue
            completed += 1
            if parsed:
                details[fight_id] = parsed
            if completed % 50 == 0:
                _save(FIGHT_DETAILS_CACHE, details)
                log(f"  fight details {completed}/{len(todo)}")
    _save(FIGHT_DETAILS_CACHE, details)
    with_metrics = sum(bool(item.get("fighters")) for item in details.values())
    log(f"Saved {len(details)} fight details ({with_metrics} with metrics)")
    return details


def main():
    client = UFCStatsClient()
    events = collect_events(client, log=lambda *a: print(*a, flush=True))
    from past_events import build_past_events
    build_past_events(log=lambda *a: print(*a, flush=True))
    if "--events-only" not in sys.argv:
        collect_fighters(client, events, log=lambda *a: print(*a, flush=True))
        if "--skip-details" not in sys.argv:
            collect_fight_details(client, events, log=lambda *a: print(*a, flush=True))
    print("DONE")


if __name__ == "__main__":
    main()
