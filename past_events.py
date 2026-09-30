"""Build a compact, deployable archive of completed UFC event results."""

import json
import os
from datetime import datetime, timezone


SOURCE_PATH = "ufcstats_events.json"
OUTPUT_PATH = os.path.join("models", "past_events.json")


def _completed_fight(fight):
    return bool(
        fight.get("outcome")
        and fight.get("method")
        and fight.get("round") is not None
        and fight.get("time")
    )


def _display_method(value):
    value = (value or "").strip()
    decisions = {
        "U-DEC": "Unanimous Decision",
        "S-DEC": "Split Decision",
        "M-DEC": "Majority Decision",
        "CNC": "No Contest",
    }
    if value in decisions:
        return decisions[value]
    if value.startswith("SUB "):
        return f"Submission — {value[4:]}"
    if value.startswith("KO/TKO "):
        return f"KO/TKO — {value[7:]}"
    return value


def _normalize_fight(fight):
    outcome = fight.get("outcome")
    fighter_1 = fight.get("f1_name")
    fighter_2 = fight.get("f2_name")
    if outcome == "f1_win":
        winner, loser, result = fighter_1, fighter_2, "win"
    elif outcome == "f2_win":
        winner, loser, result = fighter_2, fighter_1, "win"
    else:
        winner, loser, result = None, None, outcome or "draw"

    round_value = fight.get("round")
    if isinstance(round_value, float) and round_value.is_integer():
        round_value = int(round_value)
    return {
        "fight_id": fight.get("fight_id"),
        "weight_class": fight.get("weight_class"),
        "fighter_1": fighter_1,
        "fighter_2": fighter_2,
        "winner": winner,
        "loser": loser,
        "result": result,
        "method": _display_method(fight.get("method")),
        "round": round_value,
        "time": fight.get("time"),
    }


def build_past_events(source_path=SOURCE_PATH, output_path=OUTPUT_PATH, log=print):
    with open(source_path) as handle:
        raw_events = json.load(handle)

    events = []
    for event in raw_events:
        fights = [
            _normalize_fight(fight)
            for fight in event.get("fights", [])
            if _completed_fight(fight)
        ]
        if not fights:
            continue
        events.append({
            "event_id": event.get("id"),
            "name": event.get("name"),
            "date": event.get("date"),
            "location": event.get("location"),
            "source_url": event.get("url"),
            "fight_count": len(fights),
            "fights": fights,
        })

    events.sort(key=lambda event: event.get("date") or "", reverse=True)
    payload = {
        "source": "UFCStats",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event_count": len(events),
        "fight_count": sum(event["fight_count"] for event in events),
        "events": events,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_path = output_path + ".tmp"
    with open(temp_path, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(temp_path, output_path)
    log(
        f"Past-event archive: {payload['event_count']} events, "
        f"{payload['fight_count']} fights -> {output_path}"
    )
    return payload


if __name__ == "__main__":
    build_past_events()
