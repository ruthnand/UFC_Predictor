"""Build a point-in-time UFC fight dataset from ufcstats caches.

Walks every completed fight in chronological order, snapshotting each fighter's
state *before* the bout (so no future information leaks in), then updates Elo and
cumulative stats with the result. Produces:

  ufc_pit_dataset.csv   - one balanced pair of rows per fight (A-B -> 1, B-A -> 0)
  fighter_state.json    - each fighter's latest state, for live prediction
"""

import json
import os
import re

import pandas as pd

import glicko2
from pit_features import (
    ALL_NUMERIC_KEYS, all_feature_names, build_vector, method_bucket, state_to_features,
)

EVENTS_CACHE = "ufcstats_events.json"
FIGHTERS_CACHE = "ufcstats_fighters.json"
FIGHT_DETAILS_CACHE = "ufcstats_fight_details.json"
DATASET_CSV = "ufc_pit_dataset.csv"
STATE_JSON = "fighter_state.json"

ELO_START = 1500.0
ELO_K = 32.0
MIN_PRIOR = 2  # both fighters need at least this many prior bouts to form a row


def _load(path, default):
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    return default


def _duration_minutes(rnd, time_str):
    """Total fought minutes given the final round and its clock time."""
    minutes = 0.0
    if rnd:
        minutes = (rnd - 1) * 5.0
    m = re.match(r"(\d+):(\d+)", str(time_str or ""))
    if m:
        minutes += int(m.group(1)) + int(m.group(2)) / 60.0
    elif rnd:
        minutes += 5.0
    return minutes or 5.0


def _is_finish(method):
    method = (method or "").upper()
    return "KO" in method or "SUB" in method


def _new_raw(static):
    return {
        "n": 0, "w": 0, "l": 0,
        "sig_l": 0.0, "sig_a": 0.0, "td_l": 0.0, "td_a": 0.0, "sub": 0.0,
        "kd_f": 0.0, "kd_a": 0.0, "time": 0.0, "fin_f": 0, "fin_a": 0,
        "ctrl_f": 0.0, "ctrl_a": 0.0, "ctrl_time": 0.0,
        "head_l": 0.0, "head_a": 0.0,
        "body_l": 0.0, "body_a": 0.0,
        "leg_l": 0.0, "leg_a": 0.0, "target_time": 0.0,
        "glicko": glicko2.new_rating(),
        "sos_sum": 0.0,
        "adj_str_sum": 0.0, "adj_str_n": 0,
        "adj_def_sum": 0.0, "adj_def_n": 0,
        "streak": 0, "last_date": None, "elo": ELO_START,
        "height": static.get("height"), "reach": static.get("reach"),
        "stance": static.get("stance"), "dob": static.get("dob"),
        "name": static.get("name"),
    }


def _derive_state(raw):
    n, t, w, l = raw["n"], raw["time"], raw["w"], raw["l"]
    ratio = (lambda num, den: num / den if den else None)
    return {
        "elo": raw["elo"],
        "glicko_rating": raw["glicko"]["rating"],
        "glicko_rd": raw["glicko"]["rd"],
        "sos_elo": ratio(raw["sos_sum"], n),
        "adj_str_pm": ratio(raw["adj_str_sum"], raw["adj_str_n"]),
        "adj_def_pm": ratio(raw["adj_def_sum"], raw["adj_def_n"]),
        "height": raw["height"], "reach": raw["reach"],
        "stance": raw["stance"], "dob": raw["dob"], "name": raw["name"],
        "last_date": raw["last_date"],
        "experience": n,
        "win_rate": ratio(w, n),
        "streak": raw["streak"],
        "slpm": ratio(raw["sig_l"], t),
        "sapm": ratio(raw["sig_a"], t),
        "str_diff_pm": (raw["sig_l"] - raw["sig_a"]) / t if t else None,
        "td_land_p15": raw["td_l"] / t * 15 if t else None,
        "td_abs_p15": raw["td_a"] / t * 15 if t else None,
        "sub_att_p15": raw["sub"] / t * 15 if t else None,
        "kd_for_pm": ratio(raw["kd_f"], t),
        "kd_against_pm": ratio(raw["kd_a"], t),
        "ctrl_for_p15": raw["ctrl_f"] / raw["ctrl_time"] / 60 * 15
        if raw["ctrl_time"] else None,
        "ctrl_against_p15": raw["ctrl_a"] / raw["ctrl_time"] / 60 * 15
        if raw["ctrl_time"] else None,
        "head_land_pm": ratio(raw["head_l"], raw["target_time"]),
        "head_abs_pm": ratio(raw["head_a"], raw["target_time"]),
        "body_land_pm": ratio(raw["body_l"], raw["target_time"]),
        "body_abs_pm": ratio(raw["body_a"], raw["target_time"]),
        "leg_land_pm": ratio(raw["leg_l"], raw["target_time"]),
        "leg_abs_pm": ratio(raw["leg_a"], raw["target_time"]),
        "finish_rate": ratio(raw["fin_f"], w),
        "finished_against_rate": ratio(raw["fin_a"], l),
    }


def _expected(elo_a, elo_b):
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def build(log=print):
    events = _load(EVENTS_CACHE, [])
    fighters = _load(FIGHTERS_CACHE, {})
    fight_details = _load(FIGHT_DETAILS_CACHE, {})

    # Flatten completed fights and sort chronologically.
    fights = []
    for ev in events:
        d = ev.get("date")
        if not d or not re.match(r"\d{4}-\d{2}-\d{2}", str(d)):
            continue
        for f in ev.get("fights", []):
            if f.get("outcome") != "f1_win":
                continue
            if not f.get("f1_id") or not f.get("f2_id"):
                continue
            fights.append((d, ev, f))
    fights.sort(key=lambda x: x[0])
    log(f"Completed fights: {len(fights)}")

    raw_state = {}
    def get_raw(fid, name):
        if fid not in raw_state:
            static = fighters.get(fid, {"name": name})
            raw_state[fid] = _new_raw(static)
        return raw_state[fid]

    cols = all_feature_names()
    rows = []
    for d, ev, f in fights:
        a_id, b_id = f["f1_id"], f["f2_id"]
        ra = get_raw(a_id, f.get("f1_name"))
        rb = get_raw(b_id, f.get("f2_name"))

        if ra["n"] >= MIN_PRIOR and rb["n"] >= MIN_PRIOR:
            fa = state_to_features(_derive_state(ra), d)
            fb = state_to_features(_derive_state(rb), d)
            method = method_bucket(f.get("method"))
            base = {
                "date": d,
                "f1_name": f.get("f1_name"),
                "f2_name": f.get("f2_name"),
                "event_id": ev.get("id"),
                "event_name": ev.get("name"),
                "fight_id": f.get("fight_id"),
                "weight_class": f.get("weight_class"),
                "method": method,
            }
            va = dict(zip(cols, build_vector(fa, fb, keys=ALL_NUMERIC_KEYS)))
            vb = dict(zip(cols, build_vector(fb, fa, keys=ALL_NUMERIC_KEYS)))
            rows.append({**base, **va, "label": 1})
            rows.append({**base, **vb, "label": 0})

        # ----- update state with this result (A won) -----
        dur = _duration_minutes(f.get("round"), f.get("time"))
        kd = f.get("kd") or [0, 0]
        ss = f.get("sig_str") or [0, 0]
        td = f.get("td") or [0, 0]
        sb = f.get("sub_att") or [0, 0]
        detail_fighters = fight_details.get(f.get("fight_id"), {}).get("fighters", {})
        detail_a = detail_fighters.get(a_id)
        detail_b = detail_fighters.get(b_id)
        n0 = lambda x: x if isinstance(x, (int, float)) else 0.0
        finish = _is_finish(f.get("method"))

        # Pre-fight snapshots so opponent-adjusted metrics use what was known
        # going into the bout, not the post-fight update.
        pre_a = _derive_state(ra)
        pre_b = _derive_state(rb)

        ea = _expected(ra["elo"], rb["elo"])
        ra["elo"] += ELO_K * (1 - ea)
        rb["elo"] += ELO_K * (0 - (1 - ea))
        ra["glicko"], rb["glicko"] = glicko2.update_pair(ra["glicko"], rb["glicko"])

        for r, i, j, won, own_detail, opp_detail, opp_pre in (
            (ra, 0, 1, True, detail_a, detail_b, pre_b),
            (rb, 1, 0, False, detail_b, detail_a, pre_a),
        ):
            r["n"] += 1
            r["time"] += dur
            r["sig_l"] += n0(ss[i]); r["sig_a"] += n0(ss[j])
            r["td_l"] += n0(td[i]); r["td_a"] += n0(td[j])
            r["sub"] += n0(sb[i])
            r["kd_f"] += n0(kd[i]); r["kd_a"] += n0(kd[j])
            r["sos_sum"] += opp_pre["elo"]
            own_land_pm = n0(ss[i]) / dur
            own_abs_pm = n0(ss[j]) / dur
            if opp_pre["sapm"] is not None:
                r["adj_str_sum"] += own_land_pm - opp_pre["sapm"]
                r["adj_str_n"] += 1
            if opp_pre["slpm"] is not None:
                r["adj_def_sum"] += opp_pre["slpm"] - own_abs_pm
                r["adj_def_n"] += 1
            if (own_detail and opp_detail
                    and own_detail.get("control_seconds") is not None
                    and opp_detail.get("control_seconds") is not None):
                r["ctrl_f"] += own_detail["control_seconds"]
                r["ctrl_a"] += opp_detail["control_seconds"]
                r["ctrl_time"] += dur
            target_keys = ("head_landed", "body_landed", "leg_landed")
            if (own_detail and opp_detail
                    and all(own_detail.get(key) is not None for key in target_keys)
                    and all(opp_detail.get(key) is not None for key in target_keys)):
                r["head_l"] += own_detail["head_landed"]
                r["head_a"] += opp_detail["head_landed"]
                r["body_l"] += own_detail["body_landed"]
                r["body_a"] += opp_detail["body_landed"]
                r["leg_l"] += own_detail["leg_landed"]
                r["leg_a"] += opp_detail["leg_landed"]
                r["target_time"] += dur
            r["last_date"] = d
            if won:
                r["w"] += 1; r["streak"] = max(0, r["streak"]) + 1
                if finish:
                    r["fin_f"] += 1
            else:
                r["l"] += 1; r["streak"] = 0
                if finish:
                    r["fin_a"] += 1

    df = pd.DataFrame(rows)
    df.to_csv(DATASET_CSV, index=False)
    log(f"Dataset rows: {len(df)} -> {DATASET_CSV}")

    final_state = {fid: _derive_state(r) for fid, r in raw_state.items() if r["n"] >= 1}
    with open(STATE_JSON, "w") as handle:
        json.dump(final_state, handle)
    log(f"Fighter states: {len(final_state)} -> {STATE_JSON}")
    return df


if __name__ == "__main__":
    build(log=lambda *a: print(*a, flush=True))
