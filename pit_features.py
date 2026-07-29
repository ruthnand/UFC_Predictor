"""Shared point-in-time feature spec used by both training and live prediction.

A fighter's "state" holds date-independent cumulative metrics (Elo, per-minute
rates, etc.) plus raw fields (dob, last fight date, physicals). `state_to_features`
turns a state into the numeric feature dict *as of a given date* (so age and
layoff are computed relative to the fight date in training, or today at serve
time). A matchup vector is the difference of the two fighters' features.
"""

from datetime import date, datetime

import numpy as np

import glicko2

# Per-fighter numeric features; the model uses the A - B difference of each.
ALL_NUMERIC_KEYS = [
    "elo",
    "glicko_rating",        # Glicko-2 rating (uncertainty-aware Elo)
    "glicko_rd",            # Glicko-2 rating deviation (record uncertainty)
    "sos_elo",              # strength of schedule: mean opponent pre-fight Elo
    "adj_str_pm",           # opponent-adjusted striking: landed vs. what opponents typically absorb
    "adj_def_pm",           # opponent-adjusted defense: absorbed vs. what opponents typically land
    "age",
    "height",
    "reach",
    "experience",
    "win_rate",
    "streak",
    "layoff_days",
    "slpm",                 # sig strikes landed per min
    "sapm",                 # sig strikes absorbed per min
    "str_diff_pm",          # net striking per min
    "td_land_p15",          # takedowns landed per 15 min
    "td_abs_p15",           # takedowns absorbed per 15 min (grappling defense)
    "sub_att_p15",          # submission attempts per 15 min
    "kd_for_pm",            # knockdowns scored per min
    "kd_against_pm",        # knockdowns absorbed per min (chin/durability)
    "ctrl_for_p15",         # control time earned per 15 min
    "ctrl_against_p15",     # control time conceded per 15 min
    "head_land_pm",         # significant head strikes landed per min
    "head_abs_pm",          # significant head strikes absorbed per min
    "body_land_pm",         # significant body strikes landed per min
    "body_abs_pm",          # significant body strikes absorbed per min
    "leg_land_pm",          # significant leg strikes landed per min
    "leg_abs_pm",           # significant leg strikes absorbed per min
    "finish_rate",          # finishes / wins
    "finished_against_rate",  # times finished / losses (durability)
]

# Forward-chaining ablation found control + head-strike rates improve held-out
# performance, while body/leg rates add noise. Keep all rates in the dataset and
# fighter state for analysis, but train the production models on the validated
# subset.
MODEL_NUMERIC_KEYS = [
    key for key in ALL_NUMERIC_KEYS
    if key not in ("body_land_pm", "body_abs_pm", "leg_land_pm", "leg_abs_pm")
]


# Matchup-level (pair) features appended after the per-fighter differences.
PAIR_KEYS = ["stance_mismatch", "glicko_expected"]

# Canonical finish buckets for method-of-victory prediction.
METHOD_CLASSES = ["KO/TKO", "Submission", "Decision"]


def feature_names():
    return [f"{key}_diff" for key in MODEL_NUMERIC_KEYS] + PAIR_KEYS


def all_feature_names():
    return [f"{key}_diff" for key in ALL_NUMERIC_KEYS] + PAIR_KEYS


def method_bucket(method):
    """Collapse ufcstats method strings into KO/TKO / Submission / Decision."""
    text = (method or "").upper()
    if "KO" in text or "TKO" in text:
        return "KO/TKO"
    if "SUB" in text:
        return "Submission"
    if "DEC" in text:
        return "Decision"
    return None


def flip_vector(vector, features=None):
    """Orient a matchup vector from B's perspective (A-B -> B-A)."""
    features = features or feature_names()
    flipped = []
    for name, value in zip(features, vector):
        if name.endswith("_diff"):
            flipped.append(-value if value == value else value)  # keep NaN
        elif name == "glicko_expected":
            flipped.append(1.0 - value if value == value else value)
        else:
            flipped.append(value)
    return flipped


def _parse_date(value):
    if not value:
        return None
    if isinstance(value, (datetime, date)):
        return value if isinstance(value, date) and not isinstance(value, datetime) else value.date()
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _years_between(later, earlier):
    if not later or not earlier:
        return None
    return (later - earlier).days / 365.25


def state_to_features(state, as_of):
    """Materialize a fighter's numeric features as of ``as_of`` (a date/str)."""
    as_of = _parse_date(as_of)
    dob = _parse_date(state.get("dob"))
    last = _parse_date(state.get("last_date"))
    return {
        "elo": state.get("elo"),
        "glicko_rating": state.get("glicko_rating"),
        "glicko_rd": state.get("glicko_rd"),
        "sos_elo": state.get("sos_elo"),
        "adj_str_pm": state.get("adj_str_pm"),
        "adj_def_pm": state.get("adj_def_pm"),
        "age": _years_between(as_of, dob),
        "height": state.get("height"),
        "reach": state.get("reach"),
        "experience": state.get("experience"),
        "win_rate": state.get("win_rate"),
        "streak": state.get("streak"),
        "layoff_days": (as_of - last).days if (as_of and last) else None,
        "slpm": state.get("slpm"),
        "sapm": state.get("sapm"),
        "str_diff_pm": state.get("str_diff_pm"),
        "td_land_p15": state.get("td_land_p15"),
        "td_abs_p15": state.get("td_abs_p15"),
        "sub_att_p15": state.get("sub_att_p15"),
        "kd_for_pm": state.get("kd_for_pm"),
        "kd_against_pm": state.get("kd_against_pm"),
        "ctrl_for_p15": state.get("ctrl_for_p15"),
        "ctrl_against_p15": state.get("ctrl_against_p15"),
        "head_land_pm": state.get("head_land_pm"),
        "head_abs_pm": state.get("head_abs_pm"),
        "body_land_pm": state.get("body_land_pm"),
        "body_abs_pm": state.get("body_abs_pm"),
        "leg_land_pm": state.get("leg_land_pm"),
        "leg_abs_pm": state.get("leg_abs_pm"),
        "finish_rate": state.get("finish_rate"),
        "finished_against_rate": state.get("finished_against_rate"),
        "stance": state.get("stance"),
    }


def build_vector(features_a, features_b, keys=None):
    """Difference vector (A - B); missing components become NaN so the model's
    imputer can handle them. Stance mismatch is a symmetric 0/1 flag."""
    keys = keys or MODEL_NUMERIC_KEYS
    vector = []
    for key in keys:
        a, b = features_a.get(key), features_b.get(key)
        if a is None or b is None:
            vector.append(np.nan)
        else:
            vector.append(float(a) - float(b))

    sa, sb = features_a.get("stance"), features_b.get("stance")
    if sa and sb:
        vector.append(1.0 if sa != sb else 0.0)
    else:
        vector.append(0.0)

    # Glicko-2 expected score for A over B: combines both ratings AND both
    # uncertainties into a probability, which neither raw diff captures alone.
    ra, rda = features_a.get("glicko_rating"), features_a.get("glicko_rd")
    rb, rdb = features_b.get("glicko_rating"), features_b.get("glicko_rd")
    if None not in (ra, rda, rb, rdb):
        vector.append(glicko2.expected_score(
            {"rating": ra, "rd": rda}, {"rating": rb, "rd": rdb}))
    else:
        vector.append(np.nan)
    return vector
