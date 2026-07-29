import difflib
import json
import os
from datetime import date, datetime

from models import FightPredictor
from odds import OddsProvider
from pit_features import build_vector, state_to_features


def _fmt_height(inches):
    if inches is None:
        return None
    inches = int(round(float(inches)))
    return f"{inches // 12}'{inches % 12}\""


def _age_years(dob):
    if not dob:
        return None
    try:
        born = datetime.strptime(dob, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


class PredictionService:
    """Predicts fights from point-in-time fighter states (Elo, form, per-minute
    offense/defense rates), combining a calibrated gradient-boosted tree model
    and a Bayesian logistic model, plus live odds when available."""

    def __init__(self, state_file="fighter_state.json"):
        self.state_file = state_file
        self.states = {}
        self.name_to_id = {}
        self.names = []
        self._load_states()
        self.odds = OddsProvider()
        self._predictor = None

    def _load_states(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as handle:
                self.states = json.load(handle)
        except (ValueError, OSError):
            self.states = {}
        best = {}
        for fid, st in self.states.items():
            name = (st.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key not in best or (st.get("experience") or 0) > (self.states[best[key]].get("experience") or 0):
                best[key] = fid
        self.name_to_id = best
        self.names = sorted({self.states[fid].get("name") for fid in best.values()})

    @property
    def predictor(self):
        if self._predictor is None:
            predictor = FightPredictor()
            if not predictor.load():
                raise RuntimeError("Models are not trained yet. Run `python train.py`.")
            self._predictor = predictor
        return self._predictor

    def models_ready(self):
        try:
            _ = self.predictor
            return bool(self.states)
        except RuntimeError:
            return False

    def search(self, query, limit=10):
        """Prefix / fuzzy fighter name search for autocomplete."""
        q = (query or "").strip()
        if not q:
            return []
        q_lower = q.lower()
        starts = [n for n in self.names if n.lower().startswith(q_lower)]
        contains = [n for n in self.names if q_lower in n.lower() and n not in starts]
        fuzzy = difflib.get_close_matches(q, self.names, n=limit, cutoff=0.6)
        ordered = []
        for name in starts + contains + fuzzy:
            if name not in ordered:
                ordered.append(name)
            if len(ordered) >= limit:
                break
        out = []
        for name in ordered:
            fid = self.name_to_id.get(name.lower())
            st = self.states.get(fid, {})
            out.append({
                "name": name,
                "elo": round(st["elo"], 1) if st.get("elo") is not None else None,
                "experience": st.get("experience"),
                "stance": st.get("stance"),
            })
        return out

    def _resolve(self, name):
        if not name:
            return None
        key = name.strip().lower()
        if key in self.name_to_id:
            fid = self.name_to_id[key]
            return {
                "id": fid,
                "state": self.states[fid],
                "matched": self.states[fid].get("name"),
                "score": 1.0,
            }
        match = difflib.get_close_matches(name.strip(), self.names, n=1, cutoff=0.82)
        if not match:
            return None
        fid = self.name_to_id[match[0].lower()]
        score = difflib.SequenceMatcher(None, key, match[0].lower()).ratio()
        return {
            "id": fid,
            "state": self.states[fid],
            "matched": self.states[fid].get("name"),
            "score": round(score, 3),
        }

    @staticmethod
    def _summary(state):
        return {
            "name": state.get("name"),
            "elo": round(state.get("elo"), 1) if state.get("elo") is not None else None,
            "glicko": round(state.get("glicko_rating"), 1) if state.get("glicko_rating") is not None else None,
            "glicko_rd": round(state.get("glicko_rd"), 1) if state.get("glicko_rd") is not None else None,
            "experience": state.get("experience"),
            "win_rate": round(state.get("win_rate"), 3) if state.get("win_rate") is not None else None,
            "stance": state.get("stance"),
            "height": state.get("height"),
            "height_display": _fmt_height(state.get("height")),
            "reach": state.get("reach"),
            "age": _age_years(state.get("dob")),
            "last_fight": state.get("last_date"),
            "streak": state.get("streak"),
        }

    def predict(self, name_a, name_b, with_odds=True):
        ra, rb = self._resolve(name_a), self._resolve(name_b)
        if not ra or not rb:
            missing = name_a if not ra else name_b
            return {"error": "fighter not found", "query": missing}
        if ra["id"] == rb["id"]:
            return {"error": "both names resolved to the same fighter", "query": ra["matched"]}

        today = date.today().isoformat()
        fa = state_to_features(ra["state"], today)
        fb = state_to_features(rb["state"], today)
        vector = build_vector(fa, fb)
        prediction = self.predictor.predict(vector)

        final = prediction.get("final_prediction") or {}
        winner_side = final.get("winner_side", "A")
        winner_name = ra["matched"] if winner_side == "A" else rb["matched"]
        loser_name = rb["matched"] if winner_side == "A" else ra["matched"]
        method = final.get("method") or "Decision"
        win_prob = final.get("win_probability") or prediction.get("consensus_win_probability_a", 0.5)
        if winner_side == "B" and final.get("win_probability") is None:
            win_prob = 1.0 - prediction.get("consensus_win_probability_a", 0.5)
        final.update({
            "winner": winner_name,
            "loser": loser_name,
            "method": method,
            "most_likely_finish": method,
            "win_probability": float(win_prob),
            "summary": (
                f"Pick: {winner_name} ({win_prob * 100:.1f}%) — "
                f"most likely finish: {method}"
            ),
        })
        prediction["final_prediction"] = final

        result = {
            "fighter_a": ra["matched"],
            "fighter_b": rb["matched"],
            "final_prediction": final,
            "match_meta": {
                "a": {"query": name_a, "matched": ra["matched"], "score": ra["score"]},
                "b": {"query": name_b, "matched": rb["matched"], "score": rb["score"]},
            },
            "fighters": {"a": self._summary(ra["state"]), "b": self._summary(rb["state"])},
            "prediction": prediction,
        }
        if with_odds:
            result["odds"] = self._odds_block(
                ra["matched"], rb["matched"], prediction["consensus_win_probability_a"])
        return result

    def _odds_block(self, name_a, name_b, model_prob_a):
        odds = self.odds.get_matchup_odds(name_a, name_b)
        if not odds.get("available"):
            return {"available": False, "reason": odds.get("reason", "odds unavailable")}
        matchup = odds.get("matchup")
        if not matchup:
            return {"available": True, "matchup": None, "reason": odds.get("reason")}
        implied = matchup.get("implied_probability", {})
        implied_a = None
        for name, prob in implied.items():
            if name and name.split()[-1].lower() == name_a.split()[-1].lower():
                implied_a = prob
        edge = model_prob_a - implied_a if implied_a is not None else None
        blend = 0.5 * model_prob_a + 0.5 * implied_a if implied_a is not None else None
        return {
            "available": True,
            "implied_probability_a": implied_a,
            "model_edge_a": edge,
            "market_blend_probability_a": blend,
            "bookmakers": matchup.get("bookmakers"),
            "raw": matchup,
        }

    def predict_event(self, event_card):
        predictions = []
        for bout in event_card.get("bouts", []):
            red = bout.get("red", {}).get("name")
            blue = bout.get("blue", {}).get("name")
            if not red or not blue:
                continue
            try:
                outcome = self.predict(red, blue)
            except Exception as exc:  # noqa: BLE001
                outcome = {"error": f"prediction failed: {type(exc).__name__}"}
            predictions.append({
                "weight_class": bout.get("weight_class"),
                "title_bout": bout.get("title_bout"),
                "red": red,
                "blue": blue,
                "result": outcome,
            })
        return {
            "event": event_card.get("name"),
            "slug": event_card.get("slug"),
            "predictions": predictions,
        }
