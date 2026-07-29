import json
import os
import traceback

from flask import Flask, jsonify, render_template, request

from cache import cache
from events import EventScraper
from fight_history import FightHistoryExtractor
from fighter import FighterStatsExtractor
from name_resolver import FighterNameResolver
from odds import OddsProvider
from predict_service import PredictionService
from top_fighters import UFCRankingScraper

app = Flask(__name__)

_prediction_service = None
EVENTS_TTL = 15 * 60
RANKINGS_TTL = 30 * 60
ODDS_TTL = 5 * 60
EVENT_CARD_TTL = 10 * 60


def get_prediction_service():
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def _split_full_name(name):
    parts = name.split()
    if len(parts) == 1:
        return parts[0], "", ""
    if len(parts) == 2:
        return parts[0], "", parts[1]
    return parts[0], " ".join(parts[1:-1]), parts[-1]


def _get_name_args():
    full_name = request.args.get("name") or request.args.get("fighterName")
    if full_name and full_name.strip():
        return _split_full_name(full_name.strip())
    return (
        request.args.get("firstName"),
        request.args.get("middleName"),
        request.args.get("lastName"),
    )


def _resolve_or_404(firstName, middleName, lastName):
    resolved = FighterNameResolver().resolve(firstName, middleName, lastName)
    if not resolved:
        query = " ".join(p for p in (firstName, middleName, lastName) if p)
        return None, (jsonify({"error": "fighter not found", "query": query}), 404)
    return resolved, None


def _match_meta(resolved):
    return {
        "matched_name": resolved["full"],
        "exact_match": resolved["exact"],
        "query": resolved["query"],
    }


@app.errorhandler(Exception)
def handle_exception(exc):
    if hasattr(exc, "code") and isinstance(exc.code, int):
        return jsonify({"error": getattr(exc, "description", str(exc))}), exc.code
    app.logger.error("Unhandled error: %s\n%s", exc, traceback.format_exc())
    return jsonify({"error": "internal server error", "detail": type(exc).__name__}), 500


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_ready": get_prediction_service().models_ready(),
    }), 200


@app.route("/about", methods=["GET"])
def about():
    """Recruiter-facing model card: metrics, calibration, limitations."""
    meta_path = os.path.join("models", "meta.json")
    backtest_path = os.path.join("models", "backtest.json")
    metrics = {}
    backtest = {}
    if os.path.exists(meta_path):
        with open(meta_path) as handle:
            metrics = json.load(handle).get("metrics", {})
    if os.path.exists(backtest_path):
        with open(backtest_path) as handle:
            backtest = json.load(handle)

    consensus = (backtest.get("models") or {}).get("consensus") or {}
    method = metrics.get("method") or {}
    return jsonify({
        "title": "About the model",
        "tagline": (
            "A leakage-free UFC fight predictor built for an analytics portfolio: "
            "point-in-time features, calibrated ensemble probabilities, and honest evaluation."
        ),
        "problem": (
            "Predict UFC fight winners (and most-likely finish type) using only information "
            "available before each bout — no career-aggregate leakage."
        ),
        "approach": [
            "Scraped ~8,700 completed UFC fights + fighter physicals from ufcstats.com",
            "Built chronological point-in-time states: Elo, Glicko-2, strength of schedule, "
            "opponent-adjusted striking, form, layoff, age, control, and head-strike rates",
            "Trained CatBoost + Bayesian logistic regression with isotonic calibration",
            "Selected equal-weight ensemble vs stacked meta-model on a time-based holdout",
            "Separate multiclass model estimates most-likely finish: KO/TKO, Submission, Decision",
        ],
        "metrics": {
            "recent_holdout": {
                "description": "Last 20% of fights by date (most recent era)",
                "accuracy": metrics.get("stacked_test_accuracy"),
                "auc": metrics.get("stacked_test_auc"),
                "log_loss": metrics.get("stacked_test_logloss"),
                "brier": metrics.get("stacked_test_brier"),
                "tree_engine": metrics.get("tree_engine"),
                "ensemble_mode": metrics.get("ensemble_mode"),
                "n_train": metrics.get("n_train"),
                "n_test": metrics.get("n_test"),
            },
            "forward_chaining": {
                "description": backtest.get("scheme"),
                "accuracy": consensus.get("accuracy"),
                "auc": consensus.get("auc"),
                "log_loss": consensus.get("log_loss"),
                "brier": consensus.get("brier"),
                "baseline_log_loss": backtest.get("baseline_log_loss"),
                "n_evaluated": backtest.get("n_evaluated"),
                "calibration": consensus.get("calibration"),
            },
            "method_of_victory": {
                "description": (
                    "Most-likely finish among KO/TKO, Submission, Decision "
                    "(inherently noisier than winner prediction)"
                ),
                "accuracy": method.get("accuracy"),
                "log_loss": method.get("logloss"),
                "class_rate": method.get("class_rate"),
            },
        },
        "limitations": [
            "MMA is high-variance; ~65% calibrated accuracy is near the public-model / market ceiling",
            "Finish-type prediction is weaker than winner prediction — treat it as a distribution, not a certainty",
            "No historical closing odds in training; market blend happens only at serve time when ODDS_API_KEY is set",
            "Fighter states need periodic refresh after new events (see refresh_data.py / Render cron)",
            "Demo does not include injury reports, weigh-in misses, or short-notice replacement flags",
        ],
        "stack": [
            "Python / Flask",
            "ufcstats.com scrape pipeline (Playwright challenge + requests)",
            "pandas / scikit-learn / CatBoost",
            "Forward-chaining backtest + reliability curves",
        ],
    }), 200


@app.route("/fighters/search", methods=["GET"])
def fighters_search():
    q = request.args.get("q") or request.args.get("query") or ""
    limit = min(int(request.args.get("limit") or 10), 25)
    return jsonify({"query": q, "results": get_prediction_service().search(q, limit=limit)}), 200


@app.route("/fighter", methods=["GET"])
def fighter():
    firstName, middleName, lastName = _get_name_args()
    resolved, error = _resolve_or_404(firstName, middleName, lastName)
    if error:
        return error
    stats = FighterStatsExtractor().get_fighter_stats(
        resolved["first"], resolved["middle"], resolved["last"]
    )
    return jsonify(stats), 200


@app.route("/fighter/history", methods=["GET"])
def fighter_history():
    firstName, middleName, lastName = _get_name_args()
    resolved, error = _resolve_or_404(firstName, middleName, lastName)
    if error:
        return error
    history = FightHistoryExtractor().get_fight_history(
        resolved["first"], resolved["middle"], resolved["last"]
    )
    return jsonify({**_match_meta(resolved), "count": len(history), "fights": history}), 200


@app.route("/fighter/profile", methods=["GET"])
def fighter_profile():
    firstName, middleName, lastName = _get_name_args()
    resolved, error = _resolve_or_404(firstName, middleName, lastName)
    if error:
        return error
    extractor = FighterStatsExtractor()
    stats = extractor.get_fighter_stats(
        resolved["first"], resolved["middle"], resolved["last"]
    )
    history = extractor.get_fight_history(
        resolved["first"], resolved["middle"], resolved["last"]
    )
    return jsonify({**_match_meta(resolved), "stats": stats, "fight_history": history}), 200


@app.route("/rankings", methods=["GET"])
def rankings():
    division = request.args.get("division")
    key = f"rankings:{division or 'all'}"
    hit = cache.get(key)
    if hit is not None:
        return jsonify(hit), 200
    scraper = UFCRankingScraper()
    if division:
        result = scraper.get_division(division)
        if not result:
            return jsonify({"error": "division not found", "query": division}), 404
        cache.set(key, result, RANKINGS_TTL)
        return jsonify(result), 200
    payload = {"rankings": scraper.get_all_rankings()}
    cache.set(key, payload, RANKINGS_TTL)
    return jsonify(payload), 200


@app.route("/events", methods=["GET"])
def events():
    key = "events:all"
    hit = cache.get(key)
    if hit is not None:
        return jsonify(hit), 200
    payload = EventScraper().get_events()
    cache.set(key, payload, EVENTS_TTL)
    return jsonify(payload), 200


@app.route("/events/<slug>", methods=["GET"])
def event_card(slug):
    key = f"event_card:{slug}"
    hit = cache.get(key)
    if hit is not None:
        return jsonify(hit), 200
    card = EventScraper().get_event_card(slug)
    if not card:
        return jsonify({"error": "event not found", "slug": slug}), 404
    cache.set(key, card, EVENT_CARD_TTL)
    return jsonify(card), 200


@app.route("/odds", methods=["GET"])
def odds():
    key = "odds:mma"
    hit = cache.get(key)
    if hit is not None:
        return jsonify(hit), 200
    payload = OddsProvider().get_mma_odds()
    cache.set(key, payload, ODDS_TTL)
    return jsonify(payload), 200


@app.route("/backtest", methods=["GET"])
def backtest():
    path = os.path.join("models", "backtest.json")
    if not os.path.exists(path):
        return jsonify({"error": "no backtest report; run `python train.py`"}), 404
    with open(path) as handle:
        return jsonify(json.load(handle)), 200


@app.route("/predict", methods=["GET"])
def predict():
    fighter_a = request.args.get("fighterA") or request.args.get("a")
    fighter_b = request.args.get("fighterB") or request.args.get("b")
    if not fighter_a or not fighter_b:
        return jsonify({"error": "provide fighterA and fighterB query params"}), 400
    service = get_prediction_service()
    if not service.models_ready():
        return jsonify({"error": "models not trained yet; run `python train.py`"}), 503
    result = service.predict(fighter_a, fighter_b)
    status = 404 if result.get("error") == "fighter not found" else 200
    return jsonify(result), status


@app.route("/predict/event/<slug>", methods=["GET"])
def predict_event(slug):
    service = get_prediction_service()
    if not service.models_ready():
        return jsonify({"error": "models not trained yet; run `python train.py`"}), 503
    key = f"event_card:{slug}"
    card = cache.get(key)
    if card is None:
        card = EventScraper().get_event_card(slug)
        if card:
            cache.set(key, card, EVENT_CARD_TTL)
    if not card:
        return jsonify({"error": "event not found", "slug": slug}), 404
    return jsonify(service.predict_event(card)), 200


if __name__ == "__main__":
    app.run(debug=True)
