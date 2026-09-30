import json
import os
import traceback

from flask import Flask, jsonify, render_template, request

from cache import cache
from espn_mma import ESPNDataError, ESPNMMAScraper
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
ESPN_EVENTS_TTL = 5 * 60
ESPN_NEWS_TTL = 15 * 60
ESPN_SEARCH_TTL = 6 * 60 * 60
ESPN_FIGHTER_TTL = 12 * 60 * 60
PAST_EVENTS_TTL = 15 * 60


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


@app.errorhandler(ESPNDataError)
def handle_espn_error(exc):
    return jsonify({"error": "ESPN MMA data is temporarily unavailable"}), 502


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


@app.route("/espn/fighters/search", methods=["GET"])
def espn_fighters_search():
    query = (request.args.get("q") or request.args.get("query") or "").strip()
    if len(query) < 2:
        return jsonify({"error": "q must contain at least 2 characters"}), 400
    try:
        limit = min(max(int(request.args.get("limit") or 10), 1), 25)
    except ValueError:
        return jsonify({"error": "limit must be a number"}), 400
    key = f"espn:search:{query.lower()}:{limit}"
    results = cache.get(key)
    if results is None:
        results = ESPNMMAScraper().search_fighters(query, limit=limit)
        cache.set(key, results, ESPN_SEARCH_TTL)
    return jsonify({
        "source": "ESPN MMA",
        "query": query,
        "results": results,
    }), 200


@app.route("/espn/fighter", methods=["GET"])
def espn_fighter():
    query = (request.args.get("name") or request.args.get("q") or "").strip()
    athlete_id = (request.args.get("id") or "").strip()
    if not query and not athlete_id:
        return jsonify({"error": "provide a fighter name or ESPN athlete id"}), 400
    if athlete_id and not athlete_id.isdigit():
        return jsonify({"error": "ESPN athlete id must be numeric"}), 400
    cache_value = athlete_id or query.lower()
    key = f"espn:fighter:{cache_value}"
    result = cache.get(key)
    if result is None:
        result = ESPNMMAScraper().get_fighter(
            query=query or None,
            athlete_id=athlete_id or None,
        )
        if result is None:
            return jsonify({"error": "fighter not found", "query": query}), 404
        cache.set(key, result, ESPN_FIGHTER_TTL)
    return jsonify(result), 200


@app.route("/espn/events", methods=["GET"])
def espn_events():
    key = "espn:events"
    result = cache.get(key)
    if result is None:
        result = ESPNMMAScraper().get_scoreboard()
        cache.set(key, result, ESPN_EVENTS_TTL)
    return jsonify(result), 200


@app.route("/espn/news", methods=["GET"])
def espn_news():
    try:
        limit = min(max(int(request.args.get("limit") or 10), 1), 25)
    except ValueError:
        return jsonify({"error": "limit must be a number"}), 400
    key = f"espn:news:{limit}"
    result = cache.get(key)
    if result is None:
        result = ESPNMMAScraper().get_news(limit=limit)
        cache.set(key, result, ESPN_NEWS_TTL)
    return jsonify(result), 200


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


def _past_event_archive():
    key = "past_events:archive"
    archive = cache.get(key)
    if archive is not None:
        return archive
    path = os.path.join("models", "past_events.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        archive = json.load(handle)
    cache.set(key, archive, PAST_EVENTS_TTL)
    return archive


@app.route("/past-events", methods=["GET"])
def past_events():
    archive = _past_event_archive()
    if archive is None:
        return jsonify({
            "error": "no past-event archive; run `python past_events.py`",
        }), 404
    try:
        limit = min(max(int(request.args.get("limit") or 30), 1), 100)
        offset = max(int(request.args.get("offset") or 0), 0)
    except ValueError:
        return jsonify({"error": "limit and offset must be numbers"}), 400

    query = (request.args.get("q") or request.args.get("search") or "").strip().lower()
    year = (request.args.get("year") or "").strip()
    events = archive.get("events") or []
    years = sorted({
        str(event.get("date") or "")[:4]
        for event in events if event.get("date")
    }, reverse=True)
    if year:
        events = [
            event for event in events
            if str(event.get("date") or "").startswith(year)
        ]
    if query:
        matched = []
        for event in events:
            searchable = [
                event.get("name") or "",
                event.get("location") or "",
            ]
            for fight in event.get("fights") or []:
                searchable.extend([
                    fight.get("fighter_1") or "",
                    fight.get("fighter_2") or "",
                ])
            if query in " ".join(searchable).lower():
                matched.append(event)
        events = matched

    summaries = [{
        key: event.get(key)
        for key in (
            "event_id", "name", "date", "location", "source_url", "fight_count"
        )
    } for event in events[offset:offset + limit]]
    return jsonify({
        "source": archive.get("source"),
        "generated_at": archive.get("generated_at"),
        "total": len(events),
        "all_event_count": archive.get("event_count"),
        "all_fight_count": archive.get("fight_count"),
        "years": years,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < len(events),
        "events": summaries,
    }), 200


@app.route("/past-events/<event_id>", methods=["GET"])
def past_event_card(event_id):
    archive = _past_event_archive()
    if archive is None:
        return jsonify({
            "error": "no past-event archive; run `python past_events.py`",
        }), 404
    event = next(
        (
            item for item in archive.get("events") or []
            if item.get("event_id") == event_id
        ),
        None,
    )
    if event is None:
        return jsonify({"error": "past event not found", "event_id": event_id}), 404
    return jsonify(event), 200


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


@app.route("/past-predictions", methods=["GET"])
def past_predictions():
    """Leakage-safe historical picks compared with actual fight results."""
    path = os.path.join("models", "past_predictions.json")
    if not os.path.exists(path):
        return jsonify({
            "error": "no historical predictions; run `python train.py`",
        }), 404
    with open(path) as handle:
        records = json.load(handle)

    fighter = (request.args.get("fighter") or "").strip().lower()
    event_query = (request.args.get("event") or "").strip().lower()
    year = (request.args.get("year") or "").strip()
    correct_query = (request.args.get("correct") or "").strip().lower()

    if fighter:
        records = [
            row for row in records
            if fighter in (row.get("fighter_a") or "").lower()
            or fighter in (row.get("fighter_b") or "").lower()
        ]
    if event_query:
        records = [
            row for row in records
            if event_query in (row.get("event") or "").lower()
        ]
    if year:
        records = [
            row for row in records
            if str(row.get("date") or "").startswith(year)
        ]
    if correct_query in ("true", "false"):
        expected = correct_query == "true"
        records = [row for row in records if bool(row.get("correct")) == expected]

    records.sort(key=lambda row: row.get("date") or "", reverse=True)

    by_year = {}
    by_event = {}
    for row in records:
        row_year = str(row.get("date") or "")[:4]
        if row_year:
            bucket = by_year.setdefault(row_year, {"total": 0, "correct": 0})
            bucket["total"] += 1
            bucket["correct"] += int(bool(row.get("correct")))
        event_key = (
            row.get("event_id") or
            f"{row.get('date')}:{row.get('event')}"
        )
        event_bucket = by_event.setdefault(event_key, {
            "event_id": row.get("event_id"),
            "event": row.get("event"),
            "date": row.get("date"),
            "total": 0,
            "correct": 0,
        })
        event_bucket["total"] += 1
        event_bucket["correct"] += int(bool(row.get("correct")))

    year_chart = [{
        "year": key,
        "total": value["total"],
        "correct": value["correct"],
        "accuracy": value["correct"] / value["total"],
    } for key, value in sorted(by_year.items())]
    event_summary = sorted(
        ({
            **value,
            "accuracy": value["correct"] / value["total"],
        } for value in by_event.values()),
        key=lambda value: value.get("date") or "",
        reverse=True,
    )

    try:
        limit = min(max(int(request.args.get("limit") or 100), 1), 500)
        offset = max(int(request.args.get("offset") or 0), 0)
    except ValueError:
        return jsonify({"error": "limit and offset must be numbers"}), 400

    total = len(records)
    correct = sum(int(bool(row.get("correct"))) for row in records)
    return jsonify({
        "methodology": (
            "Walk-forward predictions: every displayed fight was predicted by "
            "models trained only on fights that occurred earlier in time."
        ),
        "summary": {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total if total else None,
        },
        "by_year": year_chart,
        "recent_events": event_summary[:30],
        "filters": {
            "fighter": fighter or None,
            "event": event_query or None,
            "year": year or None,
            "correct": correct_query or None,
        },
        "offset": offset,
        "limit": limit,
        "results": records[offset:offset + limit],
    }), 200


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
