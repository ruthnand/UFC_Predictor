# UFC API

A Python + Flask service that scrapes live data from the official UFC website
and layers machine-learning fight predictions on top of it. It exposes a JSON
API **and** a web dashboard for exploring stats, rankings, upcoming/past events,
betting odds, and model-driven matchup predictions.

## Features

- **Fighter stats** by name (age, division, physique, record, strikes, offense, takedowns, KOs, submissions)
- **Fight history** — full bout history (opponent, result, date, round, time, method, event)
- **Fuzzy name lookup** — misspelled or reordered names are corrected against the UFC roster
- **Rankings** — champion + top 15 for every division
- **Events** — upcoming and past events, plus full per-event fight cards
- **Live odds** — MMA moneyline odds via The Odds API, de-vigged to implied probabilities
- **Fight predictions** — two models with reasoning, built on **point-in-time features** (each fighter's Elo, recent form, streak, layoff, age, physicals, and per-minute offense/defense rates *as of the fight date* — no lookahead leakage):
  - **Gradient-boosted trees** (XGBoost, with a scikit-learn fallback), isotonic-calibrated → win probability + feature importances
  - **Bayesian logistic regression** (Laplace approximation) → win probability + 90% credible interval + per-feature log-odds contributions
- **Web dashboard** — head-to-head predictor, event-card predictions vs. market odds, and rankings

## Quick start

```bash
pip install -r requirements.txt

# (optional) enable live odds
export ODDS_API_KEY=your_the_odds_api_key

# train + backtest from the committed point-in-time dataset (fast)
python train.py

# run the app (dashboard at http://127.0.0.1:5000/)
python main.py
```

To rebuild the dataset from scratch by re-scraping ufcstats.com (which sits behind
a JavaScript anti-bot challenge), install the extra scraping deps and run the
scrape step — otherwise `train.py` just uses the committed data:

```bash
pip install -r requirements-train.txt
python -m playwright install chromium
python train.py --scrape      # solve challenge, scrape all events + fighters, rebuild, retrain
```

> On macOS, port 5000 is often taken by AirPlay Receiver (empty 403s). Disable it
> in System Settings → General → AirDrop & Handoff, or run on another port.
> XGBoost needs OpenMP (`brew install libomp` on macOS); without it the predictor
> automatically falls back to scikit-learn gradient boosting.

## Endpoints

Fighter endpoints accept either a single full name via `name` (e.g. `name=Sean Strickland`)
or split `firstName` / `middleName` / `lastName` params.

| Endpoint | Description |
|---|---|
| `GET /` | Web dashboard |
| `GET /health` | Status + whether models are trained |
| `GET /fighter` | Career stats and record |
| `GET /fighter/history` | Full fight history |
| `GET /fighter/profile` | Combined stats + fight history |
| `GET /rankings` | All divisions (or `?division=Lightweight`) |
| `GET /events` | Upcoming + past events |
| `GET /events/<slug>` | Full fight card for one event |
| `GET /odds` | Live MMA moneyline odds (implied probabilities) |
| `GET /predict?fighterA=..&fighterB=..` | Both models' predictions + reasoning + odds edge |
| `GET /predict/event/<slug>` | Predictions for every bout on a card |
| `GET /backtest` | Forward-chaining model evaluation (accuracy, AUC, log-loss, Brier, calibration) |

### Examples

```
/fighter?name=Jon Jones
/fighter/history?name=Sean Strickland
/fighter?name=shawn stricklan          # fuzzy → "Sean Strickland"
/rankings?division=Lightweight
/events
/predict?fighterA=Islam Makhachev&fighterB=Ilia Topuria
/predict/event/ufc-330
```

## How predictions work

The prediction pipeline is built on **point-in-time** data so the model never sees
the future when predicting a past fight:

1. `ufcstats_scraper.py` / `ufcstats_collect.py` scrape every completed UFC bout
   from ufcstats.com (per-fight knockdowns, sig. strikes, takedowns, sub attempts,
   control time, head/body/leg strike breakdown, method, round, time) plus each
   fighter's physicals, stance, and date of birth.
2. `ufcstats_dataset.py` walks all ~8,700 fights **in chronological order**,
   snapshotting each fighter's state *before* the bout, then updating it with the
   result. For every fighter it maintains an **Elo rating**, a **Glicko-2 rating**
   (with rating deviation, so uncertainty from short records and long layoffs is
   modeled), **opponent-adjusted metrics** (strength of schedule = mean opponent
   pre-fight Elo; striking landed/absorbed relative to what each opponent
   typically produces), win rate, current streak, layoff, age, and cumulative
   per-minute offense/defense rates (strikes landed/absorbed, takedowns
   for/against, submission attempts, knockdowns for/against, control
   for/against, target-specific head/body/leg striking, finish and
   finished-against rates). Each fight becomes a balanced pair of rows
   (A−B → 1 and B−A → 0). It also writes `fighter_state.json`, each fighter's
   latest state, used for live prediction.
3. `pit_features.py` defines the shared feature contract: a matchup vector is the
   **difference** of the two fighters' features (positive = fighter A's edge),
   plus pair-level features (stance mismatch, and the exact **Glicko-2 expected
   score**, which combines both ratings and both uncertainties). Missing
   components become `NaN`.
4. `models.py` median-imputes missing values (adding missingness indicators),
   then trains a **stacking ensemble**:
   - a gradient-boosted tree classifier (**XGBoost → CatBoost → sklearn**
     preference; hyperparameters chosen by forward-chaining grid search),
     **isotonic-calibrated** on pooled chronological out-of-fold predictions,
   - a Bayesian logistic regression fit by MAP + Laplace covariance, giving a
     predictive win probability with a credible interval, and
   - a **logistic meta-model** fit on the same out-of-fold predictions. On the
     untouched time-based holdout the meta is compared against an equal-weight
     blend of the calibrated bases, and whichever scores better is served
     (with two correlated bases and ~10k rows the equal blend currently wins —
     the classic shrinkage result).
5. `/predict` looks up both fighters' current states, builds the difference vector,
   and returns both models' probabilities, the ensemble consensus, a **final
   prediction** (`winner` + **method of victory** among KO/TKO, Submission, and
   Decision, with method probabilities), the drivers behind each pick (tree
   feature importances, Bayesian log-odds contributions), and — if
   `ODDS_API_KEY` is set — the market's **de-vigged** implied probability, the
   model's edge, and a **market-blended forecast** (50/50 model x market;
   historical odds aren't available for training, so the market enters at serve
   time rather than inside the trained stacker).

> Realistic performance: the ensemble reaches **~66% accuracy / 0.71 AUC on the
> most recent 20% of fights** and ~62.5% / 0.67 across all eras in the
> forward-chaining backtest, with log-loss well below the base-rate baseline —
> in line with strong public MMA models and the betting market itself. MMA is
> high-variance; treat ~65% calibrated as a good ceiling rather than a guarantee.

## Backtesting

`python train.py` also runs a **forward-chaining** backtest (`backtest.py`): it
always trains on past fights and validates on future ones, mirroring real use, and
saves the report to `models/backtest.json` (surfaced at `/backtest` and the
dashboard's **Backtest** tab). It reports accuracy, AUC, log-loss, and Brier score
for the tree model, the Bayesian model, the equal-weight ensemble, and the stacked
meta-model, plus a **calibration (reliability) curve** so you can see how well
predicted probabilities match observed win rates. Log-loss below the base-rate
baseline means the models add information beyond guessing.

Feature ablation is applied to the deeper fight-detail metrics: all head/body/leg
rates remain in the point-in-time dataset and fighter state, but the production
model currently uses **control for/against plus head strikes landed/absorbed**.
Those columns improved forward-chaining accuracy/AUC; adding body and leg rates
reduced held-out performance, so they are retained for analysis rather than fed
to the model blindly.

## Deploying to Render

The repo includes a `render.yaml` blueprint and a `Procfile`. Trained models and
the roster/stats caches are committed, so the service works immediately without a
slow cold-start build.

1. Push this repo to GitHub.
2. In Render: **New → Blueprint**, point it at the repo (it reads `render.yaml`).
3. (Optional) Set `ODDS_API_KEY` in the service's Environment settings to enable live odds.
4. Deploy. The dashboard is served at the service root `/`, with a health check at `/health`.

The app runs under gunicorn: `gunicorn main:app --workers 1 --threads 4 --timeout 180`
(the long timeout accommodates live scraping). XGBoost loads natively on Render's
Linux; locally it falls back to scikit-learn if OpenMP is missing.

To retrain from the committed data, run `python train.py`. To rebuild everything by
re-scraping ufcstats.com, run `python train.py --scrape` (needs
`requirements-train.txt` + `playwright install chromium`).

## Data & caching

Committed (needed at runtime, so the deployed demo works with no cold-start scrape):

- `fighter_state.json` — every fighter's latest point-in-time state (for `/predict`)
- `models/` — trained model artifacts + `backtest.json`
- `active_fighters.json` / `fighter_stats_cache.json` — used by the ufc.com fighter/history endpoints

Rebuild-only (gitignored, regenerated by `train.py --scrape` / `train.py --rebuild`):

- `ufcstats_events.json` / `ufcstats_fighters.json` — event and fighter scrape caches
- `ufcstats_fight_details.json` — per-fight control and strike-target totals
- `ufcstats_cookie.json` — cached anti-bot clearance cookie
- `ufc_pit_dataset.csv` — the built point-in-time training set
