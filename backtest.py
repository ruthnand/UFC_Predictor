import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from pit_features import feature_names
from models import BayesianLogistic, apply_stacker, build_tree_model, fit_stacker


def _calibration_curve(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    curve = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        curve.append({
            "bin_lower": float(bins[b]),
            "bin_upper": float(bins[b + 1]),
            "mean_predicted": float(np.mean(y_prob[mask])),
            "observed_rate": float(np.mean(y_true[mask])),
            "count": count,
        })
    return curve


def _metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "calibration": _calibration_curve(y_true, y_prob),
    }


def _prep_fold(Xtr, Xval):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    miss = np.where(np.isnan(Xtr).any(axis=0))[0]

    def build(X):
        flags = np.isnan(X[:, miss]).astype(float) if len(miss) else np.empty((len(X), 0))
        filled = np.where(np.isnan(X), med, X)
        return np.hstack([filled, flags])

    return build(Xtr), build(Xval)


def _time_bounds(df, n_splits):
    """Return fold boundaries without splitting fights from the same date."""
    if "date" not in df.columns:
        return np.linspace(0, len(df), n_splits + 1, dtype=int)
    dates = df["date"].astype(str).to_numpy()
    unique_dates = np.unique(dates)
    date_cuts = np.linspace(0, len(unique_dates), n_splits + 1, dtype=int)
    return np.asarray([
        len(df) if cut >= len(unique_dates)
        else int(np.searchsorted(dates, unique_dates[cut], side="left"))
        for cut in date_cuts
    ])


def _json_cell(value):
    """Convert pandas missing values to JSON-safe nulls."""
    return None if pd.isna(value) else value


class Backtester:
    """Forward-chaining (time-based) evaluation: always train on the past and
    validate on the future, mirroring how the model is used in practice. Trees
    are isotonic-calibrated on a tail of each training window."""

    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def run(self, df, features=None):
        features = features or feature_names()
        df = df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df
        X = df[features].to_numpy(dtype=float)
        y = df["label"].to_numpy(dtype=int)
        n = len(y)

        # Time-ordered folds; validate on folds 2..K, train on everything prior.
        bounds = _time_bounds(df, self.n_splits)
        oof_tree, oof_bayes, oof_stacked, covered = [], [], [], []
        covered_frames = []
        tree_engine = None

        for k in range(1, self.n_splits):
            tr_end = bounds[k]
            va_start, va_end = bounds[k], bounds[k + 1]
            if tr_end < 50 or va_end <= va_start:
                continue
            Xtr_raw, Xva_raw = X[:tr_end], X[va_start:va_end]
            ytr, yva = y[:tr_end], y[va_start:va_end]
            Xtr, Xva = _prep_fold(Xtr_raw, Xva_raw)

            # Calibration and the stacking meta-model are fit on pooled inner
            # out-of-fold predictions (base models only ever see earlier rows),
            # mirroring the production training procedure.
            inner = _time_bounds(df.iloc[:tr_end], 4)
            oof_t, oof_b, oof_y = [], [], []
            for j in range(1, 4):
                fit_end, ivs, ive = inner[j], inner[j], inner[j + 1]
                t, tree_engine = build_tree_model()
                t.fit(Xtr[:fit_end], ytr[:fit_end])
                b = BayesianLogistic().fit(Xtr[:fit_end], ytr[:fit_end])
                oof_t.append(t.predict_proba(Xtr[ivs:ive])[:, 1])
                oof_b.append(b.predict_proba(Xtr[ivs:ive]))
                oof_y.append(ytr[ivs:ive])
            ot, ob, oy = map(np.concatenate, (oof_t, oof_b, oof_y))
            cal = IsotonicRegression(out_of_bounds="clip").fit(ot, oy)
            stacker = fit_stacker(cal.transform(ot), ob, oy)

            # Refit bases on the full window, then predict the validation fold.
            tree_full, _ = build_tree_model()
            tree_full.fit(Xtr, ytr)
            tree_va = cal.transform(tree_full.predict_proba(Xva)[:, 1])

            bayes = BayesianLogistic().fit(Xtr, ytr)
            bayes_va = bayes.predict_proba(Xva)

            oof_tree.append(tree_va)
            oof_bayes.append(bayes_va)
            oof_stacked.append(apply_stacker(stacker, tree_va, bayes_va))
            covered.append(yva)
            covered_frames.append(df.iloc[va_start:va_end].copy())

        y_cov = np.concatenate(covered)
        tree_p = np.concatenate(oof_tree)
        bayes_p = np.concatenate(oof_bayes)
        stacked_p = np.concatenate(oof_stacked)
        consensus = (tree_p + bayes_p) / 2.0
        covered_df = pd.concat(covered_frames, ignore_index=True)

        # One record per real fight. The dataset stores a mirrored pair; label=1
        # is the winner-oriented row (f1 is the actual winner, f2 the loser).
        past_predictions = []
        for i, row in covered_df.iterrows():
            if int(row["label"]) != 1:
                continue
            actual_winner = _json_cell(row.get("f1_name"))
            actual_loser = _json_cell(row.get("f2_name"))
            prob_actual = float(consensus[i])
            correct = prob_actual >= 0.5
            predicted_winner = actual_winner if correct else actual_loser
            past_predictions.append({
                "date": _json_cell(row.get("date")),
                "event_id": _json_cell(row.get("event_id")),
                "event": _json_cell(row.get("event_name")) or "UFC Event",
                "fight_id": _json_cell(row.get("fight_id")),
                "weight_class": _json_cell(row.get("weight_class")),
                "fighter_a": actual_winner,
                "fighter_b": actual_loser,
                "predicted_winner": predicted_winner,
                "predicted_probability": float(max(prob_actual, 1.0 - prob_actual)),
                "actual_winner": actual_winner,
                "actual_method": _json_cell(row.get("method")),
                "correct": bool(correct),
                "model_probability_actual_winner": prob_actual,
                "tree_probability_actual_winner": float(tree_p[i]),
                "bayesian_probability_actual_winner": float(bayes_p[i]),
            })

        return {
            "n_rows": int(n),
            "n_evaluated": int(len(y_cov)),
            "n_splits": self.n_splits,
            "scheme": "forward-chaining (train on past, test on future)",
            "tree_engine": tree_engine,
            "positive_rate": float(y.mean()),
            "models": {
                "tree": _metrics(y_cov, tree_p),
                "bayesian": _metrics(y_cov, bayes_p),
                "consensus": _metrics(y_cov, consensus),
                "stacked": _metrics(y_cov, stacked_p),
            },
            "baseline_log_loss": float(log_loss(y_cov, np.full(len(y_cov), y_cov.mean()))),
            "_past_predictions": past_predictions,
        }

    def run_and_cache(self, df, path="models/backtest.json",
                      predictions_path="models/past_predictions.json"):
        report = self.run(df)
        past_predictions = report.pop("_past_predictions", [])
        report["past_predictions_count"] = len(past_predictions)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(report, handle, indent=2)
        with open(predictions_path, "w") as handle:
            json.dump(past_predictions, handle, indent=2)
        return report


if __name__ == "__main__":
    df = pd.read_csv("ufc_pit_dataset.csv")
    report = Backtester().run_and_cache(df)
    for name, m in report["models"].items():
        print(f"{name:10s} acc={m['accuracy']:.3f} auc={m['auc']:.3f} "
              f"logloss={m['log_loss']:.3f} brier={m['brier']:.3f}")
    print("baseline logloss:", round(report["baseline_log_loss"], 3))
