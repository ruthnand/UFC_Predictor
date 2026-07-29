import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from pit_features import METHOD_CLASSES, feature_names, flip_vector

# Tree engine preference: XGBoost -> CatBoost -> sklearn gradient boosting.
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_CATBOOST = False

from sklearn.ensemble import GradientBoostingClassifier


def build_tree_model(params=None):
    """Construct the gradient-boosted tree classifier with the best available
    engine (XGBoost, then CatBoost, then sklearn)."""
    params = params or {}
    if _HAS_XGB:
        model = XGBClassifier(
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.03),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            min_child_weight=params.get("min_child_weight", 3),
            reg_lambda=params.get("reg_lambda", 1.0),
            eval_metric="logloss", random_state=42,
        )
        return model, "xgboost"
    if _HAS_CATBOOST:
        # Defaults from a forward-chaining grid search (depth/lr/iterations).
        model = CatBoostClassifier(
            iterations=params.get("n_estimators", 300),
            depth=params.get("max_depth", 4),
            learning_rate=params.get("learning_rate", 0.03),
            l2_leaf_reg=params.get("reg_lambda", 3.0),
            random_seed=42, verbose=0, allow_writing_files=False,
        )
        return model, "catboost"
    model = GradientBoostingClassifier(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth", 3),
        learning_rate=params.get("learning_rate", 0.03),
        subsample=params.get("subsample", 0.9),
        random_state=42,
    )
    return model, "sklearn_gradient_boosting"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_stacker(tree_probs, bayes_probs, y):
    """Logistic meta-model that combines base-model probabilities on the
    log-odds scale. Fit only on data the base models did NOT train on."""
    X = np.column_stack([_logit(tree_probs), _logit(bayes_probs)])
    return LogisticRegression(C=1.0, max_iter=1000).fit(X, y)


def apply_stacker(meta, tree_probs, bayes_probs):
    X = np.column_stack([_logit(tree_probs), _logit(bayes_probs)])
    return meta.predict_proba(X)[:, 1]


class BayesianLogistic:
    """Bayesian logistic regression via Laplace approximation.

    A Gaussian prior on the weights is combined with the logistic likelihood.
    We find the MAP estimate with Newton's method, then approximate the
    posterior as Gaussian centered at the MAP with covariance equal to the
    inverse Hessian. This yields both a predictive win probability and a
    credible interval that reflects model uncertainty.
    """

    def __init__(self, prior_var=4.0, max_iter=100, tol=1e-8):
        self.prior_var = prior_var
        self.max_iter = max_iter
        self.tol = tol
        self.mu = None
        self.sd = None
        self.w = None      # includes intercept as w[0]
        self.cov = None

    def _standardize(self, X):
        return (X - self.mu) / self.sd

    def _augment(self, Xs):
        return np.hstack([np.ones((Xs.shape[0], 1)), Xs])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd == 0] = 1.0

        Xa = self._augment(self._standardize(X))
        n, d = Xa.shape

        # Prior precision: weak on intercept, prior_var on the rest.
        precision = np.eye(d) / self.prior_var
        precision[0, 0] = 1e-6

        w = np.zeros(d)
        for _ in range(self.max_iter):
            p = _sigmoid(Xa @ w)
            grad = Xa.T @ (p - y) + precision @ w
            R = np.clip(p * (1 - p), 1e-9, None)
            H = (Xa.T * R) @ Xa + precision
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < self.tol:
                break

        self.w = w
        p = _sigmoid(Xa @ w)
        R = np.clip(p * (1 - p), 1e-9, None)
        H = (Xa.T * R) @ Xa + precision
        self.cov = np.linalg.inv(H)
        return self

    def _predictive_logit(self, x):
        xs = (np.asarray(x, dtype=float) - self.mu) / self.sd
        xa = np.concatenate([[1.0], xs])
        mean = float(xa @ self.w)
        var = float(xa @ self.cov @ xa)
        return mean, max(var, 0.0), xs

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = []
        for x in X:
            mean, var, _ = self._predictive_logit(x)
            # Probit approximation of the logistic-normal integral.
            out.append(_sigmoid(mean / np.sqrt(1.0 + np.pi * var / 8.0)))
        return np.array(out)

    def predict_one(self, x, credible=0.90, n_samples=4000):
        mean, var, xs = self._predictive_logit(x)
        prob = float(_sigmoid(mean / np.sqrt(1.0 + np.pi * var / 8.0)))

        rng = np.random.default_rng(42)
        samples = _sigmoid(rng.normal(mean, np.sqrt(var), size=n_samples))
        lo = float(np.quantile(samples, (1 - credible) / 2))
        hi = float(np.quantile(samples, 1 - (1 - credible) / 2))

        coef_mean = self.w[1:]
        coef_sd = np.sqrt(np.clip(np.diag(self.cov)[1:], 0, None))
        contributions = coef_mean * xs
        return {
            "win_probability": prob,
            "logit_mean": mean,
            "logit_variance": var,
            "credible_interval": [lo, hi],
            "coef_mean": coef_mean,
            "coef_sd": coef_sd,
            "contributions": contributions,
        }


class FightPredictor:
    """Stacking ensemble on point-in-time features: a calibrated gradient-boosted
    tree model (XGBoost/CatBoost/sklearn) and a Bayesian logistic model are
    combined by a logistic meta-model trained on out-of-sample predictions.
    Median imputation + missingness indicators handle incomplete records."""

    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.base_features = feature_names()
        self.features = list(self.base_features)  # expanded (base + missing flags)
        self.medians = None
        self.missing_cols = []
        self.tree = None
        self.tree_engine = None
        self.tree_importances = None
        self.calibrator = None
        self.stacker = None
        self.ensemble_mode = "equal_blend"
        self.bayes = None
        self.method_model = None
        self.method_classes = list(METHOD_CLASSES)
        self.metrics = {}

    # ----- preprocessing -----
    def _fit_prep(self, Xbase):
        self.medians = np.nanmedian(Xbase, axis=0)
        self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
        has_missing = np.isnan(Xbase).any(axis=0)
        self.missing_cols = [self.base_features[i] for i in range(len(self.base_features)) if has_missing[i]]
        self.features = list(self.base_features) + [f"{c}_missing" for c in self.missing_cols]

    def _prep(self, Xbase):
        Xbase = np.asarray(Xbase, dtype=float)
        if Xbase.ndim == 1:
            Xbase = Xbase.reshape(1, -1)
        idx = [self.base_features.index(c) for c in self.missing_cols]
        flags = np.isnan(Xbase[:, idx]).astype(float) if idx else np.empty((len(Xbase), 0))
        filled = np.where(np.isnan(Xbase), self.medians, Xbase)
        return np.hstack([filled, flags])

    # ----- training -----
    def _time_split(self, df, test_frac=0.2):
        df = df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df
        cut = int(len(df) * (1 - test_frac))
        return df.iloc[:cut], df.iloc[cut:]

    def _inner_oof(self, X, y, params, n_inner=5):
        """Pooled chronological out-of-fold predictions inside a training
        window: for each inner fold, base models train only on earlier rows."""
        bounds = np.linspace(0, len(X), n_inner, dtype=int)
        oof_tree, oof_bayes, oof_y = [], [], []
        for j in range(1, n_inner - 1):
            fit_end, val_start, val_end = bounds[j], bounds[j], bounds[j + 1]
            tree, _ = build_tree_model(params)
            tree.fit(X[:fit_end], y[:fit_end])
            bayes = BayesianLogistic().fit(X[:fit_end], y[:fit_end])
            oof_tree.append(tree.predict_proba(X[val_start:val_end])[:, 1])
            oof_bayes.append(bayes.predict_proba(X[val_start:val_end]))
            oof_y.append(y[val_start:val_end])
        return (np.concatenate(oof_tree), np.concatenate(oof_bayes),
                np.concatenate(oof_y))

    def train(self, df, params=None):
        Xbase_all = df[self.base_features].to_numpy(dtype=float)
        y_all = df["label"].to_numpy(dtype=int)
        if len(np.unique(y_all)) < 2:
            raise ValueError("Training data needs both win and loss labels.")

        self._fit_prep(Xbase_all)

        # Honest holdout split by time (keeps mirrored rows of a fight together).
        train_df, test_df = self._time_split(df)
        X_tr = self._prep(train_df[self.base_features].to_numpy(dtype=float))
        y_tr = train_df["label"].to_numpy(dtype=int)
        X_te = self._prep(test_df[self.base_features].to_numpy(dtype=float))
        y_te = test_df["label"].to_numpy(dtype=int)

        # Calibrator + stacking meta-model are fit on pooled inner
        # out-of-fold predictions, keeping the outer holdout untouched for
        # evaluation and ensemble selection.
        oof_tree_raw, oof_bayes, oof_y = self._inner_oof(X_tr, y_tr, params)
        self.calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_tree_raw, oof_y)
        oof_tree = self.calibrator.transform(oof_tree_raw)
        self.stacker = fit_stacker(oof_tree, oof_bayes, oof_y)

        # Base models on the full training window; evaluate on the holdout.
        tree_tmp, self.tree_engine = build_tree_model(params)
        tree_tmp.fit(X_tr, y_tr)
        cal_te = self.calibrator.transform(tree_tmp.predict_proba(X_te)[:, 1])
        bayes_tmp = BayesianLogistic().fit(X_tr, y_tr)
        bp_te = bayes_tmp.predict_proba(X_te)

        blend_te = (cal_te + bp_te) / 2.0
        meta_te = apply_stacker(self.stacker, cal_te, bp_te)

        def _pack(probs):
            return {
                "accuracy": float(accuracy_score(y_te, (probs >= 0.5).astype(int))),
                "auc": float(roc_auc_score(y_te, probs)),
                "logloss": float(log_loss(y_te, np.clip(probs, 1e-6, 1 - 1e-6))),
                "brier": float(brier_score_loss(y_te, probs)),
            }

        tree_m, bayes_m = _pack(cal_te), _pack(bp_te)
        blend_m, meta_m = _pack(blend_te), _pack(meta_te)

        # Ensemble selection on the untouched holdout: free-weight meta vs.
        # equal-weight blend (a fixed-weight stacker). With correlated bases
        # and limited data the blend usually wins; keep whichever is better.
        self.ensemble_mode = "stacked_meta" if meta_m["logloss"] < blend_m["logloss"] else "equal_blend"
        chosen_m = meta_m if self.ensemble_mode == "stacked_meta" else blend_m

        self.metrics = {
            "n_rows": int(len(df)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "tree_engine": self.tree_engine,
            "holdout": "time-based (last 20%)",
            "ensemble_mode": self.ensemble_mode,
            "tree_test_accuracy": tree_m["accuracy"],
            "tree_test_auc": tree_m["auc"],
            "tree_test_logloss": tree_m["logloss"],
            "tree_test_brier": tree_m["brier"],
            "bayes_test_accuracy": bayes_m["accuracy"],
            "bayes_test_auc": bayes_m["auc"],
            "bayes_test_logloss": bayes_m["logloss"],
            "stacked_test_accuracy": chosen_m["accuracy"],
            "stacked_test_auc": chosen_m["auc"],
            "stacked_test_logloss": chosen_m["logloss"],
            "stacked_test_brier": chosen_m["brier"],
            "stacked_meta_holdout": meta_m,
            "equal_blend_holdout": blend_m,
            "stacker_coefficients": {
                "tree_logit": float(self.stacker.coef_[0][0]),
                "bayes_logit": float(self.stacker.coef_[0][1]),
                "intercept": float(self.stacker.intercept_[0]),
            },
        }

        # Method-of-victory model: trained on winner-oriented rows only
        # (label == 1), predicting KO/TKO / Submission / Decision.
        method_metrics = self._train_method_model(df, params)
        if method_metrics:
            self.metrics["method"] = method_metrics

        # Refit base models on all data for serving (calibrator, stacker, and
        # ensemble mode kept from the out-of-fold / holdout fits).
        X_all = self._prep(Xbase_all)
        self.tree, _ = build_tree_model(params)
        self.tree.fit(X_all, y_all)
        self.tree_importances = np.asarray(self.tree.feature_importances_, dtype=float)
        self.bayes = BayesianLogistic().fit(X_all, y_all)
        return self.metrics

    def _train_method_model(self, df, params=None):
        if "method" not in df.columns:
            return None
        win_df = df[(df["label"] == 1) & df["method"].isin(self.method_classes)].copy()
        if len(win_df) < 100 or win_df["method"].nunique() < 2:
            return None

        train_df, test_df = self._time_split(win_df)
        X_tr = self._prep(train_df[self.base_features].to_numpy(dtype=float))
        y_tr = train_df["method"].to_numpy()
        X_te = self._prep(test_df[self.base_features].to_numpy(dtype=float))
        y_te = test_df["method"].to_numpy()

        method_model, engine = build_tree_model(params)
        class_to_idx = {c: i for i, c in enumerate(METHOD_CLASSES)}
        # sklearn / xgboost need integer labels; CatBoost accepts strings.
        if engine == "catboost":
            method_model.fit(X_tr, y_tr)
            pred = method_model.predict(X_te)
            if getattr(pred, "ndim", 1) > 1:
                pred = pred.ravel()
            pred = np.asarray(pred).astype(str)
            proba = method_model.predict_proba(X_te)
            classes = list(method_model.classes_)
        else:
            y_tr_i = np.array([class_to_idx[c] for c in y_tr])
            method_model.fit(X_tr, y_tr_i)
            pred_i = method_model.predict(X_te)
            pred = np.array([METHOD_CLASSES[int(i)] for i in pred_i])
            proba = method_model.predict_proba(X_te)
            classes = list(METHOD_CLASSES)

        # Refit on all winner-oriented rows for serving.
        X_all = self._prep(win_df[self.base_features].to_numpy(dtype=float))
        y_all = win_df["method"].to_numpy()
        self.method_model, _ = build_tree_model(params)
        if engine == "catboost":
            self.method_model.fit(X_all, y_all)
            self.method_classes = [str(c) for c in self.method_model.classes_]
        else:
            y_all_i = np.array([class_to_idx[c] for c in y_all])
            self.method_model.fit(X_all, y_all_i)
            self.method_classes = list(METHOD_CLASSES)

        # Align probability columns to sklearn's lexicographic label order.
        label_order = sorted(METHOD_CLASSES)
        class_index = {c: i for i, c in enumerate(classes)}
        ordered = np.column_stack([
            proba[:, class_index[c]] if c in class_index else np.zeros(len(proba))
            for c in label_order
        ])
        return {
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "engine": engine,
            "accuracy": float(accuracy_score(y_te, pred)),
            "logloss": float(log_loss(y_te, ordered, labels=label_order)),
            "class_rate": {
                c: float((y_te == c).mean()) for c in METHOD_CLASSES
            },
        }

    # ----- persistence -----
    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        import joblib

        joblib.dump(self.tree, os.path.join(self.model_dir, "tree.joblib"))
        joblib.dump(self.calibrator, os.path.join(self.model_dir, "calibrator.joblib"))
        joblib.dump(self.stacker, os.path.join(self.model_dir, "stacker.joblib"))
        if self.method_model is not None:
            joblib.dump(self.method_model, os.path.join(self.model_dir, "method.joblib"))
        np.savez(
            os.path.join(self.model_dir, "bayes.npz"),
            mu=self.bayes.mu, sd=self.bayes.sd, w=self.bayes.w, cov=self.bayes.cov,
            importances=self.tree_importances, medians=self.medians,
        )
        with open(os.path.join(self.model_dir, "meta.json"), "w") as handle:
            json.dump({
                "base_features": self.base_features,
                "features": self.features,
                "missing_cols": self.missing_cols,
                "tree_engine": self.tree_engine,
                "ensemble_mode": self.ensemble_mode,
                "method_classes": self.method_classes,
                "metrics": self.metrics,
            }, handle, indent=2)

    def load(self):
        import joblib

        meta_path = os.path.join(self.model_dir, "meta.json")
        if not os.path.exists(meta_path):
            return False
        with open(meta_path) as handle:
            meta = json.load(handle)
        self.base_features = meta.get("base_features", feature_names())
        self.features = meta["features"]
        self.missing_cols = meta.get("missing_cols", [])
        self.tree_engine = meta["tree_engine"]
        self.ensemble_mode = meta.get("ensemble_mode", "equal_blend")
        self.method_classes = meta.get("method_classes", list(METHOD_CLASSES))
        self.metrics = meta.get("metrics", {})
        self.tree = joblib.load(os.path.join(self.model_dir, "tree.joblib"))
        cal_path = os.path.join(self.model_dir, "calibrator.joblib")
        self.calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None
        stack_path = os.path.join(self.model_dir, "stacker.joblib")
        self.stacker = joblib.load(stack_path) if os.path.exists(stack_path) else None
        method_path = os.path.join(self.model_dir, "method.joblib")
        self.method_model = joblib.load(method_path) if os.path.exists(method_path) else None
        data = np.load(os.path.join(self.model_dir, "bayes.npz"))
        self.bayes = BayesianLogistic()
        self.bayes.mu, self.bayes.sd = data["mu"], data["sd"]
        self.bayes.w, self.bayes.cov = data["w"], data["cov"]
        self.tree_importances = data["importances"]
        self.medians = data["medians"]
        return True

    # ----- inference -----
    def predict(self, vector, top_k=6):
        """`vector` is a BASE feature vector (len == len(base_features)); NaN
        allowed for unknown components."""
        x = self._prep(np.asarray(vector, dtype=float))

        raw_prob = float(self.tree.predict_proba(x)[0, 1])
        tree_prob = float(self.calibrator.transform([raw_prob])[0]) if self.calibrator else raw_prob

        order = np.argsort(self.tree_importances)[::-1]
        tree_reasons = []
        for i in order:
            if i >= len(self.base_features):
                continue
            val = float(vector[i]) if not np.isnan(vector[i]) else 0.0
            tree_reasons.append({
                "feature": self.base_features[i],
                "importance": float(self.tree_importances[i]),
                "matchup_value": val,
                "favors": "A" if val > 0 else ("B" if val < 0 else "even"),
            })
            if len(tree_reasons) >= top_k:
                break

        bx = x[0]
        bayes = self.bayes.predict_one(bx)
        contrib = bayes["contributions"]
        border = np.argsort(np.abs(contrib))[::-1]
        bayes_reasons = []
        for i in border:
            if i >= len(self.base_features):
                continue
            bayes_reasons.append({
                "feature": self.base_features[i],
                "coefficient_mean": float(bayes["coef_mean"][i]),
                "coefficient_sd": float(bayes["coef_sd"][i]),
                "contribution": float(contrib[i]),
                "favors": "A" if contrib[i] > 0 else ("B" if contrib[i] < 0 else "even"),
            })
            if len(bayes_reasons) >= top_k:
                break

        bayes_prob = bayes["win_probability"]
        if self.ensemble_mode == "stacked_meta" and self.stacker is not None:
            consensus = float(apply_stacker(self.stacker, [tree_prob], [bayes_prob])[0])
            ensemble_method = "stacked logistic meta-model (selected on holdout)"
        else:
            consensus = (tree_prob + bayes_prob) / 2.0
            ensemble_method = "equal-weight blend of calibrated bases (selected on holdout)"

        winner_is_a = consensus >= 0.5
        method_block = self._predict_method(vector if winner_is_a else flip_vector(vector))

        return {
            "consensus_win_probability_a": consensus,
            "final_prediction": {
                "winner_side": "A" if winner_is_a else "B",
                "win_probability": consensus if winner_is_a else 1.0 - consensus,
                "method": method_block["predicted_method"],
                "method_probabilities": method_block["probabilities"],
                "summary": None,  # filled by PredictionService with fighter names
            },
            "ensemble": {
                "method": ensemble_method,
                "mode": self.ensemble_mode,
                "stacked_probability_a": consensus,
                "base_probabilities": {"tree": tree_prob, "bayesian": bayes_prob},
                "meta_coefficients": self.metrics.get("stacker_coefficients"),
            },
            "xgboost": {
                "engine": self.tree_engine,
                "win_probability_a": tree_prob,
                "top_features": tree_reasons,
            },
            "bayesian": {
                "win_probability_a": bayes_prob,
                "credible_interval_90": bayes["credible_interval"],
                "top_features": bayes_reasons,
            },
            "metrics": self.metrics,
        }

    def _predict_method(self, winner_oriented_vector):
        """Predict finish method for the projected winner."""
        fallback = {c: 1.0 / len(METHOD_CLASSES) for c in METHOD_CLASSES}
        if self.method_model is None:
            return {"predicted_method": "Decision", "probabilities": fallback}

        x = self._prep(np.asarray(winner_oriented_vector, dtype=float))
        proba = self.method_model.predict_proba(x)[0]
        if self.tree_engine == "catboost" or hasattr(self.method_model, "classes_"):
            classes = [str(c) for c in getattr(self.method_model, "classes_", self.method_classes)]
        else:
            classes = list(self.method_classes)
        probs = {c: 0.0 for c in METHOD_CLASSES}
        for cls, p in zip(classes, proba):
            # Integer-trained models expose 0/1/2 class labels.
            if isinstance(cls, (int, np.integer)) or (isinstance(cls, str) and cls.isdigit()):
                cls = METHOD_CLASSES[int(cls)]
            if cls in probs:
                probs[cls] = float(p)
        predicted = max(probs, key=probs.get)
        return {"predicted_method": predicted, "probabilities": probs}


if __name__ == "__main__":
    df = pd.read_csv("ufc_pit_dataset.csv")
    predictor = FightPredictor()
    print(predictor.train(df))
    predictor.save()
    print("saved")
