"""
Phase 4 (Layer 1) — Classical Machine Learning Models.

Trains XGBoost, Random Forest, and Logistic Regression classifiers
to predict housing bubble periods from macroeconomic features.
Uses GSADF-derived labels as ground truth.
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ============================================================
# Data containers
# ============================================================

@dataclass
class ModelResult:
    name: str
    model: object
    accuracy: float
    auc: float
    confusion: np.ndarray
    report: str
    feature_importance: Optional[pd.Series] = None
    cv_scores: np.ndarray = field(default_factory=lambda: np.array([]))


# ============================================================
# Time-series aware train / test split
# ============================================================

def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_frac: float = 0.20,
) -> tuple:
    """
    Preserve temporal order: train on first 80%, test on last 20%.
    This avoids look-ahead bias.
    """
    n = len(X)
    cutoff = int(n * (1 - test_frac))
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
    return X_train, X_test, y_train, y_test


# ============================================================
# Individual model trainers
# ============================================================

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_splits: int = 5,
) -> ModelResult:
    """
    XGBoost gradient-boosted tree classifier.
    Key advantage: built-in feature importance reveals which macro indicators
    drive bubble risk — research shows interest rates explain >60% of dynamics.
    """
    logger.info("Training XGBoost …")
    scale_pos_weight = max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1))

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="roc_auc")

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fi = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
        name="importance",
    ).sort_values(ascending=False)

    result = ModelResult(
        name="XGBoost",
        model=model,
        accuracy=accuracy_score(y_test, y_pred),
        auc=roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else 0.5,
        confusion=confusion_matrix(y_test, y_pred),
        report=classification_report(y_test, y_pred, zero_division=0),
        feature_importance=fi,
        cv_scores=cv_scores,
    )
    logger.info("XGBoost: acc=%.3f | AUC=%.3f | CV-AUC=%.3f±%.3f",
                result.accuracy, result.auc, cv_scores.mean(), cv_scores.std())
    return result


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_splits: int = 5,
) -> ModelResult:
    """Random Forest ensemble — used as a comparison baseline."""
    logger.info("Training Random Forest …")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="roc_auc")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fi = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
        name="importance",
    ).sort_values(ascending=False)

    result = ModelResult(
        name="Random Forest",
        model=model,
        accuracy=accuracy_score(y_test, y_pred),
        auc=roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else 0.5,
        confusion=confusion_matrix(y_test, y_pred),
        report=classification_report(y_test, y_pred, zero_division=0),
        feature_importance=fi,
        cv_scores=cv_scores,
    )
    logger.info("Random Forest: acc=%.3f | AUC=%.3f | CV-AUC=%.3f±%.3f",
                result.accuracy, result.auc, cv_scores.mean(), cv_scores.std())
    return result


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_splits: int = 5,
) -> ModelResult:
    """
    Logistic Regression — interpretable linear baseline.
    Uses standardisation via a sklearn Pipeline so coefficients are comparable.
    """
    logger.info("Training Logistic Regression …")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=500,
            solver="lbfgs", random_state=42,
        )),
    ])
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="roc_auc")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    lr = pipe.named_steps["lr"]
    fi = pd.Series(
        np.abs(lr.coef_[0]),
        index=X_train.columns,
        name="importance",
    ).sort_values(ascending=False)

    result = ModelResult(
        name="Logistic Regression",
        model=pipe,
        accuracy=accuracy_score(y_test, y_pred),
        auc=roc_auc_score(y_test, y_prob) if y_test.nunique() > 1 else 0.5,
        confusion=confusion_matrix(y_test, y_pred),
        report=classification_report(y_test, y_pred, zero_division=0),
        feature_importance=fi,
        cv_scores=cv_scores,
    )
    logger.info("Logistic Regression: acc=%.3f | AUC=%.3f | CV-AUC=%.3f±%.3f",
                result.accuracy, result.auc, cv_scores.mean(), cv_scores.std())
    return result


# ============================================================
# Prediction helpers
# ============================================================

def predict_proba_all(
    models: dict[str, ModelResult],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a DataFrame of bubble probabilities from each model,
    indexed to X's index.
    """
    probs = {}
    for name, res in models.items():
        try:
            p = res.model.predict_proba(X)[:, 1]
            probs[name] = pd.Series(p, index=X.index)
        except Exception as exc:
            logger.warning("predict_proba failed for %s: %s", name, exc)
    return pd.DataFrame(probs)


def backtest_models(
    models: dict[str, ModelResult],
    X: pd.DataFrame,
    y: pd.Series,
    known_bubbles: list[tuple],
) -> pd.DataFrame:
    """
    Backtest: check whether each model raises a bubble flag within 6 months
    before each known bubble episode's start date.
    Returns a summary DataFrame.
    """
    rows = []
    for name, res in models.items():
        try:
            y_prob = pd.Series(
                res.model.predict_proba(X)[:, 1], index=X.index
            )
        except Exception:
            continue
        for start, end, label in known_bubbles:
            # Check average probability in the 6 months before bubble start
            window_start = pd.Timestamp(start) - pd.DateOffset(months=6)
            window_end   = pd.Timestamp(start)
            window_prob  = y_prob.loc[window_start:window_end].mean()
            rows.append({
                "model": name,
                "bubble": label,
                "avg_prob_6m_before": round(window_prob, 3) if not np.isnan(window_prob) else None,
                "flagged": window_prob > 0.5 if not np.isnan(window_prob) else False,
            })
    return pd.DataFrame(rows)


# ============================================================
# Train all models
# ============================================================

def train_all_ml_models(
    X: pd.DataFrame,
    y: pd.Series,
    save_dir: Optional[Path] = None,
) -> dict[str, ModelResult]:
    """
    Train XGBoost, Random Forest, and Logistic Regression.
    Saves models to `save_dir` if provided.
    Returns dict keyed by model name.
    """
    X_train, X_test, y_train, y_test = time_series_split(X, y)
    logger.info(
        "Train set: %d | Test set: %d | Bubble prevalence (train): %.1f%%",
        len(X_train), len(X_test), y_train.mean() * 100,
    )

    results = {
        "xgboost": train_xgboost(X_train, y_train, X_test, y_test),
        "random_forest": train_random_forest(X_train, y_train, X_test, y_test),
        "logistic_regression": train_logistic_regression(X_train, y_train, X_test, y_test),
    }

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, res in results.items():
            path = save_dir / f"{name}.joblib"
            joblib.dump(res.model, path)
            logger.info("Saved %s → %s", name, path)

    return results


def load_ml_models(save_dir: Path) -> dict:
    """Load persisted sklearn/XGBoost models from disk."""
    models = {}
    for p in save_dir.glob("*.joblib"):
        models[p.stem] = joblib.load(p)
    return models


# ============================================================
# Comparison table
# ============================================================

def model_comparison_table(results: dict[str, ModelResult]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": res.name,
            "Accuracy": round(res.accuracy, 3),
            "AUC-ROC": round(res.auc, 3),
            "CV-AUC (mean)": round(res.cv_scores.mean(), 3) if len(res.cv_scores) > 0 else None,
            "CV-AUC (std)": round(res.cv_scores.std(), 3) if len(res.cv_scores) > 0 else None,
        })
    return pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
