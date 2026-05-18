"""
Phase 4 (Layer 3) — Composite Housing Bubble Risk Score (0–100).

Combines signals from:
  - GSADF statistical test
  - LPPLS model
  - Markov regime probability
  - XGBoost classifier
  - LSTM Autoencoder anomaly score
  - LSTM Predictor divergence

Produces a weighted ensemble score with colour-coded risk levels.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default component weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "gsadf":     0.25,
    "lppls":     0.15,
    "markov":    0.15,
    "xgboost":   0.20,
    "lstm_ae":   0.15,
    "lstm_pred": 0.10,
}

# Risk level thresholds (0–100 scale)
RISK_LEVELS = {
    "Normal":   (0,   35),
    "Elevated": (35,  55),
    "High":     (55,  75),
    "Bubble":   (75, 100),
}

RISK_COLORS = {
    "Normal":   "#2ecc71",   # green
    "Elevated": "#f1c40f",   # yellow
    "High":     "#e67e22",   # orange
    "Bubble":   "#e74c3c",   # red
}


@dataclass
class RiskScoreResult:
    score: pd.Series                # composite 0–100 score indexed by date
    components: pd.DataFrame        # individual 0–1 component scores
    risk_level: pd.Series           # string risk level per date
    risk_color: pd.Series           # hex colour per date
    weights_used: dict


def _normalise_to_01(s: pd.Series, clip: bool = True) -> pd.Series:
    """Min-max normalise to [0, 1], handling NaN gracefully."""
    s = s.dropna()
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    n = (s - lo) / (hi - lo)
    if clip:
        n = n.clip(0, 1)
    return n


def _gsadf_component(gsadf_result) -> pd.Series:
    """
    Converts BSADF sequence to a 0-1 probability signal.
    Normalises relative to the 95% critical value so values above CV → > 0.5.
    """
    bsadf = gsadf_result.bsadf_sequence.dropna()
    cv95 = gsadf_result.critical_values.get("95%", 1.42)
    # Sigmoid centred at CV
    sig = 1 / (1 + np.exp(-2.5 * (bsadf - cv95)))
    sig.name = "gsadf_signal"
    return sig.clip(0, 1)


def _lppls_component(lppls_result, index: pd.DatetimeIndex) -> pd.Series:
    """
    LPPLS contributes a scalar signal (time-invariant for a given fit).
    Broadcast to the full index.
    """
    if lppls_result is None or not lppls_result.converged:
        return pd.Series(0.2, index=index, name="lppls_signal")

    score = 0.0
    if lppls_result.B < 0:
        score += 0.5        # super-exponential growth toward crash

    n = len(index)
    t_end = float(n - 1)
    time_to_tc = lppls_result.tc - t_end
    urgency = max(0.0, 1.0 - time_to_tc / (n * 0.3))
    score += 0.5 * urgency
    return pd.Series(min(score, 1.0), index=index, name="lppls_signal")


def _markov_component(markov_result, index: pd.DatetimeIndex) -> pd.Series:
    """Bubble-regime probability from the Markov model."""
    if markov_result is None:
        return pd.Series(0.0, index=index, name="markov_signal")
    br = markov_result.bubble_regime
    prob = markov_result.smoothed_probs[f"regime_{br}"]
    prob.name = "markov_signal"
    return prob.reindex(index, method="nearest").fillna(0.0).clip(0, 1)


def _ml_component(ml_results: dict, model_key: str, X_full: pd.DataFrame) -> pd.Series:
    """Bubble probability from a sklearn/XGBoost model."""
    res = ml_results.get(model_key)
    if res is None:
        return pd.Series(0.0, index=X_full.index, name=f"{model_key}_signal")
    try:
        prob = res.model.predict_proba(X_full)[:, 1]
        return pd.Series(prob, index=X_full.index, name=f"{model_key}_signal").clip(0, 1)
    except Exception:
        return pd.Series(0.0, index=X_full.index, name=f"{model_key}_signal")


def _ae_component(ae_result, index: pd.DatetimeIndex) -> pd.Series:
    """Normalised LSTM Autoencoder reconstruction error → 0-1."""
    if ae_result is None or ae_result.reconstruction_error.empty:
        return pd.Series(0.0, index=index, name="lstm_ae_signal")
    err = ae_result.reconstruction_error
    norm = _normalise_to_01(err).reindex(index, method="nearest").fillna(0.0)
    norm.name = "lstm_ae_signal"
    return norm


def _pred_component(pred_result, index: pd.DatetimeIndex) -> pd.Series:
    """LSTM Predictor divergence signal → 0-1."""
    if pred_result is None or pred_result.bubble_signal.empty:
        return pd.Series(0.0, index=index, name="lstm_pred_signal")
    sig = pred_result.bubble_signal.reindex(index, method="nearest").fillna(0.0)
    sig.name = "lstm_pred_signal"
    return sig.clip(0, 1)


def assign_risk_level(score: float) -> str:
    for level, (lo, hi) in RISK_LEVELS.items():
        if lo <= score <= hi:
            return level
    return "Bubble"


def build_composite_score(
    gsadf_result,
    lppls_result,
    markov_result,
    ml_results: dict,
    ae_result,
    pred_result,
    X_full: pd.DataFrame,
    weights: Optional[dict] = None,
) -> RiskScoreResult:
    """
    Assemble the composite Housing Bubble Risk Score.

    Parameters
    ----------
    gsadf_result  : GSADFResult
    lppls_result  : LPPLSResult
    markov_result : MarkovResult or None
    ml_results    : dict of ModelResult objects from ml_models.py
    ae_result     : LSTMAutoencoderResult or None
    pred_result   : LSTMPredictorResult or None
    X_full        : full feature matrix (used for ML model inference)
    weights       : dict of component weights (defaults to DEFAULT_WEIGHTS)
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    # Normalise weights in case user passed partial dict
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}

    full_index = X_full.index

    components = pd.DataFrame(index=full_index)
    components["gsadf"]     = _gsadf_component(gsadf_result).reindex(full_index, method="nearest").fillna(0)
    components["lppls"]     = _lppls_component(lppls_result, full_index)
    components["markov"]    = _markov_component(markov_result, full_index)
    components["xgboost"]   = _ml_component(ml_results, "xgboost", X_full)
    components["lstm_ae"]   = _ae_component(ae_result, full_index)
    components["lstm_pred"] = _pred_component(pred_result, full_index)

    # Weighted sum → scale to 0–100
    score = sum(components[k] * w[k] for k in w if k in components.columns)
    score = (score * 100).clip(0, 100)
    score.name = "bubble_risk_score"

    # Smooth with 3-month rolling average to reduce noise
    score_smooth = score.rolling(3, min_periods=1).mean()
    score_smooth.name = "bubble_risk_score"

    risk_level = score_smooth.map(assign_risk_level)
    risk_color = risk_level.map(RISK_COLORS)

    logger.info(
        "Risk Score built: current=%.1f | level=%s | date=%s",
        score_smooth.iloc[-1],
        risk_level.iloc[-1],
        score_smooth.index[-1].strftime("%Y-%m"),
    )
    return RiskScoreResult(
        score=score_smooth,
        components=components,
        risk_level=risk_level,
        risk_color=risk_color,
        weights_used=w,
    )


def score_summary(result: RiskScoreResult) -> dict:
    """Return a human-readable summary of the latest risk reading."""
    latest_date = result.score.index[-1]
    latest_score = float(result.score.iloc[-1])
    level = result.risk_level.iloc[-1]
    color = result.risk_color.iloc[-1]

    peak_date = result.score.idxmax()
    peak_score = float(result.score.max())

    component_latest = result.components.iloc[-1].to_dict()

    return {
        "latest_date": latest_date.strftime("%B %Y"),
        "score": round(latest_score, 1),
        "level": level,
        "color": color,
        "peak_date": peak_date.strftime("%B %Y"),
        "peak_score": round(peak_score, 1),
        "components": {k: round(float(v * 100), 1) for k, v in component_latest.items()},
    }
