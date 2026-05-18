"""
Phase 3 — Statistical Bubble Detection.

Implements three methods:
  1. GSADF  – Generalized Supremum ADF (Phillips, Shi & Yu 2015)
  2. LPPLS  – Log-Periodic Power Law Singularity (Johansen & Sornette)
  3. Markov – Hidden Markov Regime-Switching model (Hamilton 1994)
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ============================================================
# 1.  GSADF  (Generalized Supremum ADF)
# ============================================================

@dataclass
class GSADFResult:
    gsadf_stat: float
    bsadf_sequence: pd.Series
    critical_values: dict
    bubble_periods: list[tuple]          # list of (start, end) pd.Timestamp pairs
    bubble_flag: pd.Series               # boolean series aligned to input index


def _adf_stat(y: np.ndarray, lags: int = 1) -> float:
    """Run ADF and return the test statistic. Returns -inf on failure."""
    try:
        if len(y) < lags + 5:
            return -np.inf
        result = adfuller(y, maxlag=lags, regression="c", autolag=None)
        return float(result[0])
    except Exception:
        return -np.inf


def compute_bsadf_sequence(
    series: pd.Series,
    min_window: float = 0.20,
    lags: int = 1,
) -> pd.Series:
    """
    Compute the Backward Supremum ADF (BSADF) sequence.

    For each end-point r2, the BSADF(r2) = sup over all valid r1 of ADF(r1:r2).
    This sequence is used to date-stamp bubble origination and collapse.
    """
    log_price = np.log(series.dropna().values)
    n = len(log_price)
    idx = series.dropna().index
    min_obs = max(int(min_window * n), lags + 6)

    bsadf = np.full(n, np.nan)

    for r2 in range(min_obs, n):
        best = -np.inf
        for r1 in range(0, r2 - min_obs + 1):
            stat = _adf_stat(log_price[r1 : r2 + 1], lags=lags)
            if stat > best:
                best = stat
        bsadf[r2] = best

    return pd.Series(bsadf, index=idx, name="bsadf")


def gsadf_critical_values(n: int, cv_table: Optional[dict] = None) -> dict:
    """
    Return approximate critical values for the GSADF statistic.
    Based on Monte Carlo simulations from Phillips, Shi & Yu (2015).
    We use a lookup table indexed by sample size T.
    """
    # Hard-coded CVs from PSY (2015) Table 1 (right-tail, T=100 approx.)
    default = {"90%": 1.07, "95%": 1.42, "99%": 2.10}
    if cv_table:
        return cv_table
    # Small-sample adjustment
    if n < 60:
        return {"90%": 0.90, "95%": 1.18, "99%": 1.75}
    return default


def run_gsadf(
    series: pd.Series,
    min_window: float = 0.20,
    lags: int = 1,
    significance: str = "95%",
) -> GSADFResult:
    """
    Full GSADF test on a price series.

    Parameters
    ----------
    series : pd.Series of house prices (levels, not log)
    min_window : minimum window fraction for recursive regression
    lags : ADF lag length
    significance : critical value level ('90%', '95%', '99%')
    """
    logger.info("Running GSADF test (n=%d) …", series.dropna().shape[0])
    bsadf = compute_bsadf_sequence(series, min_window=min_window, lags=lags)
    gsadf_stat = float(bsadf.max())

    cv = gsadf_critical_values(series.dropna().shape[0])
    cv_level = cv[significance]

    bubble_flag = bsadf > cv_level

    # Date-stamp: find contiguous bubble windows (must be ≥ 3 periods)
    bubble_periods = []
    in_bubble = False
    start_date = None
    for date, flag in bubble_flag.items():
        if flag and not in_bubble:
            in_bubble = True
            start_date = date
        elif not flag and in_bubble:
            in_bubble = False
            if (date - start_date).days >= 60:
                bubble_periods.append((start_date, date))
    if in_bubble:
        bubble_periods.append((start_date, bubble_flag.index[-1]))

    logger.info(
        "GSADF stat=%.3f | CV(%s)=%.3f | Bubble periods found: %d",
        gsadf_stat, significance, cv_level, len(bubble_periods),
    )
    return GSADFResult(
        gsadf_stat=gsadf_stat,
        bsadf_sequence=bsadf,
        critical_values=cv,
        bubble_periods=bubble_periods,
        bubble_flag=bubble_flag,
    )


def gsadf_labels(gsadf_result: GSADFResult, full_index: pd.DatetimeIndex) -> pd.Series:
    """
    Return a binary Series (1=bubble, 0=normal) aligned to `full_index`.
    Used as labels for the supervised ML models.
    """
    label = pd.Series(0, index=full_index, name="bubble_label", dtype=int)
    flag = gsadf_result.bubble_flag.reindex(full_index, method="ffill").fillna(0)
    label[flag.astype(bool)] = 1
    return label


# ============================================================
# 2.  LPPLS  (Log-Periodic Power Law Singularity)
# ============================================================

@dataclass
class LPPLSResult:
    tc: float            # critical time (as ordinal float from epoch)
    m: float             # power-law exponent β
    omega: float         # log-periodic frequency ω
    phi: float           # phase offset φ
    A: float
    B: float
    C: float
    residuals: float     # sum of squared residuals
    converged: bool
    tc_date: Optional[pd.Timestamp] = None


def _lppls_fn(t: np.ndarray, tc: float, m: float, omega: float,
               phi: float, A: float, B: float, C: float) -> np.ndarray:
    """LPPLS price model: ln p(t) = A + B(tc-t)^m [1 + C·cos(ω·ln(tc-t)+φ)]"""
    dt = tc - t
    dt = np.where(dt > 0, dt, np.nan)
    power = np.power(dt, m)
    return A + B * power * (1 + C * np.cos(omega * np.log(dt) + phi))


def _lppls_residual(params: np.ndarray, t: np.ndarray, log_price: np.ndarray) -> float:
    tc, m, omega, phi = params
    if tc <= t[-1]:          # tc must be in the future
        return 1e10
    dt = tc - t
    if np.any(dt <= 0):
        return 1e10
    power = dt ** m
    cos_term = np.cos(omega * np.log(dt) + phi)

    # Conditional linear least-squares for A, B, C
    f  = power
    g  = power * cos_term
    X  = np.column_stack([np.ones(len(t)), f, g])
    try:
        coeffs, res, _, _ = np.linalg.lstsq(X, log_price, rcond=None)
        A, B, C = coeffs
        pred = A + B * f + C * g
        return float(np.sum((log_price - pred) ** 2))
    except Exception:
        return 1e10


def fit_lppls(
    series: pd.Series,
    n_trials: int = 30,
    seed: int = 42,
) -> LPPLSResult:
    """
    Fit the LPPLS model to a log-price series using differential evolution
    followed by local refinement.

    Constraints follow Johansen & Sornette (1999):
      0.1 < m (β) < 0.9
      5 < ω < 25
      0 < φ < 2π
      tc > last observation (bubble hasn't burst yet)
    """
    log_price = np.log(series.dropna().values)
    n = len(log_price)
    t0 = 0.0
    t_end = float(n - 1)

    # Convert timestamps to float ordinals (days from start)
    t = np.arange(n, dtype=float)

    tc_min = t_end + 1
    tc_max = t_end + n * 0.5   # max = 50% of series length into future

    bounds = [
        (tc_min, tc_max),   # tc
        (0.10, 0.90),       # m
        (5.0, 25.0),        # omega
        (0.0, 2 * np.pi),   # phi
    ]

    logger.info("Fitting LPPLS model (n=%d, %d trials) …", n, n_trials)
    np.random.seed(seed)

    best_res = minimize(
        _lppls_residual,
        x0=[tc_min + 10, 0.5, 9.0, 0.5],
        args=(t, log_price),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-6},
    )

    try:
        de_res = differential_evolution(
            _lppls_residual,
            bounds=bounds,
            args=(t, log_price),
            seed=seed,
            maxiter=300,
            tol=1e-7,
            workers=1,
            popsize=12,
        )
        if de_res.fun < best_res.fun:
            best_res = de_res
    except Exception:
        pass

    converged = best_res.fun < 0.5
    tc_raw, m, omega, phi = best_res.x

    # Recover A, B, C
    dt = tc_raw - t
    if np.any(dt <= 0):
        dt = np.abs(dt) + 1e-6
    power = dt ** m
    cos_term = np.cos(omega * np.log(np.maximum(dt, 1e-6)) + phi)
    X = np.column_stack([np.ones(n), power, power * cos_term])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, log_price, rcond=None)
        A, B, C = coeffs
    except Exception:
        A, B, C = 0.0, 0.0, 0.0

    # Convert tc back to a real date
    idx = series.dropna().index
    tc_date = None
    if 0 <= int(tc_raw) < len(idx) * 2:
        try:
            tc_date = idx[0] + pd.DateOffset(months=int(tc_raw))
        except Exception:
            pass

    logger.info(
        "LPPLS: tc=%.1f (≈ %s) | m=%.3f | ω=%.3f | B=%.4f | converged=%s",
        tc_raw, tc_date, m, omega, B, converged,
    )
    return LPPLSResult(
        tc=tc_raw, m=m, omega=omega, phi=phi, A=A, B=B, C=C,
        residuals=best_res.fun, converged=converged, tc_date=tc_date,
    )


def lppls_risk_score(lppls_result: LPPLSResult, current_t: float) -> float:
    """
    Translate LPPLS fit into a 0–1 bubble probability.
    High score when: B<0 (super-exponential growth), close to tc, model converged.
    """
    if not lppls_result.converged:
        return 0.0
    score = 0.0
    if lppls_result.B < 0:
        score += 0.5
    time_to_tc = lppls_result.tc - current_t
    urgency = max(0.0, 1.0 - time_to_tc / 60.0)
    score += 0.5 * urgency
    return min(score, 1.0)


# ============================================================
# 3.  Markov Regime-Switching
# ============================================================

@dataclass
class MarkovResult:
    regime_probs: pd.DataFrame       # columns: regime_0, regime_1
    bubble_regime: int               # which regime index = "bubble"
    smoothed_probs: pd.DataFrame
    transition_matrix: np.ndarray
    aic: float
    bic: float


def _identify_bubble_regime(regime_probs: pd.DataFrame, returns: pd.Series) -> int:
    """
    The bubble regime is the one with higher mean and higher variance.
    Heuristic: compute correlation of each regime probability with abs(returns).
    """
    high_vol_corr = [
        float(regime_probs[f"regime_{i}"].corr(returns.abs()))
        for i in range(regime_probs.shape[1])
    ]
    return int(np.argmax(high_vol_corr))


def run_markov_switching(
    series: pd.Series,
    k_regimes: int = 2,
    order: int = 1,
) -> Optional[MarkovResult]:
    """
    Fit a Markov-switching autoregression model.
    Identifies 'bubble' vs 'normal' market regimes with transition probabilities.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )
    except ImportError:
        logger.warning("statsmodels MarkovAutoregression not available.")
        return None

    log_ret = np.log(series.dropna()).diff().dropna() * 100
    if log_ret.shape[0] < 50:
        logger.warning("Too few observations for Markov model.")
        return None

    logger.info("Fitting Markov Regime-Switching model (k=%d) …", k_regimes)
    try:
        mod = MarkovAutoregression(
            log_ret, k_regimes=k_regimes, order=order, switching_ar=False
        )
        res = mod.fit(disp=False, maxiter=200)

        smoothed = pd.DataFrame(
            res.smoothed_marginal_probabilities,
            index=log_ret.index,
            columns=[f"regime_{i}" for i in range(k_regimes)],
        )
        filtered = pd.DataFrame(
            res.filtered_marginal_probabilities,
            index=log_ret.index,
            columns=[f"regime_{i}" for i in range(k_regimes)],
        )
        bubble_regime = _identify_bubble_regime(smoothed, log_ret)
        trans_mat = np.array(res.transition)

        logger.info(
            "Markov: AIC=%.1f | bubble regime=%d | P(stay in bubble)=%.3f",
            res.aic, bubble_regime, trans_mat[bubble_regime, bubble_regime],
        )
        return MarkovResult(
            regime_probs=filtered,
            bubble_regime=bubble_regime,
            smoothed_probs=smoothed,
            transition_matrix=trans_mat,
            aic=res.aic,
            bic=res.bic,
        )
    except Exception as exc:
        logger.error("Markov model failed: %s", exc)
        return None


def markov_bubble_probability(markov_result: MarkovResult) -> pd.Series:
    """Return the probability of being in the bubble regime at each time point."""
    br = markov_result.bubble_regime
    prob = markov_result.smoothed_probs[f"regime_{br}"]
    prob.name = "markov_bubble_prob"
    return prob


# ============================================================
# Convenience runner
# ============================================================

def run_all_statistical(
    price_series: pd.Series,
    min_window: float = 0.20,
    lags: int = 1,
    significance: str = "95%",
) -> dict:
    """
    Run GSADF, LPPLS, and Markov models on `price_series`.
    Returns a dict of results keyed by model name.
    """
    results = {}

    results["gsadf"] = run_gsadf(price_series, min_window, lags, significance)

    results["lppls"] = fit_lppls(price_series)

    markov = run_markov_switching(price_series)
    results["markov"] = markov

    return results
