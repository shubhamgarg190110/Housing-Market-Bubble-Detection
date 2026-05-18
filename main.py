"""
main.py — Full pipeline orchestrator.

Runs every phase in sequence and saves all outputs to disk.
Suitable for a one-shot run or scheduled refresh.

Usage:
    python main.py                   # full run
    python main.py --phase data      # data collection only
    python main.py --phase stats     # statistical analysis only
    python main.py --phase ml        # ML models only
    python main.py --phase score     # risk scoring only
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config


# ============================================================
# Phase runners
# ============================================================

def phase_data() -> pd.DataFrame:
    """Phase 2: Data collection & feature engineering."""
    logger.info("=" * 60)
    logger.info("PHASE 2 — Data Collection & Feature Engineering")
    logger.info("=" * 60)

    from src.data.collector import load_or_build
    from src.features.engineer import build_features

    master = load_or_build(config)
    logger.info("Master dataset: %d rows × %d columns", *master.shape)

    feat = build_features(master)
    feat.to_csv(config.DATA_PROCESSED / "features.csv")
    logger.info("Features saved → data/processed/features.csv")
    return feat


def phase_stats(feat: pd.DataFrame) -> dict:
    """Phase 3: Statistical bubble detection."""
    logger.info("=" * 60)
    logger.info("PHASE 3 — Statistical Bubble Detection")
    logger.info("=" * 60)

    from src.models.statistical import run_all_statistical, gsadf_labels

    price = feat.get("case_shiller_national", pd.Series(dtype=float)).dropna()
    if price.empty:
        logger.error("National price series not found — cannot run statistical models.")
        return {}

    stat_results = run_all_statistical(
        price,
        min_window=config.GSADF_MIN_WINDOW,
        lags=config.GSADF_LAGS,
    )

    gsadf_res = stat_results["gsadf"]
    logger.info("GSADF stat: %.3f | Bubbles detected: %d", gsadf_res.gsadf_stat, len(gsadf_res.bubble_periods))
    for start, end in gsadf_res.bubble_periods:
        logger.info("  Bubble episode: %s → %s", str(start)[:10], str(end)[:10])

    lppls_res = stat_results["lppls"]
    if lppls_res and lppls_res.converged:
        logger.info("LPPLS: tc_date=%s | B=%.4f | β=%.3f", lppls_res.tc_date, lppls_res.B, lppls_res.m)

    markov_res = stat_results["markov"]
    if markov_res:
        logger.info("Markov: AIC=%.1f | bubble_regime=%d", markov_res.aic, markov_res.bubble_regime)

    labels = gsadf_labels(gsadf_res, feat.index)
    feat["bubble_label"] = labels
    feat.to_csv(config.DATA_PROCESSED / "features_labeled.csv")
    logger.info("Labeled features saved → data/processed/features_labeled.csv")

    return stat_results


def phase_ml(feat: pd.DataFrame, stat_results: dict) -> tuple[dict, dict]:
    """Phase 4: ML + DL model training."""
    logger.info("=" * 60)
    logger.info("PHASE 4 — Machine Learning & Deep Learning Models")
    logger.info("=" * 60)

    from src.features.engineer import get_ml_feature_matrix
    from src.models.ml_models import (
        train_all_ml_models, model_comparison_table, backtest_models,
    )
    from src.models.deep_learning import train_all_dl_models

    X, y, feat_names = get_ml_feature_matrix(feat)
    if y is None or y.sum() < 5:
        logger.warning("Too few bubble labels — skipping ML training.")
        return {}, {}

    model_dir = config.DATA_PROCESSED / "models"
    ml_results = train_all_ml_models(X, y, save_dir=model_dir)

    comp = model_comparison_table(ml_results)
    logger.info("\n%s", comp.to_string(index=False))
    comp.to_csv(config.DATA_PROCESSED / "model_comparison.csv", index=False)

    bt = backtest_models(ml_results, X, y, config.BUBBLE_EPISODES)
    if not bt.empty:
        logger.info("\nBacktest results:\n%s", bt.to_string(index=False))
        bt.to_csv(config.DATA_PROCESSED / "backtest_results.csv", index=False)

    price = feat.get("case_shiller_national", pd.Series(dtype=float)).dropna()
    dl_results = train_all_dl_models(feat, price, X, y, save_dir=model_dir)

    return ml_results, dl_results


def phase_score(feat, stat_results, ml_results, dl_results) -> None:
    """Phase 4 (Layer 3): Composite risk scoring."""
    logger.info("=" * 60)
    logger.info("COMPOSITE RISK SCORE")
    logger.info("=" * 60)

    from src.features.engineer import get_ml_feature_matrix
    from src.scoring.risk_score import build_composite_score, score_summary

    gsadf_res  = stat_results.get("gsadf")
    lppls_res  = stat_results.get("lppls")
    markov_res = stat_results.get("markov")
    ae_res     = dl_results.get("lstm_ae")
    pred_res   = dl_results.get("lstm_pred")

    X_full, _, _ = get_ml_feature_matrix(feat)

    risk_res = build_composite_score(
        gsadf_res, lppls_res, markov_res,
        ml_results, ae_res, pred_res,
        X_full if not X_full.empty else feat.select_dtypes(include=[np.number]).dropna(),
    )

    summary = score_summary(risk_res)
    logger.info(
        "\n  Current Risk Score : %.1f / 100"
        "\n  Risk Level         : %s"
        "\n  As of              : %s"
        "\n  Peak score         : %.1f (%s)"
        "\n  Components         : %s",
        summary["score"], summary["level"], summary["latest_date"],
        summary["peak_score"], summary["peak_date"],
        summary["components"],
    )

    risk_res.score.to_csv(config.DATA_PROCESSED / "risk_score.csv", header=True)
    risk_res.components.to_csv(config.DATA_PROCESSED / "risk_components.csv")
    logger.info("Risk score saved → data/processed/risk_score.csv")


# ============================================================
# Entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Housing Bubble Detection Pipeline")
    parser.add_argument(
        "--phase",
        choices=["data", "stats", "ml", "score", "all"],
        default="all",
        help="Which phase to run (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    feat = pd.DataFrame()
    stat_results = {}
    ml_results = {}
    dl_results = {}

    if args.phase in ("data", "all"):
        feat = phase_data()

    if feat.empty and args.phase != "data":
        labeled_path = config.DATA_PROCESSED / "features_labeled.csv"
        plain_path   = config.DATA_PROCESSED / "features.csv"
        path = labeled_path if labeled_path.exists() else plain_path
        if path.exists():
            feat = pd.read_csv(path, index_col=0, parse_dates=True)
            logger.info("Loaded features from %s", path)
        else:
            logger.error("No feature file found. Run with --phase data first.")
            sys.exit(1)

    if args.phase in ("stats", "all"):
        stat_results = phase_stats(feat)

    if args.phase in ("ml", "all"):
        ml_results, dl_results = phase_ml(feat, stat_results)

    if args.phase in ("score", "all"):
        if stat_results:
            phase_score(feat, stat_results, ml_results, dl_results)
        else:
            logger.warning("Statistical results not available — skipping risk score.")

    logger.info("=" * 60)
    logger.info("Pipeline complete.  Launch dashboard with:")
    logger.info("  streamlit run dashboard/app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
