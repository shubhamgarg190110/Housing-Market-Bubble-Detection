"""
Housing Market Bubble Detection — Streamlit Dashboard
Capstone Project by Shubham Garg

Run:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import logging
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Housing Bubble Detector",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 16px; margin: 4px 0;
    }
    .risk-badge {
        padding: 6px 16px; border-radius: 20px;
        font-weight: bold; font-size: 1.1em; display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)


# ── Cached pipeline runner ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading data and running models… (first run ~3 min)")
def run_full_pipeline():
    """
    Execute the complete analysis pipeline once and cache the results.
    On subsequent page interactions the cache is reused (fast).
    """
    import config
    from src.data.collector import load_or_build
    from src.features.engineer import build_features, get_ml_feature_matrix
    from src.models.statistical import run_all_statistical, gsadf_labels
    from src.models.ml_models import train_all_ml_models, model_comparison_table
    from src.models.deep_learning import train_all_dl_models
    from src.scoring.risk_score import build_composite_score, score_summary

    # 1. Data
    master = load_or_build(config)

    # 2. Features
    feat = build_features(master)

    # 3. Statistical models
    price = feat.get("case_shiller_national", pd.Series(dtype=float)).dropna()
    stat_results = run_all_statistical(price)
    gsadf_res  = stat_results["gsadf"]
    lppls_res  = stat_results["lppls"]
    markov_res = stat_results["markov"]

    # 4. Labels from GSADF
    feat["bubble_label"] = gsadf_labels(gsadf_res, feat.index)

    # 5. ML models
    X, y, feat_names = get_ml_feature_matrix(feat)
    ml_results = {}
    if y is not None and y.sum() > 5:
        ml_results = train_all_ml_models(X, y)

    # 6. Deep learning (optional)
    dl_results = train_all_dl_models(feat, price, X, y) if y is not None else {}

    # 7. Composite score
    ae_res   = dl_results.get("lstm_ae")
    pred_res = dl_results.get("lstm_pred")
    X_full   = X if not X.empty else feat.select_dtypes(include=[np.number]).dropna()
    risk_res = build_composite_score(
        gsadf_res, lppls_res, markov_res,
        ml_results, ae_res, pred_res, X_full,
    )

    summary = score_summary(risk_res)
    comp_table = model_comparison_table(ml_results) if ml_results else pd.DataFrame()

    return {
        "master": master,
        "feat": feat,
        "price": price,
        "gsadf": gsadf_res,
        "lppls": lppls_res,
        "markov": markov_res,
        "ml_results": ml_results,
        "dl_results": dl_results,
        "risk": risk_res,
        "summary": summary,
        "comp_table": comp_table,
        "X_full": X_full,
        "config": config,
    }


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/House_icon_isolated.svg/240px-House_icon_isolated.svg.png", width=60)
    st.title("Housing Bubble\nDetector")
    st.caption("Capstone Project · Shubham Garg · 2026–27")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Overview", "Data Explorer", "Statistical Analysis",
         "ML Models", "City Comparison", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    significance = st.selectbox("GSADF significance level", ["90%", "95%", "99%"], index=1)
    show_bubbles = st.checkbox("Shade known bubble periods", value=True)


# ── Load pipeline (cached) ───────────────────────────────────────────────────
with st.spinner("Initialising pipeline…"):
    try:
        ctx = run_full_pipeline()
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.info("Make sure you have internet access for the first run (FRED data download).")
        st.stop()

config      = ctx["config"]
feat        = ctx["feat"]
price       = ctx["price"]
gsadf_res   = ctx["gsadf"]
lppls_res   = ctx["lppls"]
markov_res  = ctx["markov"]
ml_results  = ctx["ml_results"]
dl_results  = ctx["dl_results"]
risk_res    = ctx["risk"]
summary     = ctx["summary"]
comp_table  = ctx["comp_table"]
X_full      = ctx["X_full"]
master      = ctx["master"]

known_bubbles = config.BUBBLE_EPISODES if show_bubbles else []

from src.visualization import plots as viz

# ============================================================
#  PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":
    st.title("Housing Market Bubble Detection")
    st.markdown(
        "_Can machine learning models, combined with macroeconomic indicators, "
        "detect and predict housing market bubbles with sufficient lead time to "
        "serve as early warning signals?_"
    )
    st.divider()

    # Top metrics row
    c1, c2, c3, c4 = st.columns(4)
    color = summary["color"]
    level = summary["level"]
    with c1:
        st.metric("Current Risk Score", f"{summary['score']:.1f} / 100")
    with c2:
        st.markdown(
            f'<div class="risk-badge" style="background:{color};color:white">{level}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"As of {summary['latest_date']}")
    with c3:
        st.metric("Peak Score", f"{summary['peak_score']:.1f}", delta=f"reached {summary['peak_date']}")
    with c4:
        st.metric("Data Range", "1990 – 2025")

    st.divider()

    col_gauge, col_ts = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(
            viz.plot_risk_gauge(summary["score"], level),
            use_container_width=True,
        )
        st.markdown("**Component Breakdown (0–100)**")
        for comp, val in summary["components"].items():
            st.progress(int(val), text=f"{comp}: {val:.1f}")

    with col_ts:
        st.plotly_chart(
            viz.plot_risk_timeseries(risk_res.score, known_bubbles),
            use_container_width=True,
        )

    st.divider()
    st.subheader("House Prices with Detected Bubble Periods")
    st.plotly_chart(
        viz.plot_price_index(price, gsadf_res.bubble_periods + known_bubbles),
        use_container_width=True,
    )


# ============================================================
#  PAGE 2 — DATA EXPLORER
# ============================================================
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.caption("Explore raw and engineered features.")

    tab1, tab2, tab3 = st.tabs(["Price Indices", "Macro Indicators", "Correlation"])

    with tab1:
        city_series = {}
        for city, col in config.CITIES.items():
            s = feat.get(col, pd.Series(dtype=float)).dropna()
            if not s.empty:
                city_series[city] = s
        if city_series:
            normalise = st.checkbox("Normalise to 100 (Jan 1990)", value=True)
            st.plotly_chart(
                viz.plot_city_comparison(city_series, known_bubbles, normalise=normalise),
                use_container_width=True,
            )
        else:
            st.info("City-level data not yet available.")

    with tab2:
        macro_cols = [
            "yoy_price_growth", "price_to_rent", "price_to_income",
            "mortgage_spread", "hp_deviation", "credit_growth",
            "real_mortgage_rate", "affordability_index",
        ]
        available = [c for c in macro_cols if c in feat.columns]
        selected = st.multiselect("Select features", available, default=available[:4])
        if selected:
            import plotly.express as px
            plot_df = feat[selected].dropna()
            for col in selected:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col], mode="lines", name=col))
                if show_bubbles:
                    for start, end, label in known_bubbles:
                        fig.add_vrect(x0=str(start)[:10], x1=str(end)[:10],
                                      fillcolor="rgba(231,76,60,0.12)", line_width=0,
                                      annotation_text=label, annotation_position="top left",
                                      annotation_font_size=9)
                fig.update_layout(title=col.replace("_", " ").title(),
                                  template="plotly_white", hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.plotly_chart(
            viz.plot_correlation_heatmap(feat, top_n=18),
            use_container_width=True,
        )


# ============================================================
#  PAGE 3 — STATISTICAL ANALYSIS
# ============================================================
elif page == "Statistical Analysis":
    st.title("Statistical Bubble Detection")

    tab_gsadf, tab_lppls, tab_markov = st.tabs(["GSADF Test", "LPPLS Model", "Markov Regime-Switching"])

    with tab_gsadf:
        st.markdown("### Generalized Supremum ADF (GSADF) Test")
        st.markdown(
            "The **GSADF test** (Phillips, Shi & Yu, 2015) is the gold-standard econometric "
            "tool for detecting explosive price behaviour. The Backward Supremum ADF (BSADF) "
            "sequence crossing the critical value identifies exact bubble origination and collapse dates."
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("GSADF Statistic", f"{gsadf_res.gsadf_stat:.3f}")
        col2.metric(f"CV ({significance})", f"{gsadf_res.critical_values.get(significance, 'N/A'):.3f}")
        col3.metric("Bubble Periods Detected", len(gsadf_res.bubble_periods))
        st.plotly_chart(
            viz.plot_bsadf(gsadf_res.bsadf_sequence, gsadf_res.critical_values, known_bubbles, significance),
            use_container_width=True,
        )
        if gsadf_res.bubble_periods:
            st.subheader("Detected Bubble Episodes")
            bp_df = pd.DataFrame(gsadf_res.bubble_periods, columns=["Start", "End"])
            bp_df["Duration (months)"] = ((bp_df["End"] - bp_df["Start"]) / pd.Timedelta("30D")).round(0).astype(int)
            st.dataframe(bp_df, use_container_width=True)

    with tab_lppls:
        st.markdown("### LPPLS — Log-Periodic Power Law Singularity")
        st.markdown(
            "The **LPPLS model** (Johansen & Sornette, 1999) detects _super-exponential_ price "
            "growth with log-periodic oscillations that precede market crashes. "
            "Used by ETH Zurich's Financial Crisis Observatory."
        )
        st.latex(r"\ln p(t) = A + B(t_c - t)^{\beta}\left[1 + C\cos\left(\omega\ln(t_c - t) + \phi\right)\right]")

        if lppls_res:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("B (power-law coeff)", f"{lppls_res.B:.4f}",
                        delta="Super-exponential" if lppls_res.B < 0 else "Normal",
                        delta_color="inverse")
            col2.metric("β (exponent)", f"{lppls_res.m:.3f}")
            col3.metric("ω (frequency)", f"{lppls_res.omega:.2f}")
            col4.metric("Converged", "Yes" if lppls_res.converged else "No")
            if lppls_res.tc_date:
                st.info(f"Predicted critical time (crash window): **{str(lppls_res.tc_date)[:10]}**")
        st.plotly_chart(
            viz.plot_lppls_fit(price, lppls_res),
            use_container_width=True,
        )

    with tab_markov:
        st.markdown("### Markov Regime-Switching Model")
        st.markdown(
            "A **2-state Hidden Markov Model** (Hamilton, 1994) identifies whether the market "
            "is in a fundamentals-driven (_Normal_) or speculation-driven (_Bubble_) regime, "
            "with transition probabilities between states."
        )
        if markov_res:
            col1, col2 = st.columns(2)
            br = markov_res.bubble_regime
            tm = markov_res.transition_matrix
            col1.metric("Bubble Regime", f"Regime {br}")
            col2.metric("P(stay in bubble)", f"{tm[br, br]:.3f}")
            st.plotly_chart(
                viz.plot_markov_regimes(markov_res.smoothed_probs, br, price),
                use_container_width=True,
            )
        else:
            st.warning("Markov model results not available (requires statsmodels with regime-switching support).")


# ============================================================
#  PAGE 4 — ML MODELS
# ============================================================
elif page == "ML Models":
    st.title("Machine Learning Models")

    if not ml_results:
        st.warning("ML models were not trained (insufficient labelled data). Run the pipeline after data collection.")
    else:
        tab_comp, tab_fi, tab_ae, tab_bt = st.tabs(
            ["Model Comparison", "Feature Importance", "LSTM Autoencoder", "Backtesting"]
        )

        with tab_comp:
            st.markdown("### Model Performance Comparison")
            st.markdown(
                "Models are trained on the first **80%** of the time series and tested on the "
                "remaining **20%**, preserving temporal order to prevent look-ahead bias."
            )
            st.dataframe(comp_table, use_container_width=True)
            st.plotly_chart(viz.plot_model_comparison(comp_table), use_container_width=True)

        with tab_fi:
            st.markdown("### XGBoost Feature Importance")
            st.markdown("Which macroeconomic indicators matter most for bubble prediction?")
            xgb_res = ml_results.get("xgboost")
            if xgb_res and xgb_res.feature_importance is not None:
                st.plotly_chart(
                    viz.plot_feature_importance(xgb_res.feature_importance, top_n=20),
                    use_container_width=True,
                )
                top5 = xgb_res.feature_importance.head(5)
                st.markdown("**Top 5 predictors:**")
                for feat_name, imp in top5.items():
                    st.markdown(f"- **{feat_name}**: {imp:.4f}")

        with tab_ae:
            st.markdown("### LSTM Autoencoder — Anomaly Detection")
            st.markdown(
                "The autoencoder learns _normal_ housing market dynamics. "
                "High reconstruction error indicates the market is behaving _anomalously_ "
                "— a strong early-warning signal."
            )
            ae_res = dl_results.get("lstm_ae")
            if ae_res and not ae_res.reconstruction_error.empty:
                st.plotly_chart(
                    viz.plot_ae_error(ae_res.reconstruction_error, ae_res.threshold, known_bubbles),
                    use_container_width=True,
                )
                pct_anomalous = ae_res.anomaly_flag.mean() * 100
                st.metric("% of periods flagged as anomalous", f"{pct_anomalous:.1f}%")
            else:
                st.info("LSTM Autoencoder results not available (TensorFlow may not be installed or training was skipped).")

        with tab_bt:
            st.markdown("### Backtesting Against Known Bubble Episodes")
            st.markdown(
                "Did each model raise an early-warning signal in the **6 months before** "
                "each known bubble's start date?"
            )
            from src.models.ml_models import backtest_models
            bt_df = backtest_models(ml_results, X_full, feat.get("bubble_label", pd.Series()), known_bubbles)
            if not bt_df.empty:
                st.dataframe(bt_df, use_container_width=True)
                success_rate = bt_df["flagged"].mean() * 100
                st.metric("Early-warning success rate", f"{success_rate:.0f}%")
            else:
                st.info("Backtest not available.")


# ============================================================
#  PAGE 5 — CITY COMPARISON
# ============================================================
elif page == "City Comparison":
    st.title("City-Level Housing Analysis")
    st.markdown("NYC focus: comparing New York to national trends and other major metros.")

    city_series = {}
    for city, col in config.CITIES.items():
        s = feat.get(col, pd.Series(dtype=float)).dropna()
        if not s.empty:
            city_series[city] = s

    if not city_series:
        st.warning("City-level data not available. Check FRED connectivity and series IDs.")
    else:
        st.plotly_chart(
            viz.plot_city_comparison(city_series, known_bubbles, normalise=True),
            use_container_width=True,
        )

        st.divider()
        st.subheader("City-Level Statistics")
        rows = []
        for city, s in city_series.items():
            peak_val = s.max()
            peak_date = s.idxmax()
            recent = s.pct_change(12).iloc[-1] * 100
            since_peak = (s.iloc[-1] / peak_val - 1) * 100
            rows.append({
                "City": city,
                "Current Index": round(float(s.iloc[-1]), 1),
                "Peak Index": round(float(peak_val), 1),
                "Peak Date": str(peak_date)[:7],
                "YoY Growth (%)": round(float(recent), 1),
                "From Peak (%)": round(float(since_peak), 1),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()
        st.subheader("NYC Bubble Risk Analysis")
        nyc = feat.get("case_shiller_nyc", pd.Series(dtype=float)).dropna()
        if not nyc.empty:
            nyc_gsadf = None
            try:
                from src.models.statistical import run_gsadf
                nyc_gsadf = run_gsadf(nyc)
            except Exception:
                pass
            if nyc_gsadf:
                col1, col2 = st.columns(2)
                col1.metric("NYC GSADF Statistic", f"{nyc_gsadf.gsadf_stat:.3f}")
                col2.metric("NYC Bubble Periods", len(nyc_gsadf.bubble_periods))
                st.plotly_chart(
                    viz.plot_price_index(nyc, nyc_gsadf.bubble_periods + known_bubbles, "NYC Home Price Index (Case-Shiller)", "NYC"),
                    use_container_width=True,
                )
        else:
            st.info("NYC-specific price series not available.")


# ============================================================
#  PAGE 6 — ABOUT
# ============================================================
elif page == "About":
    st.title("About This Project")

    st.markdown("""
## Housing Market Bubble Detection
**Capstone Research Project** | Shubham Garg | April 2026

---

### Research Question
> *Can machine learning models, combined with macroeconomic indicators, detect and predict
> housing market bubbles in the U.S. with sufficient lead time to serve as early warning signals?*

---

### Methodology

| Component | Method | Reference |
|---|---|---|
| Statistical Test | GSADF (Generalized Supremum ADF) | Phillips, Shi & Yu (2015) |
| Physics Model | LPPLS (Log-Periodic Power Law) | Johansen & Sornette (1999) |
| Regime Detection | Markov Autoregression | Hamilton (1994) |
| Gradient Boosting | XGBoost with TimeSeriesSplit CV | Chen & Guestrin (2016) |
| Anomaly Detection | LSTM Autoencoder | Biagini et al. (arXiv) |
| Classification | Feedforward Neural Network | — |
| Composite Score | Weighted Ensemble (0–100) | Original contribution |

---

### Data Sources

- **FRED (Federal Reserve)** — Case-Shiller Index, mortgage rates, unemployment, CPI, GDP
- **Zillow Research** — ZHVI home values, ZORI rent index
- **BLS** — Consumer Price Index, labor statistics
- **NAR** — Housing affordability index

---

### Key Features Engineered

- Price-to-Rent Ratio *(strongest single bubble indicator)*
- Price-to-Income Ratio
- HP-Filtered Trend Deviation
- YoY Price Acceleration
- Mortgage Rate Spread
- Real Mortgage Rate
- Credit Growth Rate
- Sentiment Momentum

---

### Literature Review

- **Case & Shiller (2003)** — "Is There a Bubble in the Housing Market?"
- **Christopher Mayer** — "Housing Bubbles: A Survey" (Columbia Business School)
- **IMF Working Paper** — "Bubble Detective: City-Level Analysis of House Price Cycles"
- **Dallas Fed (2025)** — Consumer expectations and housing market exuberance
- **Biagini et al.** — Detecting asset price bubbles using deep learning (arXiv)

---

### Technology Stack

`Python 3.10+` · `pandas` · `numpy` · `statsmodels` · `scikit-learn` · `XGBoost` ·
`TensorFlow/Keras` · `Plotly` · `Streamlit` · `scipy` · `FRED API`

---

### Publication Targets

- Polygence Research Symposium
- Lumiere Research Symposium
- arXiv preprint (q-fin.GN)
- Journal of Student Research

---

*Built by Shubham Garg (Grade 10) | New York City | 2026–2027*
    """)

    st.divider()
    st.subheader("Project Timeline")
    timeline = [
        ("Sep 2026", "Phase 1", "Literature Review — research question defined"),
        ("Oct 2026", "Phase 2", "Data Collection — FRED + Zillow dataset assembled"),
        ("Nov–Dec 2026", "Phase 3", "Statistical Analysis — GSADF + LPPLS + Markov"),
        ("Jan–Feb 2027", "Phase 4", "ML Models — XGBoost + LSTM trained and validated"),
        ("Mar 2027", "Phase 5", "Dashboard — this Streamlit app deployed"),
        ("Apr 2027", "Phase 6", "Research Paper — submitted to symposium / arXiv"),
        ("Summer 2027", "Polish", "Portfolio ready for internship applications"),
    ]
    st.dataframe(
        pd.DataFrame(timeline, columns=["Period", "Phase", "Milestone"]),
        use_container_width=True, hide_index=True,
    )
