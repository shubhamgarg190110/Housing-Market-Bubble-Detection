"""
Reusable Plotly visualisation functions for the dashboard and notebooks.
All functions return plotly Figure objects that can be shown or embedded in Streamlit.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.scoring.risk_score import RISK_COLORS, RISK_LEVELS

# ---- colour palette -------------------------------------------------------
PALETTE = px.colors.qualitative.Plotly
BUBBLE_SHADE = "rgba(231, 76, 60, 0.15)"
NORMAL_SHADE = "rgba(46, 204, 113, 0.08)"


# ============================================================
# Helper
# ============================================================

def _add_bubble_shading(fig, bubble_periods: list, row: int = 1, col: int = 1):
    """Add translucent red bands for known / detected bubble periods."""
    for start, end, *label in bubble_periods:
        fig.add_vrect(
            x0=str(start)[:10], x1=str(end)[:10],
            fillcolor=BUBBLE_SHADE, line_width=0,
            annotation_text=label[0] if label else "Bubble",
            annotation_position="top left",
            annotation_font_size=10,
            row=row, col=col,
        )


# ============================================================
# 1.  House Price Index with bubble shading
# ============================================================

def plot_price_index(
    price_series: pd.Series,
    bubble_periods: list = None,
    title: str = "U.S. National Home Price Index (Case-Shiller)",
    city_label: str = "National",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index,
        y=price_series.values,
        mode="lines",
        name=city_label,
        line=dict(color="#2c3e50", width=2),
    ))
    if bubble_periods:
        _add_bubble_shading(fig, bubble_periods)
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Index Value",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ============================================================
# 2.  Multi-city comparison
# ============================================================

def plot_city_comparison(
    city_series: dict[str, pd.Series],
    bubble_periods: list = None,
    normalise: bool = True,
) -> go.Figure:
    fig = go.Figure()
    for i, (city, series) in enumerate(city_series.items()):
        if series.empty:
            continue
        y = series / series.iloc[0] * 100 if normalise else series
        fig.add_trace(go.Scatter(
            x=series.index, y=y.values,
            mode="lines", name=city,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
        ))
    if bubble_periods:
        _add_bubble_shading(fig, bubble_periods)
    norm_label = " (rebased to 100)" if normalise else ""
    fig.update_layout(
        title=f"City-Level Home Price Comparison{norm_label}",
        xaxis_title="Date",
        yaxis_title="Index" + norm_label,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ============================================================
# 3.  GSADF – BSADF sequence with critical value
# ============================================================

def plot_bsadf(
    bsadf: pd.Series,
    critical_values: dict,
    bubble_periods: list = None,
    significance: str = "95%",
) -> go.Figure:
    cv = critical_values.get(significance, 1.42)
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Scatter(
        x=bsadf.index, y=bsadf.values,
        mode="lines", name="BSADF Statistic",
        line=dict(color="#2980b9", width=1.5),
    ))
    fig.add_hline(
        y=cv, line_dash="dash", line_color="#e74c3c",
        annotation_text=f"CV ({significance}) = {cv:.2f}",
        annotation_position="bottom right",
    )
    if bubble_periods:
        _add_bubble_shading(fig, bubble_periods)

    fig.update_layout(
        title="GSADF Test — Backward Supremum ADF Sequence",
        xaxis_title="Date",
        yaxis_title="BSADF Statistic",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ============================================================
# 4.  Markov regime probabilities
# ============================================================

def plot_markov_regimes(
    regime_probs: pd.DataFrame,
    bubble_regime: int,
    price_series: pd.Series = None,
) -> go.Figure:
    n_regimes = regime_probs.shape[1]
    rows = 2 if price_series is not None else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        subplot_titles=["Regime Probabilities",
                                        "Home Price Index"] if rows == 2 else ["Regime Probabilities"])

    colors = ["#27ae60", "#e74c3c", "#3498db", "#9b59b6"]
    for i in range(n_regimes):
        col_name = f"regime_{i}"
        label = f"Regime {i} ({'Bubble' if i == bubble_regime else 'Normal'})"
        fig.add_trace(go.Scatter(
            x=regime_probs.index,
            y=regime_probs[col_name].values,
            mode="lines", name=label,
            fill="tozeroy",
            line=dict(color=colors[i % len(colors)], width=1.2),
        ), row=1, col=1)

    if price_series is not None and rows == 2:
        fig.add_trace(go.Scatter(
            x=price_series.index, y=price_series.values,
            mode="lines", name="Price Index",
            line=dict(color="#2c3e50", width=1.5),
        ), row=2, col=1)

    fig.update_layout(
        title="Markov Regime-Switching — Bubble vs Normal Probabilities",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ============================================================
# 5.  Composite Risk Score gauge + time series
# ============================================================

def plot_risk_gauge(score: float, level: str) -> go.Figure:
    color = RISK_COLORS.get(level, "#95a5a6")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(score, 1),
        title={"text": f"Housing Bubble Risk Score<br><span style='font-size:1em;color:{color}'>{level}</span>"},
        delta={"reference": 50, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 35],  "color": "#d5f5e3"},
                {"range": [35, 55], "color": "#fef9e7"},
                {"range": [55, 75], "color": "#fdebd0"},
                {"range": [75, 100],"color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    return fig


def plot_risk_timeseries(
    score: pd.Series,
    bubble_periods: list = None,
) -> go.Figure:
    fig = go.Figure()
    # Colour the line by risk level
    for level, (lo, hi) in RISK_LEVELS.items():
        mask = (score >= lo) & (score <= hi)
        seg = score[mask]
        if not seg.empty:
            fig.add_trace(go.Scatter(
                x=seg.index, y=seg.values,
                mode="markers", marker=dict(color=RISK_COLORS[level], size=4),
                name=level, legendgroup=level,
                showlegend=True,
            ))

    fig.add_trace(go.Scatter(
        x=score.index, y=score.values,
        mode="lines", line=dict(color="#7f8c8d", width=1.5),
        name="Risk Score", showlegend=False,
    ))
    for th_val, th_label in [(35, "Elevated"), (55, "High"), (75, "Bubble")]:
        fig.add_hline(y=th_val, line_dash="dot", line_color=RISK_COLORS[th_label],
                      annotation_text=th_label, annotation_position="right")
    if bubble_periods:
        _add_bubble_shading(fig, bubble_periods)
    fig.update_layout(
        title="Housing Bubble Risk Score Over Time",
        xaxis_title="Date",
        yaxis_title="Risk Score (0–100)",
        template="plotly_white",
        yaxis_range=[0, 105],
        hovermode="x unified",
    )
    return fig


# ============================================================
# 6.  Feature importance
# ============================================================

def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    title: str = "XGBoost Feature Importance",
) -> go.Figure:
    top = importance.nlargest(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index,
        orientation="h",
        marker_color="#3498db",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(300, top_n * 22),
    )
    return fig


# ============================================================
# 7.  LSTM Autoencoder reconstruction error
# ============================================================

def plot_ae_error(
    reconstruction_error: pd.Series,
    threshold: float,
    bubble_periods: list = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=reconstruction_error.index,
        y=reconstruction_error.values,
        mode="lines", name="Reconstruction Error",
        line=dict(color="#8e44ad", width=1.5),
    ))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#e74c3c",
        annotation_text=f"Anomaly threshold = {threshold:.4f}",
    )
    if bubble_periods:
        _add_bubble_shading(fig, bubble_periods)
    fig.update_layout(
        title="LSTM Autoencoder — Reconstruction Error (Anomaly Detection)",
        xaxis_title="Date",
        yaxis_title="Mean Squared Error",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ============================================================
# 8.  ML model comparison bar chart
# ============================================================

def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    metrics = ["Accuracy", "AUC-ROC"]
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        if metric not in comparison_df.columns:
            continue
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df["Model"],
            y=comparison_df[metric],
            marker_color=PALETTE[i],
        ))
    fig.update_layout(
        title="Model Performance Comparison",
        barmode="group",
        yaxis_title="Score",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        xaxis_title="Model",
    )
    return fig


# ============================================================
# 9.  LPPLS overlay on price chart
# ============================================================

def plot_lppls_fit(
    price_series: pd.Series,
    lppls_result,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values,
        mode="lines", name="Actual Price",
        line=dict(color="#2c3e50", width=2),
    ))

    if lppls_result is not None and lppls_result.converged:
        n = len(price_series.dropna())
        t = np.arange(n, dtype=float)
        tc = lppls_result.tc
        dt = tc - t
        dt = np.where(dt > 0, dt, np.nan)
        log_pred = (
            lppls_result.A
            + lppls_result.B * dt ** lppls_result.m
            * (1 + lppls_result.C * np.cos(lppls_result.omega * np.log(np.maximum(dt, 1e-9)) + lppls_result.phi))
        )
        pred = np.exp(log_pred)
        idx = price_series.dropna().index
        pred_series = pd.Series(pred, index=idx)
        fig.add_trace(go.Scatter(
            x=pred_series.index, y=pred_series.values,
            mode="lines", name="LPPLS Fit",
            line=dict(color="#e74c3c", width=1.5, dash="dash"),
        ))
        if lppls_result.tc_date:
            fig.add_vline(
                x=str(lppls_result.tc_date)[:10],
                line_dash="dot", line_color="#e74c3c",
                annotation_text=f"Predicted critical time: {str(lppls_result.tc_date)[:7]}",
            )

    fig.update_layout(
        title="LPPLS Model Fit to House Prices",
        xaxis_title="Date",
        yaxis_title="Price Index",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


# ============================================================
# 10. Correlation heatmap of features
# ============================================================

def plot_correlation_heatmap(feature_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    numeric = feature_df.select_dtypes(include=[np.number])
    if "bubble_label" in numeric.columns:
        cols = (
            numeric.corr()["bubble_label"]
            .abs()
            .nlargest(top_n + 1)
            .index.tolist()
        )
        numeric = numeric[cols]

    corr = numeric.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap",
        template="plotly_white",
        height=600,
        width=700,
    )
    return fig
