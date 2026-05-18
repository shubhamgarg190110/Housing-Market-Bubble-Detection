# Housing Market Bubble Detection
### Capstone Research Project — Shubham Garg (Grade 10)
**Using Machine Learning & Statistical Methods for Early Warning Signals**

---

## Research Question
> *Can machine learning models, combined with macroeconomic indicators, detect and predict housing market bubbles in the U.S. with sufficient lead time to serve as early warning signals?*

---

## Project Structure

```
shubham_proj/
├── config.py                    # All settings (API keys, series IDs, weights)
├── main.py                      # Full pipeline runner
├── requirements.txt
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard
├── src/
│   ├── data/
│   │   └── collector.py         # FRED + Zillow data download & caching
│   ├── features/
│   │   └── engineer.py          # Price-to-rent, HP filter, etc.
│   ├── models/
│   │   ├── statistical.py       # GSADF, LPPLS, Markov regime-switching
│   │   ├── ml_models.py         # XGBoost, Random Forest, Logistic Regression
│   │   └── deep_learning.py     # LSTM Autoencoder, LSTM Predictor, FNN
│   ├── scoring/
│   │   └── risk_score.py        # Composite 0–100 bubble risk score
│   └── visualization/
│       └── plots.py             # Plotly chart library
└── data/
    ├── raw/                     # Cached raw downloads
    └── processed/               # Features, labels, model outputs
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (downloads data, trains models)
```bash
python main.py
```

### 3. Launch the interactive dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard runs the entire pipeline on first load (≈ 3–5 minutes). Subsequent visits use cached results and load instantly.

---

## Methodology

### Phase 3 — Statistical Bubble Detection

#### GSADF Test (Generalized Supremum ADF)
The gold-standard econometric test (Phillips, Shi & Yu, 2015) for detecting explosive price behaviour.
- Tests whether price growth is inconsistent with fundamentals
- Produces a BSADF sequence that **date-stamps** bubble origination and collapse
- Critical values derived from Monte Carlo simulation

#### LPPLS Model (Log-Periodic Power Law Singularity)
Physics-inspired model used by ETH Zurich's Financial Crisis Observatory.

$$\ln p(t) = A + B(t_c - t)^{\beta}\left[1 + C\cos\left(\omega\ln(t_c - t) + \phi\right)\right]$$

- Detects super-exponential growth with log-periodic oscillations before crashes
- Estimates $t_c$ — the **critical time** (predicted crash window)
- $B < 0$ and proximity to $t_c$ → high bubble risk

#### Markov Regime-Switching Model
2-state Hidden Markov Model (Hamilton, 1994):
- **Normal regime**: fundamentals-driven price growth
- **Bubble regime**: speculation-driven, high-variance dynamics
- Outputs time-varying probability of being in each regime

---

### Phase 4 — Machine Learning Models

#### Layer 1: Classical ML
| Model | Purpose |
|-------|---------|
| **XGBoost** | Feature importance + bubble classification |
| **Random Forest** | Ensemble baseline |
| **Logistic Regression** | Interpretable linear baseline |

All trained with **TimeSeriesSplit** cross-validation (no look-ahead bias).

#### Layer 2: Deep Learning
| Model | Purpose |
|-------|---------|
| **LSTM Autoencoder** | Anomaly detection via reconstruction error |
| **LSTM Predictor** | Price forecasting; divergence = bubble signal |
| **Feedforward NN** | Binary bubble classification |

#### Layer 3: Composite Risk Score (0–100)
Weighted ensemble of all model outputs:

| Signal | Weight |
|--------|--------|
| GSADF | 25% |
| XGBoost | 20% |
| LSTM Autoencoder | 15% |
| Markov | 15% |
| LPPLS | 15% |
| LSTM Predictor | 10% |

**Risk Levels:** Normal (0–35) → Elevated (35–55) → High (55–75) → Bubble (75–100)

---

## Key Features Engineered

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| Price-to-Rent Ratio | Monthly price / rent | Strongest single bubble indicator |
| Price-to-Income Ratio | Price / household income | Affordability measure |
| HP-Filter Deviation | % above long-run trend | Overvaluation relative to fundamentals |
| YoY Price Acceleration | d²(price)/dt² | Captures reflexive price dynamics |
| Mortgage Rate Spread | Mortgage – Fed Funds | Loose credit conditions |
| Real Mortgage Rate | Nominal – inflation | True cost of borrowing |
| Credit Growth Rate | YoY monetary base growth | Lending expansion |
| Sentiment Momentum | Consumer confidence change | Speculative expectations |

---

## Data Sources (All Free & Public)

| Dataset | Source | Variables |
|---------|--------|-----------|
| Case-Shiller Index | FRED | National + city home prices |
| Mortgage Rates | FRED (Freddie Mac) | 30-year fixed rate |
| Unemployment | FRED (BLS) | Labor market health |
| CPI | FRED (BLS) | Inflation |
| Real GDP | FRED (BEA) | Economic growth |
| Housing Starts | FRED (Census) | New construction |
| Consumer Sentiment | FRED (U of M) | Expectations |
| Fed Funds Rate | FRED (Fed) | Monetary policy |
| ZHVI | Zillow Research | Metro-level home values |
| ZORI | Zillow Research | Metro-level rents |

---

## Dashboard Pages

1. **Overview** — Composite Risk Score gauge + historical trend
2. **Data Explorer** — Price indices, macro indicators, correlation heatmap
3. **Statistical Analysis** — GSADF BSADF sequence, LPPLS fit, Markov regimes
4. **ML Models** — Model comparison, XGBoost feature importance, LSTM AE errors, backtest
5. **City Comparison** — NYC vs national vs LA vs Phoenix vs Miami
6. **About** — Methodology, literature, timeline

---

## Publication Targets

- Polygence Research Symposium
- Lumiere Research Symposium
- arXiv preprint (`q-fin.GN`)
- Journal of Student Research
- Regeneron STS (if completed by Fall Grade 11)

---

## Key Literature

- **Case & Shiller (2003)** — "Is There a Bubble in the Housing Market?"
- **Phillips, Shi & Yu (2015)** — "Testing for Multiple Explosive Components" *(GSADF)*
- **Johansen & Sornette (1999)** — Log-Periodic Power Law model *(LPPLS)*
- **Hamilton (1994)** — "Time Series Analysis" *(Markov regime-switching)*
- **Christopher Mayer** — "Housing Bubbles: A Survey" (Columbia Business School)
- **IMF WP** — "Bubble Detective: City-Level Analysis of House Price Cycles"
- **Dallas Fed (2025)** — Consumer expectations and housing market exuberance
- **Biagini et al.** — Detecting asset price bubbles using deep learning (arXiv)

---