"""
Feature engineering for housing bubble detection.
Derives economic ratios and technical indicators from raw data.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.signal import lfilter

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hodrick-Prescott filter (manual, avoids statsmodels import latency)
# ---------------------------------------------------------------------------

def hp_filter(series: pd.Series, lamb: float = 1600) -> tuple[pd.Series, pd.Series]:
    """
    Return (trend, cycle) components via the HP filter.
    lamb=1600 is standard for quarterly data; use 129600 for monthly.
    """
    y = series.dropna().values
    n = len(y)
    if n < 5:
        return series, series * 0

    # Build second-difference matrix
    I = np.eye(n)
    D = np.diff(I, n=2, axis=0)
    trend = np.linalg.solve(I + lamb * D.T @ D, y)
    cycle = y - trend

    idx = series.dropna().index
    return pd.Series(trend, index=idx, name=f"{series.name}_trend"), \
           pd.Series(cycle, index=idx, name=f"{series.name}_cycle")


# ---------------------------------------------------------------------------
# Core feature functions
# ---------------------------------------------------------------------------

def price_to_rent_ratio(price: pd.Series, rent: pd.Series) -> pd.Series:
    """
    Price-to-Rent ratio: monthly home price / monthly rent.
    Historically the single strongest bubble indicator.
    A ratio > 1.3× its long-run mean signals potential overvaluation.
    """
    aligned_price, aligned_rent = price.align(rent, join="inner")
    ratio = aligned_price / aligned_rent
    ratio.name = "price_to_rent"
    return ratio


def price_to_income_ratio(price: pd.Series, income: pd.Series) -> pd.Series:
    """
    Price-to-Income ratio: home price index / median household income.
    Income is typically annual so we interpolate to monthly.
    """
    income_monthly = income.resample("MS").interpolate(method="time")
    p, i = price.align(income_monthly, join="inner")
    ratio = p / i
    ratio.name = "price_to_income"
    return ratio


def yoy_acceleration(price: pd.Series) -> pd.Series:
    """
    Year-over-year price acceleration: second derivative of YoY growth.
    Captures how fast prices are rising relative to a year ago.
    """
    yoy = price.pct_change(12) * 100
    accel = yoy.diff(3)
    accel.name = "yoy_acceleration"
    return accel


def mortgage_spread(mortgage_rate: pd.Series, fed_funds: pd.Series) -> pd.Series:
    """
    Mortgage rate minus Fed Funds rate.
    A shrinking spread signals loose credit conditions.
    """
    m, f = mortgage_rate.align(fed_funds, join="inner")
    spread = m - f
    spread.name = "mortgage_spread"
    return spread


def real_mortgage_rate(mortgage_rate: pd.Series, cpi: pd.Series) -> pd.Series:
    """Real mortgage rate = nominal rate - CPI inflation."""
    inflation = cpi.pct_change(12) * 100
    m, inf = mortgage_rate.align(inflation, join="inner")
    real = m - inf
    real.name = "real_mortgage_rate"
    return real


def credit_growth_rate(monetary_base: pd.Series, window: int = 12) -> pd.Series:
    """
    Rolling YoY growth in monetary base as a proxy for credit expansion.
    """
    growth = monetary_base.pct_change(window) * 100
    growth.name = "credit_growth"
    return growth


def hp_deviation(price: pd.Series, lamb: float = 129600) -> pd.Series:
    """
    Deviation of house prices from their HP-filtered long-run trend.
    Expressed as a percentage above/below trend.
    """
    trend, cycle = hp_filter(price, lamb=lamb)
    dev = (cycle / trend) * 100
    dev.name = "hp_deviation"
    return dev


def affordability_index(price: pd.Series, mortgage_rate: pd.Series,
                         income: pd.Series) -> pd.Series:
    """
    Simple affordability index: (income / 12) / monthly_payment(price, rate).
    Above 1.0 = affordable; below 1.0 = unaffordable.
    """
    income_monthly = income.resample("MS").interpolate(method="time")
    p, r, i = price.align(mortgage_rate, join="inner")
    p, i = p.align(income_monthly, join="inner")
    r, _ = r.align(i, join="inner")

    monthly_rate = r / 100 / 12
    n_payments = 360
    payment = (p * monthly_rate) / (1 - (1 + monthly_rate) ** (-n_payments))
    monthly_income = i / 12
    index = monthly_income / payment
    index.name = "affordability_index"
    return index


def supply_demand_ratio(building_permits: pd.Series,
                         housing_starts: pd.Series) -> pd.Series:
    """
    Permits / Starts ratio. Values > 1 indicate permits outpacing starts
    (speculative pipeline building).
    """
    p, s = building_permits.align(housing_starts, join="inner")
    ratio = p / s.replace(0, np.nan)
    ratio.name = "supply_demand_ratio"
    return ratio


def sentiment_momentum(consumer_sentiment: pd.Series, window: int = 6) -> pd.Series:
    """
    Rate of change in consumer sentiment over a 6-month rolling window.
    Surging sentiment often precedes bubble peaks.
    """
    mom = consumer_sentiment.pct_change(window) * 100
    mom.name = "sentiment_momentum"
    return mom


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling z-score: how many standard deviations above the rolling mean.
    Normalises each feature for ML models.
    """
    mean = series.rolling(window, min_periods=12).mean()
    std  = series.rolling(window, min_periods=12).std()
    z = (series - mean) / std.replace(0, np.nan)
    z.name = f"{series.name}_zscore"
    return z


# ---------------------------------------------------------------------------
# Master feature builder
# ---------------------------------------------------------------------------

def build_features(master: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts the merged monthly master DataFrame and returns an enriched
    DataFrame with all engineered features.
    """
    logger.info("Engineering features …")
    feat = master.copy()

    price = feat.get("case_shiller_national", pd.Series(dtype=float))
    rent_proxy = feat.get("zori_nyc", pd.Series(dtype=float))

    # If no rent data, derive a proxy from CPI shelter component (CPI scaled)
    if rent_proxy.empty:
        cpi = feat.get("cpi", pd.Series(dtype=float))
        rent_proxy = cpi * 3.0  # rough calibration

    mortgage = feat.get("mortgage_rate_30y", pd.Series(dtype=float))
    fed_funds = feat.get("fed_funds_rate", pd.Series(dtype=float))
    cpi       = feat.get("cpi", pd.Series(dtype=float))
    income    = feat.get("median_household_income", pd.Series(dtype=float))
    permits   = feat.get("building_permits", pd.Series(dtype=float))
    starts    = feat.get("housing_starts", pd.Series(dtype=float))
    sentiment = feat.get("consumer_sentiment", pd.Series(dtype=float))
    monetary  = feat.get("monetary_base", pd.Series(dtype=float))

    computed = {}

    if not price.empty:
        computed["yoy_price_growth"] = price.pct_change(12) * 100
        computed["yoy_acceleration"] = yoy_acceleration(price)
        computed["hp_deviation"]     = hp_deviation(price)
        computed["price_zscore"]     = rolling_zscore(price)

    if not price.empty and not rent_proxy.empty:
        ptr = price_to_rent_ratio(price, rent_proxy)
        computed["price_to_rent"]        = ptr
        computed["price_to_rent_zscore"] = rolling_zscore(ptr)

    if not price.empty and not income.empty:
        pti = price_to_income_ratio(price, income)
        computed["price_to_income"]        = pti
        computed["price_to_income_zscore"] = rolling_zscore(pti)

    if not mortgage.empty and not fed_funds.empty:
        spread = mortgage_spread(mortgage, fed_funds)
        computed["mortgage_spread"] = spread

    if not mortgage.empty and not cpi.empty:
        computed["real_mortgage_rate"] = real_mortgage_rate(mortgage, cpi)

    if not mortgage.empty and not income.empty and not price.empty:
        try:
            computed["affordability_index"] = affordability_index(price, mortgage, income)
        except Exception:
            pass

    if not monetary.empty:
        computed["credit_growth"] = credit_growth_rate(monetary)

    if not permits.empty and not starts.empty:
        computed["supply_demand_ratio"] = supply_demand_ratio(permits, starts)

    if not sentiment.empty:
        computed["sentiment_momentum"] = sentiment_momentum(sentiment)

    for col in ["unemployment_rate", "real_gdp"]:
        s = feat.get(col, pd.Series(dtype=float))
        if not s.empty:
            computed[f"{col}_yoy"] = s.pct_change(12) * 100

    for name, series in computed.items():
        feat[name] = series

    # Add lagged features for ML (lags at 1, 3, 6 months)
    core_features = [
        "yoy_price_growth", "yoy_acceleration", "hp_deviation",
        "price_to_rent", "mortgage_spread", "credit_growth",
        "sentiment_momentum", "real_mortgage_rate",
    ]
    for col in core_features:
        if col in feat.columns:
            for lag in (1, 3, 6):
                feat[f"{col}_lag{lag}"] = feat[col].shift(lag)

    feat.index.name = "date"
    logger.info("Features built: %d columns", feat.shape[1])
    return feat


def get_ml_feature_matrix(feat: pd.DataFrame, label_col: str = "bubble_label") -> tuple:
    """
    Extract X (feature matrix) and y (labels) for ML training.
    Returns (X, y, feature_names) after dropping rows with NaN.
    """
    exclude = {
        "case_shiller_national", "case_shiller_nyc", "case_shiller_la",
        "case_shiller_phoenix", "case_shiller_miami",
        "mortgage_rate_30y", "unemployment_rate", "cpi", "real_gdp",
        "housing_starts", "building_permits", "consumer_sentiment",
        "fed_funds_rate", "credit_delinquency_rate", "monetary_base",
        "median_household_income", "zhvi_national", "zori_nyc",
        label_col,
    }
    feature_cols = [c for c in feat.columns if c not in exclude]
    sub = feat[feature_cols + ([label_col] if label_col in feat.columns else [])].dropna()
    if label_col in sub.columns:
        X = sub[feature_cols]
        y = sub[label_col]
        return X, y, feature_cols
    return sub[feature_cols], None, feature_cols
