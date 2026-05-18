import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

START_DATE = "1990-01-01"
END_DATE = "2025-12-31"

FRED_SERIES = {
    "CSUSHPINSA":  "case_shiller_national",
    "NYXRSA":      "case_shiller_nyc",
    "LXXRSA":      "case_shiller_la",
    "PHXRSA":      "case_shiller_phoenix",
    "MIAXRSA":     "case_shiller_miami",
    "MORTGAGE30US":"mortgage_rate_30y",
    "UNRATE":      "unemployment_rate",
    "CPIAUCSL":    "cpi",
    "GDPC1":       "real_gdp",
    "HOUST":       "housing_starts",
    "PERMIT":      "building_permits",
    "UMCSENT":     "consumer_sentiment",
    "FEDFUNDS":    "fed_funds_rate",
    "DRCCLACBS":   "credit_delinquency_rate",
    "BOGMBASE":    "monetary_base",
    "MEHOINUSA672N": "median_household_income",
}

ZILLOW_ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)
ZILLOW_ZORI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zori/"
    "Metro_zori_uc_sfrcondomfr_sm_month.csv"
)

CITIES = {
    "National": "case_shiller_national",
    "New York": "case_shiller_nyc",
    "Los Angeles": "case_shiller_la",
    "Phoenix": "case_shiller_phoenix",
    "Miami": "case_shiller_miami",
}

GSADF_MIN_WINDOW = 0.20
GSADF_LAGS = 1
GSADF_CV_90 = 1.07
GSADF_CV_95 = 1.42
GSADF_CV_99 = 2.10

RISK_WEIGHTS = {
    "gsadf":       0.25,
    "lppls":       0.15,
    "markov":      0.15,
    "xgboost":     0.20,
    "lstm_ae":     0.15,
    "lstm_pred":   0.10,
}

BUBBLE_EPISODES = [
    ("2002-01-01", "2006-06-01", "Pre-2008 Bubble"),
    ("2020-04-01", "2022-06-01", "COVID Boom"),
]
