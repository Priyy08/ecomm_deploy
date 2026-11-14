"""Configuration and settings for the Enterprise Return Prediction API."""

import os
from pathlib import Path
from typing import Dict

# Base paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# Candidate directories for model artifacts
CANDIDATE_MODEL_DIRS = [
    os.getenv("MODELS_DIR"),
    str(PROJECT_ROOT / "artifacts" / "models"),
    str(PROJECT_ROOT / "models"),
    str(CURRENT_DIR / "artifacts" / "models"),
    str(CURRENT_DIR.parent / "artifacts" / "models"),
    "artifacts/models",
    "models",
]


def resolve_models_dir() -> Path:
    """Find and return the valid directory containing model artifacts."""
    for candidate in CANDIDATE_MODEL_DIRS:
        if candidate and os.path.isdir(candidate):
            preprocessor_path = Path(candidate) / "enterprise_preprocessor.joblib"
            if preprocessor_path.exists():
                return Path(candidate).resolve()
    # Default fallback
    return (PROJECT_ROOT / "artifacts" / "models").resolve()


MODELS_DIR = resolve_models_dir()

# Model filenames
PREPROCESSOR_FILE = "enterprise_preprocessor.joblib"
MODEL_FILES = {
    "lightgbm": "model_lightgbm.joblib",
    "xgboost": "model_xgboost.joblib",
    "random_forest": "model_random_forest.joblib",
    "logistic_regression": "model_logistic_regression.joblib",
}

# Soft Voting Ensemble weights (from validation benchmark)
ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "lightgbm": 0.35,
    "xgboost": 0.35,
    "random_forest": 0.15,
    "logistic_regression": 0.15,
}

# Optimal decision thresholds programmatically tuned on validation splits
OPTIMAL_THRESHOLDS: Dict[str, float] = {
    "ensemble": 0.54,
    "lightgbm": 0.57,
    "xgboost": 0.52,
    "random_forest": 0.57,
    "logistic_regression": 0.43,
}

# 3-Zone Operational Risk Triage Boundaries
ZONE_SAFE_UPPER = 0.35
ZONE_MODERATE_UPPER = 0.65

# Feature definitions required by ColumnTransformer
NUMERIC_FEATURES = [
    "quantity",
    "unit_price",
    "order_total",
    "shipping_cost",
    "is_bracketed_order",
    "size_chart_dwell_time_sec",
    "size_guide_clicks",
    "customer_true_return_rate",
    "sku_fit_feedback_offset",
    "delivery_delay_days",
    "first_time_brand_purchase",
    "fit_mismatch",
    "delay_remorse",
    "cod_first_brand",
]

CATEGORICAL_FEATURES = ["product_category", "payment_method"]

ALL_REQUIRED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# API Metadata
API_TITLE = "Enterprise Telemetry Return Prediction API"
API_DESCRIPTION = (
    "Production-grade FastAPI service serving high-throughput enterprise behavioral "
    "telemetry return propensity models (Soft Voting Ensemble, LightGBM, XGBoost, "
    "Random Forest, and Logistic Regression) with automated feature engineering "
    "and 3-Zone Operational Risk Triage."
)
API_VERSION = "1.0.0"
