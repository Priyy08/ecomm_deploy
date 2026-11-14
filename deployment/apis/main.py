"""FastAPI application entrypoint for Enterprise Telemetry Return Prediction."""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deployment.apis.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    ENSEMBLE_WEIGHTS,
    OPTIMAL_THRESHOLDS,
    ALL_REQUIRED_FEATURES,
    MODEL_FILES,
)
from deployment.apis.model_loader import model_manager
from deployment.apis.schemas import (
    SingleOrderRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    RegistryResponse,
    ModelMetadata,
    HealthResponse,
)
from deployment.apis.service import prediction_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deployment.apis.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads models on startup and executes warm-up."""
    logger.info("Initializing Enterprise Return Prediction API...")
    try:
        model_manager.load_artifacts()
        logger.info("All model artifacts and ColumnTransformer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model artifacts on startup: {e}")
        # Allow service to start in degraded mode if needed
    yield
    logger.info("Shutting down Enterprise Return Prediction API.")


# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for cross-origin backend and frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    tags=["General"],
    summary="API Root & Status",
    response_description="System welcome and overview of available endpoints.",
)
async def root() -> Dict[str, Any]:
    """Root endpoint providing service metadata, health, and navigation links."""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "online",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
        },
        "endpoints": {
            "health": "GET /health",
            "models_registry": "GET /models",
            "single_prediction": "POST /predict",
            "batch_prediction": "POST /predict/batch",
            "telemetry_metrics": "GET /metrics",
        },
        "default_champion_model": "Soft Voting Ensemble (LightGBM 35% + XGBoost 35% + RF 15% + LR 15%)",
    }


@app.get(
    "/health",
    tags=["Diagnostics"],
    response_model=HealthResponse,
    summary="Service Health Check",
    response_description="Detailed diagnostic health status and loaded model availability.",
)
async def health_check() -> HealthResponse:
    """Check service health and memory loading status of all model artifacts."""
    health_data = model_manager.get_health()
    return HealthResponse(**health_data)


@app.get(
    "/models",
    tags=["Registry"],
    response_model=RegistryResponse,
    summary="Model Registry & Threshold Metadata",
    response_description="List of available architectures, optimal thresholds, and ensemble weights.",
)
async def list_models() -> RegistryResponse:
    """Return catalog of available model architectures and their calibrated cutoffs."""
    models_metadata = [
        ModelMetadata(
            model_key="ensemble",
            display_name="Soft Voting Ensemble",
            optimal_threshold=OPTIMAL_THRESHOLDS["ensemble"],
            ensemble_weight=1.0,
            is_loaded=all(model_manager.loaded_status[k] for k in ENSEMBLE_WEIGHTS.keys()),
        )
    ]

    display_names = {
        "lightgbm": "LightGBM Classifier",
        "xgboost": "XGBoost Classifier",
        "random_forest": "Random Forest Classifier",
        "logistic_regression": "Logistic Regression (L2)",
    }

    for key in MODEL_FILES.keys():
        models_metadata.append(
            ModelMetadata(
                model_key=key,
                display_name=display_names.get(key, key),
                optimal_threshold=OPTIMAL_THRESHOLDS.get(key, 0.50),
                ensemble_weight=ENSEMBLE_WEIGHTS.get(key),
                is_loaded=model_manager.loaded_status.get(key, False),
            )
        )

    return RegistryResponse(
        available_models=models_metadata,
        ensemble_weights=ENSEMBLE_WEIGHTS,
        optimal_thresholds=OPTIMAL_THRESHOLDS,
        required_features=ALL_REQUIRED_FEATURES,
    )


@app.post(
    "/predict",
    tags=["Inference"],
    response_model=PredictionResponse,
    summary="Predict Return Propensity for Single Order",
    response_description="Predicted return probability, threshold cutoff, binary decision, and risk triage action.",
)
async def predict_single_order(request: SingleOrderRequest) -> PredictionResponse:
    """Predict customer return probability for an individual order transaction.

    - **Soft Voting Ensemble (Default):** Blends predictions from LightGBM, XGBoost, Random Forest, and Logistic Regression.
    - **Single Model:** You can specify `lightgbm`, `xgboost`, `random_forest`, or `logistic_regression`.
    - **Operational Triage:** Categorizes orders into Zone 1 (Safe Auto-Approve), Zone 2 (Logistics Fast-Track), or Zone 3 (Wardrobing Guardrail).
    """
    try:
        return prediction_service.predict_single(request)
    except FileNotFoundError as e:
        logger.error(f"Model artifact error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifacts are unavailable: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Inference error during single prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/predict/batch",
    tags=["Inference"],
    response_model=BatchPredictionResponse,
    summary="Batch Return Propensity Predictions",
    response_description="Vectorized predictions for multiple orders with aggregated risk distribution.",
)
async def predict_batch_orders(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Execute high-throughput vectorized predictions on a batch of up to 5,000 orders."""
    try:
        return prediction_service.predict_batch(request)
    except Exception as e:
        logger.error(f"Inference error during batch prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get(
    "/metrics",
    tags=["Diagnostics"],
    summary="Service Operational Telemetry",
    response_description="Runtime statistics, requests served, and memory state.",
)
async def get_metrics() -> Dict[str, Any]:
    """Retrieve operational telemetry metrics for monitoring."""
    health = model_manager.get_health()
    return {
        "status": health["status"],
        "uptime_seconds": health["uptime_seconds"],
        "total_inferences_served": health["total_inferences_served"],
        "models_active_count": sum(1 for v in health["models_loaded"].values() if v),
        "preprocessor_active": health["preprocessor_loaded"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("deployment.apis.main:app", host="0.0.0.0", port=8000, reload=True)
