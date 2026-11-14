"""Model artifact loader and memory manager."""

import os
import time
import logging
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd

from deployment.apis.config import (
    MODELS_DIR,
    PREPROCESSOR_FILE,
    MODEL_FILES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_REQUIRED_FEATURES,
)

logger = logging.getLogger("deployment.apis.model_loader")


class ModelManager:
    """Singleton manager for holding trained models and preprocessor in memory."""

    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.models_dir = MODELS_DIR
        self.preprocessor = None
        self.models: Dict[str, Any] = {}
        self.loaded_status: Dict[str, bool] = {
            "preprocessor": False,
            "lightgbm": False,
            "xgboost": False,
            "random_forest": False,
            "logistic_regression": False,
        }
        self.startup_time = time.time()
        self.total_inferences = 0
        self._initialized = True

    def load_artifacts(self) -> None:
        """Load preprocessor and all model artifacts from disk into RAM."""
        logger.info(f"Loading model artifacts from: {self.models_dir}")

        # 1. Load Preprocessor
        preprocessor_path = os.path.join(self.models_dir, PREPROCESSOR_FILE)
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor file not found at {preprocessor_path}"
            )
        self.preprocessor = joblib.load(preprocessor_path)
        self.loaded_status["preprocessor"] = True
        logger.info("Fitted ColumnTransformer preprocessor loaded successfully.")

        # 2. Load Models
        for key, filename in MODEL_FILES.items():
            model_path = os.path.join(self.models_dir, filename)
            if not os.path.exists(model_path):
                logger.warning(f"Model file for '{key}' not found at {model_path}")
                self.loaded_status[key] = False
                continue

            try:
                model = joblib.load(model_path)
                self.models[key] = model
                self.loaded_status[key] = True
                logger.info(f"Model '{key}' loaded successfully from {filename}.")
            except Exception as e:
                logger.error(f"Failed to load model '{key}' from {model_path}: {e}")
                self.loaded_status[key] = False

        # 3. Warm-up inference
        self.warmup()

    def warmup(self) -> None:
        """Run dummy inference through preprocessor and models to pre-initialize memory buffers."""
        try:
            dummy_data = {
                "quantity": [1.0],
                "unit_price": [50.0],
                "order_total": [50.0],
                "shipping_cost": [0.0],
                "is_bracketed_order": [0.0],
                "size_chart_dwell_time_sec": [10.0],
                "size_guide_clicks": [0.0],
                "customer_true_return_rate": [0.15],
                "sku_fit_feedback_offset": [0.0],
                "delivery_delay_days": [0.0],
                "first_time_brand_purchase": [0.0],
                "fit_mismatch": [0.0],
                "delay_remorse": [0.0],
                "cod_first_brand": [0.0],
                "product_category": ["clothing"],
                "payment_method": ["credit_card"],
            }
            df_dummy = pd.DataFrame(dummy_data)[ALL_REQUIRED_FEATURES]
            X_dummy = self.preprocessor.transform(df_dummy)

            for key, model in self.models.items():
                _ = model.predict_proba(X_dummy)
            logger.info("Warm-up inference completed successfully for all loaded models.")
        except Exception as e:
            logger.warning(f"Warm-up inference encountered minor warning: {e}")

    def get_model(self, key: str) -> Any:
        """Retrieve a loaded model by its identifier."""
        if key not in self.models:
            raise KeyError(
                f"Model '{key}' is not loaded. Available models: {list(self.models.keys())}"
            )
        return self.models[key]

    def get_health(self) -> Dict[str, Any]:
        """Return diagnostic health status."""
        return {
            "status": "healthy" if self.loaded_status["preprocessor"] and len(self.models) > 0 else "degraded",
            "models_loaded": {k: self.loaded_status.get(k, False) for k in MODEL_FILES.keys()},
            "preprocessor_loaded": self.loaded_status["preprocessor"],
            "models_dir": str(self.models_dir),
            "uptime_seconds": round(time.time() - self.startup_time, 2),
            "total_inferences_served": self.total_inferences,
        }


# Global accessor
model_manager = ModelManager()
