"""Core predictive inference service for single and batch predictions."""

import time
from typing import Dict, List, Tuple
import numpy as np

from deployment.apis.config import (
    ENSEMBLE_WEIGHTS,
    OPTIMAL_THRESHOLDS,
    ZONE_SAFE_UPPER,
    ZONE_MODERATE_UPPER,
)
from deployment.apis.model_loader import model_manager
from deployment.apis.pipeline import transform_orders
from deployment.apis.schemas import (
    ModelName,
    RiskZone,
    SingleOrderRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    OrderFeatures,
)


def determine_risk_zone_and_action(probability: float) -> Tuple[RiskZone, str]:
    """Map predicted return probability to commercial 3-Zone Operational Risk Triage."""
    if probability < ZONE_SAFE_UPPER:
        return (
            RiskZone.ZONE_1_SAFE,
            "Instant frictionless fulfillment with zero intervention.",
        )
    elif probability < ZONE_MODERATE_UPPER:
        return (
            RiskZone.ZONE_2_MODERATE,
            "Route via express logistics to prevent SLA delivery delays; include pre-printed return label.",
        )
    else:
        return (
            RiskZone.ZONE_3_HIGH,
            "Real-time checkout prompt: 'Confirm your sizing measurements to save return hassle'; "
            "require nominal return fee on multi-size apparel.",
        )


class PredictionService:
    """Service handling feature transformation, model execution, and decision logic."""

    def __init__(self):
        self.manager = model_manager

    def _ensure_loaded(self):
        """Ensure models are loaded in case lifespan was bypassed."""
        if not self.manager.loaded_status["preprocessor"]:
            self.manager.load_artifacts()

    def predict_single(self, request: SingleOrderRequest) -> PredictionResponse:
        """Execute single order return prediction."""
        self._ensure_loaded()
        t0 = time.perf_counter()
        order = request.order
        model_name = request.model_name.value

        # 1. Feature Preprocessing
        X = transform_orders(order, self.manager.preprocessor)

        # 2. Model Inference
        model_contributions = {}
        if model_name == ModelName.ENSEMBLE.value:
            ens_prob = 0.0
            for m_key, weight in ENSEMBLE_WEIGHTS.items():
                m = self.manager.get_model(m_key)
                m_p = float(m.predict_proba(X)[0, 1])
                model_contributions[m_key] = round(m_p, 4)
                ens_prob += m_p * weight

            final_prob = float(np.clip(ens_prob, 0.0, 1.0))
            default_threshold = OPTIMAL_THRESHOLDS["ensemble"]
            display_model_name = "Soft Voting Ensemble (LightGBM + XGBoost + RF + LR)"
        else:
            m = self.manager.get_model(model_name)
            final_prob = float(m.predict_proba(X)[0, 1])
            default_threshold = OPTIMAL_THRESHOLDS[model_name]
            display_model_name = model_name.replace("_", " ").title()

        # 3. Decision Logic & Cutoff
        threshold = (
            request.custom_threshold
            if request.custom_threshold is not None
            else default_threshold
        )
        will_return = bool(final_prob >= threshold)
        prediction_label = "Returned (1)" if will_return else "Delivered / Kept (0)"

        # 4. Operational Risk Triage
        risk_zone, action = determine_risk_zone_and_action(final_prob)

        # 5. Latency profiling
        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 3)

        # 6. Engineered Signals explanation
        engineered_signals = None
        if request.explain_signals:
            engineered_signals = {
                "fit_mismatch": order.fit_mismatch,
                "delay_remorse": order.delay_remorse,
                "cod_first_brand": order.cod_first_brand,
                "is_bracketed_order": float(order.is_bracketed_order),
                "customer_true_return_rate": float(order.customer_true_return_rate),
            }

        self.manager.total_inferences += 1

        return PredictionResponse(
            model_used=display_model_name,
            return_probability=round(final_prob, 4),
            decision_threshold=round(threshold, 2),
            will_return=will_return,
            prediction_label=prediction_label,
            risk_zone=risk_zone,
            operational_action=action,
            inference_latency_ms=latency_ms,
            engineered_signals=engineered_signals,
            model_contributions=model_contributions if model_contributions else None,
        )

    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Execute vectorized high-throughput batch return prediction."""
        self._ensure_loaded()
        t0 = time.perf_counter()
        orders = request.orders
        n_orders = len(orders)
        model_name = request.model_name.value

        # 1. Batch Feature Preprocessing
        X_batch = transform_orders(orders, self.manager.preprocessor)

        # 2. Vectorized Model Inference
        model_contributions_list = [{} for _ in range(n_orders)]
        if model_name == ModelName.ENSEMBLE.value:
            probs = np.zeros(n_orders, dtype=float)
            for m_key, weight in ENSEMBLE_WEIGHTS.items():
                m = self.manager.get_model(m_key)
                m_probs = m.predict_proba(X_batch)[:, 1]
                probs += m_probs * weight
                for i, p in enumerate(m_probs):
                    model_contributions_list[i][m_key] = round(float(p), 4)

            probs = np.clip(probs, 0.0, 1.0)
            default_threshold = OPTIMAL_THRESHOLDS["ensemble"]
            display_model_name = "Soft Voting Ensemble (LightGBM + XGBoost + RF + LR)"
        else:
            m = self.manager.get_model(model_name)
            probs = m.predict_proba(X_batch)[:, 1]
            default_threshold = OPTIMAL_THRESHOLDS[model_name]
            display_model_name = model_name.replace("_", " ").title()

        threshold = (
            request.custom_threshold
            if request.custom_threshold is not None
            else default_threshold
        )

        # 3. Construct individual and aggregated results
        predictions: List[PredictionResponse] = []
        returned_count = 0
        zone_1_count = 0
        zone_2_count = 0
        zone_3_count = 0

        for i, order in enumerate(orders):
            p = float(probs[i])
            will_return = bool(p >= threshold)
            if will_return:
                returned_count += 1

            risk_zone, action = determine_risk_zone_and_action(p)
            if risk_zone == RiskZone.ZONE_1_SAFE:
                zone_1_count += 1
            elif risk_zone == RiskZone.ZONE_2_MODERATE:
                zone_2_count += 1
            else:
                zone_3_count += 1

            predictions.append(
                PredictionResponse(
                    model_used=display_model_name,
                    return_probability=round(p, 4),
                    decision_threshold=round(threshold, 2),
                    will_return=will_return,
                    prediction_label="Returned (1)" if will_return else "Delivered / Kept (0)",
                    risk_zone=risk_zone,
                    operational_action=action,
                    inference_latency_ms=0.0,  # Computed at batch level
                    engineered_signals={
                        "fit_mismatch": order.fit_mismatch,
                        "delay_remorse": order.delay_remorse,
                        "cod_first_brand": order.cod_first_brand,
                    },
                    model_contributions=(
                        model_contributions_list[i] if model_contributions_list[i] else None
                    ),
                )
            )

        t1 = time.perf_counter()
        total_latency_ms = round((t1 - t0) * 1000, 2)
        per_sample_ms = round(total_latency_ms / max(1, n_orders), 4)

        for pred in predictions:
            pred.inference_latency_ms = per_sample_ms

        self.manager.total_inferences += n_orders

        return BatchPredictionResponse(
            model_used=display_model_name,
            total_orders=n_orders,
            returned_count=returned_count,
            delivered_count=n_orders - returned_count,
            zone_1_safe_count=zone_1_count,
            zone_2_moderate_count=zone_2_count,
            zone_3_high_risk_count=zone_3_count,
            average_return_probability=round(float(np.mean(probs)), 4),
            batch_latency_ms=total_latency_ms,
            per_sample_latency_ms=per_sample_ms,
            predictions=predictions,
        )


# Global service instance
prediction_service = PredictionService()
