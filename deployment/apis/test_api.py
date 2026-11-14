"""Integration and validation test suite for Enterprise Return Prediction API."""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.apis.config import ALL_REQUIRED_FEATURES, OPTIMAL_THRESHOLDS, ENSEMBLE_WEIGHTS
from deployment.apis.schemas import (
    OrderFeatures,
    SingleOrderRequest,
    BatchPredictionRequest,
    ModelName,
    RiskZone,
)
from deployment.apis.service import prediction_service
from deployment.apis.model_loader import model_manager


def run_tests():
    print("=" * 65)
    print("ENTERPRISE TELEMETRY RETURN PREDICTION — API TEST SUITE")
    print("=" * 65)

    # 1. Test Model Loading & Health Check
    print("\n[TEST 1] Verifying Model Artifacts and Preprocessor Health...")
    health = model_manager.get_health()
    if not health["preprocessor_loaded"]:
        model_manager.load_artifacts()
        health = model_manager.get_health()

    assert health["status"] == "healthy", f"Status unhealthy: {health}"
    assert health["preprocessor_loaded"] is True, "ColumnTransformer not loaded!"
    for m, is_loaded in health["models_loaded"].items():
        assert is_loaded is True, f"Model '{m}' failed to load!"
        print(f"  • {m:<20}: Loaded = {is_loaded}")
    print("  --> [PASS] All 4 models and ColumnTransformer verified in memory.")

    # 2. Test High-Risk Order Prediction
    print("\n[TEST 2] Testing High-Risk Sizing Bracketing Order (Zone 3)...")
    high_risk_order = OrderFeatures(
        product_category="clothing",
        payment_method="credit_card",
        quantity=3,
        unit_price=59.99,
        order_total=179.97,
        shipping_cost=0.0,
        is_bracketed_order=1,
        size_chart_dwell_time_sec=75.0,
        size_guide_clicks=5,
        customer_true_return_rate=0.65,
        sku_fit_feedback_offset=0.7,
        delivery_delay_days=3.0,
        first_time_brand_purchase=1,
    )
    req_high = SingleOrderRequest(
        order=high_risk_order,
        model_name=ModelName.ENSEMBLE,
        explain_signals=True,
    )
    res_high = prediction_service.predict_single(req_high)

    print(f"  Model:               {res_high.model_used}")
    print(f"  Return Probability:  {res_high.return_probability:.4f}")
    print(f"  Decision Threshold:  {res_high.decision_threshold:.2f}")
    print(f"  Prediction Label:    {res_high.prediction_label}")
    print(f"  Risk Zone:           {res_high.risk_zone.value}")
    print(f"  Operational Action:  {res_high.operational_action[:60]}...")
    print(f"  Inference Latency:   {res_high.inference_latency_ms} ms")
    print(f"  Model Breakdown:     {res_high.model_contributions}")

    assert res_high.return_probability >= 0.50
    assert res_high.will_return is True
    assert res_high.risk_zone == RiskZone.ZONE_3_HIGH
    assert res_high.engineered_signals["fit_mismatch"] == 1.0
    assert res_high.engineered_signals["delay_remorse"] == 1.0
    print("  --> [PASS] High-risk order accurately classified into Zone 3.")

    # 3. Test Low-Risk Order Prediction
    print("\n[TEST 3] Testing Low-Risk Frictionless Order (Zone 1)...")
    low_risk_order = OrderFeatures(
        product_category="grocery",
        payment_method="credit_card",
        quantity=1,
        unit_price=14.50,
        order_total=14.50,
        shipping_cost=0.0,
        is_bracketed_order=0,
        size_chart_dwell_time_sec=0.0,
        size_guide_clicks=0,
        customer_true_return_rate=0.02,
        sku_fit_feedback_offset=0.0,
        delivery_delay_days=0.0,
        first_time_brand_purchase=0,
    )
    req_low = SingleOrderRequest(order=low_risk_order, model_name=ModelName.ENSEMBLE)
    res_low = prediction_service.predict_single(req_low)

    print(f"  Return Probability:  {res_low.return_probability:.4f}")
    print(f"  Prediction Label:    {res_low.prediction_label}")
    print(f"  Risk Zone:           {res_low.risk_zone.value}")
    print(f"  Operational Action:  {res_low.operational_action}")

    assert res_low.return_probability < 0.35
    assert res_low.will_return is False
    assert res_low.risk_zone == RiskZone.ZONE_1_SAFE
    print("  --> [PASS] Low-risk order accurately classified into Zone 1.")

    # 4. Test Individual Model Routing
    print("\n[TEST 4] Testing Routing Across All Individual Model Architectures...")
    for m_enum in [
        ModelName.LIGHTGBM,
        ModelName.XGBOOST,
        ModelName.RANDOM_FOREST,
        ModelName.LOGISTIC_REGRESSION,
    ]:
        m_req = SingleOrderRequest(order=high_risk_order, model_name=m_enum)
        m_res = prediction_service.predict_single(m_req)
        print(f"  • {m_res.model_used:<25}: Prob = {m_res.return_probability:.4f} | Cutoff = {m_res.decision_threshold:.2f} | Latency = {m_res.inference_latency_ms} ms")
        assert 0.0 <= m_res.return_probability <= 1.0
    print("  --> [PASS] All individual models routed and evaluated successfully.")

    # 5. Test Batch Prediction
    print("\n[TEST 5] Testing Vectorized Batch Processing (3 Orders)...")
    batch_req = BatchPredictionRequest(
        orders=[high_risk_order, low_risk_order, high_risk_order],
        model_name=ModelName.ENSEMBLE,
    )
    batch_res = prediction_service.predict_batch(batch_req)

    print(f"  Total Orders:            {batch_res.total_orders}")
    print(f"  Returned vs Delivered:   {batch_res.returned_count} vs {batch_res.delivered_count}")
    print(f"  Risk Zone Counts:        Zone1={batch_res.zone_1_safe_count}, Zone2={batch_res.zone_2_moderate_count}, Zone3={batch_res.zone_3_high_risk_count}")
    print(f"  Batch Processing Time:   {batch_res.batch_latency_ms} ms ({batch_res.per_sample_latency_ms} ms/order)")

    assert batch_res.total_orders == 3
    assert batch_res.returned_count == 2
    assert batch_res.delivered_count == 1
    assert batch_res.zone_3_high_risk_count == 2
    assert batch_res.zone_1_safe_count == 1
    print("  --> [PASS] Vectorized batch prediction validated.")

    print("\n" + "=" * 65)
    print(">>> ALL API SERVICE & INFERENCE TESTS COMPLETED AND PASSED! <<<")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
