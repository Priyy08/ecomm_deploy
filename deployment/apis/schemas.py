"""Pydantic data models and schemas for the Return Prediction API."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ModelName(str, Enum):
    """Supported predictive model architectures."""

    ENSEMBLE = "ensemble"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"


class RiskZone(str, Enum):
    """3-Zone Operational Risk Triage categories."""

    ZONE_1_SAFE = "Zone 1: Safe Auto-Approve"
    ZONE_2_MODERATE = "Zone 2: Logistics Fast-Track & Nudge"
    ZONE_3_HIGH = "Zone 3: Wardrobing & Sizing Guardrail"


class OrderFeatures(BaseModel):
    """Input telemetry features for a single e-commerce transaction."""

    product_category: str = Field(
        ...,
        description="Merchandise category (e.g. clothing, electronics, beauty, books, grocery, home, sports, toys)",
        example="clothing",
    )
    payment_method: str = Field(
        ...,
        description="Checkout payment method (e.g. credit_card, debit_card, paypal, bnpl, gift_card, cash, cod)",
        example="credit_card",
    )
    quantity: int = Field(
        default=1,
        ge=1,
        le=500,
        description="Number of items purchased in this line item",
        example=2,
    )
    unit_price: float = Field(
        ...,
        gt=0.0,
        description="Individual item unit price in currency units ($)",
        example=49.99,
    )
    order_total: Optional[float] = Field(
        default=None,
        description="Total cart checkout amount ($). Automatically computed if omitted.",
        example=99.98,
    )
    shipping_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Shipping charge applied to order ($)",
        example=5.00,
    )
    is_bracketed_order: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if customer purchased multiple sizes/colors of the same SKU to try at home (cart bracketing)",
        example=1,
    )
    size_chart_dwell_time_sec: float = Field(
        default=0.0,
        ge=0.0,
        description="Active seconds user spent viewing the sizing chart modal before checkout",
        example=45.0,
    )
    size_guide_clicks: int = Field(
        default=0,
        ge=0,
        description="Interactive clicks on size chart filters, units, or fit recommendation tools",
        example=3,
    )
    customer_true_return_rate: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Customer's historic 12-month rolling return rate (0.0 to 1.0)",
        example=0.35,
    )
    sku_fit_feedback_offset: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Crowdsourced fit feedback offset: -1.0 (runs small), 0.0 (true to size), +1.0 (runs large)",
        example=0.5,
    )
    delivery_delay_days: float = Field(
        default=0.0,
        description="Delivery transit SLA delay in days (actual arrival minus SLA promise)",
        example=1.0,
    )
    first_time_brand_purchase: int = Field(
        default=0,
        ge=0,
        le=1,
        description="1 if this customer has never bought this brand before, else 0",
        example=1,
    )

    # Optional pre-engineered features (auto-derived if not provided)
    fit_mismatch: Optional[float] = Field(
        default=None,
        description="Compound sizing risk signal (auto-derived if omitted)",
    )
    delay_remorse: Optional[float] = Field(
        default=None,
        description="Compound logistics SLA delay signal (auto-derived if omitted)",
    )
    cod_first_brand: Optional[float] = Field(
        default=None,
        description="Compound high-risk COD first-time brand signal (auto-derived if omitted)",
    )

    @model_validator(mode="after")
    def populate_defaults_and_engineered_features(self):
        """Normalize strings and compute missing engineered features."""
        # Normalize categories to lower case
        self.product_category = self.product_category.strip().lower()
        self.payment_method = self.payment_method.strip().lower()

        # Compute order_total if missing
        if self.order_total is None:
            self.order_total = round(
                (self.quantity * self.unit_price) + self.shipping_cost, 2
            )

        # Derive compound features if omitted
        if self.fit_mismatch is None:
            self.fit_mismatch = 1.0 if (
                self.size_chart_dwell_time_sec > 35
                and abs(self.sku_fit_feedback_offset) >= 0.4
            ) else 0.0

        if self.delay_remorse is None:
            self.delay_remorse = 1.0 if max(0.0, self.delivery_delay_days) >= 2.0 else 0.0

        if self.cod_first_brand is None:
            is_cod = any(
                term in self.payment_method for term in ["cod", "cash"]
            )
            self.cod_first_brand = 1.0 if (is_cod and self.first_time_brand_purchase == 1) else 0.0

        return self


class SingleOrderRequest(BaseModel):
    """Payload for single order return prediction."""

    order: OrderFeatures
    model_name: ModelName = Field(
        default=ModelName.ENSEMBLE,
        description="Predictive model architecture to invoke (default: ensemble)",
    )
    custom_threshold: Optional[float] = Field(
        default=None,
        ge=0.05,
        le=0.95,
        description="Optional custom classification cutoff threshold to override the optimal tuned cutoff",
        example=0.54,
    )
    explain_signals: bool = Field(
        default=True,
        description="Whether to return engineered telemetry signals and individual model breakdown",
    )


class PredictionResponse(BaseModel):
    """Inference response for an individual transaction."""

    model_used: str = Field(..., description="Model architecture used for inference")
    return_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that this order will be returned",
        example=0.8842,
    )
    decision_threshold: float = Field(
        ...,
        description="Classification decision cutoff threshold used",
        example=0.54,
    )
    will_return: bool = Field(
        ...,
        description="Binary prediction: True (Returned) if probability >= threshold, else False",
        example=True,
    )
    prediction_label: str = Field(
        ...,
        description="Human-readable decision label ('Returned (1)' or 'Delivered / Kept (0)')",
        example="Returned (1)",
    )
    risk_zone: RiskZone = Field(
        ...,
        description="Commercial 3-Zone Operational Risk Triage level",
        example=RiskZone.ZONE_3_HIGH,
    )
    operational_action: str = Field(
        ...,
        description="Prescribed business action for retail fulfillment",
        example="Real-time checkout prompt: 'Confirm your sizing measurements to save return hassle'; require nominal return fee on multi-size apparel.",
    )
    inference_latency_ms: float = Field(
        ...,
        description="Model execution latency in milliseconds",
        example=1.25,
    )
    engineered_signals: Optional[Dict[str, float]] = Field(
        default=None,
        description="Key behavioral compound telemetry features computed for this transaction",
    )
    model_contributions: Optional[Dict[str, float]] = Field(
        default=None,
        description="When ensemble is queried, probabilities predicted by each constituent model",
    )


class BatchPredictionRequest(BaseModel):
    """Payload for batch order return prediction."""

    orders: List[OrderFeatures] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of order telemetry records to evaluate",
    )
    model_name: ModelName = Field(
        default=ModelName.ENSEMBLE,
        description="Predictive model architecture to invoke for all orders in batch",
    )
    custom_threshold: Optional[float] = Field(
        default=None,
        ge=0.05,
        le=0.95,
        description="Optional custom classification cutoff threshold to override the optimal tuned cutoff",
    )


class BatchPredictionResponse(BaseModel):
    """Aggregated response for batch predictions."""

    model_used: str = Field(..., description="Model architecture used for inference")
    total_orders: int = Field(..., description="Total orders evaluated in this batch")
    returned_count: int = Field(..., description="Count of orders predicted to be returned")
    delivered_count: int = Field(..., description="Count of orders predicted to be kept/delivered")
    zone_1_safe_count: int = Field(..., description="Count in Zone 1 (Safe Auto-Approve)")
    zone_2_moderate_count: int = Field(..., description="Count in Zone 2 (Logistics Fast-Track)")
    zone_3_high_risk_count: int = Field(..., description="Count in Zone 3 (Wardrobing & Guardrail)")
    average_return_probability: float = Field(..., description="Mean return probability across batch")
    batch_latency_ms: float = Field(..., description="Total batch processing latency in milliseconds")
    per_sample_latency_ms: float = Field(..., description="Average inference latency per order (ms)")
    predictions: List[PredictionResponse] = Field(..., description="Detailed prediction for each order")


class ModelMetadata(BaseModel):
    """Architecture metadata for a model in the registry."""

    model_key: str
    display_name: str
    optimal_threshold: float
    ensemble_weight: Optional[float] = None
    is_loaded: bool


class RegistryResponse(BaseModel):
    """Information on registered models and ensemble configuration."""

    available_models: List[ModelMetadata]
    ensemble_weights: Dict[str, float]
    optimal_thresholds: Dict[str, float]
    required_features: List[str]


class HealthResponse(BaseModel):
    """Service health and diagnostic status."""

    status: str
    models_loaded: Dict[str, bool]
    preprocessor_loaded: bool
    models_dir: str
    uptime_seconds: float
    total_inferences_served: int
