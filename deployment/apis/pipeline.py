"""Telemetry feature pipeline and transformer integration."""

from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from deployment.apis.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_REQUIRED_FEATURES,
)
from deployment.apis.schemas import OrderFeatures


def orders_to_dataframe(orders: Union[OrderFeatures, List[OrderFeatures]]) -> pd.DataFrame:
    """Convert one or multiple OrderFeatures objects into an ordered pandas DataFrame."""
    if isinstance(orders, OrderFeatures):
        orders_list = [orders]
    else:
        orders_list = orders

    records = []
    for o in orders_list:
        # Guarantee all required features are present
        rec = {
            "quantity": float(o.quantity),
            "unit_price": float(o.unit_price),
            "order_total": float(o.order_total),
            "shipping_cost": float(o.shipping_cost),
            "is_bracketed_order": float(o.is_bracketed_order),
            "size_chart_dwell_time_sec": float(o.size_chart_dwell_time_sec),
            "size_guide_clicks": float(o.size_guide_clicks),
            "customer_true_return_rate": float(o.customer_true_return_rate),
            "sku_fit_feedback_offset": float(o.sku_fit_feedback_offset),
            "delivery_delay_days": float(o.delivery_delay_days),
            "first_time_brand_purchase": float(o.first_time_brand_purchase),
            "fit_mismatch": float(o.fit_mismatch),
            "delay_remorse": float(o.delay_remorse),
            "cod_first_brand": float(o.cod_first_brand),
            "product_category": str(o.product_category).strip().lower(),
            "payment_method": str(o.payment_method).strip().lower(),
        }
        records.append(rec)

    df = pd.DataFrame(records)
    # Ensure strict column ordering
    return df[ALL_REQUIRED_FEATURES]


def transform_orders(
    orders: Union[OrderFeatures, List[OrderFeatures]],
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """Convert input orders to DataFrame and apply fitted ColumnTransformer."""
    df = orders_to_dataframe(orders)
    return preprocessor.transform(df)
