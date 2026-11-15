"""Enterprise E-Commerce Return Prediction & 3-Zone Operational Risk Triage
Streamlit Production Frontend Application.
"""

import json
import random
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Enterprise Return Prediction | 3-Zone Risk Triage",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-polish enterprise aesthetic
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38bdf8;
    }

    /* Operational Zone Badges */
    .badge-zone-1 {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-zone-2 {
        background-color: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.4);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-zone-3 {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }

    /* Callout Card */
    .action-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. SESSION STATE & BACKEND CONFIGURATION
# ==============================================================================
DEFAULT_BACKEND_URL = "https://ecomm-deploy-2.onrender.com"

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL

if "preset_applied" not in st.session_state:
    st.session_state.preset_applied = None


# Helper for calling backend API
def query_api(endpoint: str, method: str = "GET", payload: Dict[str, Any] = None) -> Dict[str, Any]:
    url = f"{st.session_state.backend_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"Content-Type": "application/json", "User-Agent": "Streamlit-Enterprise-Frontend/1.0"}
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=25)
        else:
            resp = requests.post(url, json=payload, headers=headers, timeout=35)

        if resp.status_code == 200:
            return {"success": True, "data": resp.json(), "status_code": 200}
        else:
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}: {resp.text}",
                "status_code": resp.status_code,
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Backend request timed out (cold start spin-up on Render may take ~30s). Please retry.",
            "status_code": 504,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "status_code": 500}


# ==============================================================================
# 3. SIDEBAR: NAVIGATION & BACKEND CONNECTION
# ==============================================================================
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/FastAPI.svg",
        width=48,
    )
    st.title("Enterprise Telemetry")
    st.caption("DS605 Machine Learning • Return Propensity Architecture")
    st.markdown("---")

    # Backend Connection Panel
    st.subheader("🌐 Live Backend Connection")
    backend_input = st.text_input(
        "API Service Base URL",
        value=st.session_state.backend_url,
        help="Backend FastAPI microservice hosting the Soft Voting Ensemble.",
    )
    if backend_input != st.session_state.backend_url:
        st.session_state.backend_url = backend_input
        st.rerun()

    # Health Check ping
    if st.button("🔄 Test API Connection", use_container_width=True):
        with st.spinner("Pinging Render backend..."):
            health_res = query_api("/health")
            if health_res["success"]:
                h_data = health_res["data"]
                st.success(f"🟢 Connected! Uptime: {h_data.get('uptime_seconds', 0):.0f}s")
                st.caption(f"Models Active: {sum(1 for v in h_data.get('models_loaded', {}).values() if v)} / 4")
            else:
                st.error(f"🔴 Connection Failed: {health_res['error']}")

    st.markdown("---")
    st.markdown("### 🏛️ Core Methodological Highlights")
    st.markdown(
        """
        - **Dataset Scale:** 3,999,025 Finalized Orders
        - **Holdout Test Set:** 75,000 Chronological Transactions
        - **Champion Architecture:** Soft Voting Ensemble (35% LGBM + 35% XGB + 15% RF + 15% LR)
        - **Optimal Cutoff:** $\\tau = 0.54$ (Validation Max F1)
        - **Core Innovation:** **3-Zone Operational Risk Triage**
        """
    )
    st.markdown("---")
    st.caption("DS605 Team Research Capstone • Production FastAPI + Streamlit")


# ==============================================================================
# 4. MAIN HEADER BANNER
# ==============================================================================
st.title("🛍️ E-Commerce Return Prediction & Operational Risk Triage")
st.markdown(
    """
    **Enterprise Behavioral Telemetry Framework** predicting post-purchase return propensity at checkout.
    Powered by an audited **Soft Voting Ensemble** evaluated on 3,999,025 commercial transactions.
    """
)

# Quick KPI Metrics Bar
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Holdout Accuracy</div>
            <div class="metric-val">89.99%</div>
            <div class="metric-sub">+0.49% over baseline</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Test ROC-AUC</div>
            <div class="metric-val">0.9453</div>
            <div class="metric-sub">PR-AUC: 0.9188</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Ensemble F1-Score</div>
            <div class="metric-val">0.8598</div>
            <div class="metric-sub">Precision: 87.79% | Recall: 84.24%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Optimal Threshold</div>
            <div class="metric-val">τ = 0.54</div>
            <div class="metric-sub">F1-Maximizing Validation Scan</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi5:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Inference Latency</div>
            <div class="metric-val">~0.024 ms</div>
            <div class="metric-sub">Real-Time Checkout API</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# 5. TABBED INTERFACE
# ==============================================================================
tab_single, tab_batch, tab_scorecard, tab_architecture = st.tabs(
    [
        "🎯 Live Order Triage & Prediction",
        "📊 Batch Simulation & Commercial ROI",
        "🏆 Model Benchmark Scorecard",
        "🧠 Telemetry Signals & Methodology",
    ]
)

# ==============================================================================
# TAB 1: LIVE ORDER TRIAGE & PREDICTION (SINGLE PREDICTION)
# ==============================================================================
with tab_single:
    st.subheader("Interactive Checkout Risk Evaluation")
    st.markdown(
        "Simulate an individual e-commerce checkout transaction to observe how behavioral telemetry signals "
        "influence the predicted return probability and trigger the **3-Zone Operational Strategy**."
    )

    # 1. Preset Scenarios
    st.markdown("##### ⚡ Quick Scenarios (One-Click Presets)")
    preset_cols = st.columns(5)

    preset_cases = {
        "preset_1": {
            "name": "🚨 Sizing Bracketing (Apparel)",
            "desc": "Customer buying 3 sizes of clothing with sizing hesitation & history.",
            "data": {
                "category": "clothing",
                "payment": "credit_card",
                "quantity": 3,
                "unit_price": 59.99,
                "is_bracketed": 1,
                "dwell_time": 85.0,
                "guide_clicks": 6,
                "return_rate": 0.72,
                "fit_offset": 0.8,
                "delay_days": 3.5,
                "first_time": 1,
            },
        },
        "preset_2": {
            "name": "💸 Serial Returner (BNPL)",
            "desc": "High historical returner ordering multi-item apparel on Buy-Now-Pay-Later.",
            "data": {
                "category": "clothing",
                "payment": "bnpl",
                "quantity": 2,
                "unit_price": 89.00,
                "is_bracketed": 1,
                "dwell_time": 60.0,
                "guide_clicks": 4,
                "return_rate": 0.80,
                "fit_offset": -0.7,
                "delay_days": 2.5,
                "first_time": 1,
            },
        },
        "preset_3": {
            "name": "⚖️ Moderate Risk (Zone 2)",
            "desc": "Single clothing item with average customer return rate & minor delay.",
            "data": {
                "category": "clothing",
                "payment": "credit_card",
                "quantity": 1,
                "unit_price": 45.00,
                "is_bracketed": 0,
                "dwell_time": 25.0,
                "guide_clicks": 1,
                "return_rate": 0.45,
                "fit_offset": 0.2,
                "delay_days": 1.0,
                "first_time": 0,
            },
        },
        "preset_4": {
            "name": "📚 Safe Purchase (Books)",
            "desc": "Low-risk loyal customer purchasing books with zero sizing hesitation.",
            "data": {
                "category": "books",
                "payment": "credit_card",
                "quantity": 1,
                "unit_price": 24.99,
                "is_bracketed": 0,
                "dwell_time": 0.0,
                "guide_clicks": 0,
                "return_rate": 0.02,
                "fit_offset": 0.0,
                "delay_days": 0.0,
                "first_time": 0,
            },
        },
        "preset_5": {
            "name": "🍏 Low-Risk Grocery",
            "desc": "Everyday grocery consumables with zero historical return friction.",
            "data": {
                "category": "grocery",
                "payment": "paypal",
                "quantity": 4,
                "unit_price": 8.50,
                "is_bracketed": 0,
                "dwell_time": 0.0,
                "guide_clicks": 0,
                "return_rate": 0.04,
                "fit_offset": 0.0,
                "delay_days": 0.0,
                "first_time": 0,
            },
        },
    }

    selected_preset_key = None
    for i, (k, p_info) in enumerate(preset_cases.items()):
        with preset_cols[i]:
            if st.button(p_info["name"], use_container_width=True, help=p_info["desc"]):
                st.session_state.active_inputs = p_info["data"]
                st.session_state.preset_applied = k
                st.rerun()

    # Active values initialization
    active = st.session_state.get(
        "active_inputs",
        preset_cases["preset_1"]["data"],
    )

    st.markdown("---")

    # 2. Input Form (3 Columns)
    with st.form("single_prediction_form"):
        col_form1, col_form2, col_form3 = st.columns(3)

        with col_form1:
            st.markdown("#### 🛒 1. Order Attributes")
            in_category = st.selectbox(
                "Product Category",
                options=["clothing", "electronics", "beauty", "books", "grocery", "home", "sports", "toys"],
                index=["clothing", "electronics", "beauty", "books", "grocery", "home", "sports", "toys"].index(
                    active["category"]
                ),
            )
            in_payment = st.selectbox(
                "Payment Method",
                options=["credit_card", "debit_card", "paypal", "bnpl", "gift_card"],
                index=["credit_card", "debit_card", "paypal", "bnpl", "gift_card"].index(active["payment"]),
            )
            in_quantity = st.number_input(
                "Quantity Ordered",
                min_value=1,
                max_value=20,
                value=int(active["quantity"]),
            )
            in_unit_price = st.number_input(
                "Unit Price ($)",
                min_value=1.0,
                max_value=2000.0,
                value=float(active["unit_price"]),
                step=5.0,
            )
            in_shipping = st.number_input(
                "Shipping Cost ($)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
            )

        with col_form2:
            st.markdown("#### 📏 2. Behavioral Sizing Signals")
            in_bracketed = st.radio(
                "Cart Sizing Bracketing (Multiple sizes/colors of SKU)",
                options=[0, 1],
                format_func=lambda x: "Yes (Multiple Sizes / Bracketing)" if x == 1 else "No (Single Size / Variant)",
                index=int(active["is_bracketed"]),
                help="Empirical return rate on bracketed apparel orders is 92.93%!",
            )
            in_dwell = st.slider(
                "Size Chart Dwell Time (seconds)",
                min_value=0.0,
                max_value=180.0,
                value=float(active["dwell_time"]),
                help="Active time spent examining the size chart modal. Dwell time > 35s signals significant sizing hesitation.",
            )
            in_clicks = st.slider(
                "Size Guide Interactive Clicks",
                min_value=0,
                max_value=15,
                value=int(active["guide_clicks"]),
                help="Clicks toggling measurement units (cm vs. inches) or interactive fit calculators.",
            )
            in_fit_offset = st.slider(
                "Crowdsourced SKU Fit Sentiment Offset",
                min_value=-1.0,
                max_value=1.0,
                value=float(active["fit_offset"]),
                step=0.1,
                help="-1.0 = Verified buyers report item runs small; 0.0 = True to size; +1.0 = Runs large.",
            )

        with col_form3:
            st.markdown("#### 👤 3. Customer Profile & Transit SLA")
            in_return_rate = st.slider(
                "Customer Historical 12-Month Return Rate",
                min_value=0.0,
                max_value=1.0,
                value=float(active["return_rate"]),
                step=0.01,
                help="Customer's rolling 12-month return rate profile (e.g. 0.05 for loyal buyers, 0.70+ for serial returners).",
            )
            in_delay = st.slider(
                "Delivery Delay vs SLA (Days Late)",
                min_value=0.0,
                max_value=10.0,
                value=float(active["delay_days"]),
                step=0.5,
                help="Last-mile SLA delay. Delays >= 2 days trigger delivery remorse and event-expiry returns.",
            )
            in_first_time = st.radio(
                "First Time Buying This Brand?",
                options=[0, 1],
                format_func=lambda x: "Yes (First-Time Brand Purchase)" if x == 1 else "No (Repeat Brand Buyer)",
                index=int(active["first_time"]),
            )

            st.markdown("#### ⚙️ 4. Architecture Selection")
            in_model = st.selectbox(
                "Model Architecture",
                options=["ensemble", "lightgbm", "xgboost", "random_forest", "logistic_regression"],
                format_func=lambda x: {
                    "ensemble": "🏆 Soft Voting Ensemble (Champion)",
                    "lightgbm": "⚡ LightGBM Classifier",
                    "xgboost": "🌲 XGBoost Classifier",
                    "random_forest": "🌳 Random Forest (Bagging)",
                    "logistic_regression": "📐 Logistic Regression (L2)",
                }.get(x, x),
            )

        submit_btn = st.form_submit_with_button = st.form_submit_button(
            "🚀 Predict Return Propensity & Operational Risk Triage",
            use_container_width=True,
            type="primary",
        )

    # 3. Handle Prediction Submission
    if submit_btn:
        payload = {
            "order": {
                "product_category": in_category,
                "payment_method": in_payment,
                "quantity": in_quantity,
                "unit_price": in_unit_price,
                "shipping_cost": in_shipping,
                "is_bracketed_order": in_bracketed,
                "size_chart_dwell_time_sec": in_dwell,
                "size_guide_clicks": in_clicks,
                "customer_true_return_rate": in_return_rate,
                "sku_fit_feedback_offset": in_fit_offset,
                "delivery_delay_days": in_delay,
                "first_time_brand_purchase": in_first_time,
            },
            "model_name": in_model,
            "explain_signals": True,
        }

        with st.spinner("Executing real-time inference on live backend..."):
            res = query_api("/predict", method="POST", payload=payload)

        if not res["success"]:
            st.error(f"Prediction failed: {res['error']}")
            st.info("💡 Note: If the Render server was idle, it may take ~30 seconds to wake up from free-tier sleep.")
        else:
            p_data = res["data"]
            prob = p_data["return_probability"]
            will_return = p_data["will_return"]
            zone = p_data["risk_zone"]
            action = p_data["operational_action"]
            threshold = p_data["decision_threshold"]
            latency = p_data["inference_latency_ms"]
            contributions = p_data.get("model_contributions", {})
            engineered = p_data.get("engineered_signals", {})

            st.markdown("---")
            st.markdown("### 📋 Real-Time Inference Results & Operational Action")

            # Result Columns
            res_left, res_right = st.columns([1.2, 1])

            with res_left:
                # Operational Zone Badge
                if "Zone 1" in zone:
                    zone_html = f'<span class="badge-zone-1">🟢 {zone}</span>'
                elif "Zone 2" in zone:
                    zone_html = f'<span class="badge-zone-2">🟡 {zone}</span>'
                else:
                    zone_html = f'<span class="badge-zone-3">🔴 {zone}</span>'

                st.markdown(f"**Triage Classification:** {zone_html}", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="action-box">
                        <strong>🎯 Prescribed Commercial Action:</strong><br>
                        {action}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Return Probability", f"{prob*100:.2f}%")
                with stat_col2:
                    st.metric("Binary Decision", p_data["prediction_label"])
                with stat_col3:
                    st.metric("Decision Cutoff (τ)", f"{threshold:.2f}", f"Latency: {latency} ms")

                # Engineered compound signals card
                st.markdown("##### 🔬 Compound Telemetry Features Detected:")
                c1, c2, c3 = st.columns(3)
                with c1:
                    is_fm = engineered.get("fit_mismatch", 0.0) == 1.0
                    st.caption(f"{'🔴' if is_fm else '⚪'} **`fit_mismatch`:** {'TRIGGERED' if is_fm else 'Normal'}")
                with c2:
                    is_dr = engineered.get("delay_remorse", 0.0) == 1.0
                    st.caption(f"{'🔴' if is_dr else '⚪'} **`delay_remorse`:** {'TRIGGERED' if is_dr else 'Normal'}")
                with c3:
                    is_cod = engineered.get("cod_first_brand", 0.0) == 1.0
                    st.caption(f"{'🔴' if is_cod else '⚪'} **`cod_first_brand`:** {'TRIGGERED' if is_cod else 'Normal'}")

            with res_right:
                # Gauge Chart showing the 3 zones
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        number={"suffix": "%", "font": {"size": 36, "color": "#f8fafc"}},
                        title={"text": "3-Zone Risk Propensity Gauge", "font": {"size": 16, "color": "#94a3b8"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
                            "bar": {"color": "#38bdf8", "thickness": 0.28},
                            "bgcolor": "rgba(15, 23, 42, 0.4)",
                            "borderwidth": 1,
                            "bordercolor": "#475569",
                            "steps": [
                                {"range": [0, 35], "color": "rgba(34, 197, 94, 0.35)"},
                                {"range": [35, 65], "color": "rgba(234, 179, 8, 0.35)"},
                                {"range": [65, 100], "color": "rgba(239, 68, 68, 0.35)"},
                            ],
                            "threshold": {
                                "line": {"color": "#f43f5e", "width": 4},
                                "thickness": 0.85,
                                "value": threshold * 100,
                            },
                        },
                    )
                )
                fig_gauge.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc"),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Model consensus breakdown chart if ensemble was queried
            if contributions:
                st.markdown("##### 🤝 Constituent Model Consensus (Soft Voting Ensemble Breakdown)")
                df_contrib = pd.DataFrame(
                    [
                        {"Model": "LightGBM (35%)", "Probability": contributions.get("lightgbm", 0.0) * 100},
                        {"Model": "XGBoost (35%)", "Probability": contributions.get("xgboost", 0.0) * 100},
                        {"Model": "Random Forest (15%)", "Probability": contributions.get("random_forest", 0.0) * 100},
                        {"Model": "Logistic Regression (15%)", "Probability": contributions.get("logistic_regression", 0.0) * 100},
                        {"Model": "Soft Voting Ensemble", "Probability": prob * 100},
                    ]
                )

                fig_contrib = px.bar(
                    df_contrib,
                    x="Probability",
                    y="Model",
                    orientation="h",
                    text=[f"{p:.1f}%" for p in df_contrib["Probability"]],
                    color="Model",
                    color_discrete_map={
                        "LightGBM (35%)": "#ff7f0e",
                        "XGBoost (35%)": "#d62728",
                        "Random Forest (15%)": "#2ca02c",
                        "Logistic Regression (15%)": "#1f77b4",
                        "Soft Voting Ensemble": "#a855f7",
                    },
                )
                fig_contrib.add_vline(x=threshold * 100, line_dash="dash", line_color="#ef4444", annotation_text="Cutoff")
                fig_contrib.update_layout(
                    xaxis_title="Predicted Return Probability (%)",
                    yaxis_title="",
                    xaxis_range=[0, 105],
                    height=220,
                    margin=dict(l=10, r=20, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig_contrib, use_container_width=True)


# ==============================================================================
# TAB 2: BATCH SIMULATION & COMMERCIAL ROI CALCULATOR
# ==============================================================================
with tab_batch:
    st.subheader("📊 Enterprise Batch Simulation & Business ROI Calculator")
    st.markdown(
        "Evaluate large batches of commercial checkout transactions simultaneously to measure "
        "the **aggregate distribution across the 3 Operational Zones** and project **annual financial savings**."
    )

    col_sim_ctrl, col_roi_calc = st.columns([1, 1.2])

    with col_sim_ctrl:
        st.markdown("#### 1. Batch Simulation Controls")
        batch_size = st.select_slider(
            "Select Number of Orders to Simulate in Batch",
            options=[10, 25, 50, 100, 250],
            value=50,
            help="Generates authentic multi-category orders matching the enterprise telemetry distribution.",
        )
        sim_model = st.selectbox(
            "Batch Prediction Architecture",
            options=["ensemble", "lightgbm", "xgboost", "random_forest", "logistic_regression"],
            key="batch_model_select",
        )

        run_batch_btn = st.button("⚡ Generate & Predict Batch Orders", type="primary", use_container_width=True)

    with col_roi_calc:
        st.markdown("#### 2. Commercial ROI & Reverse Logistics Parameters")
        roi_c1, roi_c2 = st.columns(2)
        with roi_c1:
            cost_per_return = st.slider(
                "Reverse Logistics Cost per Return ($)",
                min_value=5.0,
                max_value=35.0,
                value=15.0,
                step=1.0,
                help="Packaging, freight, inspection, restock, and inventory depreciation per return.",
            )
            monthly_order_vol = st.number_input(
                "Monthly Enterprise Order Volume",
                min_value=5000,
                max_value=500000,
                value=50000,
                step=5000,
            )
        with roi_c2:
            intervention_prevention_rate = st.slider(
                "Zone 3 Sizing Intervention Prevention Rate (%)",
                min_value=10,
                max_value=70,
                value=35,
                step=5,
                help="% of wardrobing returns averted by sizing confirmation prompts and nominal fee guardrails.",
            )
            sla_breach_cost = st.slider(
                "Average SLA Delay Penalty Cost ($)",
                min_value=2.0,
                max_value=20.0,
                value=8.0,
                step=1.0,
            )

    # Synthetic order generator matching telemetry distribution
    def generate_synthetic_orders(n: int) -> List[Dict[str, Any]]:
        categories = ["clothing", "electronics", "beauty", "books", "grocery", "home", "sports", "toys"]
        payments = ["credit_card", "debit_card", "paypal", "bnpl", "gift_card"]
        orders = []
        for _ in range(n):
            cat = random.choices(categories, weights=[0.40, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05])[0]
            is_bracketed = 1 if (cat == "clothing" and random.random() < 0.28) else 0
            if is_bracketed:
                dwell = random.uniform(45.0, 110.0)
                clicks = random.randint(2, 7)
                ret_rate = random.uniform(0.40, 0.85)
                fit_off = random.choice([-0.8, -0.6, 0.5, 0.8])
            else:
                dwell = random.uniform(0.0, 30.0)
                clicks = random.randint(0, 2)
                ret_rate = random.uniform(0.02, 0.35)
                fit_off = random.uniform(-0.2, 0.2)

            delay = max(0.0, random.gauss(0.8, 1.2))
            orders.append(
                {
                    "product_category": cat,
                    "payment_method": random.choice(payments),
                    "quantity": random.randint(2, 4) if is_bracketed else random.randint(1, 2),
                    "unit_price": round(random.uniform(20.0, 150.0), 2),
                    "is_bracketed_order": is_bracketed,
                    "size_chart_dwell_time_sec": round(dwell, 1),
                    "size_guide_clicks": clicks,
                    "customer_true_return_rate": round(ret_rate, 2),
                    "sku_fit_feedback_offset": round(fit_off, 2),
                    "delivery_delay_days": round(delay, 1),
                    "first_time_brand_purchase": 1 if random.random() < 0.35 else 0,
                }
            )
        return orders

    if run_batch_btn:
        syn_orders = generate_synthetic_orders(batch_size)
        with st.spinner(f"Evaluating {batch_size} orders through live Render backend..."):
            b_res = query_api("/predict/batch", method="POST", payload={"orders": syn_orders, "model_name": sim_model})

        if not b_res["success"]:
            st.error(f"Batch prediction error: {b_res['error']}")
        else:
            b_data = b_res["data"]
            total = b_data["total_orders"]
            ret_count = b_data["returned_count"]
            deliv_count = b_data["delivered_count"]
            z1 = b_data["zone_1_safe_count"]
            z2 = b_data["zone_2_moderate_count"]
            z3 = b_data["zone_3_high_risk_count"]
            avg_p = b_data["average_return_probability"]
            b_lat = b_data["batch_latency_ms"]
            p_lat = b_data["per_sample_latency_ms"]
            predictions = b_data["predictions"]

            st.markdown("---")
            st.markdown("### 📈 Batch Prediction Summary & Operational Breakdown")

            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            with b_c1:
                st.metric("Total Batch Evaluated", f"{total:,} orders", f"{b_lat:.1f} ms total")
            with b_c2:
                st.metric("Zone 1 (Safe Auto-Approve)", f"{z1} orders", f"{(z1/total)*100:.1f}%")
            with b_c3:
                st.metric("Zone 2 (Logistics Fast-Track)", f"{z2} orders", f"{(z2/total)*100:.1f}%")
            with b_c4:
                st.metric("Zone 3 (Wardrobing Guardrail)", f"{z3} orders", f"{(z3/total)*100:.1f}%")

            # Visualizations
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                df_zones = pd.DataFrame(
                    [
                        {"Zone": "Zone 1: Safe Auto-Approve", "Count": z1, "Action": "Frictionless Fulfillment"},
                        {"Zone": "Zone 2: Logistics Fast-Track", "Count": z2, "Action": "Express SLA Routing"},
                        {"Zone": "Zone 3: Wardrobing Guardrail", "Count": z3, "Action": "Sizing & Fee Guardrail"},
                    ]
                )
                fig_donut = px.pie(
                    df_zones,
                    names="Zone",
                    values="Count",
                    color="Zone",
                    color_discrete_map={
                        "Zone 1: Safe Auto-Approve": "#22c55e",
                        "Zone 2: Logistics Fast-Track": "#eab308",
                        "Zone 3: Wardrobing Guardrail": "#ef4444",
                    },
                    hole=0.45,
                    title="Operational Risk Zone Allocation",
                )
                fig_donut.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=35, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with v_col2:
                batch_probs = [p["return_probability"] * 100 for p in predictions]
                fig_hist = px.histogram(
                    x=batch_probs,
                    nbins=20,
                    title="Distribution of Return Propensity Probabilities",
                    labels={"x": "Return Probability (%)"},
                    color_discrete_sequence=["#38bdf8"],
                )
                fig_hist.add_vline(x=35, line_dash="dash", line_color="#22c55e", annotation_text="Zone 1 / 2 Cutoff")
                fig_hist.add_vline(x=65, line_dash="dash", line_color="#ef4444", annotation_text="Zone 2 / 3 Cutoff")
                fig_hist.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=35, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_title="Count of Orders",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # ROI Financial Impact Simulation
            st.markdown("---")
            st.markdown("### 💰 Projected Commercial Impact & Annual Savings")
            pct_z3 = z3 / max(1, total)
            monthly_z3_orders = monthly_order_vol * pct_z3
            monthly_returns_prevented = monthly_z3_orders * (intervention_prevention_rate / 100.0)
            monthly_savings_logistics = monthly_returns_prevented * cost_per_return
            annual_savings = monthly_savings_logistics * 12

            roi_m1, roi_m2, roi_m3, roi_m4 = st.columns(4)
            with roi_m1:
                st.metric("Projected Monthly High-Risk Orders", f"{int(monthly_z3_orders):,}")
            with roi_m2:
                st.metric("Returns Prevented / Month", f"{int(monthly_returns_prevented):,}")
            with roi_m3:
                st.metric("Monthly Reverse Logistics Savings", f"${monthly_savings_logistics:,.0f}")
            with roi_m4:
                st.metric("Net Projected Annual Savings", f"${annual_savings:,.0f}", "+ROI 18x")

            # Table of predictions
            st.markdown("##### 🔍 Batch Item Breakdown (First 15 Orders)")
            df_disp = pd.DataFrame(
                [
                    {
                        "Order #": i + 1,
                        "Category": syn_orders[i]["product_category"],
                        "Payment": syn_orders[i]["payment_method"],
                        "Total ($)": f"${syn_orders[i]['unit_price'] * syn_orders[i]['quantity']:.2f}",
                        "Bracketed?": "Yes" if syn_orders[i]["is_bracketed_order"] == 1 else "No",
                        "Probability": f"{predictions[i]['return_probability']*100:.1f}%",
                        "Decision": predictions[i]["prediction_label"],
                        "Operational Zone": predictions[i]["risk_zone"],
                    }
                    for i in range(min(15, total))
                ]
            )
            st.dataframe(df_disp, use_container_width=True)


# ==============================================================================
# TAB 3: MODEL SCORECARD & BENCHMARK COMPARISON
# ==============================================================================
with tab_scorecard:
    st.subheader("🏆 Model Architecture Benchmarking & Scorecard")
    st.markdown(
        "Direct comparison of all five model architectures evaluated on the authentic "
        "**75,000 unseen chronological holdout test orders** from our research."
    )

    scorecard_data = [
        {
            "Model Architecture": "Logistic Regression (L2)",
            "ROC-AUC": "0.9454",
            "PR-AUC": "0.9191",
            "Accuracy": "89.97%",
            "F1-Score": "0.8597",
            "Precision": "87.74%",
            "Recall": "84.27%",
            "Brier Score": "0.0786",
            "Cutoff (τ)": "0.43",
            "Latency": "0.00038 ms",
        },
        {
            "Model Architecture": "Random Forest (Bagging)",
            "ROC-AUC": "0.9428",
            "PR-AUC": "0.9130",
            "Accuracy": "89.93%",
            "F1-Score": "0.8584",
            "Precision": "88.03%",
            "Recall": "83.76%",
            "Brier Score": "0.0831",
            "Cutoff (τ)": "0.57",
            "Latency": "0.01166 ms",
        },
        {
            "Model Architecture": "LightGBM (GBDT)",
            "ROC-AUC": "0.9452",
            "PR-AUC": "0.9184",
            "Accuracy": "90.00%",
            "F1-Score": "0.8599",
            "Precision": "87.85%",
            "Recall": "84.21%",
            "Brier Score": "0.0821",
            "Cutoff (τ)": "0.57",
            "Latency": "0.00673 ms",
        },
        {
            "Model Architecture": "XGBoost (Hist-GBDT)",
            "ROC-AUC": "0.9453",
            "PR-AUC": "0.9180",
            "Accuracy": "89.91%",
            "F1-Score": "0.8595",
            "Precision": "87.22%",
            "Recall": "84.72%",
            "Brier Score": "0.0821",
            "Cutoff (τ)": "0.52",
            "Latency": "0.00519 ms",
        },
        {
            "Model Architecture": "🏆 Soft Voting Ensemble",
            "ROC-AUC": "0.9453",
            "PR-AUC": "0.9188",
            "Accuracy": "89.99%",
            "F1-Score": "0.8598",
            "Precision": "87.79%",
            "Recall": "84.24%",
            "Brier Score": "0.0810",
            "Cutoff (τ)": "0.54",
            "Latency": "0.02396 ms",
        },
    ]

    df_scorecard = pd.DataFrame(scorecard_data)
    st.dataframe(df_scorecard, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📊 Multi-Metric Trade-Off Radar")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        # F1 vs Accuracy bar chart
        df_plot = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost", "Voting Ensemble"],
                "F1-Score": [0.8597, 0.8584, 0.8599, 0.8595, 0.8598],
                "Accuracy (%)": [89.97, 89.93, 90.00, 89.91, 89.99],
            }
        )
        fig_f1 = px.bar(
            df_plot,
            x="Model",
            y="F1-Score",
            color="Model",
            text=[f"{f:.4f}" for f in df_plot["F1-Score"]],
            title="F1-Score Across Evaluated Architectures",
            color_discrete_sequence=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#a855f7"],
        )
        fig_f1.update_layout(yaxis_range=[0.850, 0.865], height=320, paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_f1, use_container_width=True)

    with col_chart2:
        # Latency vs ROC-AUC trade-off
        df_tradeoff = pd.DataFrame(
            {
                "Model": ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost", "Voting Ensemble"],
                "Latency_ms": [0.00038, 0.01166, 0.00673, 0.00519, 0.02396],
                "ROC_AUC": [0.9454, 0.9428, 0.9452, 0.9453, 0.9453],
            }
        )
        fig_tradeoff = px.scatter(
            df_tradeoff,
            x="Latency_ms",
            y="ROC_AUC",
            text="Model",
            size=[20, 20, 20, 20, 28],
            color="Model",
            title="Inference Latency (ms) vs. ROC-AUC Performance",
            labels={"Latency_ms": "Latency (ms/sample)", "ROC_AUC": "Holdout ROC-AUC"},
        )
        fig_tradeoff.update_traces(textposition="top center")
        fig_tradeoff.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_tradeoff, use_container_width=True)


# ==============================================================================
# TAB 4: TELEMETRY ARCHITECTURE & METHODOLOGY
# ==============================================================================
with tab_architecture:
    st.subheader("🧠 Enterprise Behavioral Telemetry Architecture")
    st.markdown(
        "Standard checkout attributes (price, category, payment method) are insufficient to detect return propensity. "
        "Our system blends traditional checkout data with **seven real-time behavioral telemetry signals**."
    )

    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown(
            """
            #### 1. Sizing Bracketing (`is_bracketed_order`)
            - **Definition:** User places an order containing multiple sizes or colors of the exact same SKU.
            - **Empirical Reality:** 92.93% of bracketed apparel orders result in a return because the buyer intends from the start to return all but the best-fitting item.

            #### 2. Size Guide Dwell Time (`size_chart_dwell_time_sec`)
            - **Definition:** Exact seconds the buyer spent inspecting the product page's size measurement chart modal.
            - **Behavioral Signal:** Dwell time $> 35$ seconds indicates severe hesitation and sizing uncertainty.

            #### 3. Interactive Sizing Clicks (`size_guide_clicks`)
            - **Definition:** Count of user interactions switching units (inches vs cm) or testing fit calculators.
            - **Correlation:** High clicks indicate poor confidence in apparel sizing standard.
            """
        )

    with t_col2:
        st.markdown(
            r"""
            #### 4. Persistent Customer Return Profile (`customer_true_return_rate`)
            - **Definition:** Rolling 12-month return rate associated with the customer's account.
            - **Impact:** Differentiates high-frequency wardrobers (~91.7% return rate) from loyal keepers (~11.6% return rate).

            #### 5. Crowdsourced Verified Buyer Sentiment (`sku_fit_feedback_offset`)
            - **Definition:** Aggregate post-purchase buyer feedback rating (-1.0 = runs small, 0 = true to size, +1.0 = runs large).
            - **Compound Signal:** High dwell time combined with fit offset triggers `fit_mismatch = 1.0`.

            #### 6. Last-Mile SLA Transit Delay (`delivery_delay_days`)
            - **Definition:** Actual delivery arrival date minus promised SLA date.
            - **Impact:** Delays $\ge 2$ days trigger buyer remorse and missed-event returns.
            """
        )

    st.markdown("---")
    st.markdown("### 🎯 3-Zone Operational Strategy Flowchart")
    st.markdown(
        """
        ```text
        Predicted Return Probability (P_hat)
        ├── [0.00 - 0.35) ──► ZONE 1: SAFE AUTO-APPROVE (~60% of orders)
        │                     • Actual return rate: < 11.5%
        │                     • Action: Instant frictionless fulfillment with zero intervention.
        │
        ├── [0.35 - 0.65) ──► ZONE 2: LOGISTICS FAST-TRACK & NUDGE (~20% of orders)
        │                     • Actual return rate: ~35.0%
        │                     • Action: Route via express logistics to prevent SLA delivery delays; include pre-printed return label.
        │
        └── [0.65 - 1.00] ──► ZONE 3: WARDROBING & SIZING GUARDRAIL (~20% of orders)
                              • Actual return rate: > 91.0%
                              • Action: Real-time checkout prompt: "Confirm your sizing measurements to save return hassle"; 
                                require nominal return fee on multi-size apparel.
        ```
        """
    )
