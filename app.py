import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
 
from inference_sales_delta import predict_sales_price_delta
from inference_rentals import predict_rental_demand
 
# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Sales & Rentals Prediction Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# -----------------------------------------------------------
# SIDEBAR MENU
# -----------------------------------------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🏡 Home",
        "📈 Sales Price Prediction",
        "🏠 Rental Demand Prediction",
        "📤 Batch Predictions",
        "🔬 SHAP Visualizations"
    ]
)
 
st.sidebar.markdown("---")
st.sidebar.caption("Developed by **Bertha GARNA**")
 
# ===========================================================
# HOME PAGE
# ===========================================================
if page == "🏡 Home":
    st.title("🏡 Sales & Rental Prediction Suite")
    st.markdown("""
        Welcome to the **Prediction Suite**, powered by:
 
        ✅ Random Forest Δ‑Model for **sales price prediction** 
        ✅ Random Forest Classifier for **rental demand prediction** 
        ✅ Fully interactive UI with **Streamlit** 
        ✅ Explainable ML with **SHAP** 
        ✅ Batch scoring & Excel export
 
        Use the left sidebar to navigate.
    """)
 
# ===========================================================
# SALES PRICE PREDICTION
# ===========================================================
elif page == "📈 Sales Price Prediction":
    st.title("📈 Sales Price Prediction (Δ‑Model)")
 
    col1, col2 = st.columns([1.5, 1])
 
    with col1:
        st.subheader("Input Home Listing Information")
        LATEST_PRICE = st.number_input("Listing Price ($)", min_value=0.0, step=1000.0, key="s_latest")
        ZIPCODE = st.text_input("ZIP Code", "12345", key="s_zip")
        listing_month = st.selectbox("Listing Month", list(range(1, 13)), key="s_month")
        is_weekend_list = st.checkbox("Weekend Listing?", key="s_weekend")
        approx_distance_to_center = st.number_input("Distance to City Center (km)", min_value=0.0, key="s_dist")
 
        if st.button("🔮 Predict Sale Price", key="s_pred_btn"):
            df = pd.DataFrame([{
                "LATEST_PRICE": LATEST_PRICE,
                "ZIPCODE": ZIPCODE,
                "listing_month": listing_month,
                "is_weekend_list": int(is_weekend_list),
                "approx_distance_to_center": approx_distance_to_center
            }])
            result = predict_sales_price_delta(df)
            st.success(f"✅ **Predicted Sale Price:** ${result.iloc[0]:,.0f}")
 
    with col2:
        st.info("Fill in details and click **Predict Sale Price**.")
 
# ===========================================================
# RENTAL DEMAND PREDICTION
# ===========================================================
elif page == "🏠 Rental Demand Prediction":
    st.title("🏠 Rental High‑Demand Prediction")
 
    col1, col2 = st.columns([1.5, 1])
 
    with col1:
        st.subheader("Rental Inputs")
        LATEST_PRICE_R = st.number_input("Listing Price ($)", min_value=0.0, key="r_latest")
        RENT_LISTED = st.number_input("Rent Listed ($/month)", min_value=0.0, key="r_rent")
        ZIPCODE_R = st.text_input("ZIP Code", "12345", key="r_zip")
        LEASE_TERM_MNTS = st.number_input("Lease Term (months)", min_value=1, key="r_lease")
        listing_month_R = st.selectbox("Listing Month", list(range(1, 13)), key="r_month")
        is_weekend_list_R = st.checkbox("Weekend Listing?", key="r_weekend")
        approx_distance_R = st.number_input("Distance to City Center (km)", min_value=0.0, key="r_dist")
 
        if st.button("🔥 Predict Rental Demand", key="r_pred_btn"):
            df = pd.DataFrame([{
                "LATEST_PRICE": LATEST_PRICE_R,
                "RENT_LISTED": RENT_LISTED,
                "ZIPCODE": ZIPCODE_R,
                "LEASE_TERM_MNTS": LEASE_TERM_MNTS,
                "listing_month": listing_month_R,
                "is_weekend_list": int(is_weekend_list_R),
                "approx_distance_to_center": approx_distance_R
            }])
 
            prob, label = predict_rental_demand(df)
            st.write(f"**Probability of High Demand:** {prob.iloc[0] * 100:.1f}%")
 
            if label.iloc[0] == 1:
                st.success("✅ High Demand Expected!")
            else:
                st.info("ℹ Normal Demand Expected")
 
    with col2:
        st.info("Fill listing info and click **Predict**.")
 
# ===========================================================
# 🚀 CSV BATCH PREDICTIONS
# ===========================================================
elif page == "📤 Batch Predictions":
    st.title("📤 Batch Scoring — Upload CSV")
 
    st.write("Upload a CSV file containing listing records for **Sales** or **Rentals**.")
 
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
 
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())
 
        model_type = st.radio("Select model:", ["Sales Δ Model", "Rental Demand Model"], key="batch_type")
 
        if st.button("⚡ Run Batch Predictions", key="batch_run"):
 
            if model_type == "Sales Δ Model":
                preds = predict_sales_price_delta(df)
                df["SALE_PRICE_PRED"] = preds
                st.success("✅ Sales predictions complete.")
 
            else:
                prob, label = predict_rental_demand(df)
                df["HIGH_DEMAND_PROB"] = prob
                df["HIGH_DEMAND_LABEL"] = label
                st.success("✅ Rental demand predictions complete.")
 
            st.write("✅ Results:", df.head())
 
            # Excel download
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False)
 
            st.download_button(
                "⬇ Download Results as Excel",
                data=buffer.getvalue(),
                file_name="batch_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
 
# ===========================================================
# 🔬 SHAP VISUALIZATIONS
# ===========================================================
elif page == "🔬 SHAP Visualizations":
    st.title("🔬 Model Explainability (SHAP)")
 
    model_choice = st.selectbox("Choose model:", ["Sales Δ Model", "Rental Demand Model"])
 
    if model_choice == "Sales Δ Model":
        pipe = joblib.load("Models/sales_delta_rf_pipeline.joblib")
        cols = pd.read_json("Models/sales_delta_rf_columns.json", typ="series").tolist()
    else:
        pipe = joblib.load("Models/rental_rf_tuned_pipeline.joblib")
        cols = pd.read_json("Models/rental_rf_columns.json", typ="series").tolist()
 
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
 
    X_dummy = pd.DataFrame([np.zeros(len(cols))], columns=cols)
    X_trans = prep.transform(X_dummy)
 
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)
 
    st.subheader("📊 Global Feature Importance")
    fig1 = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_trans, feature_names=prep.get_feature_names_out(), show=False)
    st.pyplot(fig1)
 
    st.subheader("🌊 SHAP Waterfall Plot")
    fig2 = shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value, shap_values[0], feature_names=prep.get_feature_names_out(), max_display=12
    )
    st.pyplot(bbox_inches="tight")
 
    st.subheader("🎯 SHAP Force Plot")
    shap_html = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        feature_names=prep.get_feature_names_out(),
        matplotlib=False
    )
    st.components.v1.html(shap.getjs(), height=0)
    st.components.v1.html(shap_html.html(), height=200)