import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
 
from inference_sales_delta import predict_sales_price_delta
from inference_rentals import predict_rental_demand   # ✅ Updated import
 
st.set_page_config(page_title="Sales & Rental Predictor", layout="wide")
 
st.title("🏠 Sales Price & Rental Demand Predictor")
 
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to section:", ["Single Prediction", "Batch Prediction"])
 
# -------------------------------------------
#  SINGLE PREDICTION
# -------------------------------------------
if page == "Single Prediction":
    st.subheader("📌 Enter Property Details")
 
    col1, col2 = st.columns(2)
 
    with col1:
        bedrooms = st.number_input("Bedrooms", 0, 10, 3)
        bathrooms = st.number_input("Bathrooms", 0, 10, 2)
        area = st.number_input("Area (sqft)", 200, 10000, 1200)
        age = st.number_input("Property Age (years)", 0, 100, 5)
 
    with col2:
        is_weekend = st.checkbox("Is Weekend Listing?", False)
        distance = st.number_input("Distance to City Center (km)", 0.0, 50.0, 5.0)
        parking = st.checkbox("Has Parking?", True)
        furnished = st.checkbox("Is Furnished?", False)
 
    input_df = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "area": area,
        "age": age,
        "is_weekend": int(is_weekend),
        "distance": distance,
        "parking": int(parking),
        "furnished": int(furnished),
    }])
 
    st.write("### ✅ Model Input Preview")
    st.dataframe(input_df)
 
    if st.button("Predict"):
        with st.spinner("Generating predictions..."):
           
            sale_pred = predict_sales_price_delta(input_df)
            rental_pred = predict_rental_demand(input_df)
 
        st.success(f"🏷 **Predicted Sale Price:** ${sale_pred.iloc[0]:,.0f}")
        st.info(f"📊 **Predicted Rental Demand:** {rental_pred.iloc[0]:,.0f} units")
 
 
# -------------------------------------------
#  BATCH PREDICTION
# -------------------------------------------
elif page == "Batch Prediction":
    st.subheader("📤 Batch Predictions via CSV")
 
    template_cols = [
        "bedrooms", "bathrooms", "area", "age",
        "is_weekend", "distance", "parking", "furnished"
    ]
    st.write("Required columns:", template_cols)
 
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
 
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("### ✅ File Preview")
        st.dataframe(df.head())
 
        if st.button("Run Batch Predictions"):
            with st.spinner("Running predictions…"):
 
                df["Predicted_Sale_Price"] = predict_sales_price_delta(df)
                df["Predicted_Rental_Demand"] = predict_rental_demand(df)
 
            st.success("✅ Batch predictions complete!")
            st.write(df)
 
            csv_out = df.to_csv(index=False).encode()
            st.download_button("Download Results CSV", csv_out, "predictions_output.csv")