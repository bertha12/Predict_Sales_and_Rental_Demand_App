import streamlit as st

import pandas as pd

from inference_sales_delta import predict_sales_price_delta

from inference_rentals import predict_rental_demand
st.set_page_config(

    page_title="Sales & Rentals Prediction Suite",

    layout="wide",

)
st.title("🏡 Prediction Suite: Sales Price & Rental Demand")

st.markdown("A unified ML interface powered by Random Forest models (Δ-model for Sales, RF-Classifier for Rentals).")

tabs = st.tabs(["📈 Sales Price Prediction", "🏠 Rental Demand Prediction"])
# -------------------------------------------------------------------

# 📈 TAB 1 — SALES PRICE PREDICTION (Δ‑MODEL)

# -------------------------------------------------------------------

with tabs[0]:

 

    st.header("Sales Price Prediction (Random Forest Δ-Model)")

 

    col1, col2 = st.columns(2)

 

    with col1:

        st.subheader("Property Inputs")

        LATEST_PRICE = st.number_input("Listing Price ($)", min_value=0.0, step=1000.0)

        ZIPCODE = st.text_input("ZIP Code", "12345")

        listing_month = st.selectbox("Listing Month", list(range(1, 13)))

        is_weekend_list = st.checkbox("Weekend Listing?")

        approx_distance_to_center = st.number_input("Distance to City Center (km)", min_value=0.0)

 

    with col2:

        st.subheader("Preview")

        st.info("Enter inputs on the left and click **Predict Sale Price**.")

 

    if st.button("🔮 Predict Sale Price"):

        df_sales = pd.DataFrame([{

            "LATEST_PRICE": LATEST_PRICE,

            "ZIPCODE": ZIPCODE,

            "listing_month": listing_month,

            "is_weekend_list": int(is_weekend_list),

            "approx_distance_to_center": approx_distance_to_center

        }])

 

        sales_pred = predict_sales_price_delta(df_sales)

        st.success(f"✅ **Predicted Final Sale Price:** ${sales_pred.iloc[0]:,.0f}")

 

 

# -------------------------------------------------------------------

# 🏠 TAB 2 — RENTAL DEMAND PREDICTION

# -------------------------------------------------------------------

with tabs[1]:

 

    st.header("Rental High-Demand Prediction (Random Forest Classifier)")

 

    col1, col2 = st.columns(2)

 

    with col1:

        LATEST_PRICE = st.number_input("Listing Price ($)", min_value=0.0)

        RENT_LISTED = st.number_input("Rent Listed ($ per month)", min_value=0.0)

        ZIPCODE_R = st.text_input("ZIP Code", "12345")

        LEASE_TERM_MNTS = st.number_input("Lease Term (months)", min_value=1)

        listing_month_R = st.selectbox("Listing Month", list(range(1, 13)), key="month_r")

        is_weekend_list_R = st.checkbox("Weekend Listing?", key="weekend_r")

        approx_distance_R = st.number_input("Distance to City Center (km)", min_value=0.0, key="distance_r")

 

    with col2:

        st.subheader("Preview")

        st.info("Enter inputs on the left and click **Predict Rental Demand**.")

 

    if st.button("🔥 Predict Rental Demand"):

        df_rental = pd.DataFrame([{

            "LATEST_PRICE": LATEST_PRICE,

            "RENT_LISTED": RENT_LISTED,

            "ZIPCODE": ZIPCODE_R,

            "LEASE_TERM_MNTS": LEASE_TERM_MNTS,

            "listing_month": listing_month_R,

            "is_weekend_list": int(is_weekend_list_R),

            "approx_distance_to_center": approx_distance_R

        }])

 

        prob, label = predict_rental_demand(df_rental)

 

        st.write(f"**Probability of High Demand:** {prob.iloc[0]*100:.1f}%")

 

        if label.iloc[0] == 1:

            st.success("✅ High Demand Expected!")

        else:

            st.info("ℹ️ Normal Demand")