import os

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="SuperKart Sales Predictor")
st.title("SuperKart Sales Predictor")

backend_url = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")

with st.form("single_prediction"):
    st.subheader("Single prediction")
    product_weight = st.number_input("Product weight", min_value=0.1, value=10.0)
    sugar_content = st.selectbox(
        "Sugar content", ["Low Sugar", "Regular", "No Sugar"]
    )
    allocated_area = st.number_input(
        "Allocated display area", min_value=0.0, max_value=1.0, value=0.05
    )
    product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
    store_size = st.selectbox("Store size", ["High", "Medium", "Small"])
    city_type = st.selectbox(
        "City tier", ["Tier 1", "Tier 2", "Tier 3"]
    )
    store_type = st.selectbox(
        "Store type",
        [
            "Departmental Store",
            "Supermarket Type1",
            "Supermarket Type2",
            "Food Mart",
        ],
    )
    product_id_char = st.selectbox("Product ID prefix", ["FD", "NC", "DR"])
    store_age = st.number_input("Store age in years", min_value=0, value=15)
    product_category = st.selectbox(
        "Product category",
        [
            "Frozen Foods",
            "Perishables",
            "Packaged Foods",
            "Non-Food Essentials",
            "Beverages",
            "Others",
        ],
    )
    submitted = st.form_submit_button("Predict sales")

if submitted:
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar_content,
        "Product_Allocated_Area": allocated_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city_type,
        "Store_Type": store_type,
        "Product_Id_char": product_id_char,
        "Store_Age_Years": store_age,
        "Product_Type_Category": product_category,
    }
    try:
        response = requests.post(
            f"{backend_url}/v1/predict", json=payload, timeout=30
        )
        response.raise_for_status()
        st.metric("Predicted sales", f"${response.json()['prediction']:,.2f}")
    except requests.RequestException as error:
        st.error(f"Could not get a prediction from the backend: {error}")

st.divider()
st.subheader("Batch predictions")
uploaded_csv = st.file_uploader("Upload a product CSV", type=["csv"])

if uploaded_csv is not None and st.button("Predict uploaded rows"):
    try:
        response = requests.post(
            f"{backend_url}/v1/predictbatch",
            files={"file": (uploaded_csv.name, uploaded_csv.getvalue(), "text/csv")},
            timeout=120,
        )
        response.raise_for_status()
        predictions = response.json()
        results = pd.DataFrame(
            {
                "Record Index": list(predictions.keys()),
                "Predicted Sales": list(predictions.values()),
            }
        )
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Download predictions CSV",
            results.to_csv(index=False),
            file_name="superkart_predictions.csv",
            mime="text/csv",
        )
    except requests.RequestException as error:
        st.error(f"Could not get batch predictions from the backend: {error}")