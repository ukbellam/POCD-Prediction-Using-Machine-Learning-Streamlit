import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("POCD Prediction App (Using MLP Model)")

# Load model + feature columns
model = joblib.load("mlp_pocd_pipeline.pkl")
feature_cols = joblib.load("feature_columns.pkl")

st.write("Upload a CSV file containing patient records with the same features used in training.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Reindex to match training columns
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Predict probability
    preds = model.predict_proba(df)[:, 1]

    df["POCD_Risk"] = preds

    st.success("Prediction complete!")
    st.write(df.head())

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name="pocd_predictions.csv",
        mime="text/csv",
    )
