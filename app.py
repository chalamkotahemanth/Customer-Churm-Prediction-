import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("churn_model.joblib")

st.title("📊 Telecom Customer Churn Prediction")

uploaded_file = st.file_uploader("Upload a customer CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # One-hot encode and align with model input
    df_encoded = pd.get_dummies(df)
    for col in model.feature_names_in_:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model.feature_names_in_]
    
    '''cols have lot colums and colms not in model.features_names_in_  new column kept value as 0 and total added to new table variable '''

    # Predict churn
    prediction = model.predict(df_encoded)
    df["Churn Prediction"] = prediction

    st.success("✅ Prediction Complete!")
    st.write(df[["name","gender", "Contract", "MonthlyCharges", "Churn Prediction"]])

