import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD ARTIFACTS ----------------
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # dict of LabelEncoders
selected_features = joblib.load("selected_features.pkl")  # only input features
feature_order = joblib.load("feature_order.pkl")  # order of features for input

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Financial Distress Prediction",
    layout="centered"
)

st.title("🎓 Financial Distress Prediction System")
st.write(
    "This application predicts financial distress among university students "
    "using a Random Forest model."
)

st.markdown("---")

# ---------------- USER INPUT ----------------
st.header("📋 Student Information")

user_input = {}

for feature in feature_order:
    if feature in label_encoders:
        # show selectbox for categorical features
        options = list(label_encoders[feature].classes_)
        user_input[feature] = st.selectbox(feature, options)
    else:
        # numeric features
        user_input[feature] = st.number_input(feature, value=0)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([user_input])

    # ---------------- ENCODE CATEGORICAL FEATURES ----------------
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # ---------------- SELECT FINAL FEATURES ----------------
    input_df = input_df[selected_features]  # keep only features used by the model

    # ---------------- PREDICT ----------------
    prediction = model.predict(input_df)[0]

    st.markdown("---")

    if prediction == 1:
        st.error("⚠️ High Financial Distress Detected")
    else:
        st.success("✅ Low Financial Distress Detected")
