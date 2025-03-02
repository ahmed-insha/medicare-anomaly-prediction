import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---- USER CREDENTIALS ----
USERNAME = "admin"
PASSWORD = "123"

# ---- SESSION STATE ----
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ---- LOGIN FUNCTION ----
def login():
    st.title("🔐 Login to Anomaly Detection")
    username = st.text_input("Username", value="", placeholder="Enter your username")
    password = st.text_input("Password", value="", type="password", placeholder="Enter your password")
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.authenticated = True
            st.success("✅ Login Successful! Redirecting...")
            st.rerun()
        else:
            st.error("❌ Invalid Username or Password")

# ---- FUNCTION TO COMPUTE ANOMALY FEATURES ----


def compute_anomaly_features(input_data, iso_forest, high_txn_threshold):
    # ✅ Keep only the 6 trained features
    df = pd.DataFrame(input_data, columns=["unique_procedures", "total_procedures_count", "total_counts", "age", "gender", "income"])

    # 🚀 Drop extra columns if they exist to match model training
    trained_features = iso_forest.feature_names_in_
    df = df[trained_features]  

    # ✅ Compute decision score FIRST (safe operation)
    df["decision_score"] = iso_forest.decision_function(df)

    # 🔍 Predict anomalies (ensure only trained features are passed)
    df["anomaly"] = iso_forest.predict(df[trained_features])
    df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})  # Convert to binary (1 = anomaly)

    # ✅ Business Rule-Based Anomaly Flagging
    df["business_rule_anomaly"] = 0  # Default normal

    # Rule 1: Age 0-17 with income > 0 (not needed as dataset has 65+ but kept for generalization)
    df.loc[(df["age"] < 18) & (df["income"] > 0), "business_rule_anomaly"] = 1

    # Rule 2: Unique procedures > total procedures
    df.loc[df["unique_procedures"] > df["total_procedures_count"], "business_rule_anomaly"] = 1

    # Rule 3: Total transactions = 0 or extremely high
    high_txn_threshold = df["total_counts"].quantile(0.995)
    df.loc[(df["unique_procedures"] > df["total_procedures_count"] * 1.2), "business_rule_anomaly"] = 1  # Allow 20% buffer

    # ✅ Combine Business Rules & Model Anomalies
    decision_score_threshold = df["decision_score"].quantile(0.025)  # Bottom 2.5% only
    df["final_anomaly"] = np.where(
        (df["anomaly"] == 1) | 
        ((df["business_rule_anomaly"] == 1) & (df["decision_score"] < decision_score_threshold)), 
        1, 
        0
    )

    return df




# ---- CHECK LOGIN STATUS ----
if not st.session_state.authenticated:
    login()
else:
    # ---- LOAD MODELS ----
    with open("classification_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    with open("isolation_forest_model.pkl", "rb") as file:
        iso_forest = pickle.load(file)

    # ---- SET APP CONFIG ----
    st.set_page_config(page_title="Anomaly Detection App", page_icon="🩺", layout="centered")
    st.title("💡 Medicare Anomaly Detection")
    st.markdown("Enter patient details below to predict potential anomalies in Medicare claims.")

    # ---- INPUT FORM ----
    with st.form(key='patient_form'):
        age = st.number_input("Age", min_value=65, max_value=120, step=1)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        income = st.number_input("Income", min_value=0, step=1000)
        unique_procedures = st.number_input("Unique Procedures", min_value=0, step=1)
        total_procedures_count = st.number_input("Total Procedures Count", min_value=0, step=1)
        total_counts = st.number_input("Total Number of Transactions", min_value=0, step=1)

        submit = st.form_submit_button("Predict Anomaly")

    if submit:
        # ---- PREPARE INPUT ----
        input_data = np.array([[age, gender, income, unique_procedures, total_procedures_count, total_counts]])
        
        # Compute high transaction threshold (can be precomputed if needed)
        high_txn_threshold = 100  # Adjust based on your dataset

        # Compute anomaly-related features
        computed_data = compute_anomaly_features(input_data, iso_forest, high_txn_threshold)

        # Extract decision score and anomaly flag
        decision_score = computed_data['decision_score'].values[0]
        anomaly_flag = computed_data['anomaly'].values[0]

        # Drop computed columns before prediction
        input_for_model = computed_data.drop(columns=['decision_score', 'anomaly', 'business_rule_anomaly'])

        # Predict using the classification model
        prediction = model.predict(input_for_model)

        # ---- RESULTS ----
        if prediction[0] == 1:
            st.error(f"⚠️ Anomaly Detected!\n\nDecision Score: {decision_score:.5f}")
        else:
            st.success(f"✅ Normal Case.\n\nDecision Score: {decision_score:.5f}")

        st.info("ℹ️ *Lower decision scores may indicate potential anomalies.*")
