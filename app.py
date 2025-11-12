import streamlit as st
import sklearn.compose._column_transformer as ct
if not hasattr(ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    ct._RemainderColsList = _RemainderColsList
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, r2_score, mean_absolute_error, mean_squared_error

# =============================================================
# Page Setup
# =============================================================
st.set_page_config(page_title="EMI Prediction App", layout="wide")
st.title("💰 EMI Prediction & Analysis App")

# =============================================================
# Load Models and Label Encoder
# =============================================================
@st.cache_resource
def load_models():
    models = {}
    try:
        models["Logistic Regression"] = joblib.load("logistic_regression_pipeline.pkl")
        models["XGBoost Classifier"] = joblib.load("xgboost_classifier_pipeline.pkl")
        models["LGBM Classifier"] = joblib.load("lgbm_classifier_pipeline.pkl")
        models["Linear Regression"] = joblib.load("linear_regression_pipeline.pkl")
        models["XGBoost Regressor"] = joblib.load("xgboost_regressor_pipeline.pkl")
        models["LGBM Regressor"] = joblib.load("lightgbm_regressor_pipeline.pkl")
        label_encoder = joblib.load("emi_label_encoder.pkl")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        models, label_encoder = None, None
    return models, label_encoder


models, label_encoder = load_models()

# =============================================================
# Sidebar Menu
# =============================================================
menu = st.sidebar.radio(
    "Select Option:",
    ["📊 EDA", "🎯 EMI Eligibility Prediction (Classification)", "💵 Max EMI Prediction (Regression)"]
)

# =============================================================
# 📊 Exploratory Data Analysis
# =============================================================
if menu == "📊 EDA":
    st.header("📊 Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        st.write("**Summary Statistics**")
        st.write(df.describe(include='all').T)

        # Missing values
        st.write("### Missing Values (%)")
        missing = df.isnull().mean() * 100
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            fig, ax = plt.subplots()
            sns.barplot(x=missing.values, y=missing.index, ax=ax)
            plt.xlabel("Percentage Missing")
            st.pyplot(fig)
        else:
            st.success("No missing values found!")

        # Correlation heatmap
        st.write("### Correlation Heatmap (Numeric Columns)")
        num_df = df.select_dtypes(include=np.number)
        if not num_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(num_df.corr(), cmap="coolwarm", annot=False, ax=ax)
            st.pyplot(fig)

        # Gender distribution (if column exists)
        if "gender" in df.columns:
            st.write("### Gender Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x="gender", ax=ax)
            st.pyplot(fig)

# =============================================================
# 🎯 Classification: EMI Eligibility
# =============================================================
elif menu == "🎯 EMI Eligibility Prediction (Classification)":
    st.header("🎯 EMI Eligibility Prediction")

    if models is None:
        st.warning("Models not found. Please ensure model files are in the same directory.")
    else:
        classifier_choice = st.selectbox(
            "Select Classification Model",
            ["Logistic Regression", "XGBoost Classifier", "LightGBM Classifier"]
        )

        st.subheader("Enter Applicant Details")

        # Simple form — adjust based on your dataset
        age = st.number_input("Age", min_value=18, max_value=75, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post-Graduate", "Doctorate"])
        monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, step=1000)
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer"])
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, step=0.5)
        requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, step=1000)
        requested_tenure = st.number_input("Requested Tenure (Months)", min_value=6, max_value=360, step=6)

        # Create DataFrame for model input
        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary,
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure
        }])

        if st.button("Predict Eligibility"):
            model = models[classifier_choice]
            pred_encoded = model.predict(input_data)[0]
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]

            if pred_label == "Eligible":
                st.success("✅ The applicant is Eligible for EMI.")
            elif pred_label == "Not_Eligible":
                st.warning("⚠️ The applicant is Not Eligible for EMI.")
            else:
                st.error("🚨 The applicant is High Risk.")

# =============================================================
# 💵 Regression: Max Monthly EMI Prediction
# =============================================================
elif menu == "💵 Max EMI Prediction (Regression)":
    st.header("💵 Maximum Monthly EMI Prediction")

    if models is None:
        st.warning("Models not found. Please ensure model files are in the same directory.")
    else:
        reg_choice = st.selectbox(
            "Select Regression Model",
            ["Linear Regression", "XGBoost Regressor", "LightGBM Regressor"]
        )

        st.subheader("Enter Applicant Financial Details")

        # Example subset of inputs — extend as needed
        monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, step=1000)
        current_emi_amount = st.number_input("Current EMI Amount (₹)", min_value=0, step=1000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=10)
        bank_balance = st.number_input("Bank Balance (₹)", min_value=0, step=1000)
        emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, step=1000)
        existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])

        input_data = pd.DataFrame([{
            "monthly_salary": monthly_salary,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "existing_loans": existing_loans
        }])

        if st.button("Predict Max EMI"):
            model = models[reg_choice]
            pred = model.predict(input_data)[0]
            st.success(f"💵 Estimated Maximum Affordable EMI: ₹{pred:,.2f}")




