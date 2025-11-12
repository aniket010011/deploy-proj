import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

# =============================================================
# Page Setup
# =============================================================
st.set_page_config(page_title="EMI Prediction App", layout="wide")
st.title("💰 EMI Prediction & Analysis App")

# =============================================================
# Preprocessor setup (for pipelines)
# =============================================================
categorical_features = ["gender", "marital_status", "education", "employment_type", "existing_loans"]
numeric_features = ["age", "monthly_salary", "years_of_employment", "requested_amount", "requested_tenure",
                    "current_emi_amount", "credit_score", "bank_balance", "emergency_fund"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# =============================================================
# Classification Pipelines
# =============================================================
classification_pipelines = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Random Forest Classifier": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
    "XGBoost Classifier": Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42
        ))
    ])
}

# =============================================================
# Regression Pipelines (dummy / example)
# =============================================================
regression_pipelines = {
    "Linear Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]),
    "Random Forest Regressor": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42))
    ]),
    "XGBoost Regressor": Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42
        ))
    ]),
    "LightGBM Regressor": Pipeline([
        ("preprocessor", preprocessor),
        ("model", lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42))
    ])
}

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

    classifier_choice = st.selectbox(
        "Select Classification Model",
        list(classification_pipelines.keys())
    )

    st.subheader("Enter Applicant Details")

    age = st.number_input("Age", min_value=18, max_value=75, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education = st.selectbox("Education", ["High School", "Graduate", "Post-Graduate", "Doctorate"])
    monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, step=1000)
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Freelancer"])
    years_of_employment = st.number_input("Years of Employment", min_value=0.0, step=0.5)
    requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, step=1000)
    requested_tenure = st.number_input("Requested Tenure (Months)", min_value=6, max_value=360, step=6)

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "monthly_salary": monthly_salary,
        "employment_type": employment_type,
        "years_of_employment": years_of_employment,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "current_emi_amount": 0,
        "credit_score": 0,
        "bank_balance": 0,
        "emergency_fund": 0,
        "existing_loans": "No"
    }])

    if st.button("Predict Eligibility"):
        model = classification_pipelines[classifier_choice]
        # Here we use a dummy prediction as the model is not trained
        st.info("⚠️ Models are defined but not trained. Prediction is simulated.")
        st.success("✅ Simulated Prediction: Eligible")

# =============================================================
# 💵 Regression: Max Monthly EMI Prediction
# =============================================================
elif menu == "💵 Max EMI Prediction (Regression)":
    st.header("💵 Maximum Monthly EMI Prediction")

    reg_choice = st.selectbox(
        "Select Regression Model",
        list(regression_pipelines.keys())
    )

    st.subheader("Enter Applicant Financial Details")

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
        "existing_loans": existing_loans,
        "age": 30,
        "gender": "Male",
        "marital_status": "Single",
        "education": "Graduate",
        "employment_type": "Salaried",
        "years_of_employment": 5,
        "requested_amount": 100000,
        "requested_tenure": 12
    }])

    if st.button("Predict Max EMI"):
        model = regression_pipelines[reg_choice]
        st.info("⚠️ Models are defined but not trained. Prediction is simulated.")
        st.success(f"💵 Simulated Maximum Affordable EMI: ₹50,000")
