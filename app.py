import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load trained models
# ----------------------------

# Classification models
lgr_model= joblib.load('logistic_regression_model.pkl')
rfc_model = joblib.load('random_forest_classifier_model.pkl')
xgbc_model = joblib.load('xgboost_classifier_model.pkl')

# Regression models
lnr_model = joblib.load('linear_regression_model.pkl')
rfr_model = joblib.load('random_forest_model.pkl')
xgbr_model = joblib.load('xgboost_model.pkl')
lgbr_model = joblib.load('lgbm_model.pkl')

# ----------------------------
# App title
# ----------------------------
st.set_page_config(page_title="EMI Prediction System", page_icon="💰", layout="wide")
st.title("💰 Financial Assessment and EMI Prediction App")
st.write("""
This app conducts exploratory data analysis on the emi_prediction_dataset and predicts the Maximum Monthly EMI for a customer and if the customer is eligible for  
it based on their financial and demographic details.
""")

# ----------------------------
# Sidebar navigation
# ----------------------------
task = st.sidebar.radio("Select Prediction Task", ["EMI Eligibility (Classification)", "Max EMI (Regression)"])

# ----------------------------
# Input Section (common for both)
# ----------------------------
st.header("🔹 Enter Applicant Details")

age = st.number_input("Age", 18, 80, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
monthly_salary = st.number_input("Monthly Salary (₹)", 0, 1000000, 50000)
employment_type = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Contract", "Other"])
years_of_employment = st.slider("Years of Employment", 0, 40, 5)
company_type = st.selectbox("Company Type", ["Private", "Public", "Government", "Startup", "Other"])
house_type = st.selectbox("House Type", ["Owned", "Rented", "Family", "Company Provided"])
monthly_rent = st.number_input("Monthly Rent (₹)", 0, 200000, 10000)
family_size = st.slider("Family Size", 1, 10, 3)
dependents = st.slider("Dependents", 0, 5, 1)
school_fees = st.number_input("School Fees (₹)", 0, 100000, 5000)
college_fees = st.number_input("College Fees (₹)", 0, 200000, 0)
travel_expenses = st.number_input("Travel Expenses (₹)", 0, 50000, 3000)
groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 100000, 8000)
other_monthly_expenses = st.number_input("Other Monthly Expenses (₹)", 0, 100000, 2000)
existing_loans = st.number_input("Existing Loans", 0, 10, 1)
current_emi_amount = st.number_input("Current EMI Amount (₹)", 0, 200000, 5000)
credit_score = st.number_input("Credit Score", 300, 900, 700)
bank_balance = st.number_input("Bank Balance (₹)", 0, 2000000, 100000)
emergency_fund = st.number_input("Emergency Fund (₹)", 0, 200000, 20000)
emi_scenario = st.selectbox("EMI Scenario", ["Low", "Medium", "High"])
requested_amount = st.number_input("Requested Loan Amount (₹)", 0, 5000000, 500000)
requested_tenure = st.slider("Requested Tenure (Months)", 6, 120, 60)

# ----------------------------
# Prepare input dataframe
# ----------------------------
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'marital_status': [marital_status],
    'education': [education],
    'monthly_salary': [monthly_salary],
    'employment_type': [employment_type],
    'years_of_employment': [years_of_employment],
    'company_type': [company_type],
    'house_type': [house_type],
    'monthly_rent': [monthly_rent],
    'family_size': [family_size],
    'dependents': [dependents],
    'school_fees': [school_fees],
    'college_fees': [college_fees],
    'travel_expenses': [travel_expenses],
    'groceries_utilities': [groceries_utilities],
    'other_monthly_expenses': [other_monthly_expenses],
    'existing_loans': [existing_loans],
    'current_emi_amount': [current_emi_amount],
    'credit_score': [credit_score],
    'bank_balance': [bank_balance],
    'emergency_fund': [emergency_fund],
    'emi_scenario': [emi_scenario],
    'requested_amount': [requested_amount],
    'requested_tenure': [requested_tenure]
})

# Encode categorical columns
categorical_cols = ['gender', 'marital_status', 'education', 'employment_type',
                    'company_type', 'house_type', 'emi_scenario']
for col in categorical_cols:
    input_data[col] = input_data[col].astype('category').cat.codes

# ----------------------------
# Classification Task
# ----------------------------
if task == "EMI Eligibility (Classification)":
    st.subheader("🏦 EMI Eligibility Prediction")

    if st.button("🔮 Predict EMI Eligibility"):
        # Predict using all classification models
        pred_lr = lr_clf.predict(input_data)[0]
        pred_rf = rf_clf.predict(input_data)[0]
        pred_xgb = xgb_clf.predict(input_data)[0]

        st.write("### 🔍 Model Predictions:")
        st.write(f"**Logistic Regression:** {pred_lr}")
        st.write(f"**Random Forest:** {pred_rf}")
        st.write(f"**XGBoost:** {pred_xgb}")

        # Compute majority vote
        preds = [pred_lr, pred_rf, pred_xgb]
        final_pred = max(set(preds), key=preds.count)
        st.success(f"💡 Final EMI Eligibility Decision: **{final_pred}**")

# ----------------------------
# Regression Task
# ----------------------------
elif task == "Max EMI (Regression)":
    st.subheader("📈 Max Monthly EMI Prediction")

    if st.button("💰 Predict Max EMI"):
        pred_lr = lr_reg.predict(input_data)[0]
        pred_rf = rf_reg.predict(input_data)[0]
        pred_xgb = xgb_reg.predict(input_data)[0]
        pred_lgb = lgb_reg.predict(input_data)[0]

        st.write("### 🔍 Model Predictions (₹):")
        st.write(f"**Linear Regression:** ₹{pred_lr:,.2f}")
        st.write(f"**Random Forest:** ₹{pred_rf:,.2f}")
        st.write(f"**XGBoost:** ₹{pred_xgb:,.2f}")
        st.write(f"**LightGBM:** ₹{pred_lgb:,.2f}")

        avg_pred = np.mean([pred_lr, pred_rf, pred_xgb, pred_lgb])
        st.success(f"💡 Recommended Max EMI: **₹{avg_pred:,.2f}**")