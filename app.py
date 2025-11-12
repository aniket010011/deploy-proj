import streamlit as st
import sklearn.compose._column_transformer as ct

# --- Patch sklearn version mismatch issue ---
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
# Utility: Ensure all columns expected by model are present
# =============================================================
def align_columns(input_df, model):
    """Add missing columns (as NaN/'Unknown') and drop unexpected ones."""
    input_df = input_df.copy()

    # Try to get expected feature names
    try:
        expected_cols = model.feature_names_in_
    except AttributeError:
        try:
            expected_cols = model.named_steps['preprocessor'].get_feature_names_out()
        except Exception:
            expected_cols = input_df.columns  # fallback

    # Ensure all expected columns exist
    for col in expected_cols:
        if col not in input_df.columns:
            # Use 'Unknown' for object-type features, np.nan otherwise
            input_df[col] = "Unknown"

    # Keep only expected columns
    input_df = input_df.reindex(columns=expected_cols, fill_value="Unknown")

    # Convert numeric-like columns to numeric (avoiding str nan)
    for col in input_df.columns:
        # If all values look numeric or nan
        if input_df[col].dtype == object:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except Exception:
                pass  # keep as string if fails

    return input_df
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
        missing = missi

