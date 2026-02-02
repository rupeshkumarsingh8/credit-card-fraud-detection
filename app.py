import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# ===============================
# Load Model & Scaler
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("model_xgb.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ===============================
# Title
# ===============================
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown(
    "Detect fraudulent credit card transactions using a trained **XGBoost model** with a configurable decision threshold."
)

# ===============================
# Sidebar Controls
# ===============================
st.sidebar.header("⚙️ Model Settings")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.05,
    max_value=0.9,
    value=0.2,
    step=0.05
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ===============================
# Load Dataset
# ===============================
if uploaded_file is None:
    st.warning("⬅️ Please upload a CSV file to start fraud detection.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ===============================
# Dataset Preview
# ===============================
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# ===============================
# Preprocessing
# ===============================
if "Class" in df.columns:
    X = df.drop("Class", axis=1)
else:
    X = df.copy()

# Scale only Time & Amount
X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

# ===============================
# Predictions
# ===============================
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# ===============================
# Results DataFrame
# ===============================
df_results = df.copy()
df_results["Fraud_Probability"] = y_prob
df_results["Prediction"] = y_pred

# ===============================
# Fraud Metrics
# ===============================
total_frauds = (df_results["Prediction"] == 1).sum()
total_txns = len(df_results)
fraud_rate = (total_frauds / total_txns) * 100

col1, col2, col3 = st.columns(3)

col1.metric("🚨 Fraud Transactions Detected", total_frauds)
col2.metric("📊 Total Transactions", total_txns)
col3.metric("⚠️ Fraud Rate (%)", f"{fraud_rate:.2f}%")

st.markdown("---")

# ===============================
# Display ALL Fraud Transactions
# ===============================
st.subheader("🚨 All Detected Fraud Transactions")

fraud_df = df_results[df_results["Prediction"] == 1]

if fraud_df.empty:
    st.success("🎉 No frauds detected at this threshold.")
else:
    st.dataframe(
        fraud_df,
        use_container_width=True,
        height=500
    )

    # Download button
    st.download_button(
        label="⬇️ Download Detected Frauds (CSV)",
        data=fraud_df.to_csv(index=False),
        file_name="detected_frauds.csv",
        mime="text/csv"
    )

st.markdown("---")

# ===============================
# Confusion Matrix (only if labels exist)
# ===============================
if "Class" in df.columns:
    st.subheader("🔁 Confusion Matrix")

    cm = confusion_matrix(df["Class"], y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ===============================
# Footer Note
# ===============================
st.markdown(
    "📌 **Note:** Lower threshold → higher fraud detection (higher recall) but more false alerts."
)
