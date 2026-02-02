import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# ------------------ Load Model ------------------
model = joblib.load("model_xgb.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------ Title ------------------
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown(
    "Detect fraudulent transactions using Machine Learning with adjustable decision threshold."
)

# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Model Settings")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.05,
    max_value=0.9,
    value=0.3,
    step=0.05
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ------------------ Load Data ------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# ------------------ Data Preview ------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ------------------ Preprocessing ------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Time & Amount
X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])

# ------------------ Prediction ------------------
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# ------------------ Metrics ------------------
st.subheader("📊 Model Performance")

report = classification_report(y, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

col1, col2, col3 = st.columns(3)

col1.metric("Fraud Recall", f"{df_report.loc['1','recall']:.2f}")
col2.metric("Fraud Precision", f"{df_report.loc['1','precision']:.2f}")
col3.metric("Fraud F1-Score", f"{df_report.loc['1','f1-score']:.2f}")

# ------------------ Confusion Matrix ------------------
st.subheader("🔁 Confusion Matrix")

cm = confusion_matrix(y, y_pred)

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

# ------------------ Prediction Output ------------------
st.subheader("🚨 Fraud Predictions")

df_results = df.copy()
df_results["Fraud_Probability"] = y_prob
df_results["Prediction"] = y_pred

st.dataframe(df_results.head(20))

# ------------------ Footer ------------------
st.markdown("---")
st.markdown("📌 **Note:** Lower threshold → higher fraud detection but more false alerts.")
