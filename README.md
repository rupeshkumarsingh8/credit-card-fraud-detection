# 💳 Credit Card Fraud Detection using Machine Learning

This project detects fraudulent credit card transactions using supervised machine learning models and an interactive Streamlit dashboard.

---

## 🚀 Features
- Handles highly imbalanced fraud data
- Compares Logistic Regression, Random Forest, and XGBoost
- Threshold tuning to maximize fraud recall
- Interactive Streamlit dashboard
- Real-world, business-driven evaluation metrics

---

## 🧠 Dataset
- European credit card transaction dataset
- PCA-transformed features (V1–V28)
- Highly imbalanced (fraud ≈ 0.17%)

---

## 📊 Model Performance
| Model | Fraud Recall |
|------|-------------|
| Logistic Regression | ~74% |
| Random Forest | ~85% |
| XGBoost (Final) | **~89%** |

---

## 🖥️ Streamlit Dashboard
- Adjustable decision threshold
- Confusion matrix visualization
- Fraud probability output

---

## ⚙️ Tech Stack
- Python
- scikit-learn
- XGBoost
- Pandas, NumPy
- Streamlit
- Matplotlib, Seaborn

---

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
