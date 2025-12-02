import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
import os

st.set_page_config(page_title="Fraud Detection", layout="wide")

@st.cache_resource
def load_model():
    data = joblib.load("fraud_model_final.pkl")
    return data["model"], data["X_test"], data["y_test"], data["threshold"]

model, X_test, y_test, threshold = load_model()
feature_columns = X_test.columns

with st.sidebar:
    page = st.radio("Navigation", ["🏠 Dashboard", "🔍 Single Prediction", "📊 Batch Prediction", "🎯 Model Performance"])

# DASHBOARD
if page == "🏠 Dashboard":
    st.title("🛡️ Credit Card Fraud Detection System")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Test Samples", len(y_test))
    col2.metric("Normal", (y_test == 0).sum())
    col3.metric("Fraud", (y_test == 1).sum())
    col4.metric("Fraud Rate", f"{(y_test.mean()*100):.3f}%")

    fig = px.bar(x=["Normal", "Fraud"], y=y_test.value_counts().sort_index())
    st.plotly_chart(fig, use_container_width=True)

# SINGLE PREDICTION
elif page == "🔍 Single Prediction":
    st.title("Single Transaction Prediction")
    values = [st.number_input(col, value=0.0) for col in feature_columns]
    if st.button("Predict"):
        proba = model.predict_proba(np.array(values).reshape(1,-1))[0][1]
        pred = 1 if proba >= threshold else 0
        st.write("Prediction:", "🚨 Fraud" if pred else "✔ Normal")
        st.write("Probability:", round(proba,4))

# BATCH
elif page == "📊 Batch Prediction":
    st.title("Batch Prediction")
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        data = pd.read_csv(file)
        probs = model.predict_proba(data[feature_columns])[:,1]
        preds = (probs >= threshold).astype(int)
        data["Prediction"] = preds
        data["Probability"] = probs
        st.dataframe(data)

# PERFORMANCE
elif page == "🎯 Model Performance":
    st.title("Model Performance")
    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_proba >= threshold).astype(int)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
    col5.metric("ROC-AUC", f"{roc_auc_score(y_test, y_proba):.4f}")
    col6.metric("PR-AUC", f"{average_precision_score(y_test, y_proba):.4f}")

    st.write("Confusion Matrix:")
    st.dataframe(confusion_matrix(y_test, y_pred))
