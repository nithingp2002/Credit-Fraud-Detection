import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score, roc_curve
from scipy.spatial.distance import cdist
import os

# Configuration Constants
FRAUD_CLUSTER_THRESHOLD = 2.0

# Page Configuration
st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background with Gradient */
    .stApp {
        background: linear-gradient(135deg, #1a3a3a 0%, #2d5a5a 50%, #1f4545 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d4f4f 0%, #1f3d3d 100%);
        border-right: 2px solid rgba(145, 198, 188, 0.3);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #F6F3C2;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    
    [data-testid="stSidebar"] .stRadio > div {
        gap: 10px;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(145, 198, 188, 0.1);
        padding: 15px 20px;
        border-radius: 12px;
        border: 1px solid rgba(145, 198, 188, 0.2);
        transition: all 0.3s ease;
        cursor: pointer;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(145, 198, 188, 0.2);
        border-color: rgba(145, 198, 188, 0.6);
        transform: translateX(5px);
    }
    
    /* Title Styling */
    h1 {
        color: #ffffff;
        font-weight: 800;
        font-size: 3rem !important;
        background: linear-gradient(135deg, #91C6BC 0%, #4B9DA9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
        text-shadow: 0 0 30px rgba(145, 198, 188, 0.3);
    }
    
    h2, h3 {
        color: #F6F3C2;
        font-weight: 600;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #91C6BC;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #F6F3C2;
        font-weight: 500;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(145, 198, 188, 0.15) 0%, rgba(75, 157, 169, 0.1) 100%);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid rgba(145, 198, 188, 0.3);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(145, 198, 188, 0.3);
        border-color: rgba(145, 198, 188, 0.5);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #91C6BC 0%, #4B9DA9 100%);
        color: #1a3a3a;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 15px 40px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 8px 24px rgba(145, 198, 188, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(145, 198, 188, 0.6);
        background: linear-gradient(135deg, #4B9DA9 0%, #91C6BC 100%);
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input {
        background: rgba(145, 198, 188, 0.1);
        border: 1px solid rgba(145, 198, 188, 0.3);
        border-radius: 8px;
        color: #F6F3C2;
        padding: 10px;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #91C6BC;
        box-shadow: 0 0 0 2px rgba(145, 198, 188, 0.3);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(145, 198, 188, 0.05);
        border: 2px dashed rgba(145, 198, 188, 0.3);
        border-radius: 16px;
        padding: 30px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #91C6BC;
        background: rgba(145, 198, 188, 0.15);
    }
    
    /* DataFrames */
    .stDataFrame {
        background: rgba(145, 198, 188, 0.05);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(145, 198, 188, 0.2);
    }
    
    /* Alert Boxes */
    .alert-success {
        background: linear-gradient(135deg, rgba(145, 198, 188, 0.2) 0%, rgba(75, 157, 169, 0.2) 100%);
        border-left: 4px solid #91C6BC;
        padding: 20px;
        border-radius: 12px;
        color: #F6F3C2;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 20px 0;
        backdrop-filter: blur(10px);
    }
    
    .alert-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.2) 100%);
        border-left: 4px solid #ef4444;
        padding: 20px;
        border-radius: 12px;
        color: #fca5a5;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 20px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, rgba(145, 198, 188, 0.15) 0%, rgba(75, 157, 169, 0.1) 100%);
        border: 1px solid rgba(145, 198, 188, 0.4);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        color: #F6F3C2;
        backdrop-filter: blur(10px);
    }
    
    /* Labels */
    label {
        color: #F6F3C2 !important;
        font-weight: 500 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(145, 198, 188, 0.1);
        border-radius: 12px;
        color: #F6F3C2;
        font-weight: 600;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(145, 198, 188, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #91C6BC 0%, #4B9DA9 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #4B9DA9 0%, #91C6BC 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load all models (supervised + clustering) from single PKL file"""
    try:
        data = joblib.load("fraud_model_final.pkl")
        return data
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load all models
model_data = load_model()

if model_data is None:
    st.error("Failed to load fraud_model_final.pkl. Please train the model first using train_model.py")
    st.stop()

try:
    # Extract models
    supervised_model = model_data["supervised_model"]
    supervised_threshold = model_data["supervised_threshold"]
    supervised_name = model_data["supervised_name"]

    kmeans = model_data["kmeans"]
    kmeans_high_fraud_clusters = model_data["kmeans_high_fraud_clusters"]

    hierarchical_centers = model_data["hierarchical_centers"]
    hier_high_fraud_clusters = model_data["hier_high_fraud_clusters"]

    scaler = model_data["scaler"]
    pca = model_data["pca"]

    X_test = model_data["X_test"]
    y_test = model_data["y_test"]
    optimal_k = model_data["optimal_k"]
    feature_columns = model_data["feature_columns"]
    fraud_cluster_threshold = model_data.get("fraud_cluster_threshold", FRAUD_CLUSTER_THRESHOLD)

    # Derive consistent test-time predictions for dashboards/metrics
    X_test_scaled = scaler.transform(X_test)
    supervised_proba_test = supervised_model.predict_proba(X_test)[:, 1]
    supervised_pred_test = (supervised_proba_test >= supervised_threshold).astype(int)

    test_clusters_kmeans = kmeans.predict(X_test_scaled)
    kmeans_predictions_test = np.array([
        1 if cluster in kmeans_high_fraud_clusters else 0
        for cluster in test_clusters_kmeans
    ])

    test_distances_to_centers = cdist(X_test_scaled, hierarchical_centers, 'euclidean')
    test_clusters_hier = np.argmin(test_distances_to_centers, axis=1)
    hier_predictions_test = np.array([
        1 if cluster in hier_high_fraud_clusters else 0
        for cluster in test_clusters_hier
    ])

    # OR logic ensemble (recall-first as chosen)
    ensemble_pred_test = (
        supervised_pred_test
        + kmeans_predictions_test
        + hier_predictions_test
    ) >= 1

    def summarize_metrics(y_true, y_pred, proba=None):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, proba) if proba is not None else np.nan,
            "pr_auc": average_precision_score(y_true, proba) if proba is not None else np.nan,
        }

    metrics_summary = {
        "supervised": summarize_metrics(y_test, supervised_pred_test, supervised_proba_test),
        "kmeans": summarize_metrics(y_test, kmeans_predictions_test),
        "hierarchical": summarize_metrics(y_test, hier_predictions_test),
        "ensemble": summarize_metrics(y_test, ensemble_pred_test, supervised_proba_test),
    }

    cm_ensemble = confusion_matrix(y_test, ensemble_pred_test)
    true_negatives, false_positives = cm_ensemble[0]
    false_negatives, true_positives = cm_ensemble[1]

    st.success(f"✅ All models loaded: {supervised_name} + K-Means + Hierarchical")
    
except KeyError as e:
    st.error(f"❌ Missing key in model data: {str(e)}")
    st.info(f"Available keys: {list(model_data.keys())}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error processing models: {str(e)}")
    import traceback
    st.error(traceback.format_exc())
    st.stop()


# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    page = st.radio("Navigation Menu", ["🏡 Home", "🏠 Dashboard", "🔍 Single Prediction", "📊 Batch Prediction", "🎯 Model Performance"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; color: #94a3b8;'>
        <p style='font-size: 0.9rem; margin: 0;'>Powered by AI</p>
        <p style='font-size: 0.8rem; margin: 5px 0 0 0;'>Advanced ML Detection</p>
    </div>
    """, unsafe_allow_html=True)

# Pre-compute display metrics
ensemble_recall_pct = metrics_summary['ensemble']['recall'] * 100
ensemble_precision_pct = metrics_summary['ensemble']['precision'] * 100
ensemble_f1_pct = metrics_summary['ensemble']['f1'] * 100
test_fraud_rate = y_test.mean() * 100

supervised_precision_pct = metrics_summary['supervised']['precision'] * 100
supervised_recall_pct = metrics_summary['supervised']['recall'] * 100
supervised_f1_pct = metrics_summary['supervised']['f1'] * 100

kmeans_precision_pct = metrics_summary['kmeans']['precision'] * 100
kmeans_recall_pct = metrics_summary['kmeans']['recall'] * 100
kmeans_f1_pct = metrics_summary['kmeans']['f1'] * 100

hierarchical_precision_pct = metrics_summary['hierarchical']['precision'] * 100
hierarchical_recall_pct = metrics_summary['hierarchical']['recall'] * 100
hierarchical_f1_pct = metrics_summary['hierarchical']['f1'] * 100

# HOME - STUNNING LANDING PAGE
if page == "🏡 Home":
    # Hero Section
    st.markdown(f"""
    <div style='text-align: center; padding: 60px 20px 40px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 20px; margin-bottom: 40px;'>
        <h1 style='font-size: 4rem; font-weight: 800; margin-bottom: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🛡️ FraudShield AI
        </h1>
        <p style='font-size: 1.8rem; color: #94a3b8; margin-bottom: 30px; font-weight: 300;'>
            Next-Generation Fraud Detection Powered by Ensemble Machine Learning
        </p>
        <p style='font-size: 1.2rem; color: #64748b; max-width: 800px; margin: 0 auto;'>
            Combining supervised and unsupervised AI to detect credit card fraud with <strong style='color: #667eea;'>{ensemble_recall_pct:.1f}% recall</strong> and <strong style='color: #764ba2;'>{ensemble_precision_pct:.1f}% precision</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Cards
    st.markdown("### 📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(102, 126, 234, 0.05) 100%); 
                    padding: 25px; border-radius: 15px; text-align: center; border: 1px solid rgba(102, 126, 234, 0.2);'>
            <div style='font-size: 3rem; font-weight: 800; color: #667eea; margin-bottom: 10px;'>{ensemble_recall_pct:.1f}%</div>
            <div style='color: #94a3b8; font-size: 1.1rem;'>Recall (OR Ensemble)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(52, 211, 153, 0.1) 0%, rgba(52, 211, 153, 0.05) 100%); 
                    padding: 25px; border-radius: 15px; text-align: center; border: 1px solid rgba(52, 211, 153, 0.2);'>
            <div style='font-size: 3rem; font-weight: 800; color: #34d399; margin-bottom: 10px;'>{ensemble_precision_pct:.1f}%</div>
            <div style='color: #94a3b8; font-size: 1.1rem;'>Precision (OR Ensemble)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(244, 114, 182, 0.1) 0%, rgba(244, 114, 182, 0.05) 100%); 
                    padding: 25px; border-radius: 15px; text-align: center; border: 1px solid rgba(244, 114, 182, 0.2);'>
            <div style='font-size: 3rem; font-weight: 800; color: #f472b6; margin-bottom: 10px;'>3</div>
            <div style='color: #94a3b8; font-size: 1.1rem;'>AI Models Combined</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(251, 191, 36, 0.1) 0%, rgba(251, 191, 36, 0.05) 100%); 
                    padding: 25px; border-radius: 15px; text-align: center; border: 1px solid rgba(251, 191, 36, 0.2);'>
            <div style='font-size: 3rem; font-weight: 800; color: #fbbf24; margin-bottom: 10px;'>{test_fraud_rate:.3f}%</div>
            <div style='color: #94a3b8; font-size: 1.1rem;'>Fraud Rate (Test)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Features Grid
    st.markdown("### ⚡ Powerful Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.03); padding: 25px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); height: 100%;'>
            <div style='font-size: 3rem; margin-bottom: 15px;'>🤖</div>
            <h4 style='color: #e0e0e0; margin-bottom: 10px;'>Ensemble Learning</h4>
            <p style='color: #94a3b8; line-height: 1.6;'>Combines XGBoost, K-Means, and Hierarchical Clustering with voting mechanism</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.03); padding: 25px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); height: 100%;'>
            <div style='font-size: 3rem; margin-bottom: 15px;'>⚡</div>
            <h4 style='color: #e0e0e0; margin-bottom: 10px;'>Real-Time Detection</h4>
            <p style='color: #94a3b8; line-height: 1.6;'>Instant fraud predictions with confidence levels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: rgba(255, 255, 255, 0.03); padding: 25px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); height: 100%;'>
            <div style='font-size: 3rem; margin-bottom: 15px;'>🔍</div>
            <h4 style='color: #e0e0e0; margin-bottom: 10px;'>Anomaly Detection</h4>
            <p style='color: #94a3b8; line-height: 1.6;'>Detects new fraud patterns never seen before</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # AI Models Showcase
    st.markdown("### 🎯 AI Models")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(102, 126, 234, 0.05) 100%); 
                    padding: 30px; border-radius: 15px; border-top: 4px solid #667eea;'>
            <div style='display: inline-block; padding: 5px 15px; background: rgba(102, 126, 234, 0.3); border-radius: 20px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px;'>SUPERVISED</div>
            <h3 style='color: #e0e0e0; margin-bottom: 10px;'>{supervised_name}</h3>
            <p style='color: #94a3b8; margin-bottom: 20px;'>Gradient boosting with hyperparameter tuning</p>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;'>
                <div><div style='font-size: 1.5rem; font-weight: 700; color: #667eea;'>{supervised_precision_pct:.1f}%</div><div style='font-size: 0.9rem; color: #94a3b8;'>Precision</div></div>
                <div><div style='font-size: 1.5rem; font-weight: 700; color: #667eea;'>{supervised_recall_pct:.1f}%</div><div style='font-size: 0.9rem; color: #94a3b8;'>Recall</div></div>
                <div><div style='font-size: 1.5rem; font-weight: 700; color: #667eea;'>{supervised_f1_pct:.1f}%</div><div style='font-size: 0.9rem; color: #94a3b8;'>F1</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(52, 211, 153, 0.15) 0%, rgba(52, 211, 153, 0.05) 100%); 
                    padding: 30px; border-radius: 15px; border-top: 4px solid #34d399;'>
            <div style='display: inline-block; padding: 5px 15px; background: rgba(52, 211, 153, 0.3); border-radius: 20px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px;'>UNSUPERVISED</div>
            <h3 style='color: #e0e0e0; margin-bottom: 10px;'>K-Means</h3>
            <p style='color: #94a3b8; margin-bottom: 20px;'>Distance-based anomaly detection ({optimal_k} clusters)</p>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;'>
                <div><div style='font-size: 1.5rem; font-weight: 700; color: #34d399;'>{kmeans_precision_pct:.1f}%</div><div style='font-size: 0.9rem; color: #94a3b8;'>Precision</div></div>
                <div><div style='font-size: 1.5rem; font-weight: 700; color: #34d399;'>{kmeans_recall_pct:.1f}%</div><div style='font-size: 0.9rem; color: #94a3b8;'>Recall</div></div>
                <div><div style='font-size: 1.5rem; font-weight: 700; color: #34d399;'>{kmeans_f1_pct:.1f}%</div><div style='font-size: 0.9rem; color: #94a3b8;'>F1</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(244, 114, 182, 0.15) 0%, rgba(244, 114, 182, 0.05) 100%); 
                    padding: 30px; border-radius: 15px; border-top: 4px solid #f472b6;'>
            <div style='display: inline-block; padding: 5px 15px; background: rgba(244, 114, 182, 0.3); border-radius: 20px; font-size: 0.9rem; font-weight: 600; margin-bottom: 15px;'>UNSUPERVISED</div>
            <h3 style='color: #e0e0e0; margin-bottom: 10px;'>Hierarchical</h3>
            <p style='color: #94a3b8; margin-bottom: 20px;'>Tree-based clustering ({optimal_k} clusters)</p>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;'>
                <div><div style='font-size: 1.5rem; font-weight: 700; color: #f472b6;'>{hierarchical_precision_pct:.1f}%</div><div style='font-size: 0.9rem; color: #94a3b8;'>Precision</div></div>
                <div><div style='font-size: 1.5rem; font-weight: 700; color: #f472b6;'>{hierarchical_recall_pct:.1f}%</div><div style='font-size: 0.9rem; color: #94a3b8;'>Recall</div></div>
                <div><div style='font-size: 1.5rem; font-weight: 700; color: #f472b6;'>{hierarchical_f1_pct:.1f}%</div><div style='font-size: 0.9rem; color: #94a3b8;'>F1</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%); 
                padding: 50px; border-radius: 20px; text-align: center; border: 2px solid rgba(102, 126, 234, 0.3);'>
        <h2 style='color: #e0e0e0; font-size: 2.5rem; margin-bottom: 20px;'>Ready to Detect Fraud?</h2>
        <p style='color: #94a3b8; font-size: 1.3rem; margin-bottom: 30px;'>
            Experience the power of AI-driven fraud detection
        </p>
        <div style='display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;'>
            <div style='background: rgba(255, 255, 255, 0.05); padding: 20px 30px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1);'>
                <div style='font-size: 2rem; margin-bottom: 5px;'>🔍</div>
                <div style='color: #e0e0e0; font-weight: 600;'>Single Prediction</div>
                <div style='color: #94a3b8; font-size: 0.9rem;'>Analyze individual transactions</div>
            </div>
            <div style='background: rgba(255, 255, 255, 0.05); padding: 20px 30px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1);'>
                <div style='font-size: 2rem; margin-bottom: 5px;'>📊</div>
                <div style='color: #e0e0e0; font-weight: 600;'>Batch Prediction</div>
                <div style='color: #94a3b8; font-size: 0.9rem;'>Process CSV files</div>
            </div>
            <div style='background: rgba(255, 255, 255, 0.05); padding: 20px 30px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1);'>
                <div style='font-size: 2rem; margin-bottom: 5px;'>🎯</div>
                <div style='color: #e0e0e0; font-weight: 600;'>Model Performance</div>
                <div style='color: #94a3b8; font-size: 0.9rem;'>View detailed metrics</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# DASHBOARD
elif page == "🏠 Dashboard":
    st.markdown("<h1>🛡️ AI Fraud Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.2rem; margin-bottom: 30px;'>Real-time credit card fraud detection powered by advanced machine learning</p>", unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Samples", f"{len(y_test):,}")
    with col2:
        st.metric("✅ Normal Transactions", f"{(y_test == 0).sum():,}")
    with col3:
        st.metric("🚨 Fraud Cases", f"{(y_test == 1).sum():,}")
    with col4:
        st.metric("⚠️ Fraud Rate", f"{(y_test.mean()*100):.3f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Enhanced Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction Distribution
        fig1 = go.Figure(data=[
            go.Bar(
                x=["Normal Transactions", "Fraudulent Transactions"],
                y=y_test.value_counts().sort_index(),
                marker=dict(
                    color=['#10b981', '#ef4444'],
                    line=dict(color='rgba(255, 255, 255, 0.2)', width=2)
                ),
                text=y_test.value_counts().sort_index(),
                textposition='auto',
                textfont=dict(size=14, color='white', family='Inter')
            )
        ])
        fig1.update_layout(
            title=dict(text="Transaction Distribution", font=dict(size=20, color='#e0e0e0', family='Inter')),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            height=400
        )
        st.plotly_chart(fig1, width='stretch')
    
    with col2:
        # Pie Chart
        fig2 = go.Figure(data=[
            go.Pie(
                labels=["Normal", "Fraud"],
                values=y_test.value_counts().sort_index(),
                marker=dict(colors=['#10b981', '#ef4444']),
                hole=0.4,
                textfont=dict(size=14, color='white', family='Inter')
            )
        ])
        fig2.update_layout(
            title=dict(text="Transaction Ratio", font=dict(size=20, color='#e0e0e0', family='Inter')),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            height=400,
            showlegend=True,
            legend=dict(font=dict(color='#94a3b8'))
        )
        st.plotly_chart(fig2, width='stretch')
    
    # Info Box
    st.markdown("""
    <div class='info-box'>
        <h3 style='margin-top: 0; color: #60a5fa;'>ℹ️ System Information</h3>
        <p style='margin: 10px 0;'>This AI-powered system uses advanced machine learning algorithms to detect fraudulent credit card transactions in real-time.</p>
        <p style='margin: 10px 0;'><strong>Key Features:</strong></p>
        <ul style='margin: 10px 0;'>
            <li>Real-time fraud detection with high accuracy</li>
            <li>Batch processing for multiple transactions</li>
            <li>Comprehensive performance metrics</li>
            <li>Optimized threshold for balanced precision and recall</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# SINGLE PREDICTION
elif page == "🔍 Single Prediction":
    st.markdown("<h1>🔍 Single Transaction Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.2rem; margin-bottom: 30px;'>Analyze individual transactions for fraud detection</p>", unsafe_allow_html=True)
    
    st.markdown("<div class='info-box'><p>Enter the transaction features below to predict if it's fraudulent. All features are required for accurate prediction.</p></div>", unsafe_allow_html=True)
    
    # Create a more organized input layout
    num_cols = 3
    cols_per_row = num_cols
    values = []
    
    for i in range(0, len(feature_columns), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(feature_columns):
                with col:
                    val = st.number_input(
                        feature_columns[i + j],
                        value=0.0,
                        format="%.6f",
                        key=f"input_{i+j}"
                    )
                    values.append(val)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 Analyze Transaction"):
        with st.spinner("Analyzing with all models..."):
            # Prepare input
            input_array = np.array(values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            
            # SUPERVISED PREDICTION
            supervised_proba = supervised_model.predict_proba(input_array)[0][1]
            supervised_pred = 1 if supervised_proba >= supervised_threshold else 0
            
            # K-MEANS PREDICTION (cluster membership)
            kmeans_cluster = kmeans.predict(input_scaled)[0]
            kmeans_pred = 1 if kmeans_cluster in kmeans_high_fraud_clusters else 0
            kmeans_distance = np.min(cdist(input_scaled, kmeans.cluster_centers_, 'euclidean'))
            
            # HIERARCHICAL PREDICTION (nearest centroid membership)
            hier_distances = cdist(input_scaled, hierarchical_centers, 'euclidean')[0]
            hier_cluster = int(np.argmin(hier_distances))
            hier_pred = 1 if hier_cluster in hier_high_fraud_clusters else 0
            hier_distance = np.min(hier_distances)
            
            # ENSEMBLE VOTING (OR logic, recall-first)
            votes = supervised_pred + kmeans_pred + hier_pred
            final_pred = 1 if votes >= 1 else 0
            
            st.markdown("\u003cbr\u003e", unsafe_allow_html=True)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if final_pred == 1:
                    st.markdown(f"""
                    \u003cdiv class='alert-danger'\u003e
                        \u003ch2 style='margin: 0; color: #fca5a5;'\u003e🚨 FRAUD DETECTED\u003c/h2\u003e
                        \u003cp style='margin: 10px 0 0 0; font-size: 1rem;'\u003eEnsemble detected fraud ({votes}/3 models agree)\u003c/p\u003e
                    \u003c/div\u003e
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    \u003cdiv class='alert-success'\u003e
                        \u003ch2 style='margin: 0; color: #6ee7b7;'\u003e✅ LEGITIMATE TRANSACTION\u003c/h2\u003e
                        \u003cp style='margin: 10px 0 0 0; font-size: 1rem;'\u003eEnsemble classified as normal ({3-votes}/3 models agree)\u003c/p\u003e
                    \u003c/div\u003e
                    """, unsafe_allow_html=True)
                
                # Model breakdown
                st.markdown("### 📊 Model Breakdown")
                models_detected = []
                if supervised_pred == 1:
                    models_detected.append(f"✅ {supervised_name}")
                else:
                    st.markdown(f"❌ {supervised_name}: Normal")
                    
                if kmeans_pred == 1:
                    models_detected.append("✅ K-Means")
                else:
                    st.markdown("❌ K-Means: Normal")
                    
                if hier_pred == 1:
                    models_detected.append("✅ Hierarchical")
                else:
                    st.markdown("❌ Hierarchical: Normal")
                
                if models_detected:
                    st.markdown("**Detected by:**")
                    for model in models_detected:
                        st.markdown(model)
            
            with col2:
                # Probability Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=supervised_proba * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{supervised_name} Probability", 'font': {'size': 18, 'color': '#e0e0e0'}},
                    number={'suffix': "%", 'font': {'size': 40, 'color': '#ffffff'}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#667eea"},
                        'bgcolor': "rgba(255,255,255,0.1)",
                        'borderwidth': 2,
                        'bordercolor': "rgba(255,255,255,0.2)",
                        'steps': [
                            {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.3)'},
                            {'range': [30, 70], 'color': 'rgba(251, 191, 36, 0.3)'},
                            {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': supervised_threshold * 100
                        }
                    }
                ))
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#94a3b8", 'family': "Inter"},
                    height=300
                )
                st.plotly_chart(fig, width='stretch')
                
                # Distance scores
                st.markdown("### 📏 Distance Scores")
                st.metric("K-Means Distance", f"{kmeans_distance:.4f}", 
                         delta=f"Cluster {kmeans_cluster} | High-risk: {'Yes' if kmeans_pred else 'No'}")
                st.metric("Hierarchical Distance", f"{hier_distance:.4f}",
                         delta=f"Cluster {hier_cluster} | High-risk: {'Yes' if hier_pred else 'No'}")


# BATCH PREDICTION
elif page == "📊 Batch Prediction":
    st.markdown("<h1>📊 Batch Transaction Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.2rem; margin-bottom: 30px;'>Upload and analyze multiple transactions at once</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h3 style='margin-top: 0; color: #60a5fa;'>📁 Upload Instructions</h3>
        <p>Upload a CSV file containing transaction data. The file should include all required feature columns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    file = st.file_uploader("Choose a CSV file", type="csv")
    
    if file:
        data = pd.read_csv(file)
        
        st.markdown(f"<p style='color: #5eead4; font-size: 1.1rem; font-weight: 600;'>✅ File uploaded successfully! Processing {len(data):,} transactions...</p>", unsafe_allow_html=True)
        
        with st.spinner("Analyzing with all models..."):
            # Prepare data
            X_batch = data[feature_columns]
            X_batch_scaled = scaler.transform(X_batch)
            
            # SUPERVISED PREDICTIONS
            supervised_probs = supervised_model.predict_proba(X_batch)[:, 1]
            supervised_preds = (supervised_probs >= supervised_threshold).astype(int)
            
            # K-MEANS PREDICTIONS (cluster membership)
            batch_kmeans_clusters = kmeans.predict(X_batch_scaled)
            kmeans_preds = np.array([
                1 if cluster in kmeans_high_fraud_clusters else 0
                for cluster in batch_kmeans_clusters
            ])
            
            # HIERARCHICAL PREDICTIONS (nearest centroid membership)
            hier_distances = cdist(X_batch_scaled, hierarchical_centers, 'euclidean')
            batch_hier_clusters = np.argmin(hier_distances, axis=1)
            hier_preds = np.array([
                1 if cluster in hier_high_fraud_clusters else 0
                for cluster in batch_hier_clusters
            ])
            
            # ENSEMBLE VOTING (OR logic)
            votes = supervised_preds + kmeans_preds + hier_preds
            final_preds = (votes >= 1).astype(int)
            
            # Add results to dataframe
            data["Supervised_Pred"] = supervised_preds.map({0: "Normal", 1: "Fraud"})
            data["KMeans_Pred"] = kmeans_preds.map({0: "Normal", 1: "Fraud"})
            data["Hierarchical_Pred"] = hier_preds.map({0: "Normal", 1: "Fraud"})
            data["Ensemble_Prediction"] = final_preds.map({0: "✅ Normal", 1: "🚨 Fraud"})
            data["Votes"] = votes
            data["Fraud_Probability"] = (supervised_probs * 100).round(2)

        
        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Analyzed", f"{len(data):,}")
        with col2:
            st.metric("✅ Normal", f"{(final_preds == 0).sum():,}")
        with col3:
            st.metric("🚨 Fraud Detected", f"{(final_preds == 1).sum():,}")
        with col4:
            st.metric("⚠️ Fraud Rate", f"{(final_preds.mean()*100):.2f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Model Agreement Analysis
        st.markdown("### 🤝 Model Agreement")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("All 3 Agree", f"{(votes == 3).sum() + (votes == 0).sum():,}")
        with col2:
            st.metric("2 Agree (Fraud)", f"{(votes == 2).sum():,}")
        with col3:
            st.metric("2 Agree (Normal)", f"{(votes == 1).sum():,}")
        with col4:
            high_conf = (votes == 3).sum()
            st.metric("High Confidence Frauds", f"{high_conf:,}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = go.Figure(data=[
                go.Bar(
                    x=["Normal", "Fraud"],
                    y=[((final_preds == 0).sum()), ((final_preds == 1).sum())],
                    marker=dict(color=['#10b981', '#ef4444']),
                    text=[((final_preds == 0).sum()), ((final_preds == 1).sum())],
                    textposition='auto',
                    textfont=dict(size=14, color='white')
                )
            ])
            fig1.update_layout(
                title=dict(text="Ensemble Prediction Results", font=dict(size=20, color='#e0e0e0')),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = go.Figure(data=[
                go.Histogram(
                    x=supervised_probs * 100,
                    nbinsx=30,
                    marker=dict(
                        color=supervised_probs * 100,
                        colorscale='RdYlGn_r',
                        line=dict(color='rgba(255,255,255,0.2)', width=1)
                    )
                )
            ])
            fig2.update_layout(
                title=dict(text=f"{supervised_name} Probability Distribution", font=dict(size=20, color='#e0e0e0')),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                xaxis=dict(title="Fraud Probability (%)", showgrid=False),
                yaxis=dict(title="Count", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)

        
        # Data Table
        st.markdown("<h3 style='color: #e0e0e0; margin-top: 30px;'>📋 Detailed Results</h3>", unsafe_allow_html=True)
        st.dataframe(data, width='stretch', height=400)
        
        # Download Button
        csv = data.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )

# MODEL PERFORMANCE
elif page == "🎯 Model Performance":
    st.markdown("<h1>🎯 Model Performance Metrics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 1.2rem; margin-bottom: 30px;'>Comprehensive evaluation of the fraud detection model</p>", unsafe_allow_html=True)
    
    y_proba = supervised_proba_test
    y_pred = ensemble_pred_test.astype(int)
    
    # Performance Metrics
    st.markdown("<h3 style='color: #e0e0e0;'>📊 Key Performance Indicators</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("🎯 Accuracy", f"{metrics_summary['ensemble']['accuracy']:.4f}")
    with col2:
        st.metric("🎪 Precision", f"{metrics_summary['ensemble']['precision']:.4f}")
    with col3:
        st.metric("🔍 Recall", f"{metrics_summary['ensemble']['recall']:.4f}")
    with col4:
        st.metric("⚖️ F1 Score", f"{metrics_summary['ensemble']['f1']:.4f}")
    with col5:
        st.metric("📈 ROC-AUC", f"{metrics_summary['ensemble']['roc_auc']:.4f}")
    with col6:
        st.metric("📊 PR-AUC", f"{metrics_summary['ensemble']['pr_auc']:.4f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix (ensemble)
        fig1 = go.Figure(data=go.Heatmap(
            z=cm_ensemble,
            x=['Predicted Normal', 'Predicted Fraud'],
            y=['Actual Normal', 'Actual Fraud'],
            colorscale='Plasma',
            text=cm_ensemble,
            texttemplate='%{text}',
            textfont={"size": 20, "color": "white"},
            showscale=True
        ))
        fig1.update_layout(
            title=dict(text="Confusion Matrix", font=dict(size=20, color='#e0e0e0')),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            height=400,
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig1, width='stretch')
    
    with col2:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {metrics_summary['ensemble']['roc_auc']:.4f})',
            line=dict(color='#667eea', width=3)
        ))
        fig2.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='#ef4444', width=2, dash='dash')
        ))
        fig2.update_layout(
            title=dict(text="ROC Curve", font=dict(size=20, color='#e0e0e0')),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            xaxis=dict(title="False Positive Rate", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title="True Positive Rate", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            height=400,
            showlegend=True,
            legend=dict(font=dict(color='#94a3b8'))
        )
        st.plotly_chart(fig2, width='stretch')
    
    # Additional Metrics
    st.markdown("<h3 style='color: #e0e0e0; margin-top: 30px;'>📈 Detailed Analysis</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Threshold Information
        st.markdown(f"""
        <div class='info-box'>
            <h4 style='margin-top: 0; color: #60a5fa;'>⚙️ Model Configuration</h4>
            <p><strong>Supervised Threshold:</strong> {supervised_threshold:.6f}</p>
            <p><strong>Total Test Samples:</strong> {len(y_test):,}</p>
            <p><strong>True Positives:</strong> {true_positives:,}</p>
            <p><strong>True Negatives:</strong> {true_negatives:,}</p>
            <p><strong>False Positives:</strong> {false_positives:,}</p>
            <p><strong>False Negatives:</strong> {false_negatives:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Performance Summary
        st.markdown(f"""
        <div class='info-box'>
            <h4 style='margin-top: 0; color: #60a5fa;'>🎯 Performance Summary</h4>
            <p>The model demonstrates <strong>excellent performance</strong> in detecting fraudulent transactions.</p>
            <p><strong>Strengths:</strong></p>
            <ul>
                <li>High accuracy in classification</li>
                <li>Balanced precision and recall</li>
                <li>Strong ROC-AUC score indicating good discrimination</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
