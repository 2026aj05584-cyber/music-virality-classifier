import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Set page configuration
st.set_page_config(
    page_title="Music Virality Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1DB954;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🎵 Music Virality Classifier & Evaluation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">BITS Pilani — Machine Learning Assignment 2 | End-to-End Classification & Deployment</div>', unsafe_allow_html=True)

# Required features and target
FEATURE_COLS = [
    'genre',
    'release_year',
    'duration_sec',
    'danceability',
    'energy',
    'valence',
    'tempo_bpm',
    'acousticness',
    'instrumentalness',
    'speechiness',
    'liveness',
    'loudness_db'
]
TARGET_COL = 'viral'

MODEL_MAP = {
    'Support Vector Machine': 'support_vector_machine.joblib',
    'Logistic Regression': 'logistic_regression.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DEFAULT_TEST_DATA = os.path.join(BASE_DIR, 'test_data.csv')

# --- SIDEBAR: Controls ---
st.sidebar.header("⚙️ Model & Data Configuration")

# Feature (b): Model Selection Dropdown
selected_model_name = st.sidebar.selectbox(
    "Choose Classification Model:",
    options=list(MODEL_MAP.keys()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload Test Dataset (CSV)")
# Feature (a): Dataset upload option (CSV)
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (must contain features and target column 'viral'):",
    type=["csv"]
)

use_default = False
if uploaded_file is None:
    st.sidebar.info("💡 No file uploaded yet. You can upload your test CSV or test using the pre-packaged `test_data.csv`.")
    if os.path.exists(DEFAULT_TEST_DATA):
        use_default = st.sidebar.checkbox("Use default benchmark test_data.csv", value=True)

# Load dataset
df_test = None
if uploaded_file is not None:
    try:
        df_test = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Uploaded: `{uploaded_file.name}` ({df_test.shape[0]} rows)")
    except Exception as e:
        st.error(f"Error reading uploaded CSV file: {e}")
elif use_default and os.path.exists(DEFAULT_TEST_DATA):
    df_test = pd.read_csv(DEFAULT_TEST_DATA)
    st.sidebar.info(f"Loaded default `test_data.csv` ({df_test.shape[0]} rows)")

# Validate dataset
if df_test is None:
    st.warning("👈 Please upload a test dataset CSV from the sidebar to evaluate the models.")
    st.stop()

# Check for required features and target
missing_features = [col for col in FEATURE_COLS if col not in df_test.columns]
missing_target = TARGET_COL not in df_test.columns

if missing_features or missing_target:
    st.error("Uploaded CSV is missing mandatory columns!")
    if missing_features:
        st.write("Missing features:", missing_features)
    if missing_target:
        st.write(f"Missing target column: `{TARGET_COL}`")
    st.stop()

X_test = df_test[FEATURE_COLS]
y_test = df_test[TARGET_COL]

# Load selected model pipeline
model_file = MODEL_MAP[selected_model_name]
model_path = os.path.join(MODEL_DIR, model_file)

if not os.path.exists(model_path):
    st.error(f"Model file `{model_file}` not found in `{MODEL_DIR}`. Please ensure models are trained.")
    st.stop()

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

try:
    pipeline = load_pipeline(model_path)
except Exception as e:
    st.error(f"Failed to load model pipeline: {e}")
    st.stop()

# Run inference without retraining
with st.spinner("Calculating predictions and evaluation metrics..."):
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipeline.decision_function(X_test)

# Calculate the 6 required metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

# --- TABS FOR ORGANIZED PRESENTATION ---
tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "🔍 Test Data Preview", "🏆 Benchmark Comparison"])

with tab1:
    st.subheader(f"Evaluation Metrics: {selected_model_name}")
    st.caption("All metrics are computed on the uploaded test dataset without retraining.")

    # Feature (c): Display of evaluation metrics
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("🎯 Accuracy", f"{acc:.4f}", help="Fraction of correctly classified tracks")
    col2.metric("📈 AUC Score", f"{auc:.4f}", help="Area under the ROC curve (ranking discrimination)")
    col3.metric("🎯 Precision", f"{prec:.4f}", help="Ratio of true virals among predicted virals")
    col4.metric("🔄 Recall", f"{rec:.4f}", help="Ratio of actual virals correctly detected")
    col5.metric("⚖️ F1 Score", f"{f1:.4f}", help="Harmonic mean of precision and recall")
    col6.metric("📐 MCC Score", f"{mcc:.4f}", help="Matthews Correlation Coefficient (-1 to +1)")

    st.markdown("---")

    # Feature (d): Confusion matrix or classification report
    col_cm, col_roc = st.columns([1, 1])

    with col_cm:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Not Viral (0)", "Viral (1)"],
            yticklabels=["Not Viral (0)", "Viral (1)"],
            ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        ax_cm.set_title(f"Confusion Matrix ({selected_model_name})")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_roc:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc, ax_roc = plt.subplots(figsize=(4.5, 3.8))
        ax_roc.plot(fpr, tpr, color="#1DB954", lw=2, label=f"AUC = {auc:.4f}")
        ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Receiver Operating Characteristic")
        ax_roc.legend(loc="lower right")
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    st.subheader("Detailed Classification Report")
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

with tab2:
    st.subheader("Uploaded Test Dataset Preview")
    st.write(f"Total Rows: **{df_test.shape[0]}** | Total Columns: **{df_test.shape[1]}**")
    st.dataframe(df_test.head(10), use_container_width=True)

    st.subheader("Class Distribution in Test Set")
    dist_df = pd.DataFrame({
        'Count': y_test.value_counts(),
        'Percentage': (y_test.value_counts(normalize=True) * 100).round(2).astype(str) + '%'
    })
    dist_df.index = dist_df.index.map({0: 'Not Viral (0)', 1: 'Viral (1)'})
    st.table(dist_df)

with tab3:
    st.subheader("Overall Model Comparison Table")
    st.markdown("Performance of all 5 classifiers evaluated on the same 20% test partition (2,000 samples):")

    benchmark_data = [
        {"ML Model Name": "Logistic Regression", "Accuracy": 0.7360, "AUC": 0.8081, "Precision": 0.6421, "Recall": 0.6047, "F1": 0.6229, "MCC": 0.4205},
        {"ML Model Name": "Decision Tree", "Accuracy": 0.7035, "AUC": 0.7653, "Precision": 0.5930, "Recall": 0.5659, "F1": 0.5791, "MCC": 0.3507},
        {"ML Model Name": "KNN", "Accuracy": 0.6885, "AUC": 0.7472, "Precision": 0.5714, "Recall": 0.5437, "F1": 0.5572, "MCC": 0.3174},
        {"ML Model Name": "Naive Bayes", "Accuracy": 0.6935, "AUC": 0.8062, "Precision": 0.5464, "Recall": 0.8821, "F1": 0.6748, "MCC": 0.4568},
        {"ML Model Name": "Support Vector Machine", "Accuracy": 0.7380, "AUC": 0.7808, "Precision": 0.6468, "Recall": 0.6019, "F1": 0.6236, "MCC": 0.4237}
    ]
    benchmark_df = pd.DataFrame(benchmark_data)
    st.dataframe(
        benchmark_df.style.highlight_max(subset=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"], color="#d4edda"),
        use_container_width=True
    )

    st.markdown("### Observations & Summary")
    st.markdown("""
    - **Support Vector Machine (SVM):** Highest Accuracy (**73.80%**) and Precision (**64.68%**). The RBF kernel captures non-linear audio feature boundaries effectively after standardization.
    - **Logistic Regression:** Highest AUC (**0.8081**), balanced precision/recall, and excellent generalization without overfitting.
    - **Naive Bayes:** Exceptional Recall (**88.21%**), making it an ideal high-sensitivity screener for viral tracks, though lower precision due to feature correlations (e.g. loudness vs energy).
    - **Decision Tree & KNN:** Captured local and hierarchical splits but showed lower generalization on the 12-dimensional continuous space.
    - **Overall Winner:** **Support Vector Machine (SVM)** for raw predictive accuracy and precision, and **Logistic Regression** for probability calibration and ranking discrimination (AUC).
    """)
