import streamlit as st
import pandas as pd

# config
from loan_monitoring.config import (
    PAGE_TITLE, TARGET_COLUMN, ID_COLUMN,
    FEATURE_COLUMNS, BASELINE_DATA_FILE,
    BASELINE_STATS_PATH, FEATURE_ANALYSIS_PATH,
    BASELINE_MODEL_METRICS, MODEL_PATHS
)

# data utils
from loan_monitoring.data_processing.load_data import load_csv
from loan_monitoring.data_processing.validation import validate_schema

# persistence
from loan_monitoring.utils.persistence import load_json, load_model

# visualization
from loan_monitoring.visualization.baseline_viz import (
    plot_numeric_histograms, plot_categorical_bars, plot_correlation_heatmap
)
from loan_monitoring.visualization.drift_viz import (
    plot_numeric_drift, plot_categorical_drift
)
from loan_monitoring.visualization.model_viz import (
    plot_roc_curve, plot_confusion_matrix, plot_prediction_distribution
)

# drift
from loan_monitoring.drift.drift_metrics import compute_feature_drift
from loan_monitoring.drift.prediction_drift import compute_prediction_drift
from loan_monitoring.drift.performance_eval import compute_performance_metrics
from loan_monitoring.drift.summary import generate_drift_report

# local LLM explainer
from loan_monitoring.llm.explainer import explain_drift_locally

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

# =========================
# LOAD BASELINE ARTIFACTS
# =========================
baseline_df = load_csv(BASELINE_DATA_FILE)
baseline_stats = load_json(BASELINE_STATS_PATH)
feature_analysis = load_json(FEATURE_ANALYSIS_PATH)
baseline_models = {name: load_model(path) for name, path in MODEL_PATHS.items()}

st.sidebar.markdown("### Baseline Overview")
st.sidebar.write("Rows:", baseline_df.shape[0])
st.sidebar.write("Columns:", baseline_df.shape[1])

# =========================
# BASELINE DISPLAY
# =========================
st.header("📊 Baseline Data Overview")
st.markdown("**Target distribution (Baseline):**")
st.write(baseline_df[TARGET_COLUMN].value_counts(normalize=True))

# Identify numeric & categorical for baseline
num_feats, cat_feats = (
    baseline_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
               .select_dtypes(include=["number"]).columns.tolist(),
    baseline_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
               .select_dtypes(exclude=["number"]).columns.tolist()
)

# Feature distribution plots
st.subheader("Baseline Feature Distributions")
fig1 = plot_numeric_histograms(baseline_df, num_feats)
st.pyplot(fig1)

fig2 = plot_categorical_bars(baseline_df, cat_feats)
st.pyplot(fig2)

st.subheader("Correlation Heatmap (Baseline)")
fig3 = plot_correlation_heatmap(baseline_df, num_feats)
st.pyplot(fig3)

# =========================
# MODEL PERFORMANCE (BASELINE)
# =========================
st.header("🤖 Baseline Model Performance Metrics")
model_metrics = load_json(BASELINE_MODEL_METRICS)
st.json(model_metrics)

# Optional: show baseline ROC / confusion plots
st.markdown("### Baseline Model Diagnostics Plots")
for model_name, model in baseline_models.items():
    st.markdown(f"**{model_name}**")
    # Get baseline metrics
    # For demonstration, we use the baseline dataset itself
    y_true = baseline_df[TARGET_COLUMN]
    y_proba = model.predict_proba(baseline_df.drop(columns=[ID_COLUMN, TARGET_COLUMN]))[:, 1]
    y_pred = model.predict(baseline_df.drop(columns=[ID_COLUMN, TARGET_COLUMN]))

    fig_roc = plot_roc_curve(y_true, y_proba, model_name)
    st.pyplot(fig_roc)

    fig_cm = plot_confusion_matrix(y_true, y_pred, model_name)
    st.pyplot(fig_cm)

# =========================
# UPLOAD NEW DATA
# =========================
st.header("📥 Upload New (Production) Dataset")

uploaded_file = st.file_uploader("Choose CSV", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV to analyze drift.")
    st.stop()

new_df = pd.read_csv(uploaded_file)

# Schema validation
try:
    validate_schema(
        new_df,
        expected_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        id_column=ID_COLUMN
    )
    st.success("Schema matches baseline!")
except Exception as e:
    st.error(f"Schema error: {e}")
    st.stop()

st.write("New data shape:", new_df.shape)

# =========================
# DRIFT ANALYSIS
# =========================
st.header("📉 Drift Detection Results")

feature_drift = compute_feature_drift(
    baseline_stats, new_df, ignore_columns=[ID_COLUMN, TARGET_COLUMN]
)
st.json(feature_drift)

st.subheader("Feature Drift Visual Comparison")
fig_drift_num = plot_numeric_drift(baseline_df, new_df, num_feats)
st.pyplot(fig_drift_num)

fig_drift_cat = plot_categorical_drift(baseline_df, new_df, cat_feats)
st.pyplot(fig_drift_cat)

# =========================
# PREDICTION DRIFT
# =========================
st.header("📈 Prediction Drift (Model Output Behavior)")

# Drop ID & target before computing prediction drift
X_new = new_df.drop(columns=[ID_COLUMN, TARGET_COLUMN])
pred_drift = compute_prediction_drift(X_new, MODEL_PATHS)
st.json(pred_drift)

st.subheader("Prediction Distribution Comparison")
for model_name, model in baseline_models.items():
    y_proba_base = model.predict_proba(baseline_df.drop(columns=[ID_COLUMN, TARGET_COLUMN]))[:, 1]
    y_proba_new = model.predict_proba(X_new)[:, 1]

    fig_pred = plot_prediction_distribution(y_proba_new, model_name)
    st.pyplot(fig_pred)

# =========================
# PERFORMANCE (IF LABELS AVAILABLE)
# =========================
st.header("📊 Performance on New Data (If Labels Provided)")

if TARGET_COLUMN in new_df.columns:
    perf_results = {}
    for model_name, model in baseline_models.items():
        y_true_new = new_df[TARGET_COLUMN]
        y_pred_new = model.predict(X_new)
        y_proba_new = model.predict_proba(X_new)[:, 1]
        perf_results[model_name] = compute_performance_metrics(
            y_true_new, y_pred_new, y_proba_new
        )
    st.json(perf_results)
else:
    st.info("No target column in new data — performance metrics skipped.")
    perf_results = None

# =========================
# LOCAL LLM EXPLANATION
# =========================
st.header("🤖 Drift Explanation (Local LLM)")

try:
    explanation = explain_drift_locally(
        feature_summary=feature_drift,
        prediction_summary=pred_drift,
        performance_summary=perf_results
    )
    st.write(explanation)
except Exception as llm_err:
    st.error(f"LLM explanation failed: {llm_err}")

