import sys
from pathlib import Path
# ensure project root is on sys.path so `loan_monitoring` package is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loan_monitoring.llm.explainer import explain_drift_locally

# Synthetic summaries
feature_summary = {
    "numeric": {
        "age": {"psi": 0.12, "kl_divergence": 0.05, "wasserstein_normalized": 0.02},
        "income": {"psi": 0.30, "kl_divergence": 0.12, "wasserstein_normalized": 0.18}
    },
    "categorical": {
        "employment_type": {"chi2_p_value": 0.01, "kl_divergence": 0.2},
        "region": {"chi2_p_value": 0.2, "kl_divergence": 0.02}
    }
}

prediction_summary = {
    "xgboost_model": {"mean_proba": 0.12, "entropy_mean": 0.65},
    "rf_model": {"mean_proba": 0.10, "entropy_mean": 0.60}
}

performance_summary = {
    "xgboost_model": {"accuracy": 0.78, "recall": 0.54, "roc_auc": 0.82}
}

print("--- Running explain_drift_locally() test ---")
try:
    out = explain_drift_locally(feature_summary, prediction_summary, performance_summary)
    print(out)
except Exception as e:
    print("Test failed with exception:", e)

print("--- Done ---")
