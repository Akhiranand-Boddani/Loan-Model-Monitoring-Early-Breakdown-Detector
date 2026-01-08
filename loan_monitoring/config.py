import os

# =======================
# FEATURE / TARGET
# =======================
TARGET_COLUMN = "Default"
ID_COLUMN = "LoanID"

FEATURE_COLUMNS = [
    "Age", "Income", "LoanAmount", "CreditScore",
    "MonthsEmployed", "NumCreditLines", "InterestRate",
    "LoanTerm", "DTIRatio",
    "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"
]

# =======================
# PATHS
# =======================
BASE_DATA_PATH = "data"
BASELINE_DATA_FILE = os.path.join(BASE_DATA_PATH, "baseline.csv")

ARTIFACTS_PATH = "artifacts"
BASELINE_STATS_PATH = os.path.join(ARTIFACTS_PATH, "baseline_stats/baseline_stats.json")
FEATURE_ANALYSIS_PATH = os.path.join(ARTIFACTS_PATH, "baseline_stats/feature_analysis.json")
BASELINE_MODELS_PATH = os.path.join(ARTIFACTS_PATH, "baseline_models")
BASELINE_MODEL_METRICS = os.path.join(BASELINE_MODELS_PATH, "baseline_model_metrics.json")

# specify model pkl paths
MODEL_PATHS = {
    "logistic_regression": os.path.join(BASELINE_MODELS_PATH, "logistic_regression.pkl"),
    "random_forest": os.path.join(BASELINE_MODELS_PATH, "random_forest.pkl"),
    "xgboost": os.path.join(BASELINE_MODELS_PATH, "xgboost.pkl")
}

# =======================
# DRIFT THRESHOLDS
# =======================
PSI_THRESHOLD = 0.25
KS_PVALUE_THRESHOLD = 0.05
CHI2_PVALUE_THRESHOLD = 0.05

# =======================
# STREAMLIT SETTINGS
# =======================
PAGE_TITLE = "Loan Model Monitoring & Drift Dashboard"
