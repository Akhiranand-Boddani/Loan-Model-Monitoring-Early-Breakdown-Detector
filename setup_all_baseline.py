"""
Run this script once before running your Streamlit app.
It will:
1) Create baseline statistics JSON
2) Create baseline feature analysis JSON
3) Train baseline models and save them + metrics
4) (Optionally) generate synthetic drift data for testing
"""

import os
import pandas as pd

# --- Load utilities ---
from loan_monitoring.data_processing.load_data import load_csv
from loan_monitoring.baseline.stats import create_and_save_baseline_stats
from loan_monitoring.baseline.feature_analysis import generate_and_save_feature_analysis
from loan_monitoring.baseline.model_training import train_baseline_models

# (Optional) drift generation utility
from create_drift_data import generate_drift_dataset

# --- Configuration ---
BASELINE_FILE = "data/baseline.csv"

# Where artifacts will be saved
BASELINE_STATS_JSON = "artifacts/baseline_stats/baseline_stats.json"
FEATURE_ANALYSIS_JSON = "artifacts/baseline_stats/feature_analysis.json"
BASELINE_MODELS_DIR = "artifacts/baseline_models"

# Target & ignore
TARGET_COL = "Default"
IGNORE_COLS = ["LoanID"]

# Drift generation output
GENERATE_DRIFT = True   # Set to True if you want sample drift data
DRIFT_OUTPUT_PATH = "data/drift_generated.csv"

# Numeric drift config
numeric_shifts_example = {
    "Income": ("scale", 0.7),
    "CreditScore": ("add", -40),
    "DTIRatio": ("scale", 1.3),
    "MonthsEmployed": ("scale", 0.8)
}

# Categorical drift config
categorical_shifts_example = {
    "EmploymentType": {"Salaried": 0.4, "Self-employed": 0.4, "Contract": 0.2},
    "LoanPurpose": {"Home": 0.15, "Auto": 0.15, "Education": 0.1, "Business": 0.4, "Personal": 0.2},
    "HasCoSigner": {"Yes": 0.2, "No": 0.8}
}


def ensure_directories():
    """
    Make sure all artifact directories exist before saving.
    """
    os.makedirs(os.path.dirname(BASELINE_STATS_JSON), exist_ok=True)
    os.makedirs(BASELINE_MODELS_DIR, exist_ok=True)
    print("Artifact directories ensured.")


def run_baseline_stats():
    """
    1) Load baseline data
    2) Compute baseline statistics
    """
    print("Loading baseline data from:", BASELINE_FILE)
    df = load_csv(BASELINE_FILE)

    print("Computing baseline statistics...")
    create_and_save_baseline_stats(
        df=df,
        output_path_json=BASELINE_STATS_JSON,
        ignore_columns=IGNORE_COLS + [TARGET_COL]
    )

    print("Baseline statistics saved to:", BASELINE_STATS_JSON)


def run_feature_analysis():
    """
    Compute and save feature analysis JSON.
    """
    print("Running feature analysis (correlation & mutual info)...")
    df = load_csv(BASELINE_FILE)

    generate_and_save_feature_analysis(
        df=df,
        target_col=TARGET_COL,
        ignore_columns=IGNORE_COLS,
        output_path_json=FEATURE_ANALYSIS_JSON
    )

    print("Feature analysis saved to:", FEATURE_ANALYSIS_JSON)


def run_model_training():
    """
    Train, save baseline models and metrics.
    """
    print("Training baseline models...")
    df = load_csv(BASELINE_FILE)

    train_baseline_models(
        df=df,
        target_col=TARGET_COL,
        ignore_columns=IGNORE_COLS,
        output_dir=BASELINE_MODELS_DIR
    )

    print(f"Baseline models trained and saved to: {BASELINE_MODELS_DIR}")


def run_drift_generation():
    """
    Optionally generate a sample drift dataset.
    """
    if not GENERATE_DRIFT:
        return

    print("Generating synthetic drift dataset for testing...")
    df_base = pd.read_csv(BASELINE_FILE)

    generate_drift_dataset(
        df=df_base,
        output_path=DRIFT_OUTPUT_PATH,
        numeric_shifts=numeric_shifts_example,
        categorical_shifts=categorical_shifts_example
    )

    print("Drift sample saved to:", DRIFT_OUTPUT_PATH)


if __name__ == "__main__":
    print("\n--- RUNNING SETUP FOR BASELINE MONITORING SYSTEM ---\n")

    ensure_directories()
    run_baseline_stats()
    run_feature_analysis()
    run_model_training()
    run_drift_generation()

    print("\n--- SETUP COMPLETED SUCCESSFULLY ---")
    print("You can now run: streamlit run streamlit_app.py")
