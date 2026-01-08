import pandas as pd
from typing import Dict, List

from loan_monitoring.utils.metrics import psi, ks_test, chi_square_test, wasserstein
from loan_monitoring.data_processing.load_data import detect_feature_types
from loan_monitoring.utils.persistence import load_json


def compute_feature_drift(
    baseline_stats: dict,
    df_new: pd.DataFrame,
    ignore_columns: List[str]
) -> Dict:
    """
    Compare new data vs baseline stats for each feature.
    Returns dict with drift metrics.
    """

    # detect features
    numeric_features, categorical_features = detect_feature_types(df_new, ignore_columns)

    drift_summary = {"numeric": {}, "categorical": {}}

    # numeric
    for feat in numeric_features:
        base_vals = pd.Series(baseline_stats["numeric"][feat]["values"]) \
            if "values" in baseline_stats["numeric"][feat] else None

        new_vals = df_new[feat].dropna()

        if base_vals is not None:
            drift_summary["numeric"][feat] = {
                "psi": psi(base_vals, new_vals),
                **ks_test(base_vals, new_vals),
                "wasserstein": wasserstein(base_vals, new_vals)
            }

    # categorical
    for feat in categorical_features:
        base_props = baseline_stats["categorical"][feat]["proportions"]
        new_props = df_new[feat].astype(str).value_counts(normalize=True).to_dict()

        drift_summary["categorical"][feat] = {
            **chi_square_test(base_props, new_props),
            "entropy_baseline": baseline_stats["categorical"][feat]["entropy"],
            "entropy_new": float(pd.Series(new_props).entropy())
        }

    return drift_summary
