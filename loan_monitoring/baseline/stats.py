import pandas as pd
import numpy as np
import json
from typing import List, Dict
from scipy.stats import skew, kurtosis, entropy
from loan_monitoring.data_processing.load_data import detect_feature_types
from loan_monitoring.utils.persistence import save_json

# ---------------------------------------------
# Numeric Statistics
# ---------------------------------------------
def compute_numeric_stats(df: pd.DataFrame, numeric_features: List[str]) -> Dict:
    """
    Compute baseline statistics for numeric features.

    Returns:
        Dict[str, Dict]: Numeric stats per feature
    """
    numeric_stats = {}

    for feat in numeric_features:
        col = df[feat].dropna()

        numeric_stats[feat] = {
            "count": int(col.count()),
            "mean": float(col.mean()),
            "median": float(col.median()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "q1": float(col.quantile(0.25)),
            "q3": float(col.quantile(0.75)),
            "skew": float(skew(col)),
            "kurtosis": float(kurtosis(col))
        }

    return numeric_stats


# ---------------------------------------------
# Categorical Statistics
# ---------------------------------------------
def compute_categorical_stats(
    df: pd.DataFrame, categorical_features: List[str]
) -> Dict:
    """
    Compute baseline stats for categorical features.

    Returns:
        Dict[str, Dict]: Stats per categorical feature
    """
    cat_stats = {}

    for feat in categorical_features:
        col = df[feat].astype(str).fillna("MISSING")
        counts = col.value_counts(dropna=False)
        props = col.value_counts(dropna=False, normalize=True)

        # entropy of categories
        ent = entropy(props.values, base=2)

        cat_stats[feat] = {
            "unique_values": list(counts.index),
            "counts": counts.to_dict(),
            "proportions": props.to_dict(),
            "entropy": float(ent)
        }

    return cat_stats


# ---------------------------------------------
# Combined Baseline Stats
# ---------------------------------------------
def compute_baseline_stats(df: pd.DataFrame, ignore_columns: List[str]) -> Dict:
    """
    Compute baseline statistics for the entire dataset.

    Returns:
        Dict with numeric and categorical stats
    """
    numeric_features, categorical_features = detect_feature_types(df, ignore_columns)

    numeric_stats = compute_numeric_stats(df, numeric_features)
    categorical_stats = compute_categorical_stats(df, categorical_features)

    return {
        "numeric": numeric_stats,
        "categorical": categorical_stats
    }


# ---------------------------------------------
# Entry Point
# ---------------------------------------------
def create_and_save_baseline_stats(
    df: pd.DataFrame,
    output_path_json: str,
    ignore_columns: List[str]
):
    """
    Compute baseline stats and save to JSON for later use.

    Args:
        df: Baseline dataset
        output_path_json: Where to save stats JSON
        ignore_columns: Columns to skip (ID, target)
    """
    stats = compute_baseline_stats(df, ignore_columns)

    save_json(stats, output_path_json)
    print(f"[baseline/stats] → Saved baseline stats at {output_path_json}")
