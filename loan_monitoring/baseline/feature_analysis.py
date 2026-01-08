import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
import json

from loan_monitoring.data_processing.load_data import detect_feature_types
from loan_monitoring.utils.persistence import save_json

# ---------------------------------------------
# Correlation (Numeric)
# ---------------------------------------------
def compute_numeric_correlations(
    df: pd.DataFrame,
    numeric_features: List[str],
    target_col: str
) -> Dict:
    """
    Compute Pearson correlations of numeric features with each other
    and with the target.
    """

    # correlation matrix
    corr_matrix = df[numeric_features + [target_col]].corr(method="pearson")

    # convert to nested dict
    corr_dict = corr_matrix.to_dict()
    return corr_dict


# ---------------------------------------------
# Mutual Information (Feature vs Target)
# ---------------------------------------------
def compute_mutual_information(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target_col: str
) -> Dict:
    """
    Compute mutual information of each feature with the target variable.
    """

    mi_scores = {}

    # encode target if needed
    y = df[target_col].values

    # numeric features
    for feat in numeric_features:
        # bin numeric into categories for MI
        bin_vals = pd.qcut(df[feat], q=10, duplicates="drop").codes
        mi = mutual_info_score(bin_vals, y)
        mi_scores[feat] = float(mi)

    # categorical features
    for feat in categorical_features:
        le = LabelEncoder()
        col_enc = le.fit_transform(df[feat].astype(str))
        mi = mutual_info_score(col_enc, y)
        mi_scores[feat] = float(mi)

    return mi_scores


# ---------------------------------------------
# Dataframe for Feature Analysis
# ---------------------------------------------
def create_feature_analysis(
    df: pd.DataFrame,
    target_col: str,
    ignore_columns: List[str]
) -> Dict:
    """
    Compute advanced feature statistics:
    - numeric correlation
    - mutual info
    """

    num_feats, cat_feats = detect_feature_types(df, ignore_columns)

    # numeric correlation
    num_corr = compute_numeric_correlations(df, num_feats, target_col)

    # mutual information feature vs target
    mi_scores = compute_mutual_information(df, num_feats, cat_feats, target_col)

    return {
        "numeric_correlation_matrix": num_corr,
        "mutual_information_with_target": mi_scores
    }


# ---------------------------------------------
# Entry Point
# ---------------------------------------------
def generate_and_save_feature_analysis(
    df: pd.DataFrame,
    target_col: str,
    ignore_columns: List[str],
    output_path_json: str
):
    """
    Generate feature analysis and save to disk.
    """

    analysis = create_feature_analysis(df, target_col, ignore_columns)

    save_json(analysis, output_path_json)
    print(f"[baseline/feature_analysis] → Saved feature analysis to {output_path_json}")
