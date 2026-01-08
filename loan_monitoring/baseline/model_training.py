import os
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
)
import xgboost as xgb

from loan_monitoring.utils.persistence import save_model, save_json
from loan_monitoring.data_processing.load_data import detect_feature_types

# ---------------------------------------------
# TRAIN / EVALUATION UTILITIES
# ---------------------------------------------

def compute_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict:
    """Compute basic classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def train_test_split_features(
    df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42
):
    """
    Split a DataFrame into train/test features and labels.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ---------------------------------------------
# BASELINE TRAINING
# ---------------------------------------------

def train_baseline_models(
    df: pd.DataFrame,
    target_col: str,
    ignore_columns: list,
    output_dir: str
):
    """
    Train baseline models (LogReg, RF, XGB), save them, 
    and produce metrics JSON.
    """
    # detect numeric / categorical
    num_feats, cat_feats = detect_feature_types(df, ignore_columns)

    # we will train on ALL features; encoding can be added later
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # simple train/test split
    X_train, X_test, y_train, y_test = train_test_split_features(df, target_col)

    results = {}

    # ---------------------------------------------
    # 1. Logistic Regression
    # ---------------------------------------------
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",   # handle imbalance
            max_iter=500,
            random_state=42
        ))
    ])

    lr_pipeline.fit(X_train, y_train)
    y_pred = lr_pipeline.predict(X_test)
    y_proba = lr_pipeline.predict_proba(X_test)[:, 1]

    results["logistic_regression"] = compute_class_metrics(y_test, y_pred, y_proba)

    save_model(lr_pipeline, os.path.join(output_dir, "logistic_regression.pkl"))

    # ---------------------------------------------
    # 2. Random Forest
    # ---------------------------------------------
    rf_model = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    results["random_forest"] = compute_class_metrics(y_test, y_pred, y_proba)

    save_model(rf_model, os.path.join(output_dir, "random_forest.pkl"))

    # ---------------------------------------------
    # 3. XGBoost
    # ---------------------------------------------
    # compute scale_pos_weight for XGB from train set
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=ratio,
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]

    results["xgboost"] = compute_class_metrics(y_test, y_pred, y_proba)

    save_model(xgb_model, os.path.join(output_dir, "xgboost.pkl"))

    # ---------------------------------------------
    # Save metrics to JSON
    # ---------------------------------------------
    metric_output_path = os.path.join(output_dir, "baseline_model_metrics.json")
    save_json(results, metric_output_path)
    print(f"[baseline/model_training] → Saved baseline model metrics at {metric_output_path}")

    return results
