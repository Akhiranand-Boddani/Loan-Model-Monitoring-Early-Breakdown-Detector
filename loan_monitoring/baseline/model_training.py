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
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn")
    print("Continuing without SMOTE balancing...")

from loan_monitoring.utils.persistence import save_model, save_json
from loan_monitoring.data_processing.load_data import detect_feature_types

# ---------------------------------------------
# TRAIN / EVALUATION UTILITIES
# ---------------------------------------------

def compute_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict:
    """Compute basic classification metrics."""
    # Ensure y_pred is an array, not a tuple
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
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
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y.to_numpy())


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

    from loan_monitoring.data_processing.preprocessing import build_preprocessing_pipeline

    # Build preprocessing pipeline for numeric and categorical features
    preprocessor, _ = build_preprocessing_pipeline(df, ignore_columns, target_col)

    # Prepare features and target
    drop_cols = set(ignore_columns + [target_col])
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # Fit preprocessor on entire data and transform
    preprocessor.fit(X, y)
    X_transformed = preprocessor.transform(X)
    
    print(f"\nOriginal class distribution (entire dataset):")
    print(pd.Series(y).value_counts(normalize=True))
    
    # Apply SMOTE to balance the entire dataset (upsample minority to 50% of majority)
    if SMOTE_AVAILABLE:
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_transformed, y)
        print(f"\nAfter SMOTE balancing (entire dataset, 50% ratio):")
        print(pd.Series(y_balanced).value_counts(normalize=True))
    else:
        X_balanced, y_balanced = X_transformed, y
        print("\nSMOTE not available, using original imbalanced data")
    
    # Now split the balanced data into train and test (stratified)
    X_train_balanced, X_test_balanced, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    print(f"\nTraining set class distribution (after SMOTE + split):")
    print(pd.Series(y_train).value_counts(normalize=True))
    print(f"\nTest set class distribution (after SMOTE + split):")
    print(pd.Series(y_test).value_counts(normalize=True))

    results = {}

    # ---------------------------------------------
    # 1. Logistic Regression
    # ---------------------------------------------
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='saga',
        n_jobs=-1
    )
    
    lr_model.fit(X_train_balanced, y_train)
    y_pred = lr_model.predict(X_test_balanced)
    y_proba = lr_model.predict_proba(X_test_balanced)[:, 1]

    results["logistic_regression"] = compute_class_metrics(y_test, y_pred, y_proba)

    # Save as pipeline for consistent preprocessing during inference
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", lr_model)
    ])
    save_model(lr_pipeline, os.path.join(output_dir, "logistic_regression.pkl"))
    print(f"\nLogistic Regression metrics: {results['logistic_regression']}")

    # ---------------------------------------------
    # 2. Random Forest
    # ---------------------------------------------
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_balanced, y_train)
    y_pred = rf_model.predict(X_test_balanced)
    y_proba = rf_model.predict_proba(X_test_balanced)[:, 1]

    results["random_forest"] = compute_class_metrics(y_test, y_pred, y_proba)

    # Save as pipeline for consistent preprocessing during inference
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ])
    save_model(rf_pipeline, os.path.join(output_dir, "random_forest.pkl"))
    print(f"\nRandom Forest metrics: {results['random_forest']}")

    # ---------------------------------------------
    # 3. XGBoost
    # ---------------------------------------------
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X_train_balanced, y_train)
    y_pred = xgb_model.predict(X_test_balanced)
    y_proba = xgb_model.predict_proba(X_test_balanced)[:, 1]

    results["xgboost"] = compute_class_metrics(y_test, y_pred, y_proba)

    # Save as pipeline for consistent preprocessing during inference
    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ])
    save_model(xgb_pipeline, os.path.join(output_dir, "xgboost.pkl"))
    print(f"\nXGBoost metrics: {results['xgboost']}")

    # ---------------------------------------------
    # Save metrics to JSON
    # ---------------------------------------------
    metric_output_path = os.path.join(output_dir, "baseline_model_metrics.json")
    save_json(results, metric_output_path)
    print(f"[baseline/model_training] → Saved baseline model metrics at {metric_output_path}")

    return results
