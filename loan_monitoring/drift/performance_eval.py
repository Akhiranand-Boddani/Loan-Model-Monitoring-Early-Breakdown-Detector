from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
)
from typing import Dict


def compute_performance_metrics(
    y_true, y_pred, y_proba
) -> Dict:
    """
    Compute evaluation only when target labels present.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }
