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


def page_hinkley_detector(delta=0.005, lambda_=50, alpha=0.999, stream=None):
    """
    Simple Page-Hinkley (CUSUM-like) detector generator.

    Usage:
        ph = page_hinkley_detector(delta=0.005, lambda_=0.02, alpha=0.9)
        for x in stream: ph.update(x)
        ph.reset() to reset
    Returns callable object with methods: .update(value) and .reset()
    """
    class PH:
        def __init__(self, delta, lambda_, alpha):
            self.delta = delta
            self.lambda_ = lambda_
            self.alpha = alpha
            self.mean = 0.0
            self.M = 0.0
            self.t = 0

        def update(self, x):
            self.t += 1
            self.mean = self.alpha * self.mean + (1 - self.alpha) * x
            self.M = max(0.0, self.M + x - self.mean - self.delta)
            return self.M > self.lambda_

        def reset(self):
            self.mean = 0.0
            self.M = 0.0
            self.t = 0

    return PH(delta, lambda_, alpha)
