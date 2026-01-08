import numpy as np
from scipy.stats import entropy


def probability_to_class(proba, threshold=0.5):
    """
    Convert predicted probabilities to classes using a threshold.
    """
    return (proba >= threshold).astype(int)


def prediction_entropy(proba):
    """
    Compute entropy of probability distribution.
    Low entropy = confident predictions
    High entropy = uncertain predictions
    """
    # support both 1D and 2D proba
    proba = np.array(proba)
    if proba.ndim == 1:
        # binary: [p, 1-p]
        p = proba
        probs = np.vstack([p, 1 - p]).T
    else:
        probs = proba

    # small epsilon to avoid log(0)
    eps = 1e-9
    probs = np.clip(probs, eps, 1 - eps)
    return np.apply_along_axis(entropy, 1, probs, base=2)


def class_balance(y):
    """
    Return class distribution proportions.
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {int(u): float(c / total) for u, c in zip(unique, counts)}


def confidence_stats(y_proba):
    """
    Computes:
    - average probability
    - variance
    - entropy mean
    Useful for prediction drift.
    """
    probs = np.array(y_proba)

    stats = {
        "mean_proba": float(np.mean(probs)),
        "std_proba": float(np.std(probs)),
        "entropy_mean": float(np.mean(prediction_entropy(probs))),
        "entropy_std": float(np.std(prediction_entropy(probs)))
    }
    return stats
