import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare, wasserstein_distance
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
)


# -------------------------
# Classification Metrics
# -------------------------

def classification_metrics(y_true, y_pred, y_proba):
    """
    Compute key classification metrics.
    """
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }
    return results


# -------------------------
# Drift Metrics
# -------------------------

def psi(expected, actual, buckets=10):
    """
    Population Stability Index (PSI)
    Formula: sum( (expected_pct - actual_pct) * ln(expected_pct / actual_pct) )
    """
    expected_values = np.array(expected).astype(float)
    actual_values = np.array(actual).astype(float)

    # define bins from expected
    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.quantile(expected_values, quantiles)

    # compute expected and actual counts per bucket
    exp_counts, _ = np.histogram(expected_values, bins=breakpoints)
    act_counts, _ = np.histogram(actual_values, bins=breakpoints)

    # convert to percents
    exp_perc = exp_counts / len(expected_values)
    act_perc = act_counts / len(actual_values)

    # compute PSI
    psi_val = 0.0
    for e, a in zip(exp_perc, act_perc):
        if e == 0 or a == 0:
            continue
        psi_val += (e - a) * np.log(e / a)

    return float(psi_val)


def ks_test(feature_expected, feature_actual):
    """
    Compute KS statistic and p-value between two samples.
    """
    stat, p_value = ks_2samp(feature_expected, feature_actual)
    return {"ks_stat": float(stat), "p_value": float(p_value)}


def chi_square_test(expected_counts, actual_counts):
    """
    Compute chi-square statistic given counts for categories.
    Both expected_counts and actual_counts are dicts {category: count}.
    """
    # align categories
    categories = list(set(expected_counts.keys()).union(set(actual_counts.keys())))
    exp = np.array([expected_counts.get(cat, 0) for cat in categories])
    act = np.array([actual_counts.get(cat, 0) for cat in categories])

    stat, p = chisquare(f_obs=act, f_exp=exp)
    return {"chi2_stat": float(stat), "p_value": float(p)}


def wasserstein(feature_expected, feature_actual):
    """
    Wasserstein (Earth Mover's) distance between two numeric distributions.
    """
    return float(wasserstein_distance(feature_expected, feature_actual))
