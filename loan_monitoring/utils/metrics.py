import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare, wasserstein_distance, entropy
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
)
from scipy.stats import norm, fisher_exact
import math


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


def chi_square_test(expected_proportions, actual_proportions):
    """
    Compute chi-square statistic given proportions for categories.
    Both expected_proportions and actual_proportions are dicts {category: proportion}.
    
    Normalizes to ensure sums match (required by scipy.stats.chisquare).
    """
    # Align categories from both distributions
    categories = list(set(expected_proportions.keys()).union(set(actual_proportions.keys())))
    
    # Get proportions for all categories (use 0 for missing categories)
    exp_props = np.array([expected_proportions.get(cat, 0) for cat in categories])
    act_props = np.array([actual_proportions.get(cat, 0) for cat in categories])
    
    # Normalize to ensure they sum to 1.0 (handle floating point errors)
    exp_props = exp_props / (exp_props.sum() + 1e-10)
    act_props = act_props / (act_props.sum() + 1e-10)
    
    # Convert to counts (scale to 1000 for numerical stability)
    scale = 1000
    exp_counts = exp_props * scale
    act_counts = act_props * scale
    
    # Remove categories with zero counts in both (they don't contribute to chi-square)
    valid_idx = (exp_counts > 0) | (act_counts > 0)
    exp_counts = exp_counts[valid_idx]
    act_counts = act_counts[valid_idx]
    
    # If no valid categories, return no drift
    if len(exp_counts) == 0:
        return {"chi2_stat": 0.0, "p_value": 1.0}
    
    try:
        stat, p = chisquare(f_obs=act_counts, f_exp=exp_counts)
        return {"chi2_stat": float(stat), "p_value": float(p)}
    except ValueError as e:
        # If still fails, return safe defaults
        return {"chi2_stat": 0.0, "p_value": 1.0, "error": str(e)}


def wasserstein(feature_expected, feature_actual):
    """
    Wasserstein (Earth Mover's) distance between two numeric distributions.
    """
    return float(wasserstein_distance(feature_expected, feature_actual))


def kl_divergence(expected, actual, bins=30):
    """
    Kullback-Leibler (KL) Divergence between two distributions.
    Measures how much one distribution differs from expected distribution.
    
    Formula: sum( P(x) * log(P(x) / Q(x)) )
    
    Interpretation:
    - 0 = identical distributions
    - < 0.1 = minor drift
    - 0.1-0.25 = moderate drift
    - > 0.25 = significant drift
    """
    expected_arr = np.array(expected).astype(float)
    actual_arr = np.array(actual).astype(float)
    
    # Create histograms with shared bins
    min_val = min(expected_arr.min(), actual_arr.min())
    max_val = max(expected_arr.max(), actual_arr.max())
    
    hist_exp, _ = np.histogram(expected_arr, bins=bins, range=(min_val, max_val))
    hist_act, _ = np.histogram(actual_arr, bins=bins, range=(min_val, max_val))
    
    # Normalize to probabilities
    p = hist_exp / (hist_exp.sum() + 1e-10)
    q = hist_act / (hist_act.sum() + 1e-10)
    
    # Compute KL divergence with small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    kl_div = np.sum(p * np.log(p / q))
    
    return float(max(0, kl_div))  # KL divergence is always >= 0


# -------------------------
# Additional statistical helpers
# -------------------------
def wilson_ci(successes, n, alpha=0.05):
    """Compute Wilson score confidence interval for a proportion.

    Returns (lower, upper).
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    z = norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    adj = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return (max(0.0, lower), min(1.0, upper))


def two_proportion_z_test(succ1, n1, succ2, n2):
    """Two-proportion z-test; returns two-sided p-value and z-statistic.
    If counts are small, caller may prefer Fisher exact test.
    """
    if n1 <= 0 or n2 <= 0:
        return 1.0, 0.0
    p1 = succ1 / n1
    p2 = succ2 / n2
    p_pool = (succ1 + succ2) / (n1 + n2)
    denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if denom == 0:
        return 1.0, 0.0
    z = (p1 - p2) / denom
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return p_value, z


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR correction. Returns list of adjusted p-values in original order."""
    n = len(pvals)
    if n == 0:
        return []
    # pair with original indices
    indexed = sorted([(p, i) for i, p in enumerate(pvals)], key=lambda x: x[0])
    adjusted = [0.0] * n
    prev_adj = 0.0
    for rank, (p, i) in enumerate(indexed, start=1):
        adj = p * n / rank
        adj = min(adj, 1.0)
        # ensure monotonicity
        adj = max(adj, prev_adj)
        adjusted[i] = adj
        prev_adj = adj
    # enforce monotonic decreasing if needed
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    return adjusted

