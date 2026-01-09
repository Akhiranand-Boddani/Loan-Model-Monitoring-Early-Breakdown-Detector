import pandas as pd
import numpy as np
from typing import Dict, List
from scipy.stats import entropy

from loan_monitoring.utils.metrics import psi, ks_test, chi_square_test, wasserstein, kl_divergence
from loan_monitoring.data_processing.load_data import detect_feature_types
from loan_monitoring.utils.persistence import load_json
from loan_monitoring.data_processing.load_data import load_csv
from loan_monitoring.config import BASELINE_DATA_FILE


# Drift detection thresholds (industry standard)
DRIFT_THRESHOLDS = {
    "psi": 0.25,              # PSI > 0.25 = significant drift
    "ks_p_value": 0.05,       # KS p-value < 0.05 = significant drift
    "chi2_p_value": 0.05,     # Chi-square p-value < 0.05 = significant drift
    "wasserstein": 0.1,       # Wasserstein distance > 0.1 = drift (normalized)
    "kl_divergence": 0.1      # KL div > 0.1 = drift
}


def compute_feature_drift(
    baseline_stats: dict,
    df_new: pd.DataFrame,
    ignore_columns: List[str],
    thresholds: dict = None,
    flagging_policy: str = "any"
) -> Dict:
    """
    Comprehensive feature drift detection using multiple statistical tests.
    
    Returns:
        Dict with numeric & categorical drift metrics, p-values, and drift flags.
    """

    # Use provided thresholds or fall back to defaults
    thresholds = thresholds or DRIFT_THRESHOLDS

    # Use provided thresholds or fall back to defaults
    thresholds = thresholds or DRIFT_THRESHOLDS

    # load empirical baseline dataset if available
    try:
        baseline_df = load_csv(BASELINE_DATA_FILE)
    except Exception:
        baseline_df = None

    # detect feature types
    numeric_features, categorical_features = detect_feature_types(df_new, ignore_columns)

    drift_summary = {}

    # =====================
    # NUMERIC FEATURES
    # =====================
    for feat in numeric_features:
        if feat not in baseline_stats.get("numeric", {}):
            continue
        base_stats = baseline_stats["numeric"][feat]
        new_vals = df_new[feat].dropna().values.astype(float)
        
        if len(new_vals) == 0:
            continue  # Skip if no valid values
        
        # Use empirical baseline samples when available to avoid synthetic-vs-real mismatch
        if baseline_df is not None and feat in baseline_df.columns:
            base_vals = baseline_df[feat].dropna().values.astype(float)
            # sample to match size of new_vals for fair comparison
            if len(base_vals) > len(new_vals) and len(new_vals) > 0:
                base_vals = np.random.choice(base_vals, size=len(new_vals), replace=False)
            elif len(base_vals) < len(new_vals) and len(base_vals) > 0:
                # upsample baseline if needed
                base_vals = np.random.choice(base_vals, size=len(new_vals), replace=True)
        else:
            # Fallback: approximate baseline distribution from stored stats
            n_samples = max(len(new_vals), 1000)
            base_mean = base_stats.get("mean", 0)
            base_std = base_stats.get("std", 1)
            base_min = base_stats.get("min", base_mean - 3 * base_std)
            base_max = base_stats.get("max", base_mean + 3 * base_std)
            base_vals = np.random.normal(loc=base_mean, scale=base_std, size=n_samples)
            base_vals = np.clip(base_vals, base_min, base_max)
        
        # Compute all drift metrics
        try:
            psi_score = psi(base_vals, new_vals)
            ks_result = ks_test(base_vals, new_vals)
            wasserstein_dist = wasserstein(base_vals, new_vals)
            kl_div = kl_divergence(base_vals, new_vals)
        except Exception as e:
            # If any metric fails, skip this feature
            continue
        
        # Determine feature range for normalization
        try:
            base_min = float(np.min(base_vals))
            base_max = float(np.max(base_vals))
        except Exception:
            base_min = base_stats.get("min", 0)
            base_max = base_stats.get("max", base_min + 1)
        feat_range = base_max - base_min
        wasserstein_normalized = wasserstein_dist / (feat_range + 1e-6) if feat_range > 0 else 0
        
        # Determine per-metric flags
        flags = {
            "psi": psi_score > thresholds["psi"],
            "ks_p": ks_result["p_value"] < thresholds["ks_p_value"],
            "wasserstein": wasserstein_normalized > thresholds["wasserstein"],
            "kl": kl_div > thresholds["kl_divergence"]
        }

        # Evaluate according to chosen flagging policy
        def policy_evaluate(flags_dict, policy="any"):
            # convert bools to ints
            vals = list(map(int, flags_dict.values()))
            total = sum(vals)
            if policy == "any":
                return total >= 1
            if policy == "ks_and_effect":
                # require KS and at least one effect-size (psi/wasserstein/kl)
                return flags_dict.get("ks_p", False) and (flags_dict.get("psi", False) or flags_dict.get("wasserstein", False) or flags_dict.get("kl", False))
            if policy == "majority":
                # require majority of the 4 metrics
                return total >= 2
            # default to any
            return total >= 1

        is_drifted = policy_evaluate(flags, flagging_policy)
        
        drift_summary[feat] = {
            "feature_type": "numeric",
            "is_drift": is_drifted,
            "metrics": {
                "psi": round(psi_score, 4),
                "ks_statistic": round(ks_result["ks_stat"], 4),
                "ks_p_value": round(ks_result["p_value"], 6),
                "wasserstein_distance": round(wasserstein_dist, 4),
                "wasserstein_normalized": round(wasserstein_normalized, 4),
                "kl_divergence": round(kl_div, 4)
            },
            "metric_flags": flags,
            "thresholds": {
                "psi_threshold": thresholds["psi"],
                "ks_p_threshold": thresholds["ks_p_value"],
                "wasserstein_threshold": thresholds["wasserstein"],
                "kl_threshold": thresholds["kl_divergence"]
            },
            "baseline_stats": {
                "mean": round(base_stats.get("mean", 0), 4),
                "std": round(base_stats.get("std", 0), 4),
                "min": round(base_stats.get("min", 0), 4),
                "max": round(base_stats.get("max", 0), 4)
            },
            "new_stats": {
                "mean": round(float(new_vals.mean()), 4),
                "std": round(float(new_vals.std()), 4),
                "min": round(float(new_vals.min()), 4),
                "max": round(float(new_vals.max()), 4)
            }
        }

    # =====================
    # CATEGORICAL FEATURES
    # =====================
    for feat in categorical_features:
        if feat not in baseline_stats.get("categorical", {}):
            continue
            
        base_stats = baseline_stats["categorical"][feat]
        base_props = base_stats.get("proportions", {})
        
        new_props = df_new[feat].astype(str).value_counts(normalize=True).to_dict()
        
        # Chi-square test on proportions (function handles normalization)
        chi2_result = chi_square_test(base_props, new_props)
        
        # Align categories for KL divergence
        all_categories = list(set(base_props.keys()).union(set(new_props.keys())))
        base_props_aligned = np.array([base_props.get(cat, 1e-10) for cat in all_categories])
        new_props_aligned = np.array([new_props.get(cat, 1e-10) for cat in all_categories])
        
        # Normalize
        base_props_aligned = base_props_aligned / base_props_aligned.sum()
        new_props_aligned = new_props_aligned / new_props_aligned.sum()
        
        kl_cat = kl_divergence(base_props_aligned, new_props_aligned)
        
        # Flag as drift if p-value < threshold or high KL divergence
        is_drifted = (
            chi2_result["p_value"] < thresholds["chi2_p_value"] or
            kl_cat > thresholds["kl_divergence"]
        )
        
        # Compute entropy for new data
        new_props_values = np.array(list(new_props.values()))
        new_props_values = new_props_values / new_props_values.sum()  # Normalize
        entropy_new = float(entropy(new_props_values))
        
        drift_summary[feat] = {
            "feature_type": "categorical",
            "is_drift": is_drifted,
            "metrics": {
                "chi2_statistic": round(chi2_result["chi2_stat"], 4),
                "chi2_p_value": round(chi2_result["p_value"], 6),
                "kl_divergence": round(kl_cat, 4),
                "entropy_baseline": round(base_stats.get("entropy", 0), 4),
                "entropy_new": round(entropy_new, 4)
            },
            "thresholds": {
                "chi2_p_threshold": thresholds["chi2_p_value"],
                "kl_threshold": thresholds["kl_divergence"]
            },
            "baseline_proportions": {cat: round(prop, 4) for cat, prop in list(base_props.items())[:5]},
            "new_proportions": {cat: round(prop, 4) for cat, prop in list(new_props.items())[:5]}
        }

    return drift_summary


def compute_conditional_numeric_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame, feature: str, target_col: str):
    """
    Compute conditional (per-class) drift metrics for a numeric feature comparing baseline and new datasets.
    Returns dict keyed by class label with psi and KS results.
    """
    results = {}
    for cls in sorted(baseline_df[target_col].dropna().unique()):
        base_vals = baseline_df.loc[baseline_df[target_col] == cls, feature].dropna().values.astype(float)
        new_vals = new_df.loc[new_df[target_col] == cls, feature].dropna().values.astype(float)
        if len(base_vals) < 10 or len(new_vals) < 10:
            results[cls] = {"psi": None, "ks_stat": None, "p_value": None}
            continue
        try:
            psi_score = psi(base_vals, new_vals)
            ks_res = ks_test(base_vals, new_vals)
            results[cls] = {"psi": round(float(psi_score), 4), "ks_stat": round(float(ks_res.get('ks_stat', 0)), 4), "p_value": round(float(ks_res.get('p_value', 1)), 6)}
        except Exception:
            results[cls] = {"psi": None, "ks_stat": None, "p_value": None}
    return results


def compute_conditional_categorical_drift(baseline_df: pd.DataFrame, new_df: pd.DataFrame, feature: str, target_col: str):
    """
    Compute conditional (per-class) categorical drift: chi2 + KL per class.
    """
    results = {}
    for cls in sorted(baseline_df[target_col].dropna().unique()):
        base_props = baseline_df.loc[baseline_df[target_col] == cls, feature].astype(str).value_counts(normalize=True).to_dict()
        new_props = new_df.loc[new_df[target_col] == cls, feature].astype(str).value_counts(normalize=True).to_dict()
        try:
            chi2 = chi_square_test(base_props, new_props)
            # align categories
            all_cats = list(set(base_props.keys()).union(set(new_props.keys())))
            bp = np.array([base_props.get(c, 1e-10) for c in all_cats])
            np_ = np.array([new_props.get(c, 1e-10) for c in all_cats])
            bp = bp / bp.sum()
            np_ = np_ / np_.sum()
            klv = kl_divergence(bp, np_)
            results[cls] = {"chi2_p": round(float(chi2.get('p_value', 1)), 6), "chi2_stat": round(float(chi2.get('chi2_stat', 0)), 4), "kl": round(float(klv), 4)}
        except Exception:
            results[cls] = {"chi2_p": None, "chi2_stat": None, "kl": None}
    return results
