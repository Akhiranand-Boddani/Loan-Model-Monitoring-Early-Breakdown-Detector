"""
Threshold sensitivity reporter.
Usage:
    python tools/threshold_report.py --new data/drift_sample_1.csv

This script loads baseline stats, computes feature drift metrics against a new dataset,
and reports current metric values plus the threshold value required to avoid flagging
for each metric.
"""
import argparse
import json
import numpy as np
import pandas as pd
import sys
import os

# ensure repository root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from loan_monitoring.config import BASELINE_STATS_PATH, ID_COLUMN, TARGET_COLUMN
from loan_monitoring.utils.persistence import load_json
from loan_monitoring.drift.drift_metrics import compute_feature_drift, DRIFT_THRESHOLDS
from loan_monitoring.data_processing.load_data import load_csv


def required_thresholds_for_feature(feat_metrics):
    """Return a dict of required thresholds per metric to avoid flag."""
    res = {}
    metrics = feat_metrics.get("metrics", {})
    # PSI: need threshold > psi to avoid flag
    psi = metrics.get("psi")
    if psi is not None:
        res["psi_required"] = float(psi)
    # KS p-value: to avoid flag, threshold must be <= p_value
    ks_p = metrics.get("ks_p_value")
    if ks_p is not None:
        res["ks_p_required"] = float(ks_p)
    # Wasserstein normalized: need threshold > value
    w = metrics.get("wasserstein_normalized")
    if w is not None:
        res["wasserstein_required"] = float(w)
    # KL divergence: need threshold > value
    kl = metrics.get("kl_divergence")
    if kl is not None:
        res["kl_required"] = float(kl)
    return res


def print_report(baseline_stats, df_new, ignore_columns=None):
    ignore_columns = ignore_columns or []
    drift_summary = compute_feature_drift(baseline_stats, df_new, ignore_columns=ignore_columns)
    print("Feature, PSI, PSI_thr, KS_p, KS_thr, Wasser_norm, Wasser_thr, KL, KL_thr, Flagging_metrics")
    for feat, info in drift_summary.items():
        if info.get("feature_type") != "numeric":
            continue
        m = info.get("metrics", {})
        psi_v = m.get("psi")
        ks_p = m.get("ks_p_value")
        w = m.get("wasserstein_normalized")
        kl = m.get("kl_divergence")
        flags = []
        if psi_v is not None and psi_v > DRIFT_THRESHOLDS.get("psi", 0.25):
            flags.append("psi")
        if ks_p is not None and ks_p < DRIFT_THRESHOLDS.get("ks_p_value", 0.05):
            flags.append("ks_p")
        if w is not None and w > DRIFT_THRESHOLDS.get("wasserstein", 0.1):
            flags.append("wasserstein")
        if kl is not None and kl > DRIFT_THRESHOLDS.get("kl_divergence", 0.1):
            flags.append("kl")
        req = required_thresholds_for_feature(info)
        print(f"{feat}, {psi_v}, {DRIFT_THRESHOLDS.get('psi')}, {ks_p}, {DRIFT_THRESHOLDS.get('ks_p_value')}, {w}, {DRIFT_THRESHOLDS.get('wasserstein')}, {kl}, {DRIFT_THRESHOLDS.get('kl_divergence')}, {','.join(flags)}")
        print("  Required thresholds to avoid flag:", req)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", help="Path to new CSV file", required=False)
    args = parser.parse_args()

    baseline_stats = load_json(BASELINE_STATS_PATH)

    new_path = args.new or "data/drift_sample_1.csv"
    df_new = load_csv(new_path)

    print_report(baseline_stats, df_new, ignore_columns=[ID_COLUMN, TARGET_COLUMN])
