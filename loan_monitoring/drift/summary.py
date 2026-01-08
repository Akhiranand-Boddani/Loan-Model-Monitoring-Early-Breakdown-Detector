from typing import Dict
import numpy as np


def importance_rank(drift_dict: Dict, top_k: int = 5) -> Dict:
    """
    Pick the top K drifting features by PSI / chi2.
    """
    ranked = {}

    for feat_type, feats in drift_dict.items():
        vals = []
        for f, scores in feats.items():
            # numeric: psi strongest
            if feat_type == "numeric":
                vals.append((f, scores.get("psi", 0)))
            else:
                vals.append((f, scores.get("chi2_stat", 0)))
        ranked[feat_type] = sorted(vals, key=lambda x: x[1], reverse=True)[:top_k]

    return ranked


def generate_drift_report(
    feature_drift: Dict,
    prediction_drift: Dict,
    performance_drift: Dict = None
) -> Dict:
    """
    Collate all drift components into a dict for UI/LLM.
    """
    summary = {
        "top_numeric_drift": importance_rank(feature_drift["numeric"]),
        "top_categorical_drift": importance_rank(feature_drift["categorical"]),
        "prediction_output_changes": prediction_drift
    }

    if performance_drift:
        summary["performance_changes"] = performance_drift

    return summary
