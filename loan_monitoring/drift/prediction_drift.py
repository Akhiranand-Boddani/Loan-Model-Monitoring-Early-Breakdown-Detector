import numpy as np
import pandas as pd
from typing import Dict

from loan_monitoring.utils.scoring import confidence_stats, prediction_entropy
from loan_monitoring.utils.persistence import load_model


def compute_prediction_drift(
    df_new: pd.DataFrame,
    model_paths: Dict[str, str]
) -> Dict:
    """
    Compute prediction output drift for each model.
    """
    pred_drift = {}

    for model_name, mpath in model_paths.items():
        model = load_model(mpath)

        # get probabilities
        y_proba = model.predict_proba(df_new)[:, 1]
        stats = confidence_stats(y_proba)

        pred_drift[model_name] = {
            "mean_proba": stats["mean_proba"],
            "std_proba": stats["std_proba"],
            "entropy_mean": stats["entropy_mean"],
            "entropy_std": stats["entropy_std"]
        }

    return pred_drift
