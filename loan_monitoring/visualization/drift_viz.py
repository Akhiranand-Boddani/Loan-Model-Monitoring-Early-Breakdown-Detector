import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List

sns.set_style("whitegrid")


# ---------------------------------------------
# Baseline vs Drift Numeric Comparison
# ---------------------------------------------
def plot_numeric_drift(
    df_baseline: pd.DataFrame,
    df_drift: pd.DataFrame,
    numeric_features: List[str],
    bins: int = 30,
    figsize: tuple = (12, 8)
):
    """
    Plot numeric histograms for baseline vs drift.
    """
    n_feats = len(numeric_features)
    n_cols = 2
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, feat in enumerate(numeric_features):
        sns.histplot(df_baseline[feat].dropna().astype(float).to_numpy(), color="blue", label="baseline", bins=bins, ax=axes[i], alpha=0.5)
        sns.histplot(df_drift[feat].dropna().astype(float).to_numpy(), color="red", label="drift", bins=bins, ax=axes[i], alpha=0.5)
        axes[i].set_title(f"{feat} (Baseline vs Drift)")
        axes[i].legend()

    plt.tight_layout()
    return fig


# ---------------------------------------------
# Baseline vs Drift Categorical Comparison
# ---------------------------------------------
def plot_categorical_drift(
    df_baseline: pd.DataFrame,
    df_drift: pd.DataFrame,
    categorical_features: List[str],
    figsize: tuple = (12, 8)
):
    """
    Plot side-by-side bar charts comparing baseline vs drift.
    """
    fig, axes = plt.subplots(len(categorical_features), 1, figsize=figsize)

    for i, feat in enumerate(categorical_features):
        base_props = df_baseline[feat].value_counts(normalize=True) * 100
        drift_props = df_drift[feat].value_counts(normalize=True) * 100

        df_plot = pd.DataFrame({
            "baseline": base_props,
            "drift": drift_props
        }).fillna(0)

        df_plot.plot(kind="bar", ax=axes[i])
        axes[i].set_title(feat)
        axes[i].set_ylabel("Percentage")
        axes[i].legend(["Baseline", "Drift"])

    plt.tight_layout()
    return fig


# ---------------------------------------------
# Prediction Probability Comparison
# ---------------------------------------------
def plot_prediction_comparison(
    baseline_probas: np.ndarray,
    drift_probas: np.ndarray,
    model_name: str,
    bins: int = 30,
    figsize: tuple = (8, 6)
):
    """
    Plot predicted probability distribution for baseline vs drift.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(baseline_probas, color="blue", label="baseline", bins=bins, ax=ax, alpha=0.5)
    sns.histplot(drift_probas, color="red", label="drift", bins=bins, ax=ax, alpha=0.5)
    ax.set_title(f"Prediction Distribution ({model_name})")
    ax.legend()

    return fig
