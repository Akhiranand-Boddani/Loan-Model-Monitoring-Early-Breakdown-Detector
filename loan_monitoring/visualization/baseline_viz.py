import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List

sns.set_style("whitegrid")


# ---------------------------------------------
# Numeric Feature Distribution
# ---------------------------------------------
def plot_numeric_histograms(
    df: pd.DataFrame,
    numeric_features: List[str],
    bins: int = 30,
    figsize: tuple = (12, 8)
):
    """
    Plot histograms of numeric features.
    Returns a figure object.
    """
    n_feats = len(numeric_features)
    n_cols = 3
    n_rows = int(np.ceil(n_feats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    i = -1
    for i, feat in enumerate(numeric_features):
        sns.histplot(df[feat].dropna().astype(float).to_numpy(), bins=bins, ax=axes[i])
        axes[i].set_title(feat)

    # hide empty axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


# ---------------------------------------------
# Categorical Feature Bar Plots
# ---------------------------------------------
def plot_categorical_bars(
    df: pd.DataFrame,
    categorical_features: List[str],
    figsize: tuple = (12, 8)
):
    """
    Plot bar plots for categorical features.
    """
    fig, axes = plt.subplots(len(categorical_features), 1, figsize=figsize)

    for i, feat in enumerate(categorical_features):
        counts = df[feat].value_counts(normalize=True) * 100
        sns.barplot(x=counts.index, y=counts.values, ax=axes[i])
        axes[i].set_title(feat)
        axes[i].set_ylabel("Percentage")
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


# ---------------------------------------------
# Correlation Heatmap
# ---------------------------------------------
def plot_correlation_heatmap(
    df: pd.DataFrame,
    numeric_features: List[str],
    figsize: tuple = (12, 10)
):
    """
    Plot correlation matrix heatmap for numeric features only.
    Filters to only include numeric columns to avoid errors.
    """
    # Ensure we only have numeric features
    numeric_cols = df[numeric_features].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None  # Not enough numeric features for correlation
    
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar_kws={"label": "Correlation"})
    ax.set_title("Numeric Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ---------------------------------------------
# Model Calibration Plot
# ---------------------------------------------
def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    figsize: tuple = (8, 6)
):
    """
    Plot a calibration curve for model predictions.
    """
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(mean_pred, frac_pos, marker='o')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")

    return fig
