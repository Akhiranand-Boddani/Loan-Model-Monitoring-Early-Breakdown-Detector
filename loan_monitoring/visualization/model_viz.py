# --- Model Diagnostics Plots ---
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

def plot_roc_curve(y_true, y_proba, model_name):
    """
    Plot ROC curve for a model.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    return fig

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot confusion matrix for a model.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'],
           title=f'Confusion Matrix - {model_name}', ylabel='True label', xlabel='Predicted label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def plot_prediction_distribution(y_proba, model_name):
    """
    Plot distribution of predicted probabilities for a model.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(y_proba, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'Prediction Probability Distribution - {model_name}')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    return fig

# Visualization interface functions for Streamlit app
from loan_monitoring.visualization.baseline_viz import (
    plot_numeric_histograms, plot_categorical_bars, plot_correlation_heatmap
)
from loan_monitoring.visualization.drift_viz import (
    plot_numeric_drift, plot_categorical_drift, plot_prediction_comparison
)

def show_baseline_distribution(df_baseline, numeric_features, categorical_features):
    """
    Display baseline distribution plots in Streamlit.
    """
    import streamlit as st
    st.header("Baseline Distribution Plots")
    fig1 = plot_numeric_histograms(df_baseline, numeric_features)
    st.pyplot(fig1)

    fig2 = plot_categorical_bars(df_baseline, categorical_features)
    st.pyplot(fig2)

def show_drift_comparison(df_baseline, df_drift, numeric_features):
    """
    Display baseline vs drift comparison plots in Streamlit.
    """
    import streamlit as st
    st.header("Baseline vs Drift Comparison")
    fig3 = plot_numeric_drift(df_baseline, df_drift, numeric_features)
    st.pyplot(fig3)
