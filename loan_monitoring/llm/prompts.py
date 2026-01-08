FEATURE_DRIFT_PROMPT = """
Feature Drift Analysis Summary

Top Numeric Drifted Features:
{numeric_table}

Top Categorical Drifted Features:
{categorical_table}

Prediction Output Summary:
{prediction_summary}

Performance Changes (if available):
{performance_summary}

Interpret these results and explain what they indicate
about the quality and reliability of the model on the new data.
"""

EXPLAINATION_PREFIX = "Provide a clear explanation of the drift analysis results."
