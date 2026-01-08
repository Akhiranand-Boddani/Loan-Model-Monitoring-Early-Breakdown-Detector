import pandas as pd
import numpy as np
from typing import Tuple

def apply_numeric_shift(
    df: pd.DataFrame, column: str, shift_type: str = "scale", factor: float = 1.2
) -> pd.Series:
    """
    Apply a drift transformation to numeric feature.
      - shift_type = "scale": multiply values by factor (>1 increases, <1 decreases)
      - shift_type = "add": add constant factor
    """
    if shift_type == "scale":
        return df[column] * factor
    elif shift_type == "add":
        return df[column] + factor
    else:
        raise ValueError(f"Unknown shift_type: {shift_type}")


def apply_categorical_shift(
    df: pd.DataFrame, column: str, new_dist: dict
) -> pd.Series:
    """
    Change categorical distribution by resampling according to new distribution.
    new_dist should be dict {category: probability}, and sum(probs)=1.
    """
    categories = list(new_dist.keys())
    probs = list(new_dist.values())
    n = len(df)

    return np.random.choice(categories, size=n, p=probs)


def generate_drift_dataset(
    df: pd.DataFrame,
    output_path: str,
    numeric_shifts: dict,
    categorical_shifts: dict,
    seed: int = 42
):
    """
    Generates a drifted version of baseline dataset.

    Args:
        df: baseline DataFrame
        output_path: where to save drift CSV
        numeric_shifts: dict like {"Income": ("scale", 0.8), "CreditScore": ("add", -30)}
        categorical_shifts: dict like {"EmploymentType": {"Salaried": 0.3, "Self-employed": 0.5, ...}}
        seed: random state
    """
    np.random.seed(seed)

    df_drift = df.copy()

    # Apply numeric shifts
    for col, (shift_type, factor) in numeric_shifts.items():
        if col in df_drift.columns:
            df_drift[col] = apply_numeric_shift(df_drift, col, shift_type, factor)

    # Apply categorical shifts
    for col, new_dist in categorical_shifts.items():
        if col in df_drift.columns:
            df_drift[col] = apply_categorical_shift(df_drift, col, new_dist)

    # Save to CSV
    df_drift.to_csv(output_path, index=False)
    print(f"Drift dataset saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    baseline_file = "data/baseline.csv"
    df_base = pd.read_csv(baseline_file)

    # Numeric shifts — e.g., simulate economic downturn
    numeric_shifts_example = {
        "Income": ("scale", 0.7),         # Reduce incomes by 30%
        "CreditScore": ("add", -40),      # Lower credit scores by 40 points
        "DTIRatio": ("scale", 1.3),       # Increase DTI by 30%
        "MonthsEmployed": ("scale", 0.8)  # Reduce employment stability
    }

    # Categorical shifts — adjust proportions
    categorical_shifts_example = {
        "EmploymentType": {
            "Salaried": 0.4,
            "Self-employed": 0.4,
            "Contract": 0.2
        },
        "LoanPurpose": {
            "Home": 0.15,
            "Auto": 0.15,
            "Education": 0.1,
            "Business": 0.4,
            "Personal": 0.2
        },
        "HasCoSigner": {
            "Yes": 0.2,
            "No": 0.8
        }
    }

    generate_drift_dataset(
        df=df_base,
        output_path="data/drift_generated.csv",
        numeric_shifts=numeric_shifts_example,
        categorical_shifts=categorical_shifts_example
    )
