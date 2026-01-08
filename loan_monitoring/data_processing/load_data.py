import pandas as pd
from typing import Tuple, List


def load_csv(
    filepath: str,
    expected_columns: List[str] = None,
    required_target: str = None
) -> pd.DataFrame:
    """
    Load a CSV file from the given filepath.

    Args:
        filepath (str): Path to CSV file.
        expected_columns (List[str], optional):
            Expected columns for schema validation.
        required_target (str, optional):
            Name of the target column that must be present.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """

    df = pd.read_csv(filepath)

    # Basic sanity checks
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing expected columns: {missing_cols} in {filepath}"
            )

    if required_target:
        if required_target not in df.columns:
            raise ValueError(
                f"Target column '{required_target}' not found in {filepath}"
            )

    return df


def detect_feature_types(
    df: pd.DataFrame,
    ignore_columns: List[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Detect numeric and categorical features.

    Args:
        df (pd.DataFrame): Input dataset.
        ignore_columns (List[str], optional):
            Columns to ignore (e.g., LoanID, Target).

    Returns:
        Tuple[List[str], List[str]]:
            A tuple of (numeric_features, categorical_features).
    """

    if ignore_columns is None:
        ignore_columns = []

    numeric_features = []
    categorical_features = []

    for col in df.columns:
        if col in ignore_columns:
            continue

        # pandas dtype check
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features


if __name__ == "__main__":
    # Example quick test
    filepath = "./loan_drift_monitoring/data/baseline.csv"
    df = load_csv(filepath)
    num_feats, cat_feats = detect_feature_types(df, ignore_columns=["LoanID", "Default"])

    print("Loaded dataframe:", df.shape)
    print("Numeric:", num_feats)
    print("Categorical:", cat_feats)
