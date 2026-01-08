import pandas as pd
from typing import List


class SchemaValidationError(Exception):
    """Custom exception for schema validation errors."""
    pass


def validate_schema(
    df: pd.DataFrame,
    expected_columns: List[str],
    target_column: str,
    id_column: str = None
) -> bool:
    """
    Validate that the given DataFrame has the expected schema.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        expected_columns (List[str]): List of expected feature columns
            (excluding target and ID columns if provided).
        target_column (str): Name of the target column.
        id_column (str, optional): Name of the ID column.

    Returns:
        bool: True if the DataFrame is valid.

    Raises:
        SchemaValidationError: If the schema is invalid.
    """

    # All expected columns + target + id
    required_columns = set(expected_columns + [target_column])
    if id_column:
        required_columns.add(id_column)

    missing_columns = required_columns - set(df.columns)
    extra_columns = set(df.columns) - required_columns

    if missing_columns:
        raise SchemaValidationError(
            f"Missing required columns: {missing_columns}"
        )

    if extra_columns:
        raise SchemaValidationError(
            f"Extra unexpected columns: {extra_columns}"
        )

    # Basic type checks
    for col in expected_columns:
        if df[col].isna().all():
            raise SchemaValidationError(f"Column '{col}' is entirely NA")

    # Passed all checks
    return True
