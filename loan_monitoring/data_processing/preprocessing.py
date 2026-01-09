import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, List

from loan_monitoring.data_processing.load_data import detect_feature_types

# ---------------------------------------------------
# 1) Global Preprocessing Helpers
# ---------------------------------------------------

def get_feature_lists(
    df: pd.DataFrame,
    ignore_columns: List[str],
    target_col: str
) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical feature lists.
    """
    # ignore target and ID columns
    feature_columns = [c for c in df.columns if c not in ignore_columns + [target_col]]

    # detect types
    numeric_features, categorical_features = detect_feature_types(df, ignore_columns + [target_col])

    # ensure we only return allowed features
    numeric_features = [c for c in numeric_features if c in feature_columns]
    categorical_features = [c for c in categorical_features if c in feature_columns]

    return numeric_features, categorical_features


# ---------------------------------------------------
# 2) Preprocessing Pipeline Builder
# ---------------------------------------------------

def build_preprocessing_pipeline(
    df: pd.DataFrame,
    ignore_columns: List[str],
    target_col: str
) -> Tuple[Pipeline, List[str]]:
    """
    Builds a preprocessing pipeline with imputation and encoding.

    For:
    - Numeric: impute median
    - Categorical: impute most frequent + one-hot

    Returns:
        pipeline: sklearn ColumnTransformer pipeline
        feature_names: list of processed feature names
    """

    num_feats, cat_feats = get_feature_lists(df, ignore_columns, target_col)

    # impute numeric -> median, then keep as is
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    # impute categorical -> most frequent, then one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_feats),
            ("cat", categorical_transformer, cat_feats)
        ]
    )

    # Only return the pipeline and the feature lists (feature names after fitting, if needed, should be extracted after fit)
    return preprocessor, num_feats + cat_feats
