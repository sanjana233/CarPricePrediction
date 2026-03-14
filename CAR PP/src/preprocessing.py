"""
=============================================================================
    Car Price Prediction - Data Preprocessing Pipeline
    ===================================================
    Creates sklearn Pipelines for numeric & categorical features.
    Combines them with ColumnTransformer for clean, reproducible transforms.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(X: pd.DataFrame):
    """
    Build a ColumnTransformer that:
      • Imputes + Scales numeric features
      • Imputes + OneHot-encodes categorical features
    
    Returns:
        preprocessor (ColumnTransformer): fitted-ready transformer
        numeric_features (list): column names
        categorical_features (list): column names
    """
    # Identify numeric vs categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    print(f"\n📊 Numeric features    ({len(numeric_features)}): {numeric_features}")
    print(f"🔤 Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # --- Numeric pipeline: impute missing → scale ---
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # --- Categorical pipeline: impute missing → one-hot encode ---
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # --- Combine both pipelines ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        remainder="drop"   # drop any columns not explicitly listed
    )
    
    return preprocessor, numeric_features, categorical_features
