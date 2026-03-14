"""
=============================================================================
    Car Price Prediction - Utility Functions
    =========================================
    Shared helpers for data loading, cleaning, and feature engineering.
=============================================================================
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV dataset and return a pandas DataFrame.
    Handles common encoding issues automatically.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    
    df = pd.read_csv(path)
    print(f"✅ Dataset loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def display_basic_info(df: pd.DataFrame) -> None:
    """Print shape, dtypes, null counts, and sample rows."""
    print("\n" + "=" * 60)
    print("  DATASET OVERVIEW")
    print("=" * 60)
    print(f"\n📐 Shape          : {df.shape}")
    print(f"📊 Total values   : {df.size}")
    print(f"🔢 Numeric cols   : {df.select_dtypes(include='number').columns.tolist()}")
    print(f"🔤 Categorical    : {df.select_dtypes(include='object').columns.tolist()}")
    
    print("\n--- Null Values ---")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls.sum() > 0 else "No null values found ✅")
    
    print("\n--- Duplicates ---")
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows: {dup_count}")
    
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    print("\n--- First 5 Rows ---")
    print(df.head())


# ──────────────────────────────────────────────
#  DATA CLEANING
# ──────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the car dataset:
      1. Drop the ID column (not useful for prediction).
      2. Convert Levy to numeric (replace '-' with NaN, then fill with median).
      3. Parse Engine volume (strip ' Turbo', convert to float).
      4. Parse Mileage (strip ' km', convert to int).
      5. Convert Prod. year to int.
      6. Convert Cylinders to int.
      7. Fix Doors column (map string labels to ints).
      8. Drop duplicates.
    """
    df = df.copy()
    
    # --- Drop ID (unique identifier, no predictive power) ---
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
    
    # --- Levy: replace '-' with NaN, fill with median ---
    if "Levy" in df.columns:
        df["Levy"] = df["Levy"].replace("-", np.nan)
        df["Levy"] = pd.to_numeric(df["Levy"], errors="coerce")
        df["Levy"].fillna(df["Levy"].median(), inplace=True)
    
    # --- Engine volume: extract numeric portion & create Turbo flag ---
    if "Engine volume" in df.columns:
        df["Turbo"] = df["Engine volume"].astype(str).str.contains("Turbo", case=False).astype(int)
        df["Engine volume"] = (
            df["Engine volume"]
            .astype(str)
            .str.replace("Turbo", "", case=False, regex=False)
            .str.strip()
        )
        df["Engine volume"] = pd.to_numeric(df["Engine volume"], errors="coerce")
        df["Engine volume"].fillna(df["Engine volume"].median(), inplace=True)
    
    # --- Mileage: strip ' km' and convert ---
    if "Mileage" in df.columns:
        df["Mileage"] = (
            df["Mileage"]
            .astype(str)
            .str.replace("km", "", case=False, regex=False)
            .str.strip()
            .str.replace(",", "")
        )
        df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
        df["Mileage"].fillna(df["Mileage"].median(), inplace=True)
    
    # --- Cylinders ---
    if "Cylinders" in df.columns:
        df["Cylinders"] = pd.to_numeric(df["Cylinders"], errors="coerce")
        df["Cylinders"].fillna(df["Cylinders"].median(), inplace=True)
        df["Cylinders"] = df["Cylinders"].astype(int)
    
    # --- Prod. year ---
    if "Prod. year" in df.columns:
        df["Prod. year"] = pd.to_numeric(df["Prod. year"], errors="coerce")
        # Create a derived feature: age of car
        current_year = 2026
        df["Car_Age"] = current_year - df["Prod. year"]
    
    # --- Doors: convert '02-Mar' / '04-May' / '>5' to ints ---
    if "Doors" in df.columns:
        door_map = {"02-Mar": 3, "04-May": 5, ">5": 6}
        df["Doors"] = df["Doors"].map(door_map).fillna(5).astype(int)
    
    # --- Leather interior: convert to binary ---
    if "Leather interior" in df.columns:
        df["Leather interior"] = df["Leather interior"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    
    # --- Price: make sure it is numeric ---
    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df.dropna(subset=["Price"], inplace=True)
    
    # --- Airbags: ensure numeric ---
    if "Airbags" in df.columns:
        df["Airbags"] = pd.to_numeric(df["Airbags"], errors="coerce")
        df["Airbags"].fillna(0, inplace=True)
        df["Airbags"] = df["Airbags"].astype(int)
    
    # --- Drop duplicates ---
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before - after > 0:
        print(f"🗑️  Removed {before - after} duplicate rows.")
    
    # --- Drop columns not useful for modelling ---
    # 'Model' has too many unique values (high cardinality) → drop
    if "Model" in df.columns:
        df.drop(columns=["Model"], inplace=True)
    
    print(f"✅ Cleaning complete. Shape: {df.shape}")
    return df


# ──────────────────────────────────────────────
#  FEATURE / TARGET SPLIT
# ──────────────────────────────────────────────

def split_features_target(df: pd.DataFrame, target: str = "Price"):
    """Return (X, y) after separating the target column."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def get_train_test(X, y, test_size=0.2, random_state=42):
    """80-20 stratified-ish split with fixed random seed for reproducibility."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
