"""
=============================================================================
    Car Price Prediction - Exploratory Data Analysis (EDA)
    ======================================================
    Generates and saves EDA visualisations to the outputs/ folder.
    Run:  python notebooks/eda.py
=============================================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root so we can import src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_data, display_basic_info, clean_data

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data set 2.csv")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colour palette
sns.set_style("whitegrid")
PALETTE = "viridis"


def save_fig(fig, name: str):
    """Save a matplotlib figure to the outputs directory."""
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📸 Saved: {path}")


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def main():
    # ── 1.  Load & display basic info ──
    df_raw = load_data(DATA_PATH)
    display_basic_info(df_raw)
    
    # ── 2.  Clean ──
    df = clean_data(df_raw)
    
    # ── 3.  Describe numeric columns ──
    print("\n--- Statistical Summary ---")
    print(df.describe())
    
    # ═══════════════════════════════════════════
    #  PLOT 1 – Correlation Heatmap
    # ═══════════════════════════════════════════
    # Insight: Helps identify multicollinearity and features strongly
    #          correlated with Price (e.g., Engine vol, Cylinders, Levy).
    print("\n🔥 Generating Correlation Heatmap …")
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Numeric Features", fontsize=16, pad=15)
    save_fig(fig, "01_correlation_heatmap.png")
    
    # ═══════════════════════════════════════════
    #  PLOT 2 – Price Distribution
    # ═══════════════════════════════════════════
    # Insight: Price is heavily right-skewed.  A log-transform could help
    #          some models converge faster, but tree-based models handle it.
    print("📊 Generating Price Distribution …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw price
    sns.histplot(df["Price"], bins=60, kde=True, color="#4C72B0", ax=axes[0])
    axes[0].set_title("Price Distribution")
    axes[0].set_xlabel("Price")
    
    # Log price – easier to read
    sns.histplot(np.log1p(df["Price"]), bins=60, kde=True, color="#DD8452", ax=axes[1])
    axes[1].set_title("Log(Price + 1) Distribution")
    axes[1].set_xlabel("log(Price)")
    
    fig.suptitle("Target Variable Analysis", fontsize=15, y=1.02)
    fig.tight_layout()
    save_fig(fig, "02_price_distribution.png")
    
    # ═══════════════════════════════════════════
    #  PLOT 3 – Box Plots (numeric vs Price)
    # ═══════════════════════════════════════════
    # Insight: Box plots reveal outliers in Mileage, Engine volume, etc.
    print("📦 Generating Box Plots …")
    num_cols = ["Levy", "Engine volume", "Mileage", "Cylinders", "Airbags", "Car_Age"]
    num_cols = [c for c in num_cols if c in df.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, col in enumerate(num_cols):
        ax = axes[idx // 3][idx % 3]
        sns.boxplot(y=df[col], color="#55A868", ax=ax)
        ax.set_title(f"Box Plot – {col}")
    fig.suptitle("Outlier Detection via Box Plots", fontsize=15, y=1.01)
    fig.tight_layout()
    save_fig(fig, "03_box_plots.png")
    
    # ═══════════════════════════════════════════
    #  PLOT 4 – Feature vs Price scatter/box
    # ═══════════════════════════════════════════
    # Insight: Newer cars (low Car_Age) → higher price.
    #          More cylinders → higher price.
    #          Mileage effect is non-linear.
    print("📈 Generating Feature-vs-Price Plots …")
    scatter_features = ["Car_Age", "Mileage", "Engine volume", "Levy"]
    scatter_features = [c for c in scatter_features if c in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, col in enumerate(scatter_features):
        ax = axes[idx // 2][idx % 2]
        ax.scatter(df[col], df["Price"], alpha=0.15, s=8, color="#C44E52")
        ax.set_xlabel(col)
        ax.set_ylabel("Price")
        ax.set_title(f"{col} vs Price")
    fig.suptitle("Feature vs Price Analysis", fontsize=15, y=1.01)
    fig.tight_layout()
    save_fig(fig, "04_feature_vs_price.png")
    
    # ═══════════════════════════════════════════
    #  PLOT 5 – Categorical distributions
    # ═══════════════════════════════════════════
    # Insight: HYUNDAI, TOYOTA, MERCEDES dominate. Sedan & Jeep most common.
    print("📊 Generating Categorical Feature Plots …")
    cat_cols = ["Manufacturer", "Category", "Fuel type", "Gear box type", "Drive wheels", "Wheel"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    for idx, col in enumerate(cat_cols):
        ax = axes[idx // 2][idx % 2]
        top = df[col].value_counts().head(15)
        sns.barplot(x=top.values, y=top.index, palette="mako", ax=ax)
        ax.set_title(f"Top 15 – {col}")
        ax.set_xlabel("Count")
    fig.suptitle("Categorical Feature Distribution", fontsize=15, y=1.01)
    fig.tight_layout()
    save_fig(fig, "05_categorical_distributions.png")
    
    # ═══════════════════════════════════════════
    #  PLOT 6 – Average price by Manufacturer (top 15)
    # ═══════════════════════════════════════════
    print("💰 Generating Avg Price by Manufacturer …")
    if "Manufacturer" in df.columns:
        avg = df.groupby("Manufacturer")["Price"].mean().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=avg.values, y=avg.index, palette="rocket", ax=ax)
        ax.set_title("Average Price by Top 15 Manufacturers", fontsize=14)
        ax.set_xlabel("Average Price")
        save_fig(fig, "06_avg_price_by_manufacturer.png")
    
    # ═══════════════════════════════════════════
    #  PLOT 7 – Average price by Fuel type
    # ═══════════════════════════════════════════
    print("⛽ Generating Avg Price by Fuel Type …")
    if "Fuel type" in df.columns:
        avg = df.groupby("Fuel type")["Price"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=avg.index, y=avg.values, palette="crest", ax=ax)
        ax.set_title("Average Price by Fuel Type", fontsize=14)
        ax.set_ylabel("Average Price")
        plt.xticks(rotation=30)
        save_fig(fig, "07_avg_price_by_fueltype.png")
    
    print("\n✅ EDA complete – all plots saved in outputs/")


if __name__ == "__main__":
    main()
