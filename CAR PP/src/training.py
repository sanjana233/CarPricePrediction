"""
=============================================================================
    Car Price Prediction - Model Training & Evaluation
    ===================================================
    Phases 4-6:  Train ▸ Evaluate ▸ Tune ▸ Save

    Run:  python src/training.py
=============================================================================
"""

import os
import sys
import warnings
import time

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ── project imports ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_data, clean_data, split_features_target, get_train_test
from src.preprocessing import build_preprocessor

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "..", "data set 2.csv")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
#  EVALUATION HELPER
# ──────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute MAE, MSE, RMSE, R² for a fitted model."""
    y_pred = model.predict(X_test)
    
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    
    print(f"\n{'─' * 50}")
    print(f"  {model_name}")
    print(f"{'─' * 50}")
    print(f"  MAE   : {mae:,.2f}")
    print(f"  MSE   : {mse:,.2f}")
    print(f"  RMSE  : {rmse:,.2f}")
    print(f"  R²    : {r2:.4f}")
    
    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4)
    }


# ══════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════

def main():
    # ────────────────────────────────────────
    # 1.  Load & Clean
    # ────────────────────────────────────────
    print("=" * 60)
    print("  PHASE 1-3 : Load → Clean → Preprocess")
    print("=" * 60)
    
    df = load_data(DATA_PATH)
    df = clean_data(df)
    
    # Remove extreme price outliers (top/bottom 1%) to improve model quality
    q_low  = df["Price"].quantile(0.01)
    q_high = df["Price"].quantile(0.99)
    df = df[(df["Price"] >= q_low) & (df["Price"] <= q_high)]
    print(f"After outlier removal: {df.shape}")
    
    X, y = split_features_target(df, target="Price")
    X_train, X_test, y_train, y_test = get_train_test(X, y)
    
    # Build preprocessor from training data
    preprocessor, num_feats, cat_feats = build_preprocessor(X_train)
    
    # ────────────────────────────────────────
    # 2.  PHASE 4 – Model Building
    # ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 4 : Model Building & Evaluation")
    print("=" * 60)
    
    models = {
        "Linear Regression":         LinearRegression(),
        "Decision Tree Regressor":   DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor":   RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=200, random_state=42),
    }
    
    results = []
    best_score = -np.inf
    best_model_name = None
    best_pipeline = None
    
    for name, model in models.items():
        print(f"\n🏋️  Training {name} …")
        t0 = time.time()
        
        # Wrap in a Pipeline so preprocessing + model are bundled
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        pipe.fit(X_train, y_train)
        elapsed = time.time() - t0
        
        # Evaluate
        metrics = evaluate_model(pipe, X_test, y_test, name)
        metrics["Train Time (s)"] = round(elapsed, 2)
        results.append(metrics)
        
        # Cross-validation (3-fold for speed)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)
        print(f"  CV R² (3-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        metrics["CV R² Mean"] = round(cv_scores.mean(), 4)
        
        if metrics["R2"] > best_score:
            best_score = metrics["R2"]
            best_model_name = name
            best_pipeline = pipe
    
    # ── Comparison Table ──
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON TABLE")
    print("=" * 80)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print(f"\n✅ Comparison saved to outputs/model_comparison.csv")
    
    # ════════════════════════════════════════
    #  PHASE 5 – Hyperparameter Tuning
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 5 : Hyperparameter Tuning (RandomForest)")
    print("=" * 60)
    
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [15, 25, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
    }
    
    rf_pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
    
    print("🔍 Running GridSearchCV (this may take a few minutes) …")
    grid = GridSearchCV(
        rf_pipe,
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid.fit(X_train, y_train)
    
    print(f"\n🏆 Best Parameters: {grid.best_params_}")
    print(f"   Best CV R²    : {grid.best_score_:.4f}")
    
    tuned_metrics = evaluate_model(grid.best_estimator_, X_test, y_test, "Tuned Random Forest")
    
    # ── Decide best overall model ──
    if tuned_metrics["R2"] >= best_score:
        best_pipeline = grid.best_estimator_
        best_model_name = "Tuned Random Forest"
        best_score = tuned_metrics["R2"]
    
    # ════════════════════════════════════════
    #  FEATURE IMPORTANCE
    # ════════════════════════════════════════
    print("\n📊 Generating Feature Importance Plot …")
    try:
        fitted_model = best_pipeline.named_steps["model"]
        fitted_pre   = best_pipeline.named_steps["preprocessor"]
        
        # Build feature names from the transformer
        ohe = fitted_pre.named_transformers_["cat"].named_steps["encoder"]
        cat_names = ohe.get_feature_names_out(cat_feats).tolist()
        all_names = num_feats + cat_names
        
        importances = fitted_model.feature_importances_
        indices = np.argsort(importances)[-20:]  # top 20
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], color="#4C72B0")
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([all_names[i] for i in indices])
        ax.set_xlabel("Importance")
        ax.set_title("Top 20 Feature Importances (Best Model)", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "08_feature_importance.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  📸 Saved: outputs/08_feature_importance.png")
    except Exception as e:
        print(f"  ⚠️  Could not plot feature importance: {e}")
    
    # ════════════════════════════════════════
    #  PHASE 6 – Save Best Model
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 6 : Save Best Model")
    print("=" * 60)
    
    model_path = os.path.join(MODEL_DIR, "best_car_price_model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"✅ Best model ({best_model_name}) saved → {model_path}")
    print(f"   Test R²: {best_score:.4f}")
    
    # ── Demo: load & predict ──
    USD_TO_INR = 85.0
    print("\n🔮 Demo — load model & predict on one sample:")
    loaded_model = joblib.load(model_path)
    sample = X_test.iloc[[0]]
    pred = loaded_model.predict(sample)[0]
    actual = y_test.iloc[0]
    print(f"   Predicted : ${pred:,.0f}  →  ₹{pred * USD_TO_INR:,.0f} INR")
    print(f"   Actual    : ${actual:,.0f}  →  ₹{actual * USD_TO_INR:,.0f} INR")
    
    print("\n🎉 Training pipeline complete!")


if __name__ == "__main__":
    main()
