# 🚗 Car Price Prediction — End-to-End Machine Learning Project

> A production-ready, modular ML pipeline that predicts used-car prices using a **19 000+ record Kaggle-style dataset** with features like manufacturer, mileage, engine volume, fuel type, and more.

---

## 📑 Table of Contents
1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Description](#3-dataset-description)
4. [Project Structure](#4-project-structure)
5. [Methodology](#5-methodology)
6. [Results](#6-results)
7. [How to Run](#7-how-to-run)
8. [Streamlit Web App](#8-streamlit-web-app)
9. [Conclusion](#9-conclusion)
10. [Future Scope](#10-future-scope)

---

## 1. Introduction

Buying or selling a used car can be a daunting task — **how do you know the fair price?**  
This project builds a complete machine-learning pipeline that ingests car listing data, performs thorough exploratory data analysis, engineers meaningful features, trains and compares multiple regression models, tunes hyper-parameters, and deploys the best model behind a **Streamlit** web interface.

The entire workflow follows **industry best practices**: modular code, reproducible pipelines,
cross-validation, proper handling of categorical features, and version-controlled model artifacts.

---

## 2. Problem Statement

Given a set of attributes for a used car (manufacturer, category, mileage, engine volume, fuel type, etc.), **predict its market selling price**.

This is a **supervised regression** problem.  
The target variable is **Price** (continuous, in currency units).

---

## 3. Dataset Description

| Item | Detail |
|---|---|
| **Source** | Kaggle-style car listing dataset |
| **Records** | ~19 000 |
| **Features** | 18 (including the target) |

### Key Features

| Feature | Type | Description |
|---|---|---|
| Manufacturer | Categorical | Brand (TOYOTA, BMW, …) |
| Prod. year | Numeric | Year of manufacture |
| Category | Categorical | Sedan, Jeep, Hatchback, … |
| Leather interior | Binary | Yes / No |
| Fuel type | Categorical | Petrol, Diesel, Hybrid, … |
| Engine volume | Numeric | Engine size in litres |
| Mileage | Numeric | Odometer reading in km |
| Cylinders | Numeric | Number of cylinders |
| Gear box type | Categorical | Automatic, Manual, Tiptronic, … |
| Drive wheels | Categorical | Front, Rear, 4×4 |
| Wheel | Categorical | Left / Right-hand drive |
| Color | Categorical | Body colour |
| Airbags | Numeric | Number of airbags |
| Levy | Numeric | Tax/levy amount |
| **Price** | **Numeric** | **Target variable** |

### Engineered Features

| Feature | Origin |
|---|---|
| Car_Age | `2026 − Prod. year` |
| Turbo | Extracted from Engine volume string |

---

## 4. Project Structure

```
CAR PP/
│
├── data set 1.csv          # Raw dataset 1
├── data set 2.csv          # Raw dataset 2 (primary, has Price)
├── requirements.txt        # Python dependencies
├── README.md               # ← You are here
│
├── src/
│   ├── __init__.py
│   ├── utils.py            # Data loading, cleaning, feature engineering
│   ├── preprocessing.py    # sklearn Pipelines & ColumnTransformer
│   └── training.py         # Model training, evaluation, tuning, saving
│
├── notebooks/
│   └── eda.py              # Exploratory Data Analysis (generates plots)
│
├── models/
│   └── best_car_price_model.joblib   # Saved best model (after training)
│
├── outputs/
│   ├── 01_correlation_heatmap.png
│   ├── 02_price_distribution.png
│   ├── 03_box_plots.png
│   ├── 04_feature_vs_price.png
│   ├── 05_categorical_distributions.png
│   ├── 06_avg_price_by_manufacturer.png
│   ├── 07_avg_price_by_fueltype.png
│   ├── 08_feature_importance.png
│   └── model_comparison.csv
│
└── app.py                  # Streamlit web app
```

---

## 5. Methodology

### 5.1 Data Understanding & EDA
- Loaded the dataset and inspected shape, types, null values.
- Generated **7 visualisation sets**: correlation heatmap, price distribution (raw + log), box plots for outlier detection, feature-vs-price scatter, categorical distributions, average price by manufacturer, average price by fuel type.

### 5.2 Data Preprocessing
- **Missing values**: Levy `'-'` → median imputation; other numeric NaNs → median.
- **Duplicates**: Removed.
- **Feature engineering**: Created `Car_Age`, extracted `Turbo` flag.
- **Encoding**: OneHotEncoder for categorical features (handle_unknown="ignore").
- **Scaling**: StandardScaler on numeric features.
- **Pipeline**: All transforms wrapped in `sklearn.pipeline.Pipeline` + `ColumnTransformer`.
- **Split**: 80 / 20 train-test.

### 5.3 Model Building
Trained four regressors, each wrapped in the preprocessing pipeline:

| # | Model | Key Params |
|---|---|---|
| 1 | Linear Regression | — |
| 2 | Decision Tree | `random_state=42` |
| 3 | Random Forest | 100 trees, `n_jobs=-1` |
| 4 | Gradient Boosting | 200 estimators |

### 5.4 Evaluation Metrics
- **MAE** – Mean Absolute Error
- **MSE** – Mean Squared Error
- **RMSE** – Root Mean Squared Error
- **R² Score** – Coefficient of Determination
- **3-fold Cross-Validation R²**

### 5.5 Hyperparameter Tuning
Used `GridSearchCV` on Random Forest with parameters:
- `n_estimators`: [100, 200]
- `max_depth`: [15, 25, None]
- `min_samples_split`: [2, 5]
- `min_samples_leaf`: [1, 2]

### 5.6 Model Saving
Best performing model saved with `joblib` to `models/best_car_price_model.joblib`.

---

## 6. Results

> **Actual metrics are printed and saved to `outputs/model_comparison.csv` when you run `python src/training.py`.**

Random Forest and Gradient Boosting typically achieve the best R² scores on this dataset, with the **Tuned Random Forest** often coming out on top after hyperparameter optimisation.

Feature importance analysis consistently highlights **Car_Age**, **Engine volume**, **Mileage**, **Levy**, and **Manufacturer** as the most impactful features.

---

## 7. How to Run

### Prerequisites
- Python 3.10+
- pip

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run EDA
```bash
python notebooks/eda.py
```
All plots saved to `outputs/`.

### 3. Train models
```bash
python src/training.py
```
Trains all models, tunes RF, saves best to `models/`.

### 4. Launch the web app
```bash
streamlit run app.py
```

---

## 8. Streamlit Web App

The web app provides a **clean sidebar** with inputs for every car attribute.  
Click **Predict** and the estimated price is displayed prominently.

Features:
- Cached model loading for instant predictions.
- Input summary expander.
- Clean gradient-styled UI.
- Proper error handling if no model exists yet.

---

## 9. Conclusion

- **Tree-based ensembles** (Random Forest, Gradient Boosting) significantly outperform Linear Regression for this dataset, which is expected given the non-linear relationships and mixed feature types.
- **Feature engineering** (Car_Age, Turbo flag) and proper handling of the noisy Levy / Mileage columns had a measurable positive impact.
- The full pipeline is **production-ready**: a single `.joblib` file bundles preprocessing and prediction.

---

## 10. Future Scope

| Area | Idea |
|---|---|
| **Advanced Models** | XGBoost, LightGBM, CatBoost |
| **Deep Learning** | Tabular neural nets (TabNet) |
| **NLP** | Use Model name text embeddings instead of dropping |
| **Deployment** | Dockerise + deploy on AWS / GCP / Heroku |
| **CI/CD** | GitHub Actions for automated retraining |
| **MLOps** | MLflow / Weights & Biases experiment tracking |
| **Data** | Scrape live listings for continuous model improvement |

---

> **Author**: ML Engineering Project  
> **Stack**: Python · pandas · scikit-learn · Streamlit · matplotlib · seaborn  
> **License**: MIT
