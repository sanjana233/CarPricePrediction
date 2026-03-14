@echo off
title Car Price Prediction - Full Pipeline
color 0A

echo ============================================================
echo    CAR PRICE PREDICTION - COMPLETE PIPELINE
echo ============================================================
echo.

:: ── Step 1: Install Dependencies ──
echo [1/4] Installing dependencies...
echo ------------------------------------------------------------
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies!
    pause
    exit /b 1
)
echo Done.
echo.

:: ── Step 2: Run EDA ──
echo [2/4] Running Exploratory Data Analysis...
echo ------------------------------------------------------------
python notebooks/eda.py
if %errorlevel% neq 0 (
    echo ERROR: EDA script failed!
    pause
    exit /b 1
)
echo.

:: ── Step 3: Train Models ──
echo [3/4] Training Models + Hyperparameter Tuning...
echo ------------------------------------------------------------
echo (This may take a few minutes)
python src/training.py
if %errorlevel% neq 0 (
    echo ERROR: Training script failed!
    pause
    exit /b 1
)
echo.

:: ── Step 4: Launch Streamlit App ──
echo ============================================================
echo [4/4] Launching Streamlit Web App...
echo ============================================================
echo.
echo The app will open in your browser shortly.
echo Press Ctrl+C in this window to stop the server.
echo.
streamlit run app.py

pause
