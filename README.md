⌚ Wearable Health Devices Performance Prediction

A machine learning project that analyzes and predicts the Performance Score of wearable health devices using multiple regression models and feature importance analysis.

📌 Project Overview

This project focuses on understanding how hardware features, sensor accuracy, connectivity options, pricing, and user satisfaction impact the overall performance score of wearable health devices such as smartwatches and fitness trackers.

We apply Exploratory Data Analysis (EDA) and compare multiple machine learning models to predict device performance and identify the most influential features.

🎯 Objectives

1. Analyze relationships between device features and performance
2. Visualize feature correlations and performance distribution
3. Build and compare regression models
4. Identify key drivers of performance using feature importance
5. Save trained models for future inference

📂 Dataset

Key Features Include:
1. Price (USD)
2. Battery life (hours)
3. Heart rate, sleep & activity accuracy (%)
4. GPS accuracy (meters)
5. Health sensors count
6. Connectivity features (WiFi, Bluetooth, NFC, LTE)
7. Customer satisfaction rating
8. Brand and category details

Target Variable: Performance_Score

🛠️ Tech Stack

1. Python
2. Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
3. ML Models: Linear Regression, Random Forest Regressor, XGBoost Regressor

🔍 Project Workflow

1. Data Loading & Cleaning
    Dropped identifiers (Device Name, Model, Test Date)
    Converted connectivity features into binary columns
    Handled missing numerical values using median imputation
2. Exploratory Data Analysis
    Performance score distribution
    Correlation heatmap of numerical features
3. Data Preprocessing
    Standard scaling for numerical features
    One-hot encoding for categorical features
    ColumnTransformer + Pipeline
4. Model Training
    Train/Test split (80/20)
    5-fold cross-validation
    Evaluation using MAE, MSE, and R²
5. Model Evaluation & Interpretation
    Actual vs Predicted scatter plots
    Feature importance for tree-based models

📊 Visualizations Generated

1. Correlation Matrix Heatmap
2. Performance Score Distribution
3. Actual vs Predicted Plots
    Linear Regression
    Random Forest
    XGBoost
4. Top 10 Important Features
    Random Forest
    XGBoost

📌 XGBoost showed the highest R² score and most accurate predictions.

🔑 Key Insights

1. Sensor accuracy is the strongest predictor of performance
2. Devices with advanced health tracking features perform better
3. Customer satisfaction has a meaningful impact on performance
4. Price alone does not guarantee high performance
5. Battery life and GPS accuracy play moderate roles

🚀 How to Run the Project
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
python app.py

📌 Future Improvements

1. Hyperparameter tuning using GridSearchCV
2. SHAP values for explainability
3. Deployment using Streamlit or Flask
4. Time-based performance trend analysis
