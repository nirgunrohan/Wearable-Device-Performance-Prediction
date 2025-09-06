import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Load and Prepare the Data ---
df = pd.read_csv('wearable_health_devices_performance_upto_26june2025.csv')

# Drop identifier columns
df_model = df.drop(['Test_Date', 'Device_Name', 'Model'], axis=1)

# --- Handle Connectivity_Features (split into multiple binary columns) ---
# Extract unique connectivity options
connectivity_options = ['WiFi', 'Bluetooth', 'NFC', 'LTE']

for option in connectivity_options:
    df_model[f'Has_{option}'] = df_model['Connectivity_Features'].apply(
        lambda x: 1 if option in str(x) else 0
    )

# Drop original text column after transformation
df_model = df_model.drop('Connectivity_Features', axis=1)

# Features (X) and Target (y)
X = df_model.drop('Performance_Score', axis=1)
y = df_model['Performance_Score']


# --- 2. Exploratory Data Analysis ---
plt.figure(figsize=(10, 6))
sns.histplot(df['Performance_Score'], kde=True)
plt.title('Distribution of Performance Score')
plt.xlabel('Performance Score')
plt.ylabel('Frequency')
plt.savefig('performance_score_distribution.png')
plt.close()

numerical_features_for_heatmap = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_features_for_heatmap].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_heatmap.png')
plt.close()


# --- 3. Data Preprocessing ---
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Fix missing values (without inplace warning)
X = X.copy()
for col in numerical_features:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# Transformers
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# --- 4. Model Training and Evaluation ---
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    # Train
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

    results[name] = {
        'MAE': mae,
        'MSE': mse,
        'R-squared': r2,
        'CV R-squared': cv_scores.mean()
    }

    # Save Model
    joblib.dump(pipeline, f"{name.replace(' ', '_')}_model.pkl")

    # Scatter Plot Actual vs Predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.xlabel("Actual Performance Score")
    plt.ylabel("Predicted Performance Score")
    plt.title(f"Actual vs Predicted ({name})")
    plt.savefig(f"{name.replace(' ', '_')}_prediction_scatter.png")
    plt.close()

    # Feature Importance (only for tree-based models)
    if hasattr(model, "feature_importances_"):
        feature_names = (numerical_features.tolist() +
                         list(pipeline.named_steps['preprocessor']
                              .named_transformers_['cat']
                              .get_feature_names_out(categorical_features)))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices][:10], y=np.array(feature_names)[indices][:10])
        plt.title(f"Top 10 Important Features ({name})")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.savefig(f"{name.replace(' ', '_')}_feature_importance.png")
        plt.close()


# --- 5. Display Results ---
print("\n--- Model Evaluation Results ---")
for name, metrics in results.items():
    print(f"\n--- {name} ---")
    print(f"Mean Absolute Error: {metrics['MAE']:.4f}")
    print(f"Mean Squared Error: {metrics['MSE']:.4f}")
    print(f"R-squared: {metrics['R-squared']:.4f}")
    print(f"Cross-validated R-squared: {metrics['CV R-squared']:.4f}")

print("\nAll models saved as .pkl files, and evaluation plots have been generated.")
