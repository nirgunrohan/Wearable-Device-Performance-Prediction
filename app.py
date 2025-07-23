import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- 1. Load and Prepare the Data ---
# Load the dataset from the CSV file
df = pd.read_csv('wearable_health_devices_performance_upto_26june2025.csv')

# For modeling, we'll drop columns that are identifiers or have too many unique values
df_model = df.drop(['Test_Date', 'Device_Name', 'Model'], axis=1)

# Separate the features (X) from the target variable (y)
X = df_model.drop('Performance_Score', axis=1)
y = df_model['Performance_Score']


# --- 2. Exploratory Data Analysis (Optional: for generating plots) ---
# Distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['Performance_Score'], kde=True)
plt.title('Distribution of Performance Score')
plt.xlabel('Performance Score')
plt.ylabel('Frequency')
plt.savefig('performance_score_distribution.png')
plt.close()

# Correlation heatmap for numerical features
numerical_features_for_heatmap = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_features_for_heatmap].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_heatmap.png')
plt.close()


# --- 3. Data Preprocessing ---
# Identify categorical and numerical features for preprocessing
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Handle missing values in numerical features by filling with the median
# We create a copy to avoid a SettingWithCopyWarning
X = X.copy()
for col in numerical_features:
    if X[col].isnull().sum() > 0:
        X[col].fillna(X[col].median(), inplace=True)

# Create preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Use ColumnTransformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# --- 4. Model Training and Evaluation ---
# Define the models you want to train
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42) # A powerful gradient boosting model
}

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store the evaluation results
results = {}

# Loop through each model to train and evaluate it
for name, model in models.items():
    # Create a machine learning pipeline that combines preprocessing and the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])

    # Train the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = pipeline.predict(X_test)

    # Calculate the evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store the results
    results[name] = {'MAE': mae, 'MSE': mse, 'R-squared': r2}


# --- 5. Display Results ---
print("--- Model Evaluation Results ---")
for name, metrics in results.items():
    print(f"\n--- {name} ---")
    print(f"Mean Absolute Error: {metrics['MAE']:.4f}")
    print(f"Mean Squared Error: {metrics['MSE']:.4f}")
    print(f"R-squared: {metrics['R-squared']:.4f}")