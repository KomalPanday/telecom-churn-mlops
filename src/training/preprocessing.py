"""Preprocessing utilities for model training."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from src.data_preprocessing import preprocessor

# Load dataset
df = pd.read_csv(r"D:\Komal_360digitmg\Project\telecom-churn-mlops\data\raw\Telcom-churn.csv")

print("✅ Dataset Loaded")

# -----------------------------
# Remove customerID
# -----------------------------
df.drop("customerID", axis=1, inplace=True)

# -----------------------------
# Handle TotalCharges
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# -----------------------------
# Convert target variable
# -----------------------------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -----------------------------
# Split features and target
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# -----------------------------
# Identify column types
# -----------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

print("\nCategorical Columns:")
print(categorical_cols)

print("\nNumerical Columns:")
print(numerical_cols)

# -----------------------------
# Preprocessing pipelines
# -----------------------------
numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# -----------------------------
# Combine preprocessors
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Fit and transform
# -----------------------------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\n✅ Preprocessing Completed")
print("Training Shape:", X_train_processed.shape)
print("Testing Shape:", X_test_processed.shape)

# -----------------------------
# Save preprocessor
# -----------------------------
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("\n✅ Preprocessor saved successfully")
