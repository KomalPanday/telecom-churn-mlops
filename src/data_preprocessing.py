import pandas as pd
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(r"D:\Komal_360digitmg\Project\telecom-churn-mlops\data\raw\Telcom-churn.csv")

# -----------------------------
# Clean target
# -----------------------------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# -----------------------------
# Pipelines
# -----------------------------
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# -----------------------------
# Preprocessor
# -----------------------------
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# -----------------------------
# Fit preprocessor ONLY
# -----------------------------
preprocessor.fit(X)

# -----------------------------
# Save preprocessor
# -----------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(preprocessor, "models/preprocessor.pkl")

print("Preprocessor saved successfully")