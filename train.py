import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data/raw/Telcom-churn.csv")
df = df.rename(columns={
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Contract": "contract",
    "Churn": "churn",
})
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
df = df.dropna(subset=["total_charges"])
df["churn"] = df["churn"].map({"No": 0, "Yes": 1})

# Features
X = df[["tenure", "monthly_charges", "total_charges", "contract"]]
y = df["churn"]

# Column types
num_cols = ["tenure", "monthly_charges", "total_charges"]
cat_cols = ["contract"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Save NEW model
joblib.dump(pipeline, "models/model.pkl")

print("NEW PIPELINE SAVED")
