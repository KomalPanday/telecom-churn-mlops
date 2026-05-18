import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\Komal_360digitmg\Project\telecom-churn-mlops\data\raw\Telcom-churn.csv")

print("✅ Dataset Loaded Successfully")

# ----------------------------
# 1. Dataset Shape
# ----------------------------
print("\nDataset Shape:")
print(df.shape)

# ----------------------------
# 2. Missing Values
# ----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------
# 3. Data Types
# ----------------------------
print("\nData Types:")
print(df.dtypes)

# ----------------------------
# 4. Duplicate Rows
# ----------------------------
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ----------------------------
# 5. Required Columns Check
# ----------------------------
required_columns = [
    "customerID",
    "gender",
    "tenure",
    "MonthlyCharges",
    "Churn"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    print("\n❌ Missing Columns:", missing_columns)
else:
    print("\n✅ All required columns exist")

print("\n✅ Data Validation Completed")
