import os
import pandas as pd
from sqlalchemy import create_engine

# 1. Database Connection Configurations
# Syntax: postgresql://username:password@host:port/database_name
DB_USER = "postgres"
DB_PASSWORD = "postgres123"  # Replace with the password you set during installation
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "telecom_mlops"

# 2. Define File Paths
# Replace 'telecom_churn.csv' with the actual path or name of your CSV file
CSV_FILE_PATH = r"D:\Komal_360digitmg\Project\telecom-churn-mlops\data\raw\Telcom-churn.csv"
TABLE_NAME = "customer_churn"

def load_csv_to_postgres():
    try:
        # Check if CSV exists
        if not os.path.exists(CSV_FILE_PATH):
            print(f"❌ Error: CSV file not found at {CSV_FILE_PATH}")
            return

        print("📖 Reading CSV file...")
        df = pd.read_csv(CSV_FILE_PATH)
        
        # Clean column names (Optional but recommended: replaces spaces with underscores and lowercase)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        print("🔌 Connecting to PostgreSQL...")
        # Create SQLAlchemy engine
        connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(connection_string)

        print(f"📥 Loading data into table '{TABLE_NAME}'...")
        # Load data using pandas to_sql
        # if_exists='replace' drops the table if it exists and creates a new one.
        # Use if_exists='append' if you want to add data to an existing table.
        df.to_sql(name=TABLE_NAME, con=engine, if_exists='replace', index=False)
        
        print(f"✅ Success! Loaded {len(df)} rows into '{TABLE_NAME}' table successfully.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    load_csv_to_postgres()
