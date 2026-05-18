from fastapi import FastAPI
import joblib
import psycopg2
from api.db import fetch_customer_data
from src.processing.schema import apply_schema


model = joblib.load("models/model.pkl")

app = FastAPI(
    title="Churn Prediction API",
    description="FastAPI MLOps Pipeline",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


def to_python_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


@app.get("/predict/{customerid}")
def predict_customer(customerid: str):

    try:
        print("Customer ID:", customerid)

        df = fetch_customer_data(customerid)
        if df is None or df.empty:
            return {"error": "No data found for customer"}

        df = apply_schema(df)

        print("RAW DATA:", df)

        expected_cols = [
            "tenure",
            "monthly_charges",
            "total_charges",
            "contract"
        ]

        df = df[expected_cols]

        # -------------------
        # PREDICTION
        # -------------------
        prediction = model.predict(df)[0]
        tenure = int(to_python_scalar(df["tenure"].iloc[0]))
        monthly_charges = float(to_python_scalar(df["monthly_charges"].iloc[0]))
        total_charges = float(to_python_scalar(df["total_charges"].iloc[0]))
        contract = str(to_python_scalar(df["contract"].iloc[0]))
        prediction = int(to_python_scalar(prediction))

        # -------------------
        # SAVE TO DATABASE (MISSING PART)
        # -------------------
        conn = psycopg2.connect(
            host="localhost",
            database="telecom_mlops",
            user="postgres",
            password="postgres123",
            port=5432
        )

        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions_log2 (
                customerid,
                tenure,
                monthly_charges,
                total_charges,
                contract,
                prediction
            ) VALUES (%s,%s,%s,%s,%s,%s)
        """, (
            customerid,
            tenure,
            monthly_charges,
            total_charges,
            contract,
            prediction
        ))

        conn.commit()
        conn.close()

        return {
            "customerid": customerid,
            "prediction": prediction
        }

    except Exception as e:
        return {"error": str(e)}
