from fastapi import FastAPI
import joblib
import pandas as pd

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI()

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("models/model.pkl")

# -----------------------------
# Home route
# -----------------------------
@app.get("/")
def home():
    return {"message": "Telecom Churn Prediction API"}

# -----------------------------
# Prediction route
# -----------------------------
@app.post("/predict")
def predict(data: dict):

    # Convert input to dataframe
    df = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(df)[0]

    result = "Churn" if prediction == 1 else "No Churn"

    return {"prediction": result}