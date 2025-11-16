import os
import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Churn Prediction API")
Instrumentator().instrument(app).expose(app)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/best_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model_name = saved["model_name"]
pipeline = saved["pipeline"]

# Define request schema
class Customer(BaseModel):
    Tenure: float
    PreferredLoginDevice: str 
    CityTier: str 
    WarehouseToHome: float
    PreferredPaymentMode: str 
    Gender: str 
    HourSpendOnApp: float
    NumberOfDeviceRegistered: str 
    PreferedOrderCat: str
    SatisfactionScore: str 
    MaritalStatus: str 
    NumberOfAddress: float
    Complain: int
    OrderAmountHikeFromlastYear: float
    CouponUsed: float
    OrderCount: float
    DaySinceLastOrder: float
    CashbackAmount: float

@app.post("/predict")
def predict(Customer: Customer):
    df = pd.DataFrame([Customer.dict()])

    pred = pipeline.predict(df)[0]
    proba = pipeline.predict_proba(df)[0, 1]
  
    return {
        "model": model_name,
        "prediction": int(pred),
        "probability": float(proba)
    }

# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "Hello World"}