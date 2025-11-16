import os
import sys
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn

from src.dict_vectorizer_transformer import DictVectorizerTransformer

#def load_model(path):
#    with open(path, "rb") as f_in:
#        saved = pickle.load(f_in)
#    return saved["pipeline"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")

def load_model_from_registry(model_name="ecommerce-churn-model", stage="Production"):
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    return model, model_name

def predict_single(customer, model, model_name):
    df = pd.DataFrame([customer])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]
  
    return {
        "model": model_name,
        "prediction": int(pred),
        "probability": float(proba)
    }

if __name__ == "__main__":
    #model = load_model(path="models/best_model.pkl")

    model, model_name = load_model_from_registry()

    example = {
        'PreferredLoginDevice':'Computer',
        'PreferredPaymentMode':'Debit Card',
        'PreferedOrderCat':'Grocery',
        'MaritalStatus':'Single',
        'CityTier':'3',
        'Gender':'Male',
        'NumberOfDeviceRegistered':'3',
        'SatisfactionScore':'3',
        'Tenure':6.0,
        'WarehouseToHome':3.178054,
        'HourSpendOnApp':2.0,
        'NumberOfAddress':1.098612,
        'OrderAmountHikeFromlastYear':11.0,
        'CouponUsed':0.693147,
        'OrderCount':1.94591,
        'DaySinceLastOrder':0.0,
        'CashbackAmount':5.616771,
        'Complain':0
    }

    print(predict_single(example, model, model_name))
