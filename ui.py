import streamlit as st
import requests

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("🛍️ E-commerce Churn Prediction")
st.markdown("Enter customer details below to predict churn probability.")

# Input fields
PreferredLoginDevice = st.selectbox("Preferred Login Device", ["Mobile", "Computer"])
PreferredPaymentMode = st.selectbox("Preferred Payment Mode", ["Credit Card", "Debit Card", "UPI", "Cash on Delivery"])
PreferedOrderCat = st.selectbox("Preferred Order Category", ["Grocery", "Laptop & Accessory", "Mobile", "Fashion"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
CityTier = st.selectbox("City Tier", ["1", "2", "3"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfDeviceRegistered = st.selectbox("Number of Devices Registered", ["1", "2", "3", "4", "5"])
SatisfactionScore = st.selectbox("Satisfaction Score", ["1", "2", "3", "4", "5"])
Tenure = st.number_input("Tenure (months)", min_value=0.0, value=6.0)
WarehouseToHome = st.number_input("Distance Warehouse to Home (km)", min_value=0.0, value=3.18)
HourSpendOnApp = st.number_input("Hours Spent on App", min_value=0.0, value=2.0)
NumberOfAddress = st.number_input("Number of Addresses (log-transformed)", min_value=0.0, value=1.098612)
OrderAmountHikeFromlastYear = st.number_input("Order Amount Hike From Last Year (%)", min_value=0.0, value=11.0)
CouponUsed = st.number_input("Coupons Used (log-transformed)", min_value=0.0, value=0.693147)
OrderCount = st.number_input("Order Count (log-transformed)", min_value=0.0, value=1.94591)
DaySinceLastOrder = st.number_input("Days Since Last Order", min_value=0.0, value=0.0)
CashbackAmount = st.number_input("Cashback Amount", min_value=0.0, value=5.616771)
Complain = st.selectbox("Has Complained?", [0, 1])

# Submit button
if st.button("Predict Churn"):
    payload = {
        "PreferredLoginDevice": PreferredLoginDevice,
        "PreferredPaymentMode": PreferredPaymentMode,
        "PreferedOrderCat": PreferedOrderCat,
        "MaritalStatus": MaritalStatus,
        "CityTier": CityTier,
        "Gender": Gender,
        "NumberOfDeviceRegistered": NumberOfDeviceRegistered,
        "SatisfactionScore": SatisfactionScore,
        "Tenure": Tenure,
        "WarehouseToHome": WarehouseToHome,
        "HourSpendOnApp": HourSpendOnApp,
        "NumberOfAddress": NumberOfAddress,
        "OrderAmountHikeFromlastYear": OrderAmountHikeFromlastYear,
        "CouponUsed": CouponUsed,
        "OrderCount": OrderCount,
        "DaySinceLastOrder": DaySinceLastOrder,
        "CashbackAmount": CashbackAmount,
        "Complain": Complain
    }

    try:
        res = requests.post("https://ecommerce-churn-prediction-qylu.onrender.com/predict", json=payload)
        result = res.json()
        st.success(f"Prediction: {'Churn' if result['prediction'] == 1 else 'No Churn'}")
        st.metric(label="Churn Probability", value=f"{result['probability']:.2%}")
        st.caption(f"Model used: {result['model']}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")