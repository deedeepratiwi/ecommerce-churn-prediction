import requests

url = "http://localhost:8000/predict"

payload = {
    "PreferredLoginDevice":"Computer",
    "PreferredPaymentMode":"Debit Card",
    "PreferedOrderCat":"Grocery",
    "MaritalStatus":"Single",
    "CityTier":"3",
    "Gender":"Male",
    "NumberOfDeviceRegistered":"3",
    "SatisfactionScore":"3",
    "Tenure":6.0,
    "WarehouseToHome":3.178054,
    "HourSpendOnApp":2.0,
    "NumberOfAddress":1.098612,
    "OrderAmountHikeFromlastYear":11.0,
    "CouponUsed":0.693147,
    "OrderCount":1.94591,
    "DaySinceLastOrder":0.0,
    "CashbackAmount":5.616771,
    "Complain":0
}

res = requests.post(url, json=payload)
print(res.json())