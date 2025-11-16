#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os

INPUT_PATH = os.path.join(os.path.dirname(__file__), '../data/e-commerce-dataset.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../data/processed_data.csv')

def preprocess(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    # --- Load Data ---
    df = pd.read_csv(input_path, sep=';')

    # --- Fix Data Types ---
    cat_fix = ['CustomerID', 'CityTier', 'NumberOfDeviceRegistered', 'SatisfactionScore']
    df[cat_fix] = df[cat_fix].astype('object')

    # --- Fill NULLs with Median Values ---
    numerical = ['Tenure','WarehouseToHome','HourSpendOnApp',
                 'NumberOfAddress', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                 'OrderCount', 'DaySinceLastOrder']

    for col in numerical:
        df[col] = df[col].fillna(df[col].median())

    # --- Handle Outliers ---
    cols_to_log = ['CouponUsed', 'OrderCount', 'WarehouseToHome', 'DaySinceLastOrder', 'CashbackAmount', 'NumberOfAddress']

    for col in cols_to_log:
        df[col] = np.log1p(df[col])


    # --- Standardize Categorical Values
    df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({
        'Phone':'Mobile Phone'
    })

    df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
        'CC':'Credit Card',
        'COD':'Cash on Delivery'
    })

    df['PreferedOrderCat'] = df['PreferedOrderCat'].replace({
        'Mobile':'Mobile Phone'
    })

    # --- Save Processed Data ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}, shape: {df.shape}")

if __name__ == "__main__":
    preprocess()
