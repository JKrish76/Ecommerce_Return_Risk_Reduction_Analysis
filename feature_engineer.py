# feature_engineer.py

import pandas as pd
from datetime import datetime
import numpy as np

def create_engineering_features(df):
    """
    Creates time-based and target-encoded features for the model.
    """
    df_new = df.copy()

    # --- 1. Delivery Time Delta Feature ---
    # Convert dates to datetime objects
    for col in ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']:
        # Use errors='coerce' to handle any remaining non-date strings
        df_new[col] = pd.to_datetime(df_new[col], errors='coerce') 

    # Calculate delivery delta (Actual - Estimated) in days
    df_new['delivery_delta_days'] = (
        df_new['order_delivered_customer_date'] - df_new['order_estimated_delivery_date']
    ).dt.total_seconds() / (60*60*24)
    df_new['delivery_delta_days'] = df_new['delivery_delta_days'].fillna(0) # Fill NaNs (e.g., non-delivered) with 0

    # Calculate actual delivery time (Purchase to Delivery) in days
    df_new['actual_delivery_days'] = (
        df_new['order_delivered_customer_date'] - df_new['order_purchase_timestamp']
    ).dt.total_seconds() / (60*60*24)
    # Fill missing values with the mean for now
    df_new['actual_delivery_days'].fillna(df_new['actual_delivery_days'].mean(), inplace=True)


    # --- 2. Target Encoding Features (Risk Metrics) ---
    
    # Avg Return Rate by Category
    category_mean_return = df_new.groupby('product_category_name')['Is_Return'].mean()
    df_new['category_avg_return_rate'] = df_new['product_category_name'].map(category_mean_return)
    df_new['category_avg_return_rate'].fillna(df_new['Is_Return'].mean(), inplace=True) # Fill NaNs with overall mean

    # Avg Return Rate by Seller
    seller_mean_return = df_new.groupby('seller_id')['Is_Return'].mean()
    df_new['seller_avg_return_rate'] = df_new['seller_id'].map(seller_mean_return)
    df_new['seller_avg_return_rate'].fillna(df_new['Is_Return'].mean(), inplace=True) # Fill NaNs with overall mean


    # --- 3. Simple Numerical Features (Renaming for clarity) ---
    df_new['price_per_item'] = df_new['price']
    df_new['freight_value_per_item'] = df_new['freight_value']

    print("✅ Feature engineering complete.")
    return df_new