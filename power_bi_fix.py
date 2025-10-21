# FIX_FOR_POWER_BI.py

import pandas as pd
from feature_engineer import create_engineering_features # Import your Day 2 function

# 1. Load the existing master table
try:
    df_master = pd.read_csv('df_master_day1.csv', low_memory=False)
except FileNotFoundError:
    print("Error: df_master_day1.csv not found.")
    exit()

# 2. Apply the feature engineering (creates delivery_delta_days and other features)
df_powerbi_master = create_engineering_features(df_master)

# 3. Save the new, feature-rich file
df_powerbi_master.to_csv('df_master_for_powerbi.csv', index=False)
print("✅ New file 'df_master_for_powerbi.csv' created with all engineered features.")