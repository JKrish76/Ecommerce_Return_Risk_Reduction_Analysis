# Corrected code to test preprocess_and_split.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# Import configuration and feature engineering logic
from config import NUMERICAL_FEATURES, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE 
from feature_engineer import create_engineering_features

def preprocess_and_split(df_master):
    """
    Applies final cleaning, scaling, and splits the data into train/test sets.
    
    Returns: X_train_scaled, X_test_scaled, y_train, y_test, scaler, imputer
    """
    
    # 1. Apply feature engineering
    df_engineered = create_engineering_features(df_master)
    
    # 2. Select Features and Target
    # We drop rows where any core feature or the target variable is missing
    df_model = df_engineered.dropna(subset=NUMERICAL_FEATURES + [TARGET_COLUMN]).copy()
    
    X = df_model[NUMERICAL_FEATURES]
    y = df_model[TARGET_COLUMN]

    # 3. Impute Missing Values (Handle NaNs gracefully)
    # We fit the imputer on the full feature set (X) before splitting
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 4. Train/Test Split
    # Stratify by y ensures the return rate is consistent across train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 5. Feature Scaling (Essential for Logistic Regression)
    # Fit the scaler ONLY on the training data to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- Preprocessing Summary ---")
    print(f"Training Samples: {X_train_scaled.shape[0]}")
    print(f"Testing Samples: {X_test_scaled.shape[0]}")
    print("Data is scaled and ready for modeling.")
    
    # Return the scaled data, plus the scaler and imputer objects (needed for prediction on Day 3)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, imputer

# ----------------------------------------------------------------------------------
# --------------------- CORRECT WAY TO TEST THE FUNCTION ---------------------------
# ----------------------------------------------------------------------------------

# 1. Define the correct file path (using a correct raw string)
FILE_PATH = r'C:\Users\JAYAKRISHNAN\Desktop\Data Analyst Intership\Project 1\df_master_day1.csv'

try:
    # 2. Load the DataFrame
    df_master = pd.read_csv(FILE_PATH, low_memory=False)
    
    # 3. Ensure the target column is the correct type
    df_master['Is_Return'] = df_master['Is_Return'].astype(int)

    # 4. Call the function with the DataFrame
    X_train, X_test, y_train, y_test, scaler, imputer = preprocess_and_split(df_master)
    
    # Optional: Print the shapes of the output arrays to confirm the split
    print(f"\nFinal Training Data Shape: {X_train.shape}")
    print(f"Final Testing Data Shape: {X_test.shape}")
    
except FileNotFoundError:
    print(f"\nFATAL ERROR: Could not find the file at {FILE_PATH}")
    print("Please ensure your Day 1 script ran and saved the file 'df_master_day1.csv' in the correct location.")

# If you were running this file to prepare for Day 3, you would NOT include the print(preprocess_and_split(...)) line.
# Instead, you would use the variables (X_train, etc.) in the Day 3 script (model_predictor.py).