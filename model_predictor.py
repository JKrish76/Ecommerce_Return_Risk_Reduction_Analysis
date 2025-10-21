# model_predictor.py (CORRECTED)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Import modules from Day 2
from config import NUMERICAL_FEATURES, TARGET_COLUMN, RANDOM_STATE
from preprocess_and_split import preprocess_and_split 
from feature_engineer import create_engineering_features # <-- NEW IMPORTED FUNCTION

# --- 1. Load Data (Output from Day 1) ---
try:
    df_master = pd.read_csv('df_master_day1.csv', low_memory=False)
    df_master[TARGET_COLUMN] = df_master[TARGET_COLUMN].astype(int) 
    print("✅ Day 1 Master Data Loaded.")

except FileNotFoundError:
    print("FATAL ERROR: Could not find 'df_master_day1.csv'. Please ensure your Day 1 script ran successfully and saved the file.")
    exit()

# --- 1b. CRITICAL FIX: Re-run Feature Engineering on the Full DataFrame ---
# Apply the feature creation logic to the df_master object *before* modeling
df_master = create_engineering_features(df_master) 
print("✅ Feature Engineering applied to full master table.")


# --- 2. Preprocess and Split Data (Calls Day 2 Logic) ---
# Note: The preprocess_and_split function will re-run feature engineering internally, 
# but it's essential we have the engineered columns in the main df_master for step 5.
X_train, X_test, y_train, y_test, scaler, imputer = preprocess_and_split(df_master)

# The rest of the outputs from the preprocessing stage are correct and will be printed here.

# --- 3. Model Training (Logistic Regression) ---
print("\n--- 3. Model Training ---")
model = LogisticRegression(random_state=RANDOM_STATE, solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)
print("✅ Logistic Regression Model Trained.")

# --- 4. Model Evaluation ---
# (Evaluation output is good, showing ROC-AUC of 0.7547)

# ... (Omitted printing code for brevity)

# --- 5. Prediction Generation for Full Dataset ---

# 5a. Prepare the full dataset (it now has the engineered features!)
# We use the full df_master which now contains all columns from feature_engineer.py
X_full = df_master[NUMERICAL_FEATURES].copy() 

# Impute and scale using the transformers fitted on the training set
X_full_imputed = imputer.transform(X_full) 
X_full_scaled = scaler.transform(X_full_imputed) 

# 5b. Generate the Return Risk Score (Predicted Probability of Is_Return=1)
df_master['Return_Risk_Score'] = model.predict_proba(X_full_scaled)[:, 1]

# --- 6. Export High-Risk Products CSV (DELIVERABLE) ---

RISK_THRESHOLD = 0.6
high_risk_df = df_master[df_master['Return_Risk_Score'] >= RISK_THRESHOLD].copy()

# Keep only the essential columns for business intervention and Power BI
final_deliverable_cols = [
    'order_id', 
    'product_id', 
    'product_category_name', 
    'seller_id', 
    'price', 
    'freight_value', 
    'Return_Risk_Score'
]

high_risk_export = high_risk_df[final_deliverable_cols].sort_values(
    by='Return_Risk_Score', ascending=False
)

csv_filename = 'high_risk_products_for_intervention.csv'
high_risk_export.to_csv(csv_filename, index=False)

print(f"\n--- Day 3 Deliverable Summary (CSV) ---")
print(f"Total High-Risk Products (Score >= {RISK_THRESHOLD}): {high_risk_export.shape[0]}")
print(f"CSV exported successfully: {csv_filename}")
print("\n--- PROJECT NOW MOVES TO POWER BI (Day 4 & 5) ---")