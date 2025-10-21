# config.py

# --- Modeling Configuration ---

# Numerical features used for prediction
NUMERICAL_FEATURES = [
    'price_per_item',
    'freight_value_per_item',
    'delivery_delta_days',          
    'actual_delivery_days',         
    'category_avg_return_rate',     
    'seller_avg_return_rate',       
    'order_item_id',                
]

# Target variable column name
TARGET_COLUMN = 'Is_Return'

# Test set size for train_test_split
TEST_SIZE = 0.3

# Random seed for reproducibility
RANDOM_STATE = 42