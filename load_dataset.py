import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_style("whitegrid")

# Define file paths (update these if your files are in a different folder)
file_paths = {
    'orders': r'C:\Users\JAYAKRISHNAN\Desktop\Data Analyst Intership\Project 1\olist_customers_dataset.csv\olist_orders_dataset.csv',
    'order_items': r'C:\Users\JAYAKRISHNAN\Desktop\Data Analyst Intership\Project 1\olist_customers_dataset.csv\olist_order_items_dataset.csv',
    'products': r'C:\Users\JAYAKRISHNAN\Desktop\Data Analyst Intership\Project 1\olist_customers_dataset.csv\olist_products_dataset.csv',
    'reviews': r'C:\Users\JAYAKRISHNAN\Desktop\Data Analyst Intership\Project 1\olist_customers_dataset.csv\olist_order_reviews_dataset.csv',
}

# Load the datasets
data = {}
for name, path in file_paths.items():
    try:
        # Use low_memory=False to avoid DtypeWarning, especially on large files
        data[name] = pd.read_csv(path, low_memory=False)
        print(f"Loaded {name} dataset. Shape: {data[name].shape}")
    except FileNotFoundError:
        print(f"ERROR: File not found at {path}. Please check your file paths.")
        # Exit or raise error if critical file is missing
        raise

orders_df = data['orders']
items_df = data['order_items']
products_df = data['products']
reviews_df = data['reviews']


# --- 1. Join Order Items with Products ---
# This links the item (and its seller) to the product category
df_master = pd.merge(
    items_df,
    products_df[['product_id', 'product_category_name']],
    on='product_id',
    how='left'
)
print(f"After joining with Products. Shape: {df_master.shape}")


# --- 2. Join with Orders ---
# This brings in the timestamp data (for delivery analysis later)
df_master = pd.merge(
    df_master,
    orders_df[['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']],
    on='order_id',
    how='left'
)
print(f"After joining with Orders. Shape: {df_master.shape}")


# --- 3. Join with Reviews ---
# IMPORTANT: Only the *first* review submitted per order is relevant for the score.
# We'll select the latest review submission, but since there's one main review,
# we'll simplify and drop duplicates based on order_id.

# Sort by creation date and keep the latest one (in case of multiple review submissions)
reviews_df.sort_values(by='review_creation_date', ascending=False, inplace=True)
reviews_df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)

df_master = pd.merge(
    df_master,
    reviews_df[['order_id', 'review_score']],
    on='order_id',
    how='left'
)
print(f"Final Master Analysis Table shape: {df_master.shape}")
print(df_master.head())

# --- 4. Create the Is_Return Target Variable ---
# Return/Bad Experience is defined as review_score <= 2
# Note: We are using a 5-star scale, so 1 or 2 stars is a severe failure.

df_master['Is_Return'] = np.where(df_master['review_score'] <= 2, 1, 0)

# Calculate the overall "return rate"
overall_return_rate = df_master['Is_Return'].mean() * 100
print(f"\nOverall 'Bad Experience' Rate (Review Score <= 2): {overall_return_rate:.2f}%")

# Quick check on review score distribution
print("\nReview Score Distribution:")
print(df_master['review_score'].value_counts(normalize=True).mul(100).round(2))

# --- 5a. Analyze Return Rate by Category ---

# Calculate the bad experience rate (mean of Is_Return) for each category
category_returns = df_master.groupby('product_category_name')['Is_Return'].agg(
    ['count', 'mean']
).rename(columns={'count': 'Total_Items', 'mean': 'Return_Rate'}).sort_values(by='Return_Rate', ascending=False)

# Filter for categories with a significant volume (e.g., > 100 items sold)
category_returns_filtered = category_returns[category_returns['Total_Items'] >= 100]

# Select the top 10 categories with the highest return rate
top_10_high_risk_categories = category_returns_filtered.head(10)

print("\nTop 10 High-Risk Categories (by Bad Experience Rate):")
print(top_10_high_risk_categories.to_markdown(floatfmt=".2%"))

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(
    y=top_10_high_risk_categories.index.str.replace('_', ' ').str.title(),
    x=top_10_high_risk_categories['Return_Rate'],
    # FIX: Assign the y-axis variable to 'hue' and set 'legend=False'
    hue=top_10_high_risk_categories.index.str.replace('_', ' ').str.title(), # Explicitly set hue
    legend=False,                                                              # Suppress the redundant legend
    palette="Reds_d"
)
plt.title('Top 10 High-Risk Categories (Review Score <= 2)', fontsize=16)
# ... (rest of the plot settings remains the same)
plt.show()

# --- 5b. Analyze Return Rate by Seller ---

# Calculate the bad experience rate for each seller
seller_returns = df_master.groupby('seller_id')['Is_Return'].agg(
    ['count', 'mean']
).rename(columns={'count': 'Total_Sales', 'mean': 'Return_Rate'}).sort_values(by='Return_Rate', ascending=False)

# Filter for sellers with a significant volume (e.g., > 50 sales)
seller_returns_filtered = seller_returns[seller_returns['Total_Sales'] >= 50]

# Select the top 10 sellers with the highest return rate
top_10_high_risk_sellers = seller_returns_filtered.head(10)

print("\nTop 10 High-Risk Sellers (by Bad Experience Rate):")
print(top_10_high_risk_sellers.to_markdown(floatfmt=".2%"))

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(
    y=top_10_high_risk_sellers.index,
    x=top_10_high_risk_sellers['Return_Rate'],
    # FIX: Assign the y-axis variable to 'hue' and set 'legend=False'
    hue=top_10_high_risk_sellers.index, # Explicitly set hue
    legend=False,                       # Suppress the redundant legend
    palette="Oranges_d"
)
plt.title('Top 10 High-Risk Sellers (Review Score <= 2)', fontsize=16)
# ... (rest of the plot settings remains the same)
plt.show()