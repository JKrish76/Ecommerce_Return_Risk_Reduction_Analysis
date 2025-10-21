import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for plots
sns.set_style("whitegrid")
pd.options.display.float_format = '{:.2f}'.format

# --- 1. Define File Paths (CRUCIAL STEP) ---
# **Action:** Update this BASE_DIR to the folder containing your Olist CSV files.
BASE_DIR = r'C:\Users\JAYAKRISHNAN\Desktop\Data Analyst Intership\Project 1\olist_customers_dataset'

file_paths = {
    'orders': BASE_DIR + r'\olist_orders_dataset.csv',
    'order_items': BASE_DIR + r'\olist_order_items_dataset.csv',
    'products': BASE_DIR + r'\olist_products_dataset.csv',
    'reviews': BASE_DIR + r'\olist_order_reviews_dataset.csv',
}

# --- 2. Load the Datasets ---
data = {}
for name, path in file_paths.items():
    try:
        # Use low_memory=False for safety on large CSVs
        data[name] = pd.read_csv(path, low_memory=False)
        print(f"Loaded {name} dataset. Shape: {data[name].shape}")
    except FileNotFoundError:
        print(f"FATAL ERROR: File not found at {path}. Please check your BASE_DIR and file names.")
        exit()

orders_df = data['orders']
items_df = data['order_items']
products_df = data['products']
reviews_df = data['reviews']

# --- 3. Join the Essential CSVs into a Master Table ---

# Start with Order Items (transaction level)
df_master = items_df.copy()

# A. Join with Products to get Category
df_master = pd.merge(
    df_master,
    products_df[['product_id', 'product_category_name']],
    on='product_id',
    how='left'
)

# B. Join with Orders to get Customer Info and Dates
df_master = pd.merge(
    df_master,
    orders_df[['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']],
    on='order_id',
    how='left'
)

# C. Join with Reviews to get the Score (Our Return Proxy)
# Pre-process reviews: Keep only one review per order (the latest one, though usually only one exists)
reviews_df.sort_values(by='review_creation_date', ascending=False, inplace=True)
reviews_df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)

df_master = pd.merge(
    df_master,
    reviews_df[['order_id', 'review_score']],
    on='order_id',
    how='left'
)

print(f"\n✅ Master Analysis Table Created. Final Shape: {df_master.shape}")

# --- 4. Create the Target Variable ('Is_Return') ---

# Define 'Return' / 'Bad Experience' as a review score <= 2 (on a 5-star scale)
df_master['Is_Return'] = np.where(df_master['review_score'] <= 2, 1, 0)

# Calculate the overall "return rate"
overall_return_rate = df_master['Is_Return'].mean() * 100
print(f"Overall 'Bad Experience' Rate (Review Score <= 2): {overall_return_rate:.2f}%")


# --- 5. Essential EDA ---

# A. Return Rate by Category
category_returns = df_master.groupby('product_category_name')['Is_Return'].agg(
    ['count', 'mean']
).rename(columns={'count': 'Total_Items', 'mean': 'Return_Rate'}).sort_values(by='Return_Rate', ascending=False)

# Filter for categories with significant volume (> 100 items)
category_returns_filtered = category_returns[category_returns['Total_Items'] >= 100]
top_10_high_risk_categories = category_returns_filtered.head(10)

print("\n--- Top 10 High-Risk Categories (Initial Diagnosis) ---")
print(top_10_high_risk_categories.to_markdown(floatfmt=".2%")) # Use .2% for rate, .0f for count if possible

# B. Return Rate by Seller
seller_returns = df_master.groupby('seller_id')['Is_Return'].agg(
    ['count', 'mean']
).rename(columns={'count': 'Total_Sales', 'mean': 'Return_Rate'}).sort_values(by='Return_Rate', ascending=False)

# Filter for sellers with significant volume (> 50 sales)
seller_returns_filtered = seller_returns[seller_returns['Total_Sales'] >= 50]
top_10_high_risk_sellers = seller_returns_filtered.head(10)

print("\n--- Top 10 High-Risk Sellers (Initial Diagnosis) ---")
print(top_10_high_risk_sellers.to_markdown(floatfmt=".2%"))


# --- 6. Save Master Table for Day 2/3 ---
# This is CRITICAL for the rest of your pipeline (Day 2/3 scripts) to work
df_master.to_csv('df_master_day1.csv', index=False)
print("\n✅ Master DataFrame saved as 'df_master_day1.csv' for Day 2/3 modeling.")

# --- Visualization (Optional but Recommended) ---
# Visualization (Cleaned to fix the FutureWarning)
def plot_top_risks(data, title, color_palette):
    plt.figure(figsize=(10, 6))
    
    # We use the index (Category/Seller ID) for y and hue
    sns.barplot(
        y=data.index,
        x=data['Return_Rate'],
        hue=data.index,
        legend=False,
        palette=color_palette
    )
    plt.title(title, fontsize=14)
    plt.xlabel('Bad Experience Rate (%)', fontsize=12)
    plt.ylabel(data.index.name.replace('_', ' ').title(), fontsize=12)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    plt.tight_layout()
    plt.show()

# Run the visualization functions
plot_top_risks(
    top_10_high_risk_categories.copy(), 
    'Top 10 High-Risk Categories', 
    'Reds_d'
)
plot_top_risks(
    top_10_high_risk_sellers.copy(), 
    'Top 10 High-Risk Sellers', 
    'Oranges_d'
)