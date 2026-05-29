import os
import numpy as np
import pandas as pd


def generate_b2b_data(num_rows=50000, seed=42):
    """
    Generates a production-grade synthetic dataset for Metro Agent's
    B2B Smart Order Quantity Recommender engine.
    """
    np.random.seed(seed)

    # 1. Categorical Choices
    client_segments = ['Small Grocery', 'Medium Supermarket', 'Regional Distributor']
    product_categories = ['Packaged Foods', 'Beverages', 'Household Care', 'Fresh Produce']

    # Randomly assign categories based on typical retail distributions
    client_segment_col = np.random.choice(client_segments, size=num_rows, p=[0.5, 0.35, 0.15])
    product_category_col = np.random.choice(product_categories, size=num_rows, p=[0.4, 0.3, 0.2, 0.1])

    # 2. Base Scale mapping depending on store size metrics
    # Distributors are huge, small groceries are small
    size_map = {'Small Grocery': (50, 200), 'Medium Supermarket': (201, 1500), 'Regional Distributor': (1501, 5000)}
    store_sizes = [np.random.randint(size_map[seg][0], size_map[seg][1]) for seg in client_segment_col]

    # 3. Generating Features
    unit_prices = np.round(np.random.uniform(1.5, 120.0, size=num_rows), 2)

    # Establish historic baseline relative to client scale
    hist_averages = []
    for seg in client_segment_col:
        if seg == 'Small Grocery':
            hist_averages.append(np.random.randint(10, 100))
        elif seg == 'Medium Supermarket':
            hist_averages.append(np.random.randint(101, 800))
        else:
            hist_averages.append(np.random.randint(801, 4000))
    hist_averages = np.array(hist_averages)

    # Add natural fluctuations for last month's purchase metrics
    last_month_qty = np.maximum(5, hist_averages + np.random.randint(-20, 21, size=num_rows))
    stockout_days = np.random.choice([0, 1, 2, 3, 5, 7, 10], size=num_rows, p=[0.6, 0.15, 0.1, 0.05, 0.04, 0.04, 0.02])

    # Market & Environmental Signals
    regional_demand = np.round(np.random.uniform(0.7, 1.4, size=num_rows), 2)
    is_holiday = np.random.choice([0, 1], size=num_rows, p=[0.8, 0.2])

    # Supply Chain Constraints
    lead_times = np.random.randint(1, 11, size=num_rows)
    delivery_reliability = np.round(np.random.uniform(0.75, 1.0, size=num_rows), 2)
    discounts = np.random.choice([0.0, 0.05, 0.10, 0.20], size=num_rows, p=[0.7, 0.15, 0.1, 0.05])
    increments = np.random.choice([1, 5, 10, 25], size=num_rows, p=[0.3, 0.4, 0.2, 0.1])
    credit_utilization = np.round(np.random.uniform(0.0, 0.95, size=num_rows), 2)

    # 4. Constructing the Target Variable using structural business logic
    # Base baseline derived from standard demand velocity
    base_target = hist_averages * regional_demand

    # Adjust upward if holidays are active or heavy stockouts occurred
    base_target += (is_holiday * (hist_averages * 0.15))
    base_target += (stockout_days * 15) + (lead_times * 5)

    # Boost capacity based on promotional factors
    base_target *= (1.0 + discounts)

    # Depress purchasing capacity if credit parameters are maxed out
    base_target *= (1.0 - (credit_utilization * 0.2))

    # Ensure minimum sanity baseline floor and cast data types
    optimal_qty = np.maximum(10, np.round(base_target)).astype(int)

    # Restrict volume down to regional packaging limits (snapping to packaging increment bounds)
    optimal_qty = (optimal_qty // increments) * increments
    optimal_qty = np.maximum(increments, optimal_qty)  # maintain package floor boundaries

    # 5. Assembly into DataFrame
    df = pd.DataFrame({
        'client_segment': client_segment_col,
        'client_store_size_sqm': store_sizes,
        'product_category': product_category_col,
        'unit_price_usd': unit_prices,
        'historical_monthly_avg': hist_averages,
        'last_month_order_qty': last_month_qty,
        'stockout_days_last_month': stockout_days,
        'regional_demand_index': regional_demand,
        'is_holiday_season': is_holiday,
        'lead_time_days': lead_times,
        'supplier_delivery_reliability': delivery_reliability,
        'applied_discount_percentage': discounts,
        'minimum_order_increment': increments,
        'client_credit_limit_utilized': credit_utilization,
        'optimal_order_quantity': optimal_qty
    })

    return df


if __name__ == '__main__':
    print("🚀 Initializing Synthetic Dataset Generator for UC015...")
    dataset = generate_b2b_data(num_rows=50000)

    # Secure storage location pathways
    output_dir = os.path.join('data', 'raw')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'raw_data.csv')
    dataset.to_csv(output_path, index=False)

    print(f"✅ Success! Generated {dataset.shape[0]} rows and saved to: {output_path}")
    print("\n--- Quick Schema Inspection ---")
    print(dataset.head(3))