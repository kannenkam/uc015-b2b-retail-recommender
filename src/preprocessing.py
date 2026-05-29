import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_preprocessing_pipeline(raw_data_path):
    """
    Loads raw data, performs validation, sets up a robust preprocessing
    pipeline, splits data, and saves transformed sets for model training.
    """
    print("📋 Loading raw dataset for preprocessing validation...")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Missing input raw data artifact at: {raw_data_path}")

    df = pd.read_csv(raw_data_path)

    # 1. In-Line Production Data Validation
    # Ensure no unexpected null values exist before training begins
    if df.isnull().sum().sum() > 0:
        print("⚠️ Warning: Null fields detected. Executing defensive drop logic...")
        df = df.dropna()

    # 2. Separate Target Vector from Features Matrix
    X = df.drop(columns=['optimal_order_quantity'])
    y = df['optimal_order_quantity']

    # 3. Categorize features by data types
    categorical_features = ['client_segment', 'product_category']
    numerical_features = [
        'client_store_size_sqm', 'unit_price_usd', 'historical_monthly_avg',
        'last_month_order_qty', 'stockout_days_last_month', 'regional_demand_index',
        'is_holiday_season', 'lead_time_days', 'supplier_delivery_reliability',
        'applied_discount_percentage', 'minimum_order_increment', 'client_credit_limit_utilized'
    ]

    # 4. Construct Individual Pipeline Transformers
    # Categorical pipeline: transforms text labels into binary column configurations
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Numerical pipeline: Scales features using statistics resilient to outliers
    num_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())
    ])

    # 5. Bundle Transformers into a unified ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numerical_features),
            ('cat', cat_transformer, categorical_features)
        ])

    # 6. Execute Production Train/Test Split (80% training, 20% validation evaluation)
    print("✂️ Splitting data into Train (80%) and Test (20%) partitions...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Fit and Transform Data
    print("⚡ Executing pipeline transformations...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Convert back to clean DataFrames to maintain structural feature tracking
    cat_encoder = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    final_columns = numerical_features + list(cat_encoder)

    X_train_final = pd.DataFrame(X_train_transformed, columns=final_columns)
    X_test_final = pd.DataFrame(X_test_transformed, columns=final_columns)

    # 8. Save artifacts locally for subsequent step consumption
    processed_dir = os.path.join('data', 'processed')
    models_dir = 'models'
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Save the split processed datasets
    X_train_final.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test_final.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)

    # CRITICAL: Save the preprocessor transformer object so we can use it during deployment/live demo
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.pkl'))

    print(f"✅ Success! Processed files and 'preprocessor.pkl' saved successfully.")
    print(f"📊 Processed Feature Matrix Shape: {X_train_final.shape}")


if __name__ == '__main__':
    raw_path = os.path.join('data', 'raw', 'raw_data.csv')
    build_preprocessing_pipeline(raw_path)