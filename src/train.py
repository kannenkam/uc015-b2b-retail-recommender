import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_production_model():
    """
    Loads preprocessed datasets, trains a Random Forest regressor natively
    on local CPU compute, evaluates its KPIs, and serializes the final model artifact.
    """
    print("🏋️ Loading preprocessed training and evaluation data partitions...")
    processed_dir = os.path.join('data', 'processed')

    # Load feature matrices and target arrays
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()

    print("🤖 Initializing Random Forest Engine natively...")
    # Optimized parameters to keep the binary file well under GitHub's 100MB limit
    model = RandomForestRegressor(
        n_estimators=40,  # Reduced from 100 to 40 to shrink file size significantly
        max_depth=10,  # Reduced from 15 to 10 to limit depth complexity
        random_state=42,
        n_jobs=-1
    )

    print("🚀 Training the forest model instance (this will take a few seconds)...")
    model.fit(X_train, y_train)

    print("📊 Evaluating model predictions against test partition KPIs...")
    predictions = model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {
        "Mean_Absolute_Error_MAE": round(float(mae), 2),
        "Root_Mean_Squared_Error_RMSE": round(float(rmse), 2),
        "R2_Score": round(float(r2), 4)
    }

    print("\n================ MODEL PERFORMANCE SYSTEM LABELS ================")
    print(json.dumps(metrics, indent=4))
    print("=================================================================\n")

    # Save metrics log for portfolio audit trail
    metrics_path = os.path.join('models', 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"📝 Performance metrics log saved to: {metrics_path}")

    # Serialize and save the final trained model binary
    model_output_path = os.path.join('models', 'lightgbm_model.pkl')

    # Note: We keep the filename as 'lightgbm_model.pkl' for now so we don't have to change
    # our upcoming deployment/inference file paths. It works exactly the same!
    joblib.dump(model, model_output_path)
    print(f"📦 Final model binary serialized successfully at: {model_output_path}")


if __name__ == '__main__':
    train_production_model()