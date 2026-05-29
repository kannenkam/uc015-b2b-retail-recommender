import os
import joblib
import pandas as pd
import numpy as np
import warnings  # Add this

# Suppress scikit-learn numpy feature name warnings during production inference
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class B2BRecommenderInference:
    def __init__(self):
        """
        Initializes the inference engine by loading the serialized preprocessor
        and model artifacts from disk.
        """
        self.models_dir = 'models'
        self.preprocessor_path = os.path.join(self.models_dir, 'preprocessor.pkl')
        #self.model_path = os.path.join(self.models_dir, 'lightgbm_model.pkl')
        self.model_path = os.path.join(self.models_dir, 'lightgbm_model.pkl')
        # Load the production artifacts
        if not os.path.exists(self.preprocessor_path) or not os.path.exists(self.model_path):
            raise FileNotFoundError("🚨 Production model artifacts missing! Please run src/train.py first.")

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        print("✅ Core Inference Engine successfully loaded into memory.")

    def predict_quantity(self, raw_input_data: dict) -> float:
        """
        Accepts raw, un-encoded user input dictionary, processes it through
        the pipeline, and returns an optimized order quantity recommendation.
        """
        # Convert single raw input dict into a pandas DataFrame row
        df_input = pd.DataFrame([raw_input_data])

        # Transform text features and numerical metrics using our preprocessor
        X_processed = self.preprocessor.transform(df_input)

        # Generate the recommendation prediction
        prediction = self.model.predict(X_processed)[0]

        # Post-processing: Ensure business rules prevent negative recommendation values
        return max(0.0, round(float(prediction), 2))


if __name__ == '__main__':
    # Real-world verification test case
    print("🔬 Testing live inference loop with sample corporate customer request...")

    sample_request = {
        'client_segment': 'Regional Distributor',
        'product_category': 'Beverages',
        'client_store_size_sqm': 450.0,
        'unit_price_usd': 12.50,
        'historical_monthly_avg': 1200.0,
        'last_month_order_qty': 1150.0,
        'stockout_days_last_month': 2,
        'applied_discount_percentage': 10.0,
        'is_holiday_season': 0,  # 0 for No, 1 for Yes
        'lead_time_days': 4,
        'minimum_order_increment': 50,
        'regional_demand_index': 1.05,
        'client_credit_limit_utilized': 0.65,  # 65% utilization
        'supplier_delivery_reliability': 0.98  # 98% on-time delivery
    }

    # Run pipeline prediction
    engine = B2BRecommenderInference()
    recommended_qty = engine.predict_quantity(sample_request)

    print("\n================ LIVE PRODUCTION PREDICTION ================")
    print(f"🛒 Incoming Request Details: {sample_request}")
    print(f"📦 Recommended Order Target Volume: {recommended_qty} units")
    print("============================================================\n")