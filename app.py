import gradio as gr
from src.predict import B2BRecommenderInference

# Initialize our core inference engine wrapper
try:
    engine = B2BRecommenderInference()
except Exception as e:
    print(f"🚨 Failed to initialize inference engine: {e}")
    engine = None


def generate_recommendation(
        client_segment, product_category, client_store_size_sqm, unit_price_usd,
        historical_monthly_avg, last_month_order_qty, stockout_days_last_month,
        applied_discount_percentage, is_holiday, lead_time_days,
        minimum_order_increment, regional_demand_index,
        client_credit_limit_utilized, supplier_delivery_reliability
):
    """
    Bridges the web UI input values directly to the back-end machine learning inference engine.
    """
    if engine is None:
        return "🚨 Error: Model artifacts could not be loaded on the server."

    # Pack the user's UI inputs into the precise dictionary format our preprocessor expects
    payload = {
        'client_segment': client_segment,
        'product_category': product_category,
        'client_store_size_sqm': float(client_store_size_sqm),
        'unit_price_usd': float(unit_price_usd),
        'historical_monthly_avg': float(historical_monthly_avg),
        'last_month_order_qty': float(last_month_order_qty),
        'stockout_days_last_month': int(stockout_days_last_month),
        'applied_discount_percentage': float(applied_discount_percentage),
        'is_holiday_season': 1 if is_holiday else 0,
        'lead_time_days': int(lead_time_days),
        'minimum_order_increment': int(minimum_order_increment),
        'regional_demand_index': float(regional_demand_index),
        'client_credit_limit_utilized': float(client_credit_limit_utilized) / 100.0,  # Convert % slider to decimal
        'supplier_delivery_reliability': float(supplier_delivery_reliability) / 100.0  # Convert % slider to decimal
    }

    # Run calculation
    recommended_qty = engine.predict_quantity(payload)

    # Return a stylized HTML string to display beautifully in the web dashboard
    return f"""
    <div style="text-align: center; padding: 20px; background-color: #f0fdf4; border-radius: 10px; border: 2px solid #bbf7d0;">
        <h3 style="color: #166534; margin: 0; font-family: sans-serif;">🎯 Recommended Order Volume</h3>
        <p style="font-size: 36px; font-weight: bold; color: #15803d; margin: 10px 0; font-family: sans-serif;">
            {recommended_qty:,} <span style="font-size: 18px; font-weight: normal;">Units</span>
        </p>
        <small style="color: #166534; font-family: sans-serif;">Optimized target calculated via Random Forest Ensemble</small>
    </div>
    """


# Build the professional Gradio UI Layout Dashboard
with gr.Blocks(theme=gr.themes.Soft(), title="B2B Demand Optimization Engine") as demo:
    gr.Markdown(
        """
        # 📈 B2B Retail Demand Optimization Dashboard
        **Enterprise Intelligent Replenishment & Order Volume Recommendation System**

        *Adjust the corporate client parameters and logistics metrics below to calculate real-time inventory target optimization recommendations.*
        """
    )

    with gr.Row():
        # Column 1: Client Profile & Pricing Context
        with gr.Column():
            gr.Markdown("### 🏢 Client & Product Profile")
            client_segment = gr.Dropdown(
                label="Client Segment",
                choices=["Small Grocery", "Supermarket", "Regional Distributor", "National Chain"],
                value="Regional Distributor"
            )
            product_category = gr.Dropdown(
                label="Product Category",
                choices=["Groceries", "Beverages", "Electronics", "Apparel", "Home & Kitchen"],
                value="Beverages"
            )
            client_store_size_sqm = gr.Number(label="Store Size (sqm)", value=450.0)
            unit_price_usd = gr.Slider(label="Unit Price ($ USD)", minimum=0.50, maximum=250.0, value=12.50, step=0.50)
            applied_discount_percentage = gr.Slider(label="Applied Discount (%)", minimum=0.0, maximum=50.0, value=10.0,
                                                    step=0.5)

        # Column 2: Historical Velocity & Inventory Performance
        with gr.Column():
            gr.Markdown("### 📊 Historical Run-Rates & Stock Risks")
            historical_monthly_avg = gr.Number(label="Historical Monthly Avg Qty", value=1200.0)
            last_month_order_qty = gr.Number(label="Last Month Order Qty", value=1150.0)
            stockout_days_last_month = gr.Slider(label="Stockout Days (Last Month)", minimum=0, maximum=30, value=2,
                                                 step=1)
            is_holiday = gr.Checkbox(label="Is Active Holiday Peak Season?", value=False)

        # Column 3: Logistics & Operational Credit Constraints
        with gr.Column():
            gr.Markdown("### 🚚 Supply Chain & Risk Indicators")
            lead_time_days = gr.Slider(label="Supplier Lead Time (Days)", minimum=1, maximum=21, value=4, step=1)
            minimum_order_increment = gr.Number(label="Minimum Order Increment Units", value=50)
            regional_demand_index = gr.Slider(label="Regional Demand Macro Index", minimum=0.5, maximum=2.0, value=1.05,
                                              step=0.01)
            client_credit_limit_utilized = gr.Slider(label="Client Credit Limit Utilized (%)", minimum=0.0,
                                                     maximum=100.0, value=65.0, step=1.0)
            supplier_delivery_reliability = gr.Slider(label="Supplier Delivery Reliability (%)", minimum=50.0,
                                                      maximum=100.0, value=98.0, step=1.0)

    # Output Row
    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=2):
            submit_btn = gr.Button("🚀 Compute Optimized Order Target", variant="primary")
        with gr.Column(scale=3):
            output_display = gr.HTML(value="""
            <div style="text-align: center; padding: 20px; background-color: #f8fafc; border-radius: 10px; border: 2px dashed #cbd5e1; color: #64748b; font-family: sans-serif;">
                Click the compute button to run machine learning optimization inference
            </div>
            """)

    # Set up event trigger
    submit_btn.click(
        fn=generate_recommendation,
        inputs=[
            client_segment, product_category, client_store_size_sqm, unit_price_usd,
            historical_monthly_avg, last_month_order_qty, stockout_days_last_month,
            applied_discount_percentage, is_holiday, lead_time_days,
            minimum_order_increment, regional_demand_index,
            client_credit_limit_utilized, supplier_delivery_reliability
        ],
        outputs=output_display
    )

if __name__ == '__main__':
    # Launch app locally
    demo.launch(server_name="127.0.0.1", server_port=7860)