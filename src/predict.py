import pandas as pd
import joblib
import os
import datetime


# ==========================================
# 1. LOAD SAVED MODEL
# ==========================================

print("Loading trained model...")
pipeline = joblib.load("../models/sales_model.pkl")
print("Model loaded successfully!")


# ==========================================
# 2. DEFINE NEW ORDER(S) TO PREDICT
# ==========================================
order_date = datetime.date(2027, 6, 24)

new_order = pd.DataFrame({

    "Quantity": [5],
    "Discount": [0],
    "Profit": [45.0],
    "Segment": ["Corporate"],
    "Region": ["East"],
    "Ship Mode": ["First Class"],
    "Sub-Category": ["Tables"],
    "Order_Year": [order_date.year],
    "Order_Month": [order_date.month],
    "Order_Day": [order_date.day],
    "Order_DayOfWeek": [order_date.weekday()]
})


# ==========================================
# 3. MAKE PREDICTION
# ==========================================

predicted_sales = pipeline.predict(new_order)

print("\n================================")
print("NEW ORDER SALES PREDICTION")
print("================================")
print(f"Predicted Sales Amount: {predicted_sales[0]:.2f}")