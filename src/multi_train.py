import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("Loading dataset...")
df = pd.read_csv("../data/sales.csv", encoding="latin1")

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month
df["Order_Day"] = df["Order Date"].dt.day
df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek

os.makedirs("../models", exist_ok=True)


# ==========================================
# 2. FEATURES / TARGET (same for every model - only the TRAINING ROWS differ)
# ==========================================
# Every model below uses the exact same input form / feature columns.
# What changes is WHICH ROWS of history each one is trained on. This
# means a user always fills in the same form, but picking a different
# "filter" actually calls a genuinely different trained model underneath.

FEATURES = [
    "Quantity", "Discount", "Profit", "Segment", "Region",
    "Ship Mode", "Sub-Category", "Product Name", "Order_Year",
    "Order_Month", "Order_Day", "Order_DayOfWeek"
]
TARGET = "Sales"

CATEGORICAL_FEATURES = ["Segment", "Region", "Ship Mode", "Sub-Category", "Product Name"]


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ],
        remainder="passthrough"
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_and_save(name, subset_df, save_path):
    """Trains one Random Forest on subset_df and saves it. Returns metrics."""

    X = subset_df[FEATURES]
    y = subset_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    joblib.dump(pipeline, save_path)

    print(f"\n=== {name} ===")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")
    print(f"Saved to: {save_path}")

    return {"name": name, "rows": len(subset_df), "mae": mae, "rmse": rmse, "r2": r2}


# ==========================================
# 3. DEFINE THE FOUR DATA SLICES
# ==========================================

results = []

# --- Model 1: General model - trained on ALL historical data ---
results.append(train_and_save(
    "General Model (all data)",
    df,
    "../models/model_general.pkl"
))

# --- Model 2: Most Recent Year - trained ONLY on the latest year on record ---
latest_year = df["Order_Year"].max()
last_year_df = df[df["Order_Year"] == latest_year]
results.append(train_and_save(
    f"Recent Year Model ({latest_year} only)",
    last_year_df,
    "../models/model_recent_year.pkl"
))

# --- Model 3: High Profit - trained ONLY on the top 25% most profitable orders ---
profit_threshold = df["Profit"].quantile(0.75)
high_profit_df = df[df["Profit"] >= profit_threshold]
results.append(train_and_save(
    f"High Profit Model (top 25%, Profit >= ${profit_threshold:.2f})",
    high_profit_df,
    "../models/model_high_profit.pkl"
))

# --- Model 4: Best Seller - trained ONLY on the single most-sold Sub-Category ---
best_seller_category = df.groupby("Sub-Category")["Quantity"].sum().idxmax()
best_seller_df = df[df["Sub-Category"] == best_seller_category]
results.append(train_and_save(
    f"Best Seller Model ({best_seller_category} only)",
    best_seller_df,
    "../models/model_best_seller.pkl"
))

# --- Model 5: Loss Risk - trained ONLY on orders that historically LOST money ---
# 34% of all historical orders had negative Profit (avg -$85). This model
# specializes in understanding what a "high loss-risk" order looks like.
loss_risk_df = df[df["Profit"] < 0]
results.append(train_and_save(
    "Loss Risk Model (Profit < 0 orders only)",
    loss_risk_df,
    "../models/model_loss_risk.pkl"
))

# --- Model 6: High Discount - trained ONLY on orders with >20% discount ---
# Heavily discounted orders (>20% off) average -$100 profit vs +$70 for
# no-discount orders - a dramatic difference. This model specializes in
# pricing behavior once heavy discounting is involved.
high_discount_df = df[df["Discount"] > 0.20]
results.append(train_and_save(
    "High Discount Model (Discount > 20% orders only)",
    high_discount_df,
    "../models/model_high_discount.pkl"
))


# ==========================================
# 4. SUMMARY TABLE
# ==========================================

print("\n\n================================")
print("SUMMARY - ALL 6 MODELS")
print("================================")
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))

# Save which category was the best seller, so the app can display it
with open("../models/best_seller_category.txt", "w") as f:
    f.write(best_seller_category)

print(f"\nBest seller category saved as: {best_seller_category}")
print("\nAll models trained and saved successfully!")