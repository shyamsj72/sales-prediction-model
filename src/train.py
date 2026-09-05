import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

print("\nDataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)


# ==========================================
# 2. CHECK DATA
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
print(df.info())

# ==========================================
# 3. CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs", exist_ok=True)


# ==========================================
# 4. DATA PREPROCESSING
# ==========================================

df["Order Date"] = pd.to_datetime(df["Order Date"])

df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month
df["Order_Day"] = df["Order Date"].dt.day
df["Order_DayOfWeek"] = df["Order Date"].dt.dayofweek


# ==========================================
# 5. SELECT FEATURES
# ==========================================
# Dropped columns:
#   Row ID, Order ID, Customer ID/Name, Product ID/Name -> unique IDs, not predictive
#   Country -> only one value in this dataset
#   City, State, Postal Code -> too many unique values for this dataset size
#   Category -> only one value here (this file is furniture-only)
#   Ship Date -> mostly redundant with Order Date for this purpose

features = [
    "Quantity",
    "Discount",
    "Profit",
    "Segment",
    "Region",
    "Ship Mode",
    "Sub-Category",
    "Order_Year",
    "Order_Month",
    "Order_Day",
    "Order_DayOfWeek"
]

target = "Sales"

X = df[features]
y = df[target]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# ==========================================
# 6. IDENTIFY COLUMN TYPES
# ==========================================

categorical_features = [
    "Segment",
    "Region",
    "Ship Mode",
    "Sub-Category"
]

numerical_features = [
    "Quantity",
    "Discount",
    "Profit",
    "Order_Year",
    "Order_Month",
    "Order_Day",
    "Order_DayOfWeek"
]


# ==========================================
# 7. PREPROCESS CATEGORICAL DATA
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 8. CREATE MACHINE LEARNING MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 9. CREATE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 10. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 11. TRAIN MODEL
# ==========================================

print("\nTraining model...")
pipeline.fit(X_train, y_train)
print("Model training completed!")


# ==========================================
# 12. MAKE PREDICTIONS
# ==========================================

y_pred = pipeline.predict(X_test)


# ==========================================
# 13. MODEL EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")


# ==========================================
# 14. ACTUAL VS PREDICTED
# ==========================================

results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nActual vs Predicted:")
print(results.head(10))


# ==========================================
# 15. SAVE RESULTS
# ==========================================

results.to_csv("../outputs/prediction_results.csv", index=False)


# ==========================================
# 16. ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual Sales vs Predicted Sales")
plt.savefig("../outputs/actual_vs_predicted.png")
plt.show()


# ==========================================
# 17. SAVE MODEL
# ==========================================

joblib.dump(pipeline, "../models/sales_model.pkl")

print("\nModel saved successfully!")

