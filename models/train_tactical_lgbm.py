import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import joblib
import hashlib
from datetime import datetime

# === Load prepared data ===
horizons = [10, 20, 30]

lgbm_data = pd.read_csv("data_processed/lgbm_tactical_processed_data.csv")
# === Get top items ===
# Sort again to be safe
lgbm_data = lgbm_data.sort_values("date_time")

# Use 80% for training, 20% for validation
split_idx = int(len(lgbm_data) * 0.8)
train_df = lgbm_data.iloc[:split_idx]
valid_df = lgbm_data.iloc[split_idx:]

features = [col for col in lgbm_data.columns if col not in ["date_time", "target"]]

X_train = train_df[features]
y_train = train_df["target"]
X_valid = valid_df[features]
y_valid = valid_df["target"]

# Create LightGBM datasets
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_valid, label=y_valid)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1
}

# Train with early stopping
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=["train", "valid"],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)

# Predict and evaluate
y_pred = model.predict(X_valid)
rmse = mean_squared_error(y_valid, y_pred)
print(f"Validation RMSE: {rmse:.4f}")

model.save_model("./models/model_versions/lgbm_short_horizon.txt")

