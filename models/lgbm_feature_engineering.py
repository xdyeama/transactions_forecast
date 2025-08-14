import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np

# Load raw data
df = pd.read_csv("./data_processed/data.csv", parse_dates=["date_time"])

# Sort by time
df = df.sort_values("date_time").reset_index(drop=True)

# Aggregate: count sales per item per timestamp
sales_df = df.groupby(["date_time", "item_id"]).size().reset_index(name="sales")

# Lag and rolling features
def create_lag_features(data, lags, rolling_windows):
    for lag in lags:
        data[f"lag_{lag}"] = data.groupby("item_id")["sales"].shift(lag)
    for window in rolling_windows:
        data[f"roll_mean_{window}"] = (
            data.groupby("item_id")["sales"]
                .shift(1)
                .rolling(window=window)
                .mean()
        )
    return data

sales_df = create_lag_features(sales_df, lags=[1, 2, 3], rolling_windows=[3, 5, 10])

# Merge weather/holiday features
df_merged = pd.merge(
    sales_df,
    df.drop(columns="sales", errors="ignore"),
    on=["date_time", "item_id"],
    how="left"
)

# Encode item_id to numeric
df_merged["item_id"] = df_merged["item_id"].astype("category").cat.codes

# Drop rows with NaN (from lag/rolling features)
df_merged = df_merged.dropna().reset_index(drop=True)

# Target: short horizon (e.g., 10 min ahead)
horizon = 10
df_merged["target"] = df_merged.groupby("item_id")["sales"].shift(-horizon)
df_merged = df_merged.dropna(subset=["target"])

# === Save processed dataset ===
df_merged.to_csv("./data_processed/lgbm_processed_data.csv", index=False)

print("Feature engineering complete. File saved to ./data_processed/lgbm_processed_data.csv")
