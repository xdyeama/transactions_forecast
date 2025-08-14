import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# 1. Load and preprocess data
df = pd.read_csv('./data_processed/data.csv', parse_dates=['date_time'])
df['date'] = df['date_time'].dt.date

# 2. Aggregate to daily item-level sales
df['count'] = 1
daily_sales = (df.groupby(['item_id', 'date'])
                 .agg({'count': 'sum'})
                 .reset_index()
                 .rename(columns={'count': 'sales'}))

# 3. Create lag and rolling features per item_id
# 3. Create lag and rolling features per item_id
daily_sales = daily_sales.sort_values(['item_id', 'date'])

for lag in [1, 2, 3, 7]:
    daily_sales[f'lag_{lag}'] = daily_sales.groupby('item_id')['sales'].shift(lag)

# Fix: Apply rolling within each item_id group
daily_sales['roll_mean_3'] = (
    daily_sales.groupby('item_id')['sales']
    .shift(1)
    .rolling(3)
    .mean()
    .reset_index(level=0, drop=True)
)

daily_sales['roll_mean_7'] = (
    daily_sales.groupby('item_id')['sales']
    .shift(1)
    .rolling(7)
    .mean()
    .reset_index(level=0, drop=True)
)

# Convert roll_mean columns to float
daily_sales['roll_mean_3'] = daily_sales['roll_mean_3'].astype(float)
daily_sales['roll_mean_7'] = daily_sales['roll_mean_7'].astype(float)


# 4. Add calendar features
daily_sales['date'] = pd.to_datetime(daily_sales['date'])
daily_sales['weekday'] = daily_sales['date'].dt.weekday
daily_sales['day'] = daily_sales['date'].dt.day
daily_sales['month'] = daily_sales['date'].dt.month

# Drop rows with missing lag/rolling values
daily_sales.dropna(inplace=True)

# 5. Split data into train/test based on time
# Example: train before last 7 days; test = last 7 days for forecasting
max_date = daily_sales['date'].max()
train_df = daily_sales[daily_sales['date'] < max_date - pd.Timedelta(days=7)]
test_df = daily_sales[daily_sales['date'] >= max_date - pd.Timedelta(days=7)]

features = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'roll_mean_3', 'roll_mean_7',
            'weekday', 'day', 'month']
target = 'sales'

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

# 6. Train LightGBM model
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_valid = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}

gbm = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    valid_names=["train", "valid"],
    num_boost_round=500,
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)

# 7. Evaluate performance
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print("Test MAE:", mean_absolute_error(y_test, y_pred))

# 8. Predict next day's sales for the item
# Assuming last row for that specific item exists:
# Reconstruct recent sales history
recent = daily_sales[daily_sales['item_id'] == 2].sort_values('date').tail(7)

# Create next-day features manually
next_features = {
    'lag_1': recent.iloc[-1]['sales'],
    'lag_2': recent.iloc[-2]['sales'],
    'lag_3': recent.iloc[-3]['sales'],
    'lag_7': recent.iloc[-7]['sales'],
    'roll_mean_3': recent['sales'].iloc[-4:-1].mean(),  # lag 1-3
    'roll_mean_7': recent['sales'].iloc[:-1].mean(),     # lag 1-7
    'weekday': (recent.iloc[-1]['date'] + pd.Timedelta(days=1)).weekday(),
    'day': (recent.iloc[-1]['date'] + pd.Timedelta(days=1)).day,
    'month': (recent.iloc[-1]['date'] + pd.Timedelta(days=1)).month
}

next_df = pd.DataFrame([next_features])
pred_next = gbm.predict(next_df)[0]
print(f"Predicted next-day sales for item 2: {pred_next:.2f}")

gbm.save_model("./models/model_versions/lgbm_strategic_model.txt")