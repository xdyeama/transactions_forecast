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
daily_sales = daily_sales.sort_values(['item_id', 'date'])
for lag in [1, 2, 3, 7]:
    daily_sales[f'lag_{lag}'] = daily_sales.groupby('item_id')['sales'].shift(lag)
daily_sales['roll_mean_3'] = daily_sales.groupby('item_id')['sales'].shift(1).rolling(3).mean()
daily_sales['roll_mean_7'] = daily_sales.groupby('item_id')['sales'].shift(1).rolling(7).mean()

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

train_df.to_csv('data_processed/strategic_train.csv', index=False)
test_df.to_csv('data_processed/strategic_test.csv', index=False)