import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

# 1. Load the dataset
df = pd.read_csv('Tesla.csv')

# 2. Feature Engineering (from your PDF)
# Create date-based features
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df['month'] = df['Date'].dt.month
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

# Create price-based features
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']

# 3. Create the Target Variable
# Predict if the next day's close price is higher (1) or lower (0)
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Drop the last row because it has no target
df = df.dropna()

# 4. Prepare Data for Training
# Define the features and target
features_df = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

# Scale the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)

# 5. Train the XGBoost Model
model = XGBClassifier()
model.fit(features_scaled, target)

# 6. Save the Model and Scaler
joblib.dump(model, 'stock_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ XGBoost model (stock_model.pkl) and scaler (scaler.pkl) saved successfully!")