import numpy as np
import xgboost as xgb
from data_utils import load_and_preprocess
from sklearn.metrics import mean_squared_error

# 1. Load Data
DATA_PATH = '/workspaces/aeroguard-fl/data/raw/train_FD001.txt'
X, y = load_and_preprocess(DATA_PATH)
X_flattened = X.reshape(X.shape[0], -1)

# 2. The "Sub-20" Settings
model = xgb.XGBRegressor(
    n_estimators=1000,      # High limit, but we'll stop early
    learning_rate=0.05,     # Slightly faster to find the sweet spot
    max_depth=8,            # One level deeper for complex sensor patterns
    subsample=0.8,
    colsample_bytree=0.8,   # Only use 80% of sensors per tree (prevents obsession)
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)

print("🚀 Attempting to break the Sub-20 RMSE barrier...")
model.fit(X_flattened, y)

# 3. Quick Check (On the training data for now)
y_pred = model.predict(X_flattened)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"\n" + "🌟" * 10)
print(f"NEW RMSE: {rmse:.2f}")
print("🌟" * 10)