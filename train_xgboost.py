import numpy as np
import xgboost as xgb
from data_utils import load_and_preprocess
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load data
print("🚀 Loading data for XGBoost...")
X, y = load_and_preprocess('/workspaces/aeroguard-fl/data/raw/train_FD001.txt')
X_flattened = X.reshape(X.shape[0], -1)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

# 3. Train XGBoost
print("⚙️ Training XGBoost Regressor...")
# We use 'hist' tree_method for speed on larger datasets
model = xgb.XGBRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=6, 
    tree_method='hist', 
    random_state=42
)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n--- XGBOOST RESULTS ---")
print(f"✅ RMSE: {rmse:.2f} cycles")
print(f"✅ R2 Score: {r2:.4f}")