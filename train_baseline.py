import numpy as np
import pandas as pd
from data_utils import load_and_preprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# 1. Load the processed data
# Note: Using the path verified in your previous terminal output
DATA_PATH = '/workspaces/aeroguard-fl/data/raw/train_FD001.txt'

print("🚀 Loading and Windowing Data...")
try:
    X, y = load_and_preprocess(DATA_PATH)
except FileNotFoundError:
    print(f"❌ Error: Could not find data at {DATA_PATH}. Check your folders!")
    exit()

# 2. Flatten for Random Forest: (Samples, Window, Features) -> (Samples, Window*Features)
# This converts the 3D data into a 2D format the model can read
X_flattened = X.reshape(X.shape[0], -1)

# 3. Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

# 4. Train the Model
print(f"⚙️ Training Random Forest on {X_train.shape[0]} samples...")
# n_jobs=-1 uses all your CPU cores for faster training
model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n" + "="*30)
print(f"--- BASELINE RESULTS ---")
print(f"✅ RMSE: {rmse:.2f} cycles")
print(f"✅ R2 Score: {r2:.4f}")
print("="*30)

# 6. Sanity Check
print(f"\nSample Prediction: Model predicted {y_pred[0]:.1f}, Actual was {y_test[0]:.1f}")