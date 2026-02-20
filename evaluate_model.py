import numpy as np
from data_utils import load_and_preprocess
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Paths verified from your previous terminal sessions
TRAIN_PATH = 'data/raw/train_FD001.txt'
TEST_PATH = 'data/raw/test_FD001.txt'
TRUTH_PATH = 'data/raw/RUL_FD001.txt'

print("📂 Loading data...")
X_train, y_train = load_and_preprocess(TRAIN_PATH, is_test=False)
X_test, y_test = load_and_preprocess(TEST_PATH, is_test=True, ground_truth_path=TRUTH_PATH)

# These work now because the data is returned as NumPy arrays
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=100.0))
])

print("⚖️ Training...")
model.fit(X_train_flat, y_train)

print("🚀 Evaluating...")
y_pred = model.predict(X_test_flat)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n✅ FINAL RESULTS")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")