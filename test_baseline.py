from pipeline import NASADataPipeline
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error

# 1. Initialize the professional pipeline
pipe = NASADataPipeline()

# 2. Process data using the Class methods
print("📊 Processing Training Data...")
X_train, y_train = pipe.fit_and_transform('data/raw/train_FD001.txt')

print("📊 Processing Test Data...")
X_test, y_test = pipe.transform_test('data/raw/test_FD001.txt', 'data/raw/RUL_FD001.txt')

# 3. Quick Baseline Model (The same Ridge we used before)
model = Ridge(alpha=100.0)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# 4. Results
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"\n✅ Day 1 Baseline Verified!")
print(f"Target RMSE: 35.59")
print(f"Your RMSE: {rmse:.2f}")