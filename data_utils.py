import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(file_path, window_size=30, is_test=False, ground_truth_path=None):
    # 1. Setup Column Names
    col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)

    # 2. Target Calculation (RUL) with Professional Clipping
    if not is_test:
        max_cycle = df.groupby('id')['cycle'].transform('max')
        # Clipping at 125 stops the model from wasting energy on 'perfect health' noise
        df['RUL'] = (max_cycle - df['cycle']).clip(upper=125)
    
    # 3. Feature Engineering: Rolling Means
    sensor_cols = [f's{i}' for i in range(1, 22)]
    for col in sensor_cols:
        df[f'{col}_rolling_mean'] = df.groupby('id')[col].transform(lambda x: x.rolling(window=window_size).mean())
    
    # Clean up NaNs created by the windowing
    df = df.ffill().bfill()

    # 4. Scaling
    feature_cols = sensor_cols + [f'{c}_rolling_mean' for c in sensor_cols]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 5. Return logic (Fixed Indentation & NumPy conversion)
    if not is_test:
        return df[feature_cols].values, df['RUL'].values
    
    # For Test: Extract only the last snapshot for each engine
    X_test_df = df.groupby('id').last().reset_index()[feature_cols]
    X_test = X_test_df.values 
    
    if ground_truth_path:
        y_test = pd.read_csv(ground_truth_path, header=None).iloc[:, 0].values
        # Apply the same clipping to ground truth for an honest comparison
        y_test = np.clip(y_test, None, 125)
        return X_test, y_test
    
    return X_test