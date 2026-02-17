import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(file_path, window_size=30, is_test=False, ground_truth_path=None):
    """
    Professional-grade preprocessing for NASA CMAPSS dataset.
    Includes RUL calculation, rolling features, and scaling.
    """
    # 1. Define Column Names
    col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    
    # 2. Load Data
    # Using sep=r'\s+' because the NASA files are space-delimited
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)

    # 3. Handle Remaining Useful Life (RUL)
    if not is_test:
        # For training: RUL = Max Cycle - Current Cycle
        max_cycle = df.groupby('id')['cycle'].transform('max')
        df['RUL'] = max_cycle - df['cycle']
    elif ground_truth_path:
        # For testing: We eventually merge with the RUL_FD001.txt values
        pass 

    # 4. Feature Engineering: Rolling Means
    # This turns 'noisy sensors' into 'clear trends'
    sensor_cols = [f's{i}' for i in range(1, 22)]
    
    for col in sensor_cols:
        # Groupby 'id' is mandatory so engine data doesn't cross-contaminate
        df[f'{col}_rolling_mean'] = df.groupby('id')[col].transform(
            lambda x: x.rolling(window=window_size).mean()
        )
    
    # Fill the NaNs at the start of each window so the model doesn't crash
    df = df.ffill().bfill()

    # 5. Scaling
    # We scale both the raw sensors and our new rolling features
    feature_cols = sensor_cols + [f'{c}_rolling_mean' for c in sensor_cols]
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df

def get_latest_cycles(df):
    """
    Extracts the last recorded cycle for each engine in the test set.
    """
    return df.groupby('id').last().reset_index()