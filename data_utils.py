import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(file_path, window_size=30):
    # 1. Load Data
    col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    
    # 2. Calculate RUL (Remaining Useful Life) - Our Target
    # We find the max cycle for each engine and subtract current cycle
    max_cycle = df.groupby('id')['cycle'].transform('max')
    df['RUL'] = max_cycle - df.cycle
    
    # 3. Scale Features (MinMax Scaling is standard for Sensors)
    scaler = MinMaxScaler()
    feature_cols = [c for c in df.columns if c not in ['id', 'cycle', 'RUL']]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 4. Create Sliding Windows
    X, y = [], []
    for engine_id in df['id'].unique():
        engine_data = df[df['id'] == engine_id]
        if len(engine_data) < window_size:
            continue # Professional handle: skip engines smaller than window
            
        for i in range(len(engine_data) - window_size + 1):
            # Window of sensor data
            X.append(engine_data.iloc[i:i+window_size][feature_cols].values)
            # The label is the RUL at the END of the window
            y.append(engine_data.iloc[i+window_size-1]['RUL'])
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    path = '/workspaces/aeroguard-fl/data/raw/train_FD001.txt'
    X, y = load_and_preprocess(path)
    print(f"✅ Data Processed. Shape: {X.shape}, Labels: {y.shape}")