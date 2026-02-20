import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class NASADataPipeline:
    def __init__(self, window_size=30, clip_value=125):
        self.window_size = window_size
        self.clip_value = clip_value
        self.scaler = MinMaxScaler()
        self.feature_cols = []

    def load_data(self, file_path):
        """Standardized loading for CMAPSS format."""
        col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
        return pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)

    def calculate_rul(self, df):
        """Calculates RUL and applies aviation-standard clipping."""
        max_cycle = df.groupby('id')['cycle'].transform('max')
        df['RUL'] = (max_cycle - df['cycle']).clip(upper=self.clip_value)
        return df

    def add_features(self, df):
        """Engineering rolling means for temporal patterns."""
        sensor_cols = [f's{i}' for i in range(1, 22)]
        for col in sensor_cols:
            df[f'{col}_rolling_mean'] = df.groupby('id')[col].transform(
                lambda x: x.rolling(window=self.window_size).mean()
            )
        # Handle the NaNs created by the rolling window
        return df.ffill().bfill()

    def fit_and_transform(self, train_path):
        """Fits the scaler on TRAIN data and returns features/labels."""
        df = self.load_data(train_path)
        df = self.calculate_rul(df)
        df = self.add_features(df)
        
        # Define features: original sensors + rolling features
        self.feature_cols = [f's{i}' for i in range(1, 22)] + \
                            [f's{i}_rolling_mean' for i in range(1, 22)]
        
        df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])
        return df[self.feature_cols].values, df['RUL'].values

    def transform_test(self, test_path, truth_path):
        """Uses the ALREADY FITTED scaler to transform test data."""
        df = self.load_data(test_path)
        df = self.add_features(df)
        
        # We only predict on the last timestamp of each engine in the test set
        X_test = df.groupby('id').last().reset_index()[self.feature_cols]
        X_test_scaled = self.scaler.transform(X_test) # Critical: .transform, not .fit_transform
        
        # Load the ground truth RUL
        y_test = pd.read_csv(truth_path, header=None).iloc[:, 0].values
        y_test = np.clip(y_test, None, self.clip_value)
        
        return X_test_scaled, y_test