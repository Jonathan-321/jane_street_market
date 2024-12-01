# src/utils/metrics.py
import numpy as np

def weighted_r2_score(y_true, y_pred, weights):
    """
    Calculate weighted R² score as per competition metric
    """
    numerator = np.sum(weights * (y_true - y_pred) ** 2)
    denominator = np.sum(weights * y_true ** 2)
    r2 = 1 - numerator / denominator
    return r2

# src/data/preprocessor.py
class DataPreprocessor:
    """Handle feature engineering and data preprocessing"""
    def __init__(self):
        self.feature_means = {}
        self.feature_stds = {}
    
    def handle_missing_values(self, df):
        """
        Strategy for handling missing values based on sample analysis:
        - Forward fill within same symbol_id
        - Fill remaining with 0
        """
        return df.groupby('symbol_id').ffill().fillna(0)
    
    def create_time_features(self, df):
        """
        Create time-based features from date_id and time_id
        """
        # Time differences
        df['time_diff'] = df.groupby('symbol_id')['time_id'].diff()
        
        # Rolling statistics per symbol
        for window in [5, 10, 20]:
            df[f'rolling_mean_{window}'] = df.groupby('symbol_id')['responder_6'].rolling(window).mean().reset_index(0, drop=True)
        
        return df

# src/features/feature_generator.py
class FeatureGenerator:
    """Generate features for the model"""
    def __init__(self):
        self.feature_history = {}
    
    def create_lag_features(self, df, lags=[1, 2, 3]):
        """
        Create lag features for each responder
        """
        for lag in lags:
            for col in [f'responder_{i}' for i in range(9)]:
                df[f'{col}_lag_{lag}'] = df.groupby('symbol_id')[col].shift(lag)
        return df
    
    def create_technical_features(self, df):
        """
        Create technical indicators based on responder values
        """
        # Moving averages
        windows = [5, 10, 20]
        for window in windows:
            df[f'ma_{window}'] = df.groupby('symbol_id')['responder_6'].rolling(window).mean().reset_index(0, drop=True)
            
        # Volatility
        df['volatility'] = df.groupby('symbol_id')['responder_6'].rolling(10).std().reset_index(0, drop=True)
        
        return df