# src/data/preprocessor.py

import polars as pl
import numpy as np
from typing import List, Dict, Optional

class MarketDataPreprocessor:
    """
    Preprocessor for Jane Street Market Data
    """
    def __init__(self):
        self.feature_stats = {}
        self.symbol_stats = {}
        
    def fit(self, df: pl.DataFrame) -> None:
        """
        Calculate statistics needed for preprocessing
        """
        # Calculate feature statistics
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        for col in feature_cols:
            self.feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'missing_pct': df[col].null_count() / len(df)
            }
            
        # Calculate symbol-level statistics
        self.symbol_stats = df.groupby('symbol_id').agg([
            pl.col('responder_6').mean().alias('mean_response'),
            pl.col('responder_6').std().alias('std_response'),
            pl.count().alias('count')
        ])
    
    def handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Strategy for handling missing values:
        1. Forward fill within same symbol_id
        2. Fill remaining with 0
        """
        # Forward fill within groups
        df = df.sort(['symbol_id', 'date_id', 'time_id'])\
               .groupby('symbol_id')\
               .agg([
                   pl.all().forward_fill()
               ])\
               .fill_null(0)
        
        return df
    
    def normalize_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize features to zero mean and unit variance
        """
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        for col in feature_cols:
            if col in self.feature_stats:
                mean = self.feature_stats[col]['mean']
                std = self.feature_stats[col]['std']
                if std > 0:
                    df = df.with_columns([
                        ((pl.col(col) - mean) / std).alias(f"{col}_normalized")
                    ])
        
        return df
    
    def create_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create time-based features
        """
        df = df.with_columns([
            # Time differences
            pl.col('time_id').diff().over('symbol_id').alias('time_diff'),
            
            # Session progress
            (pl.col('time_id') - pl.col('time_id').min()) / \
            (pl.col('time_id').max() - pl.col('time_id').min()).alias('session_progress')
        ])
        
        return df
    
    def create_rolling_features(self, df: pl.DataFrame, 
                              windows: List[int] = [5, 10, 20]) -> pl.DataFrame:
        """
        Create rolling window features
        """
        for window in windows:
            df = df.with_columns([
                pl.col('responder_6')\
                  .rolling_mean(window_size=window)\
                  .over('symbol_id')\
                  .alias(f'rolling_mean_{window}'),
                  
                pl.col('responder_6')\
                  .rolling_std(window_size=window)\
                  .over('symbol_id')\
                  .alias(f'rolling_std_{window}')
            ])
        
        return df
    
    def transform(self, df: pl.DataFrame, create_features: bool = True) -> pl.DataFrame:
        """
        Apply all preprocessing steps
        """
        df = self.handle_missing_values(df)
        
        if create_features:
            df = self.normalize_features(df)
            df = self.create_time_features(df)
            df = self.create_rolling_features(df)
        
        return df