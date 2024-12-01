# src/data/data_loader.py

import polars as pl
import pandas as pd
from pathlib import Path

class JaneStreetDataLoader:
    """Data loader for Jane Street Market Data"""
    def __init__(self):
        self.data_path = Path('../data/raw/jane-street-real-time-market-data-forecasting')
    
    def load_metadata(self):
        """Load feature and responder metadata"""
        features_meta = pd.read_csv(self.data_path / 'features.csv')
        responders_meta = pd.read_csv(self.data_path / 'responders.csv')
        return features_meta, responders_meta
    
    def load_training_sample(self, n_partitions=2):
        """Load first n partitions of training data"""
        frames = []
        for i in range(n_partitions):
            try:
                df = pl.read_parquet(self.data_path / 'train.parquet' / f'partition_id={i}')
                frames.append(df)
            except Exception as e:
                print(f"Error loading partition {i}: {e}")
        
        if frames:
            return pl.concat(frames)
        return None
    
    def get_feature_columns(self):
        """Get list of feature columns"""
        return [f'feature_{str(i).zfill(2)}' for i in range(79)]

    def analyze_data_structure(self, df: pl.DataFrame) -> dict:
        """Analyze basic structure of the dataset"""
        structure = {
            'shape': df.shape,
            'features': len([col for col in df.columns if col.startswith('feature_')]),
            'responders': len([col for col in df.columns if col.startswith('responder_')]),
            'unique_dates': len(df['date_id'].unique()),
            'unique_times': len(df['time_id'].unique()),
            'unique_symbols': len(df['symbol_id'].unique()),
            'missing_values': {
                col: df[col].null_count()
                for col in df.columns
                if df[col].null_count() > 0
            }
        }
        return structure

    def analyze_time_structure(self, df):
            """
            Analyze temporal structure of the data using Polars syntax
            """
            try:
                # Get basic time information
                date_range = (df.select(pl.col('date_id').min()).item(), 
                            df.select(pl.col('date_id').max()).item())
                time_range = (df.select(pl.col('time_id').min()).item(), 
                            df.select(pl.col('time_id').max()).item())
                
                # Calculate observations per day
                daily_stats = (df.group_by('date_id')
                            .agg(pl.count().alias('count'))
                            .select(
                                pl.col('count').mean().alias('avg_obs_per_day'),
                                pl.col('count').min().alias('min_obs_per_day'),
                                pl.col('count').max().alias('max_obs_per_day')
                            ))
                
                # Calculate symbols per timepoint
                symbol_stats = (df.group_by(['date_id', 'time_id'])
                                .agg(pl.n_unique('symbol_id').alias('symbol_count'))
                                .select(
                                    pl.col('symbol_count').mean().alias('avg_symbols'),
                                    pl.col('symbol_count').min().alias('min_symbols'),
                                    pl.col('symbol_count').max().alias('max_symbols')
                                ))
                
                return {
                    'date_range': date_range,
                    'time_range': time_range,
                    'avg_observations_per_day': float(daily_stats.select('avg_obs_per_day').item()),
                    'min_observations_per_day': float(daily_stats.select('min_obs_per_day').item()),
                    'max_observations_per_day': float(daily_stats.select('max_obs_per_day').item()),
                    'avg_symbols_per_timepoint': float(symbol_stats.select('avg_symbols').item()),
                    'min_symbols_per_timepoint': float(symbol_stats.select('min_symbols').item()),
                    'max_symbols_per_timepoint': float(symbol_stats.select('max_symbols').item())
                }
            except Exception as e:
                print(f"Error in time structure analysis: {e}")
                return {
                    'error': str(e)
                }

    def analyze_feature_correlations(self, df: pl.DataFrame, target_col='responder_6') -> pd.DataFrame:
        """Analyze feature correlations with target"""
        feature_cols = self.get_feature_columns()
        correlations = []
        
        for col in feature_cols:
            if df[col].null_count() / len(df) < 0.5:  # Only analyze features with <50% missing
                corr = df[col].corr(df[target_col])
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
        
        corr_df = pd.DataFrame(correlations)
        return corr_df.sort_values('abs_correlation', ascending=False)
    

    