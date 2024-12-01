import os
import logging
from pathlib import Path
import polars as pl
import kaggle_evaluation.jane_street_inference_server

from src.utils.logger import setup_logger
from src.data.data_loader import DataLoader
from src.models.model_factory import ModelFactory

# Setup logging
logger = setup_logger()

class MarketPredictor:
    def __init__(self):
        self.data_loader = DataLoader()
        self.model_factory = ModelFactory()
        self.model = None
        self.lags_ = None

    def predict(self, test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
        """Make predictions for the current batch."""
        # Save lags data when available (at time_id == 0)
        if lags is not None:
            self.lags_ = lags

        try:
            # Make predictions
            predictions = test.select(
                'row_id',
                pl.lit(0.0).alias('responder_6'),  # Replace with actual predictions
            )

            # Validate output format
            if isinstance(predictions, pl.DataFrame):
                assert predictions.columns == ['row_id', 'responder_6']
            else:
                raise TypeError('Predictions must be a Polars DataFrame')
            
            assert len(predictions) == len(test)
            
            return predictions

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            # Return zero predictions as fallback
            return test.select(
                'row_id',
                pl.lit(0.0).alias('responder_6'),
            )

def main():
    # Initialize predictor
    predictor = MarketPredictor()
    
    # Initialize inference server
    inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(
        predictor.predict
    )

    # Run local gateway or serve based on environment
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            (
                'data/raw/test.parquet',
                'data/raw/lags.parquet',
            )
        )

if __name__ == "__main__":
    main()