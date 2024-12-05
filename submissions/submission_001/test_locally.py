import kaggle_evaluation.jane_street_inference_server as jsis
from inference_server import PredictionServer

def test_submission():
    # Create server instance
    server = PredictionServer()
    
    # Create test features
    test_features = {
        'feature_0': [0.1],
        'feature_1': [0.2],
        'date_id': [100],
        'time_id': [10],
        'symbol_id': [1]
    }
    
    # Test single prediction
    result = server.predict(test_features)
    print(f"Test prediction result: {result}")
    
    # Test with evaluation server
    evaluation_server = jsis.JaneStreetInferenceServer()
    evaluation_server.predict_setup(server.predict)
    print("Evaluation server setup complete")

if __name__ == "__main__":
    test_submission() 