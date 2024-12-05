import yaml
import kaggle
from pathlib import Path

def submit_solution():
    """Submit solution to Kaggle"""
    # Load configuration
    with open('submission_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Prepare submission files
    files = {
        'inference_server.py': 'submission/inference_server.py',
        'model_development.py': 'submission/model_development.py',
        'requirements.txt': 'submission/requirements.txt'
    }
    
    # Create submission
    submission = kaggle.api.competition_submit(
        files,
        message="Regime-based model with risk controls",
        competition='jane-street-market-prediction'
    )
    
    return submission

if __name__ == "__main__":
    submit_solution() 