import os
import shutil
from pathlib import Path

class SubmissionManager:
    def __init__(self, submission_id: str):
        self.submission_id = submission_id
        self.base_path = Path(f"submissions/submission_{submission_id}")
        
    def prepare_submission(self):
        """Prepare submission directory"""
        # Create directories if they don't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "models").mkdir(exist_ok=True)
        
        # Copy necessary files
        self._copy_common_files()
        
        print(f"Submission {self.submission_id} prepared at {self.base_path}")
    
    def _copy_common_files(self):
        """Copy common utilities to submission directory"""
        common_path = Path("submissions/common")
        for file in common_path.glob("*.py"):
            if file.name != "__init__.py":
                shutil.copy2(file, self.base_path)

    def validate_submission(self):
        """Check if submission has all required files"""
        required_files = [
            "inference_server.py",
            "requirements.txt"
        ]
        
        missing = []
        for file in required_files:
            if not (self.base_path / file).exists():
                missing.append(file)
        
        if missing:
            print(f"Missing required files: {missing}")
            return False
        return True