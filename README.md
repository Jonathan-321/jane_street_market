
```markdown
# Jane Street Market Prediction

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/jane_street_market.git
cd jane_street_market
```

### 2. Set Up Virtual Environment

#### For Linux/Mac:
```bash
python -m venv venv
source venv/bin/activate
```

#### For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Competition Data

Download the competition data from Kaggle and extract the files to `data/raw/`:

```bash
kaggle competitions download -c jane-street-real-time-market-data-forecasting
```

Extract the following files to `data/raw/`:
- `train.parquet`
- `test.parquet`
- `features.csv`
- `responders.csv`

Processed data will be saved to `data/processed/`.

### 5. Update Requirements File

Ensure your `requirements.txt` includes the following dependencies:

```python
# requirements.txt
polars>=0.20.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
jupyter>=1.0.0
pytest>=7.0.0
```

### 6. Run Notebooks

Run the notebooks in the following order:

1. `notebooks/01_initial_eda.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_model_experiments.ipynb`

## File Structure

```
jane_street_market/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── notebooks/
│   ├── 01_initial_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── base_features.py
│   │   ├── regime_features.py
│   │   └── technical_features.py
│   └── models/
│       ├── __init__.py
│       └── model.py
└── tests/
    └── __init__.py
```

