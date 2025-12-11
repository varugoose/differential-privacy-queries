import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self):
        self.data_dir = Path(__file__).parent
        
    def load_adult_dataset(self):
        """Load UCI Adult dataset"""
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                   'marital_status', 'occupation', 'relationship', 'race', 'sex',
                   'capital_gain', 'capital_loss', 'hours_per_week', 
                   'native_country', 'income']
        
        # Load data
        data = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
        data = data.dropna()
        
        # Add some derived binary features for queries
        data['high_income'] = (data['income'] == '>50K').astype(int)
        data['senior'] = (data['age'] >= 65).astype(int)
        data['middle_aged'] = ((data['age'] >= 40) & (data['age'] < 65)).astype(int)
        data['young_adult'] = ((data['age'] >= 18) & (data['age'] < 40)).astype(int)
        data['college_educated'] = data['education'].isin(['Bachelors', 'Masters', 'Doctorate']).astype(int)
        
        print(f"Loaded {len(data)} records from Adult dataset")
        return data
    
    def generate_synthetic_data(self, n_records=10000):
        """Generate synthetic data with known correlations"""
        np.random.seed(42)
        
        # Base age distribution
        age = np.random.normal(45, 15, n_records).clip(18, 80).astype(int)
        
        # Create correlated features
        data = pd.DataFrame({
            'age': age,
            'income_high': np.where(age > 40, 
                                   np.random.binomial(1, 0.6, n_records),
                                   np.random.binomial(1, 0.2, n_records)),
            'has_disease_x': np.where(age > 50,
                                     np.random.binomial(1, 0.4, n_records),
                                     np.random.binomial(1, 0.1, n_records)),
            'has_disease_y': np.where(age > 60,
                                     np.random.binomial(1, 0.5, n_records),
                                     np.random.binomial(1, 0.15, n_records)),
            'employed': np.where((age >= 25) & (age <= 65),
                               np.random.binomial(1, 0.8, n_records),
                               np.random.binomial(1, 0.3, n_records))
        })
        
        return data