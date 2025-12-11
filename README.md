# Optimizing Privacy Budget for Multiple Correlated Queries: Beyond Naive Composition in Differential Privacy

## Overview
This project demonstrates how advanced composition techniques can achieve 40-50% better accuracy than naive methods when answering multiple statistical queries under differential privacy.

## Key Results
- **45% error reduction** at 20 queries
- **48% error reduction** at 50 queries  
- SVT handles **100+ threshold queries** (vs 10 with naive)

## Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/differential-privacy-queries.git
cd differential-privacy-queries

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run all experiments
python main.py
```

## Methods Implemented
1. **Naive Laplace Composition** - Baseline method splitting ε equally
2. **Advanced Composition** - Uses √k scaling with Gaussian mechanism
3. **Sparse Vector Technique** - Pays privacy cost only when needed
4. **Histogram Release** - One-shot release for unlimited queries

## Dataset
UCI Adult Income Dataset (32,561 records)
- Automatically downloaded when running the code
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)

## Project Structure
- `mechanisms/` - Different DP mechanism implementations
- `experiments/` - Experiment runner and query generation
- `visualization/` - Plotting utilities
- `data/` - Data loading and preprocessing
- `results/` - Generated plots and results

## Requirements
- Python 3.8+
- numpy
- pandas
- matplotlib
- seaborn
- scipy

## Author
Ryan Varughese

## License
MIT