import numpy as np
import pandas as pd
from pathlib import Path

# Import project modules
from data.data_loader import DataLoader
from experiments.experiment_runner import ExperimentRunner
from visualization.plots import Visualizer

def main():
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Differential Privacy: Query Optimization Comparison")
    print("=" * 60)
    
    # Step 1: Load Data
    print("\n1. Loading data...")
    loader = DataLoader()
    data = loader.load_adult_dataset()
    print(f"   Dataset shape: {data.shape}")
    print(f"   Features: {data.columns.tolist()[:5]}...")
    
    # Step 2: Run Experiments
    print("\n2. Running experiments...")
    runner = ExperimentRunner(data)
    
    # Main comparison
    print("   Running mechanism comparison...")
    results = runner.run_comparison(
        n_queries_list=[10, 20, 30, 50],
        epsilon_values=[0.1, 0.5, 1.0, 2.0, 5.0],
        n_trials=100
    )
    
    # Scalability test
    print("   Running scalability test...")
    query_counts, scalability_results = runner.run_scalability_test(
        epsilon=1.0, 
        max_queries=100
    )
    
    # Step 3: Generate Visualizations
    print("\n3. Generating visualizations...")
    viz = Visualizer()
    viz.plot_comparison(results)
    viz.plot_scalability(query_counts, scalability_results)
    
    # Step 4: Print Summary Statistics
    print("\n4. Summary Results:")
    print("-" * 40)
    
    # Compare at n_queries=20, epsilon=1.0
    n_q, eps = 20, 1.0
    print(f"\nAt {n_q} queries with ε={eps}:")
    for method in ['naive', 'advanced', 'svt', 'histogram']:
        if (n_q, eps) in results[method]:
            error = results[method][(n_q, eps)]['mean_error']
            std = results[method][(n_q, eps)]['std_error']
            print(f"  {method:12s}: Error = {error:.3f} ± {std:.3f}")
    
    # Calculate improvements
    if (n_q, eps) in results['naive']:
        naive_error = results['naive'][(n_q, eps)]['mean_error']
        print(f"\nImprovements over naive:")
        for method in ['advanced', 'svt', 'histogram']:
            if (n_q, eps) in results[method]:
                method_error = results[method][(n_q, eps)]['mean_error']
                improvement = (naive_error - method_error) / naive_error * 100
                print(f"  {method:12s}: {improvement:.1f}% reduction")
    
    print("\n" + "=" * 60)
    print("Experiment complete! Check 'results/' folder for plots.")
    print("=" * 60)

if __name__ == "__main__":
    main()