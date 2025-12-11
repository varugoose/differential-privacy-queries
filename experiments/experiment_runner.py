import numpy as np
from mechanisms.naive_laplace import NaiveLaplace
from mechanisms.advanced_composition import AdvancedComposition
from mechanisms.sparse_vector import SparseVectorTechnique
from mechanisms.histogram_release import HistogramRelease
from experiments.query_generator import QueryGenerator

class ExperimentRunner:
    def __init__(self, data):
        self.data = data
        self.query_gen = QueryGenerator()
        
    def run_comparison(self, n_queries_list=[10, 20, 30, 50], 
                       epsilon_values=[0.1, 0.5, 1.0, 2.0, 5.0],
                       n_trials=100):
        results = {
            'naive': {},
            'advanced': {},
            'svt': {},
            'histogram': {}
        }
        
        for n_queries in n_queries_list:
            print(f"\nTesting with {n_queries} queries...")
            queries = self.query_gen.generate_correlated_queries(n_queries)
            
            for epsilon in epsilon_values:
                print(f"  Epsilon = {epsilon}")
                
                # Initialize mechanisms
                mechanisms = {
                    'naive': NaiveLaplace(epsilon),
                    'advanced': AdvancedComposition(epsilon),
                    'svt': SparseVectorTechnique(epsilon),
                    'histogram': HistogramRelease(epsilon)
                }
                
                for name, mechanism in mechanisms.items():
                    errors = []
                    
                    for trial in range(n_trials):
                        noisy_answers, true_answers = mechanism.answer_queries(
                            self.data, queries
                        )
                        
                        # Calculate error (handling SVT output)
                        if name == 'svt':
                            valid_errors = []
                            for noisy, true in zip(noisy_answers, true_answers):
                                if isinstance(noisy, tuple) and noisy[1] is not None:
                                    valid_errors.append(abs(noisy[1] - true))
                            if valid_errors:
                                errors.append(np.mean(valid_errors))
                        else:
                            trial_errors = [abs(n - t) for n, t in 
                                          zip(noisy_answers, true_answers)]
                            errors.append(np.mean(trial_errors))
                    
                    # Store results
                    key = (n_queries, epsilon)
                    if key not in results[name]:
                        results[name][key] = {}
                    
                    results[name][key] = {
                        'mean_error': np.mean(errors),
                        'std_error': np.std(errors),
                        'median_error': np.median(errors)
                    }
        
        return results
    
    def run_scalability_test(self, epsilon=1.0, max_queries=100):
        query_counts = list(range(5, max_queries + 1, 5))
        results = {
            'naive': [],
            'advanced': [],
            'histogram': []
        }
        
        for n_queries in query_counts:
            queries = self.query_gen.generate_correlated_queries(n_queries)
            
            # Naive Laplace
            naive = NaiveLaplace(epsilon)
            noisy, true = naive.answer_queries(self.data, queries)
            naive_error = np.mean([abs(n - t) for n, t in zip(noisy, true)])
            results['naive'].append(naive_error)
            
            # Advanced Composition
            advanced = AdvancedComposition(epsilon)
            noisy, true = advanced.answer_queries(self.data, queries)
            advanced_error = np.mean([abs(n - t) for n, t in zip(noisy, true)])
            results['advanced'].append(advanced_error)
            
            # Histogram (constant error)
            histogram = HistogramRelease(epsilon)
            noisy, true = histogram.answer_queries(self.data, queries)
            hist_error = np.mean([abs(n - t) for n, t in zip(noisy, true)])
            results['histogram'].append(hist_error)
        
        return query_counts, results