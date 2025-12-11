import numpy as np
from .base_mechanism import BaseMechanism

class HistogramRelease(BaseMechanism):
    """One-shot histogram release mechanism"""
    
    def __init__(self, epsilon_total, delta=1e-5, bins=None):
        super().__init__(epsilon_total, delta)
        self.bins = bins if bins is not None else np.arange(18, 81, 5)
        
    def answer_queries(self, data, queries):
        """Release noisy histogram once, answer all queries from it"""
        # Create noisy histogram
        hist, bin_edges = np.histogram(data['age'], bins=self.bins)
        
        # Add noise to each bin
        sensitivity = 1  # Each person contributes to exactly one bin
        noisy_hist = hist + np.random.laplace(0, sensitivity/self.epsilon_total, len(hist))
        noisy_hist = np.maximum(noisy_hist, 0)  # Ensure non-negative
        
        # Normalize to get distribution
        total = np.sum(noisy_hist)
        if total > 0:
            noisy_dist = noisy_hist / total
        else:
            noisy_dist = np.ones_like(noisy_hist) / len(noisy_hist)
        
        results = []
        true_answers = []
        
        for query_func, query_type in queries:
            # Get true answer
            true_answer = query_func(data)
            true_answers.append(true_answer)
            
            # Answer from histogram
            if 'average' in str(query_func.__name__):
                # Reconstruct average from histogram
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                noisy_answer = np.sum(bin_centers * noisy_dist)
            else:
                # For count queries, use histogram directly
                noisy_answer = np.sum(noisy_hist)
            
            results.append(noisy_answer)
        
        return results, true_answers