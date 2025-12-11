import numpy as np
from .base_mechanism import BaseMechanism

class SparseVectorTechnique(BaseMechanism):
    """Sparse Vector Technique for threshold queries"""
    
    def __init__(self, epsilon_total, delta=1e-5, threshold=40):
        super().__init__(epsilon_total, delta)
        self.threshold = threshold
        
    def answer_queries(self, data, queries):
        """Answer threshold queries using SVT"""
        # Split epsilon: half for threshold, half for queries
        epsilon_threshold = self.epsilon_total / 3
        epsilon_queries = 2 * self.epsilon_total / 3
        
        # Add noise to threshold once
        noisy_threshold = self.threshold + self.laplace_noise(1, epsilon_threshold)
        
        results = []
        true_answers = []
        num_above = 0
        max_above = 10  # Limit number of "above" answers
        
        for query_func, query_type in queries:
            # Get true answer
            true_answer = query_func(data)
            true_answers.append(true_answer)
            
            # Add noise to query
            sensitivity = self.calculate_sensitivity(query_type)
            if query_type == 'average' and len(data) > 0:
                sensitivity = sensitivity / len(data)
            
            # Use remaining budget for queries
            if num_above < max_above:
                noise = self.laplace_noise(2 * sensitivity, epsilon_queries)
                noisy_query = true_answer + noise
                
                if noisy_query >= noisy_threshold:
                    results.append(('Above', noisy_query))
                    num_above += 1
                else:
                    results.append(('Below', None))
            else:
                results.append(('Budget Exhausted', None))
        
        return results, true_answers