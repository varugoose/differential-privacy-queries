import numpy as np
from .base_mechanism import BaseMechanism

class AdvancedComposition(BaseMechanism):
    def answer_queries(self, data, queries):
        """Use advanced composition theorem with Gaussian noise"""
        n_queries = len(queries)
        
        # Advanced composition: For k queries, we can use ε/√(2k ln(1/δ)) per query
        epsilon_per_query = self.epsilon_total / np.sqrt(2 * n_queries * np.log(1/self.delta))
        
        results = []
        true_answers = []
        
        for query_func, query_type in queries:
            # Get true answer
            true_answer = query_func(data)
            true_answers.append(true_answer)
            
            # Use Gaussian noise for better composition
            sensitivity = self.calculate_sensitivity(query_type)
            if query_type == 'average' and len(data) > 0:
                sensitivity = sensitivity / len(data)
            
            noise = self.gaussian_noise(sensitivity, epsilon_per_query, 
                                       self.delta/n_queries)
            noisy_answer = true_answer + noise
            
            results.append(noisy_answer)
        
        return results, true_answers