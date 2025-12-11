import numpy as np
from .base_mechanism import BaseMechanism

class NaiveLaplace(BaseMechanism):
    def answer_queries(self, data, queries):
        n_queries = len(queries)
        epsilon_per_query = self.epsilon_total / n_queries
        
        results = []
        true_answers = []
        
        for query_func, query_type in queries:
            # Get true answer
            true_answer = query_func(data)
            true_answers.append(true_answer)
            
            # Add Laplace noise
            sensitivity = self.calculate_sensitivity(query_type)
            if query_type == 'average' and len(data) > 0:
                sensitivity = sensitivity / len(data)
            
            noise = self.laplace_noise(sensitivity, epsilon_per_query)
            noisy_answer = true_answer + noise
            
            results.append(noisy_answer)
        
        return results, true_answers