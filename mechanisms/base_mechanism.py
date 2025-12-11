import numpy as np
from abc import ABC, abstractmethod

class BaseMechanism(ABC):
    """Base class for all DP mechanisms"""
    
    def __init__(self, epsilon_total, delta=1e-5):
        self.epsilon_total = epsilon_total
        self.delta = delta
        
    @abstractmethod
    def answer_queries(self, data, queries):
        """Answer a list of queries with DP"""
        pass
    
    def laplace_noise(self, sensitivity, epsilon):
        """Generate Laplace noise"""
        scale = sensitivity / epsilon
        return np.random.laplace(0, scale)
    
    def gaussian_noise(self, sensitivity, epsilon, delta):
        """Generate Gaussian noise"""
        sigma = sensitivity * np.sqrt(2 * np.log(1.25/delta)) / epsilon
        return np.random.normal(0, sigma)
    
    def calculate_sensitivity(self, query_type):
        """Calculate sensitivity for different query types"""
        if query_type == 'count':
            return 1
        elif query_type == 'average':
            # Assuming bounded age [18, 80]
            return 62  # max change in average
        else:
            return 1