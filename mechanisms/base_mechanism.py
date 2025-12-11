import numpy as np
from abc import ABC, abstractmethod

class BaseMechanism(ABC):
    def __init__(self, epsilon_total, delta=1e-5):
        self.epsilon_total = epsilon_total
        self.delta = delta
        
    @abstractmethod
    def answer_queries(self, data, queries):
        pass
    
    def laplace_noise(self, sensitivity, epsilon):
        scale = sensitivity / epsilon
        return np.random.laplace(0, scale)
    
    def gaussian_noise(self, sensitivity, epsilon, delta):
        sigma = sensitivity * np.sqrt(2 * np.log(1.25/delta)) / epsilon
        return np.random.normal(0, sigma)
    
    def calculate_sensitivity(self, query_type):
        if query_type == 'count':
            return 1
        elif query_type == 'average':
            # Assuming bounded age [18, 80]
            return 62  
        else:
            return 1