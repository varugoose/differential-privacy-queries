class QueryGenerator:
    """Generate different types of queries for testing"""
    
    @staticmethod
    def generate_correlated_queries(n_queries=20):
        """Generate correlated age queries"""
        queries = []
        
        # Overall average
        queries.append(
            (lambda d: d['age'].mean(), 'average')
        )
        
        # Conditional averages - these are correlated
        conditions = ['high_income', 'senior', 'middle_aged', 'young_adult', 
                     'college_educated']
        
        for i in range(min(n_queries - 1, len(conditions))):
            condition = conditions[i]
            queries.append(
                (lambda d, c=condition: d[d[c] == 1]['age'].mean() 
                 if any(d[c] == 1) else d['age'].mean(), 
                 'average')
            )
        
        # Add more complex queries if needed
        remaining = n_queries - len(queries)
        for i in range(remaining):
            # Random combination queries
            queries.append(
                (lambda d: d.sample(frac=0.5)['age'].mean(), 'average')
            )
        
        return queries
    
    @staticmethod
    def generate_threshold_queries(n_queries=20):
        """Generate threshold queries for SVT"""
        queries = []
        
        for i in range(n_queries):
            # Different age thresholds
            threshold_age = 30 + i
            queries.append(
                (lambda d, t=threshold_age: d[d['age'] >= t]['age'].mean()
                 if any(d['age'] >= t) else 0,
                 'average')
            )
        
        return queries