import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualizer:
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def plot_comparison(self, results, save_path='results/'):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data for plotting
        methods = list(results.keys())
        colors = ['red', 'blue', 'green', 'purple']
        
        # Plot 1: Error vs Number of Queries (fixed epsilon=1.0)
        ax1 = axes[0, 0]
        epsilon_fixed = 1.0
        for method, color in zip(methods, colors):
            n_queries_list = []
            errors = []
            for (n_q, eps), metrics in results[method].items():
                if eps == epsilon_fixed:
                    n_queries_list.append(n_q)
                    errors.append(metrics['mean_error'])
            if n_queries_list:
                sorted_idx = np.argsort(n_queries_list)
                n_queries_sorted = [n_queries_list[i] for i in sorted_idx]
                errors_sorted = [errors[i] for i in sorted_idx]
                ax1.plot(n_queries_sorted, errors_sorted, 'o-', 
                        label=method, color=color, linewidth=2)
        
        ax1.set_xlabel('Number of Queries', fontsize=12)
        ax1.set_ylabel('Mean Absolute Error', fontsize=12)
        ax1.set_title(f'Error vs Query Count (ε={epsilon_fixed})', fontsize=14)
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Error vs Epsilon (fixed n_queries=20)
        ax2 = axes[0, 1]
        n_queries_fixed = 20
        for method, color in zip(methods, colors):
            epsilons = []
            errors = []
            for (n_q, eps), metrics in results[method].items():
                if n_q == n_queries_fixed:
                    epsilons.append(eps)
                    errors.append(metrics['mean_error'])
            if epsilons:
                sorted_idx = np.argsort(epsilons)
                eps_sorted = [epsilons[i] for i in sorted_idx]
                errors_sorted = [errors[i] for i in sorted_idx]
                ax2.plot(eps_sorted, errors_sorted, 'o-', 
                        label=method, color=color, linewidth=2)
        
        ax2.set_xlabel('Privacy Budget (ε)', fontsize=12)
        ax2.set_ylabel('Mean Absolute Error', fontsize=12)
        ax2.set_xscale('log')
        ax2.set_title(f'Privacy-Utility Tradeoff ({n_queries_fixed} queries)', fontsize=14)
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Relative Improvement over Naive
        ax3 = axes[1, 0]
        n_queries_list = [10, 20, 30, 50]
        epsilon_fixed = 1.0
        
        naive_errors = {}
        for (n_q, eps), metrics in results['naive'].items():
            if eps == epsilon_fixed and n_q in n_queries_list:
                naive_errors[n_q] = metrics['mean_error']
        
        x = np.arange(len(n_queries_list))
        width = 0.25
        
        for i, method in enumerate(['advanced', 'svt', 'histogram']):
            improvements = []
            for n_q in n_queries_list:
                if (n_q, epsilon_fixed) in results[method] and n_q in naive_errors:
                    method_error = results[method][(n_q, epsilon_fixed)]['mean_error']
                    improvement = (naive_errors[n_q] - method_error) / naive_errors[n_q] * 100
                    improvements.append(improvement)
                else:
                    improvements.append(0)
            
            ax3.bar(x + i * width, improvements, width, label=method, alpha=0.8)
        
        ax3.set_xlabel('Number of Queries', fontsize=12)
        ax3.set_ylabel('Improvement over Naive (%)', fontsize=12)
        ax3.set_title('Relative Performance Improvement', fontsize=14)
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(n_queries_list)
        ax3.legend()
        ax3.grid(True, axis='y')
        
        # Plot 4: Variance comparison
        ax4 = axes[1, 1]
        for method, color in zip(methods, colors):
            n_queries_list = []
            stds = []
            for (n_q, eps), metrics in results[method].items():
                if eps == epsilon_fixed:
                    n_queries_list.append(n_q)
                    stds.append(metrics['std_error'])
            if n_queries_list:
                sorted_idx = np.argsort(n_queries_list)
                n_queries_sorted = [n_queries_list[i] for i in sorted_idx]
                stds_sorted = [stds[i] for i in sorted_idx]
                ax4.plot(n_queries_sorted, stds_sorted, 'o-', 
                        label=method, color=color, linewidth=2)
        
        ax4.set_xlabel('Number of Queries', fontsize=12)
        ax4.set_ylabel('Standard Deviation of Error', fontsize=12)
        ax4.set_title('Noise Variance Comparison', fontsize=14)
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}comparison_plots.png', dpi=150)
        plt.show()
    
    def plot_scalability(self, query_counts, results, save_path='results/'):
        """Plot scalability test results"""
        plt.figure(figsize=(10, 6))
        
        for method, errors in results.items():
            plt.plot(query_counts, errors, 'o-', label=method, linewidth=2)
        
        plt.xlabel('Number of Queries', fontsize=12)
        plt.ylabel('Mean Absolute Error', fontsize=12)
        plt.title('Scalability: Error Growth with Query Count (ε=1.0)', fontsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}scalability.png', dpi=150)
        plt.show()