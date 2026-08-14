import numpy as np

def fit_bradley_terry(comparisons, n_items, learning_rate, n_iterations):
    """
    Estimate Bradley-Terry strength parameters using gradient ascent.
    
    :param comparisons: List of (winner_idx, loser_idx) tuples
    :param n_items: Total number of items
    :param learning_rate: Step size for gradient ascent
    :param n_iterations: Number of optimization iterations
    :return: numpy array of shape (n_items,) containing centered strength parameters
    """
    # 1. Initialize all beta parameters to 0
    beta = np.zeros(n_items)
    
    # Numerically stable sigmoid function
    def sigmoid(x):
        # Prevent overflow for large negative numbers
        pos_mask = x >= 0
        z = np.zeros_like(x, dtype=float)
        z[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
        exp_x = np.exp(x[~pos_mask])
        z[~pos_mask] = exp_x / (1.0 + exp_x)
        return z

    # 2. Optimization loop
    for _ in range(n_iterations):
        grads = np.zeros(n_items)
        
        # Accumulate gradients from all comparisons
        for winner, loser in comparisons:
            diff = beta[winner] - beta[loser]
            p = sigmoid(diff)
            
            # Error term (1 - p)
            error = 1.0 - p
            grads[winner] += error
            grads[loser] -= error
            
        # Update parameters via gradient ascent
        beta += learning_rate * grads
        
        # 3. Center parameters by subtracting their mean (identifiability constraint)
        beta -= np.mean(beta)
        
    return beta