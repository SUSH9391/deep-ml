import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Verifies the correctness of analytically computed gradients by comparing 
    them with numerically approximated gradients using centered finite difference.
    """
    # Ensure inputs are numpy arrays
    x = np.array(x, dtype=float)
    analytical_grad = np.array(analytical_grad, dtype=float)
    
    # Initialize numerical gradient array with the same shape as x
    numerical_grad = np.zeros_like(x)
    
    # Iterator to handle both 1D and multi-dimensional arrays
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]
        
        # Compute f(x + epsilon)
        x[idx] = old_value + epsilon
        f_plus = f(x)
        
        # Compute f(x - epsilon)
        x[idx] = old_value - epsilon
        f_minus = f(x)
        
        # Restore the original value
        x[idx] = old_value
        
        # Calculate the centered finite difference
        numerical_grad[idx] = (f_plus - f_minus) / (2 * epsilon)
        
        it.iternext()
        
    # Calculate norms
    norm_diff = np.linalg.norm(numerical_grad - analytical_grad)
    norm_num = np.linalg.norm(numerical_grad)
    norm_ana = np.linalg.norm(analytical_grad)
    
    # Handle the edge case where both gradients are zero vectors
    if norm_num == 0.0 and norm_ana == 0.0:
        relative_error = 0.0
    else:
        # Calculate relative error
        relative_error = norm_diff / (norm_num + norm_ana)
        
    return numerical_grad, relative_error