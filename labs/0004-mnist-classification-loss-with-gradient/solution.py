import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    # Your implementation here
    
    N, C = preds.shape
    
    # Corrected variable name and `np.arange` typo
    true_prob = preds[np.arange(N), target]
    
    # Compute average cross-entropy loss with numerical stability
    loss = -np.mean(np.log(np.clip(true_prob, 1e-15, 1.0)))
    
    # Fixed syntax parenthesis and divide the full batch gradient by N
    one_hot = np.eye(C)[target]
    grad = (preds - one_hot) / N
    
    # Return both required outputs: loss value and gradient
    
    return loss, grad