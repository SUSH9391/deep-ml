import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    T = attn_weights.shape[0]
    
    if method == 'tril':
        # Construct a lower-triangular mask of 1s
        mask = np.tril(np.ones((T, T)))
        
        # Multiply element-wise with the attention weights
        masked_weights = attn_weights * mask
        
        # Re-normalize each row so it sums to 1
        row_sums = masked_weights.sum(axis=-1, keepdims=True)
        res = masked_weights / row_sums
        
    elif method == 'triu':
        # Construct an upper-triangular boolean mask above the diagonal (k=1)
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        
        # Take the log to get original pre-softmax-like scores
        scores = np.log(attn_weights)
        
        # Set masked positions to -inf
        scores[mask] = -np.inf
        
        # Apply softmax along the last axis (numerically stable)
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        res = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
    else:
        raise ValueError("Method must be 'tril' or 'triu'")
        
    # Return as a nested python list
    return res.tolist()