import torch
import torch.nn.functional as f
import math
def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)
    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2,-1))
    scaled_scores = scores / math.sqrt(d_k)
    attention_weights = f.softmax(scaled_scores, dim = 1)
    output = torch.matmul(attention_weights, V)
    
    pass
    return output
