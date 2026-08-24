import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = np.dot(X,W_q)
    K = np.dot(X,W_k)
    V = np.dot(X,W_v)
    pass
    return Q, K, V
def softmax(X: np.ndarray, axis : int = -1) -> np.ndarray:
    exp_x = np.exp(X - np.max(X,axis = axis, keepdims = True))
    return exp_x / np.sum(exp_x, axis = axis, keepdims = True)
def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)/ np.sqrt(d_k))
    weights = softmax(scores, axis = -1)
    output = np.matmul(weights, V)
    return output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int , W_o: np.ndarray = None) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape
    assert d_model % n_heads ==0
    d_k = d_model //n_heads
    Q_heads = Q.reshape(seq_len, n_heads, d_k).swapaxes(0,1)
    K_heads = K.reshape(seq_len, n_heads, d_k).swapaxes(0,1)
    V_heads = V.reshape(seq_len, n_heads, d_k).swapaxes(0,1)
    head_outputs = self_attention(Q_heads,K_heads,V_heads)
    concat_output = head_outputs.swapaxes(0,1).reshape(seq_len, d_model)
    if W_o is None:
        W_o = np.eye(d_model)
    output = np.dot(concat_output, W_o)
    return output