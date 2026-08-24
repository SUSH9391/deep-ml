import numpy as np
import math
def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)
def softmax(scores): 
	max_scores = np.max(scores, axis = -1, keepdims= True)
	exp_scores =np.exp(scores - max_scores)
	return exp_scores / np.sum(exp_scores, axis =-1, keepdims = True)
def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	# Your code here
	d_k = Q.shape[-1]
	scores = np.dot(Q, np.swapaxes(K, -2, -1)) / np.sqrt(d_k)
	score_mask = scores + mask
	#softmax helper function
	
	attention_weights = softmax(score_mask)
	output = np.dot(attention_weights, V) # returns dot product of softmax and matrix V
	pass
	return output