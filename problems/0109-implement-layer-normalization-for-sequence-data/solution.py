import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	mean = np.mean(X, axis = -1, keepdims = True)
	var = np.var(X, axis = -1 , keepdims = True)
	x_normal_vector = (X - mean) / np.sqrt(var + epsilon)
	out = gamma * x_normal_vector +beta
	# Your code here
	pass
	return out