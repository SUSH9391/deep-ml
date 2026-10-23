import numpy as np

def cosine_similarity(v1, v2):
    """
    Calculate the cosine_similarity of two vectors.
    Args:
        v1 (numpy.ndarray): 1D array representing the first vector.
        v2 (numpy.ndarray): 1D array representing the second vector.
    Returns:
        The cosine_similarity of the two vectors.
    """
    # Fix the denominator grouping and handle potential division by zero
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return 0.0
        
    res = np.dot(v1, v2) / norm_product
    return float(res)