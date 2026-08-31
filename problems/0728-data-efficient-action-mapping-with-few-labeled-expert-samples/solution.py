import numpy as np

def map_latent_to_real(latent_labeled: list, real_labeled: list, latent_query: list) -> np.ndarray:
    """
    Fit an affine map from latent actions to real actions using a small labeled set,
    and apply it to a batch of latent queries.
    """
    # Convert lists to numpy arrays
    X = np.array(latent_labeled, dtype=float)
    Y = np.array(real_labeled, dtype=float)
    X_query = np.array(latent_query, dtype=float)
    
    # Augment the labeled latent actions with a column of ones to account for the bias term
    N = X.shape[0]
    X_aug = np.hstack([X, np.ones((N, 1))])
    
    # Solve for the weights and bias using Least Squares
    # X_aug @ W_aug = Y -> W_aug contains both the slope and the bias
    W_aug, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
    
    # Augment the latent queries similarly
    M = X_query.shape[0]
    X_query_aug = np.hstack([X_query, np.ones((M, 1))])
    
    # Predict the real actions by applying the learned affine map
    predicted_real = X_query_aug @ W_aug
    
    return predicted_real
    