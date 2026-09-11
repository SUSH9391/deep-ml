import numpy as np

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    weights = np.array(initial_weights, dtype=float)
    bias = float(initial_bias)
    features = np.array(features, dtype=float)
    labels = np.array(labels, dtype=float)
    
    mse_values = []
    
    for _ in range(epochs):
        # Forward pass
        z = np.dot(features, weights) + bias
        predictions = 1 / (1 + np.exp(-z))
        
        # Compute MSE loss before update
        errors = predictions - labels
        mse = np.mean(errors ** 2)
        mse_values.append(round(float(mse), 4))
        
        # Backward pass (gradients for MSE with sigmoid)
        derivative_factor = errors * predictions * (1 - predictions)
        weights_gradients = (2 / len(labels)) * np.dot(features.T, derivative_factor)
        bias_gradients = (2 / len(labels)) * np.sum(derivative_factor)
        
        # Gradient descent update
        weights -= learning_rate * weights_gradients
        bias -= learning_rate * bias_gradients
        
    return np.round(weights, 4).tolist(), round(float(bias), 4), mse_values