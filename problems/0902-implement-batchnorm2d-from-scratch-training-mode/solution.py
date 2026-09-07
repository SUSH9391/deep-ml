import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # Compute mean and population variance over batch and spatial dimensions
    mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
    var = torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False)
    
    # Normalize
    x_hat = (x - mean) / torch.sqrt(var + eps)
    
    # Reshape gamma and beta for broadcasting
    gamma_reshaped = gamma.view(1, -1, 1, 1)
    beta_reshaped = beta.view(1, -1, 1, 1)
    
    # Scale and shift
    y = gamma_reshaped * x_hat + beta_reshaped
    return y