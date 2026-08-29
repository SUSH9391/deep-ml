import torch

def cross_entropy(logits, targets):
    # Compute the log-normalizer stably using logsumexp
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    
    # Extract the logits corresponding to the target classes
    N = logits.shape[0]
    correct_class_logits = logits[torch.arange(N), targets]
    
    # Compute the negative log-likelihood for each sample
    losses = log_sum_exp - correct_class_logits
    
    # Return the mean loss across the batch
    return losses.mean()