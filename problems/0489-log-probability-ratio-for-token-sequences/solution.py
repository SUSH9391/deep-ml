import numpy as np

def compute_log_prob_ratios(
    policy_log_probs: np.ndarray,
    reference_log_probs: np.ndarray,
    mask: np.ndarray
) -> dict:
    """
    Compute log-probability ratios between policy and reference models.
    """
    # 1. Per-token log-probability ratios (zero out padded positions)
    per_token_ratios = (policy_log_probs - reference_log_probs) * mask
    
    # 2. Sequence-level log-probability ratios (sum over valid tokens for each sequence)
    sequence_ratios = np.sum(per_token_ratios, axis=1)
    
    # 3. Mean sequence log-probability ratio across the batch
    mean_sequence_ratio = float(np.mean(sequence_ratios))
    
    return {
        'per_token_log_ratios': per_token_ratios,
        'sequence_log_ratios': sequence_ratios,
        'mean_sequence_log_ratio': mean_sequence_ratio
    }