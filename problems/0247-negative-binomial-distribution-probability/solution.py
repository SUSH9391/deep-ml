import math

def negative_binomial_pmf(k: int, r: int, p: float) -> float:
    """
    Calculate the probability of observing exactly k failures
    before achieving r successes in independent Bernoulli trials.
    
    Args:
        k: Number of failures (non-negative integer)
        r: Number of successes required (positive integer)
        p: Probability of success on each trial (0 < p <= 1)
        
    Returns:
        Probability P(X = k) rounded to 5 decimal places
    """
    # Calculate the number of combinations: (k + r - 1) choose k
    combinations = math.comb(k + r - 1, k)
    
    # Calculate the probability
    probability = combinations * (p ** r) * ((1 - p) ** k)
    
    # Return the result rounded to 5 decimal places
    return round(probability, 5)