import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
    # Convert to a numpy array for easy mathematical operations
    joint = np.array(joint_prob)
    
    # 1. Compute marginal probabilities P(X) and P(Y)
    px = np.sum(joint, axis=1) # Sum across columns for each row
    py = np.sum(joint, axis=0) # Sum across rows for each column
    
    # 2. Compute the expected independent joint probability P(X)*P(Y)
    px_py = np.outer(px, py)
    
    # 3. Create a mask to only compute log for non-zero joint probabilities (avoids log(0) errors)
    mask = joint > 0
    
    # 4. Apply the Mutual Information formula: sum( P(x,y) * log(P(x,y) / (P(x)*P(y))) )
    mi = np.sum(joint[mask] * np.log(joint[mask] / px_py[mask]))
    
    return float(mi)