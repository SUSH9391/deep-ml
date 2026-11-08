import numpy as np
from scipy.optimize import linprog

def lad_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    N, D = X.shape
    # Decision vector v = [w (D parameters), e (N residuals)]
    # Objective: minimize sum(e_n)
    c = np.zeros(D + N)
    c[D:] = 1.0
    
    A_ub = []
    b_ub = []
    
    for i in range(N):
        # Constraint 1: w^T x_n - y_n <= e_n  =>  x_n * w - e_n <= y_n
        row1 = np.zeros(D + N)
        row1[:D] = X[i]
        row1[D + i] = -1.0
        A_ub.append(row1)
        b_ub.append(y[i])
        
        # Constraint 2: -(w^T x_n - y_n) <= e_n  =>  -x_n * w - e_n <= -y_n
        row2 = np.zeros(D + N)
        row2[:D] = -X[i]
        row2[D + i] = -1.0
        A_ub.append(row2)
        b_ub.append(-y[i])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # w is unbounded, e_n >= 0
    bounds = [(None, None)] * D + [(0, None)] * N
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return res.x[:D]

def huber_fit(X: np.ndarray, y: np.ndarray, delta: float, n_iter: int) -> np.ndarray:
    N, D = X.shape
    # Start from ordinary least-squares solution
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    for _ in range(n_iter):
        r = y - X @ w
        abs_r = np.abs(r)
        
        # Compute weights s_n
        s = np.ones_like(r)
        mask = abs_r > 0
        s[mask] = np.minimum(1.0, delta / abs_r[mask])
        
        # Weighted least squares via square root of weights
        S_sqrt = np.sqrt(s)
        X_weighted = X * S_sqrt[:, np.newaxis]
        y_weighted = y * S_sqrt
        
        w, _, _, _ = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)
        
    return w