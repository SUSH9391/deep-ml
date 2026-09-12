import numpy as np

def off_policy_nstep_td(
    episodes: list,
    behavior_policy: list,
    target_policy: list,
    num_states: int,
    num_actions: int,
    n: int,
    alpha: float,
    gamma: float
) -> np.ndarray:
    """
    Off-policy n-step TD prediction for state values using importance sampling.
    """
    # 1. Initialize all state values to zero.
    V = np.zeros(num_states)
    
    # 2. Process episodes in order.
    for episode in episodes:
        T = len(episode)
        
        # Process time steps from the beginning to the end.
        for t in range(T):
            max_step = min(t + n, T)
            
            # 3. Compute the appropriate n-step return
            G = 0.0
            for i in range(t, max_step):
                _, _, r = episode[i]
                G += (gamma ** (i - t)) * r
                
            # Bootstrap if we haven't reached the end of the episode
            if t + n < T:
                next_s, _, _ = episode[t + n]
                G += (gamma ** n) * V[next_s]
                
            # 4. Compute the importance sampling correction ratio for the n-step window
            rho = 1.0
            for i in range(t, max_step):
                s, a, _ = episode[i]
                prob_pi = target_policy[s][a]
                prob_b = behavior_policy[s][a]
                rho *= (prob_pi / prob_b)
                
            # 5. Apply the TD update
            s_t, _, _ = episode[t]
            V[s_t] += alpha * rho * (G - V[s_t])
            
    return V