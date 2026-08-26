import numpy as np

def temperature_decay(
    schedule_type: str,
    initial_temp: float,
    current_step: int,
    total_steps: int,
    final_temp: float = 0.01,
    decay_rate: float = 0.95
) -> float:
    # Handle edge case where total_steps is 0 to avoid division by zero
    total_steps = max(1, total_steps)
    current_step = min(max(0, current_step), total_steps)
    
    if schedule_type == 'constant':
        return initial_temp
        
    elif schedule_type == 'linear':
        # Gradually slide from initial down to final linearly
        fraction = current_step / total_steps
        temp = initial_temp - (initial_temp - final_temp) * fraction
        return max(final_temp, temp)
        
    elif schedule_type == 'exponential':
        # Multiply by decay_rate at each step
        temp = initial_temp * (decay_rate ** current_step)
        return max(final_temp, temp)
        
    elif schedule_type == 'cosine':
        # Smooth cosine curve mapping initial_temp down to final_temp
        cosine_decay = final_temp + 0.5 * (initial_temp - final_temp) * (
            1 + np.cos(np.pi * current_step / total_steps)
        )
        return max(final_temp, cosine_decay)
        
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")