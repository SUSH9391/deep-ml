def estimate_decode_latency(num_params, bytes_per_param, bandwidth_bytes_per_s):
    # Return [latency_ms, tokens_per_sec]
    # 1. Total size of model weights in bytes
    total_bytes = num_params * bytes_per_param
    
    # 2. Time in seconds to stream weights for one token
    latency_seconds = total_bytes / bandwidth_bytes_per_s
    
    # 3. Convert latency to milliseconds
    latency_ms = latency_seconds * 1000.0
    
    # 4. Calculate tokens per second throughput
    tokens_per_sec = 1.0 / latency_seconds if latency_seconds > 0 else 0.0
    
    return [latency_ms, tokens_per_sec]
    pass