def calculate_batch_health(predictions: list, confidence_threshold: float = 0.5) -> dict:
    """
    Calculate health metrics for a batch prediction job.
    
    Args:
        predictions: list of prediction results, each a dict with 'status' and optionally 'confidence'
        confidence_threshold: threshold below which a prediction is considered low confidence
    
    Returns:
        dict with keys: 'success_rate', 'avg_confidence', 'low_confidence_rate'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if not predictions:
        return {}
    total = len(predictions)
    successes = [p for p in predictions if p.get('status') == "success"]
    no_of_successes = len(successes)
    success_rate = round((no_of_successes / total) * 100, 2)
    if no_of_successes == 0:
        return {
             'success_rate': success_rate,
        'avg_confidence': 0.0,
        'low_confidence_rate': 0.0
        }
    confidence = [p['confidence'] for p in successes]
    avg_confidence = round((sum(confidence) / no_of_successes) * 100 , 2)
    low_count = sum(1 for c in confidence if c < confidence_threshold)
    low_confidence_rate = round((low_count / no_of_successes) * 100, 2)
    
    return {
        'success_rate': success_rate,
        'avg_confidence': avg_confidence,
        'low_confidence_rate': low_confidence_rate
    }
    pass