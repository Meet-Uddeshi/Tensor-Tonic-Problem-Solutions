# Expected Value (Discrete Distribution)
# Topic: Probability and Statistics

import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")

    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Probabilities must sum to 1")

    return float(np.dot(x, p))
