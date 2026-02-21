import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x = np.asarray(x, dtype=float)
    pos = 1.0 / (1.0 + np.exp(-x))      
    neg = np.exp(x) / (1.0 + np.exp(x)) 
    result = np.where(x >= 0, pos, neg)
    if result.shape == ():
        return float(result)
    return result
