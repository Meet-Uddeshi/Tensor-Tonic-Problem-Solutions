# RMSProp Optimizer (Single Update Step)
# Topic: Optimization

import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    if w.shape != g.shape or w.shape != s.shape:
        raise ValueError("w, g, and s must have the same shape")

    s = beta * s + (1.0 - beta) * (g * g)

    w = w - lr * g / (np.sqrt(s) + eps)
    if w.shape == ():
        return float(w), float(s)
    return w, s