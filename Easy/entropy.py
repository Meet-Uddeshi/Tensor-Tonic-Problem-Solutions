# Entropy of node
# Topic: Classic ML

import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)

    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    return float(max(entropy, 0.0))