# Adam Optimizer Step
# Topic: Optimization

import numpy as np

def adam_step(param, grad, m, v, t,
              lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Returns:
        param_new, m_new, v_new
    """
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    if not (param.shape == grad.shape == m.shape == v.shape):
        raise ValueError("param, grad, m, and v must have the same shape")
    t_for_bias = int(t) if int(t) > 0 else 1

    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad * grad)

    beta1_t = beta1 ** t_for_bias
    beta2_t = beta2 ** t_for_bias
    m_hat = m / (1.0 - beta1_t)
    v_hat = v / (1.0 - beta2_t)

    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param, m, v