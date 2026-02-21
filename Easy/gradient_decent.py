# Gradient Descent for a 1D Quadratic

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    # history = []
    for _ in range(steps):
        derivative = (2*a*x + b)
        x = x - lr * derivative
        history.append(x)
    return x
    # return x, history     
    