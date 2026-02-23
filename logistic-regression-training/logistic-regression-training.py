import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    # Ensure numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    N, D = X.shape
    
    # Initialize parameters
    w = np.zeros(D)
    b = 0.0
    
    for _ in range(steps):
        # Forward pass
        z = X @ w + b            # (N,)
        p = _sigmoid(z)          # (N,)
        
        # Gradients
        error = p - y            # (N,)
        dw = (X.T @ error) / N   # (D,)
        db = np.sum(error) / N   # scalar
        
        # Parameter update
        w -= lr * dw
        b -= lr * db
    
    return w, float(b)
    pass