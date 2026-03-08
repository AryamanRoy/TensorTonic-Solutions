import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.asarray(x,dtype = float)
    if x.ndim == 0:
        x = x.reshape(1)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    pass