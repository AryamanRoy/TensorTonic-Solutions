import numpy as np

def clip_gradients(g, max_norm):
    # Convert to numpy array (preserve dtype if already array)
    g = np.array(g)
    
    # If max_norm is non-positive, do nothing
    if max_norm <= 0:
        return g
    
    # Compute L2 norm
    norm = np.sqrt(np.sum(g * g))
    
    # Clip only if needed
    if norm > max_norm and norm > 0:
        g = g * (max_norm / norm)
    
    return g