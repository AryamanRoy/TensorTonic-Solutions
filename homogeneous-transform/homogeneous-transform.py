import numpy as np

def apply_homogeneous_transform(T, points):
    
    """
    Apply 4x4 homogeneous transform to 3D point(s).
    
    Args:
        T: np.ndarray of shape (4,4)
        points: (3,) or (N,3)
    
    Returns:
        Transformed points: (3,) or (N,3)
    """
    T = np.asarray(T)
    p = np.asarray(points)
    
    # Check if single point
    single = False
    if p.ndim == 1:
        p = p.reshape(1, 3)
        single = True
    
    # Convert to homogeneous coordinates: (N,4)
    ones = np.ones((p.shape[0], 1), dtype=p.dtype)
    p_h = np.concatenate([p, ones], axis=1)
    
    # Apply transform (vectorized)
    p_transformed = p_h @ T.T
    
    # Drop homogeneous coordinate
    p_transformed = p_transformed[:, :3]
    
    # Return original shape
    if single:
        return p_transformed[0]
    return p_transformed