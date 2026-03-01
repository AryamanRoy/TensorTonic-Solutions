def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.array(X, dtype=float)

    n, d = X.shape

    mean = np.mean(X, axis=0)
    Xc = X - mean

    C = (Xc.T @ Xc) / (n - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    idx = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, idx[:k]]

    X_proj = Xc @ W
    
    return X_proj.tolist()