import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)

    N,M = A.shape
    result = np.empty((M, N), dtype = A.dtype)

    for i in range(N):
        result[:, i] = A[i]

    return result
    pass
