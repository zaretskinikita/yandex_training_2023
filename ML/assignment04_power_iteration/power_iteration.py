import numpy as np

def get_dominant_eigenvalue_and_eigenvector(A, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    
    r = np.random.rand(A.shape[0])
    for i in range(num_steps):
        r = A.dot(r) / np.sum(A.dot(r))
    mu = float(r.T.dot(A.dot(r)) / r.dot(r))    
    return mu, r / (np.sum(r ** 2))**0.5