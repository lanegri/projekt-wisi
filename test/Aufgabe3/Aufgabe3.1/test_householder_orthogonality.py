import numpy as np


def householder_transformation(A):
    """
    Führt die Householder-Transformation auf der Matrix A durch.
    Gibt die orthogonale Matrix Q und die obere Dreiecksmatrix R zurück, so dass A = QR.
    """
    (m, n) = A.shape
    Q = np.eye(m)
    R = A.copy()

    for k in range(n):
        # Create the Householder vector
        x = R[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x)
        u = x - e1
        v = u / np.linalg.norm(u)

        # Create the Householder matrix
        Q_k = np.eye(m)
        Q_k[k:, k:] -= 2.0 * np.outer(v, v)

        # Apply the transformation
        R = Q_k @ R
        Q = Q @ Q_k.T

    return Q, R


def solve_least_squares_householder(A, b):
    Q, R = householder_transformation(A)
    Q_T_b = Q.T @ b
    n = A.shape[1]
    b1 = Q_T_b[:n]

    # Solve the upper triangular system Rx = b1
    x = np.linalg.solve(R[:n, :], b1)
    return x


# Beispiel
A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
b = np.array([1, -2, 7], dtype=float)

x = solve_least_squares_householder(A, b)
print("Lösung mit Householder-Orthogonalisierung:", x)
