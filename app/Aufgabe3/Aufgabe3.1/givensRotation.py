import numpy as np

def givens_rotation(A):
    """
    Führt die Givens-Rotation auf der Matrix A durch.
    Gibt die orthogonale Matrix Q und die obere Dreiecksmatrix R zurück, so dass A = QR.
    """
    (m, n) = A.shape
    Q = np.eye(m)
    R = A.copy()

    for j in range(n):
        for i in range(m-1, j, -1):
            if R[i, j] != 0:
                r = np.hypot(R[i-1, j], R[i, j])
                c = R[i-1, j] / r
                s = -R[i, j] / r

                G = np.eye(m)
                G[[i-1, i], [i-1, i]] = c
                G[i-1, i] = -s
                G[i, i-1] = s

                R = G @ R
                Q = Q @ G.T

    return Q, R

def solve_least_squares_givens(A, b):
    Q, R = givens_rotation(A)
    Q_T_b = Q.T @ b
    n = A.shape[1]
    b1 = Q_T_b[:n]

    # Solve the upper triangular system Rx = b1
    x = np.linalg.solve(R[:n, :], b1)
    return x

# Beispiel
A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
b = np.array([1, -2, 7], dtype=float)

x = solve_least_squares_givens(A, b)
print("Lösung mit Givens-Rotation:", x)
