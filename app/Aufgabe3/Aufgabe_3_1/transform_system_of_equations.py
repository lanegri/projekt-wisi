import numpy as np


def householder_transformation(in_matrix):
    """
        Führt die Householder-Transformation auf der Matrix A durch.
        Gibt die orthogonale Matrix Q und die obere Dreiecksmatrix R zurück, so dass A = QR.
    """
    (m, n) = in_matrix.shape
    Q = np.eye(m)
    R = in_matrix.copy()

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


def givens_rotation(in_matrix):
    """
    Führt die Givens-Rotation auf der Matrix A durch.
    Gibt die orthogonale Matrix Q und die obere Dreiecksmatrix R zurück, so dass A = QR.
    """
    (m, n) = in_matrix.shape
    Q = np.eye(m)
    R = in_matrix.copy()

    for j in range(n):
        for i in range(m - 1, j, -1):
            if R[i, j] != 0:
                r = np.hypot(R[i - 1, j], R[i, j])
                c = R[i - 1, j] / r
                s = -R[i, j] / r

                G = np.eye(m)
                G[[i - 1, i], [i - 1, i]] = c
                G[i - 1, i] = -s
                G[i, i - 1] = s

                R = G @ R
                Q = Q @ G.T

    return Q, R


# # Beispiel
# A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
# out_Q, out_R = householder_transformation(A)
# print("Lösung mit Householder-Orthogonalisierung:")
# print(f"Q: \n {out_Q}")
# print(f"R: \n {out_R}")
#
#
# Beispiel
B = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
out_Q, out_R = givens_rotation(B)
print("Lösung mit Givens-Rotation:")
print(f"Q: \n {out_Q}")
print(f"R: \n {out_R}")
np.set_printoptions(precision=1, suppress=True)
# Input matrix
A = np.array([
    [3, -6],
    [4, -8],
    [0, 1]
], dtype=float)

# Print input matrix
print('The A matrix is equal to:\n', A)
# Compute QR decomposition using Givens rotation
(QA, RA) = givens_rotation(A)

# Print orthogonal matrix Q
print('\n The Q matrix is equal to:\n', QA)

# Print upper triangular matrix R
print('\n The R matrix is equal to:\n', RA)

B = np.array([
    [2, -1, -2],
    [-4, 6, 3],
    [-4, -2, 8]
], dtype=float)

# Print input matrix
print('The A matrix is equal to:\n', B)
# Compute QR decomposition using Givens rotation
(QB, RB) = givens_rotation(B)

# Print orthogonal matrix Q
print('\n The Q matrix is equal to:\n', QB)

# Print upper triangular matrix R
print('\n The R matrix is equal to:\n', RB)
