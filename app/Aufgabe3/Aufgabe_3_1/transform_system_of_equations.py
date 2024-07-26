from fractions import Fraction

import numpy as np


def matrix_to_fractions(matrix):
    # Convert each element in the matrix to a Fraction and simplify
    fraction_matrix = np.vectorize(lambda x: Fraction(x).limit_denominator())(matrix)
    return fraction_matrix


def householder_vector(a_k):
    """
        Householder Vektor für Vektor a_k berechnen.
    """
    norm_x = np.linalg.norm(a_k)
    if norm_x == 0:
        return a_k  # Vektor so zurückgeben, um null division zu vermeiden

    e = np.zeros_like(a_k)
    e[0] = np.copysign(norm_x, -a_k[0])
    v = a_k - e
    v_norm = np.linalg.norm(v)
    if v_norm != 0:
        v = v / v_norm
    return v


def householder_matrix(m, v):
    """
        Householder Matrix H_k mit Hilfe von Vektor v berechnen.
    """
    # H_k
    return np.eye(m) - 2 * np.outer(v, v)


def q_r_with_householder_transformation(A):
    """
        Führt die Householder-Transformation auf der Matrix A durch.
        Gibt die orthogonale Matrix Q und die obere Dreiecksmatrix R zurück, so dass A = QR.
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for k in range(min(m, n)):
        # Householder Vektor für den k-te Spalten-Vektor berechnen
        a_k = R[k:, k]
        v = householder_vector(a_k)

        # Householder Matrix H_k berechnen
        H_k = householder_matrix(m - k, v)
        # H_k expandieren
        H = np.eye(m)
        H[k:, k:] = H_k

        # Q und R transformieren
        R = H @ R
        Q = Q @ H.T

    return Q, R


def givens_rotation(a, b):
    """Berechnet die Givens-Rotationsparameter c und s."""
    if b == 0:
        c = 1
        s = 0
    else:
        r = np.hypot(a, b)
        c = a / r
        s = -b / r
    return c, s


def q_r_with_givens_rotation(A):
    """
    Führt die Givens-Rotation auf der Matrix A durch.
    Gibt die orthogonale Matrix Q und die obere Dreiecksmatrix R zurück, so dass A = QR.
    """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()

    for j in range(n):
        for i in range(m - 1, j, -1):
            if R[i, j] != 0:
                c, s = givens_rotation(R[i - 1, j], R[i, j])

                G = np.eye(m)
                G[i - 1, i - 1] = c
                G[i - 1, i] = -s
                G[i, i - 1] = s
                G[i, i] = c

                R = G @ R
                Q = Q @ G.T

    return Q, R

np.set_printoptions(precision=1, suppress=True)

# Beispiel
print('Beispiel 1')
A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
Q_A_H, R_A_H = q_r_with_householder_transformation(A)
print("Lösung mit Householder-Orthogonalisierung:")
print(f"Q: \n {Q_A_H}")
print(f"R: \n {R_A_H}")


# Beispiel
A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
Q_A_G, R_A_G = q_r_with_givens_rotation(A)
print("Lösung mit Givens-Rotation:")
print(f"Q: \n {Q_A_G}")
print(f"R: \n {R_A_G}")

print('Beispiel 2')

# Input matrix
B = np.array([
    [3, -6],
    [4, -8],
    [0, 1]
], dtype=float)

# Print input matrix
print('B = :\n', B)
print("Lösung mit Householder-Orthogonalisierung:")
Q_B_H, R_B_H = q_r_with_householder_transformation(A)
print(f"Q: \n {Q_A_H}")
print(f"R: \n {R_A_H}")

B = np.array([
    [3, -6],
    [4, -8],
    [0, 1]
], dtype=float)

# Compute QR decomposition using Givens rotation
Q_B_G, R_B_G = q_r_with_givens_rotation(B)

# Print orthogonal matrix Q
print('\n Q = \n', Q_B_G)

# Print upper triangular matrix R
print('\n R = \n', R_B_G)
