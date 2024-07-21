import numpy as np

from app.Aufgabe2.Aufgabe_2_2.gauss_normal_equations import solve_upper_triangular
from app.Aufgabe3.Aufgabe_3_1.givens_rotation import givens_rotation
from app.Aufgabe3.Aufgabe_3_1.householder_orthogonality import householder_transformation


def solve_least_squares_householder(A, b):
    Q, R = householder_transformation(A)
    Q_T_b = Q.T @ b
    n = A.shape[1]
    b1 = Q_T_b[:n]

    # Solve the upper triangular system Rx = b1
    x = solve_upper_triangular(R[:n, :], b1)
    return x


def solve_least_squares_givens(A, b):
    Q, R = givens_rotation(A)
    Q_T_b = Q.T @ b
    n = A.shape[1]
    b1 = Q_T_b[:n]

    # Solve the upper triangular system Rx = b1
    x = solve_upper_triangular(R[:n, :], b1)
    return x


# Beispiel
A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
b = np.array([1, -2, 7], dtype=float)

x_householder = solve_least_squares_householder(A, b)
print("Lösung mit Householder-Orthogonalisierung:", x_householder)

x_givens = solve_least_squares_givens(A, b)
print("Lösung mit Givens-Rotation:", x_givens)
