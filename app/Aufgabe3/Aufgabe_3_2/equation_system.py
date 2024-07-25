import numpy as np

from app.Aufgabe2.Aufgabe_2_2.gauss_normal_equations import solve_upper_triangular
from app.Aufgabe3.Aufgabe_3_1.transform_system_of_equations import q_r_with_householder_transformation, q_r_with_givens_rotation


def solve_least_squares_householder(matrix_A, matrix_b):
    Q, R = q_r_with_householder_transformation(matrix_A)
    Q_T_b = Q.T @ matrix_b
    n = matrix_A.shape[1]
    b1 = Q_T_b[:n]

    # Solve the upper triangular system Rx = b1
    x = solve_upper_triangular(R[:n, :], b1)
    return x


def solve_least_squares_givens(matrix_A, matrix_b):
    Q, R = q_r_with_givens_rotation(matrix_A)
    Q_T_b = Q.T @ matrix_b
    n = matrix_A.shape[1]
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
