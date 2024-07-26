import numpy as np

from app.Aufgabe2.Aufgabe_2_1.cholesky_decomposition import cholesky_decomposition
from app.utils.checkup_utils import check_is_symmetric, check_positiv_definit


def solve_upper_triangular(R, b):
    """
        Löst das Gleichungssystem Rx = b für eine obere Dreiecksmatrix R.
    """
    n = len(b)
    x = np.zeros_like(b)

    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]

    return x


def solve_lower_triangular(L, b):
    """
        Löst das Gleichungssystem Lx = b für eine untere Dreiecksmatrix L.
    """
    n = len(b)
    x = np.zeros_like(b)

    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]

    return x


def solve_with_gauss_normal_equations(A, b):
    """
        Löst das lineare Ausgleichsproblem durch Lösen der Gaußschen Normalengleichungen.
    """
    # Berechne A^T * A und A^T * b
    A_T_A = A.T @ A
    A_T_b = A.T @ b

    # Berechne die Cholesky-Zerlegung von A_T_A
    L = cholesky_decomposition(A_T_A)

    # Löse L * L^T * x = A_T_b
    # Schritt 1: Löse L * y = A_T_b
    y = solve_lower_triangular(L, A_T_b)

    # Schritt 2: Löse L^T * x = y
    x = solve_upper_triangular(L.T, y)

    return A_T_A, A_T_b, L, y, x


print("Lösung des linearen Ausgleichungsproblems mit Gauß: \n")


def out_solve_with_gauss_normal_equations():
    """
        Test Gauß linearen Ausgleichung der Matrix A
    """
    A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    b = np.array([1, -2, 7], dtype=float)

    A_T_A, A_T_b, L, y, x = solve_with_gauss_normal_equations(A, b)

    print(f"A_T_A = {A_T_A} \n")
    print(f"A_T_b = {A_T_b} \n")
    print(f"Cholesky-Zerlegung von A_T_A = {L} \n")
    print(f"Vorwärt einsetzen = {y} \n")
    print("Rückwärts einsetzen / Lösung mit Gauß:", x)


out_solve_with_gauss_normal_equations()
