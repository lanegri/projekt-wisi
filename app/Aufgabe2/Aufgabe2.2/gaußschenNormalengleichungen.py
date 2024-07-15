import numpy as np

def ist_symmetrisch(A):
    """
    Überprüft, ob die Matrix A symmetrisch ist.
    """
    return np.allclose(A, A.T)

def cholesky_zerlegung(A):
    """
    Berechnet die Cholesky-Zerlegung einer symmetrischen und positiv definiten Matrix.
    Gibt die untere Dreiecksmatrix L zurück, so dass A = L * L.T.
    """
    n = A.shape[0]
    L = np.zeros_like(A)

    for k in range(n):
        sum_k = sum(L[k, j] ** 2 for j in range(k))
        L[k, k] = (A[k, k] - sum_k) ** 0.5
        for i in range(k + 1, n):
            sum_ik = sum(L[i, j] * L[k, j] for j in range(k))
            L[i, k] = (A[i, k] - sum_ik) / L[k, k]

    return L

def ist_positiv_definit(A):
    """
    Überprüft, ob die Matrix A positiv definit ist.
    """
    if not ist_symmetrisch(A):
        return False

    try:
        _ = cholesky_zerlegung(A)
        return True
    except:
        return False

def solve_upper_triangular(R, b):
    """
    Löst das Gleichungssystem Rx = b für eine obere Dreiecksmatrix R.
    """
    n = len(b)
    x = np.zeros_like(b)

    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
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

def solve_least_squares_normal_equations(A, b):
    """
    Löst das lineare Ausgleichsproblem durch Lösen der Gaußschen Normalengleichungen.
    """
    # Berechne A^T * A und A^T * b
    A_T_A = A.T @ A
    A_T_b = A.T @ b

    # Überprüfe, ob A_T_A positiv definit ist
    if not ist_positiv_definit(A_T_A):
        raise ValueError("Matrix A^T * A ist nicht positiv definit.")

    # Berechne die Cholesky-Zerlegung von A_T_A
    L = cholesky_zerlegung(A_T_A)

    # Löse L * L^T * x = A_T_b
    # Schritt 1: Löse L * y = A_T_b
    y = solve_lower_triangular(L, A_T_b)

    # Schritt 2: Löse L^T * x = y
    x = solve_upper_triangular(L.T, y)

    return x

# Beispiel
A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
b = np.array([1, -2, 7], dtype=float)

x = solve_least_squares_normal_equations(A, b)
print("Lösung des linearen Ausgleichsproblems:", x)