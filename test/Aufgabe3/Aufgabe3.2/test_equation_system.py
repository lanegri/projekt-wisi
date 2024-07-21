import numpy as np


def ist_symmetrisch(A):
    """
    Überprüft, ob die Matrix A symmetrisch ist.
    """
    n = len(A)
    for i in range(n):
        for j in range(n):
            if A[i][j] != A[j][i]:
                return False
    return True


def cholesky_zerlegung(A):
    """
    Berechnet die Cholesky-Zerlegung einer symmetrischen und positiv definiten Matrix.
    Gibt die untere Dreiecksmatrix L zurück, so dass A = L * L.T.
    """
    n = len(A)
    L = np.zeros((n, n))

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


def givens_rotation(A):
    """
    Führt die Givens-Rotation auf der Matrix A durch.
    Gibt die orthogonale Matrix Q und die obere Dreiecksmatrix R zurück, so dass A = QR.
    """
    (m, n) = A.shape
    Q = np.eye(m)
    R = A.copy()

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
