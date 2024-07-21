import numpy as np


def ist_symmetrisch(A):
    return np.allclose(A, A.T)


def cholesky_zerlegung(A):
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
    if not ist_symmetrisch(A):
        return False

    try:
        _ = cholesky_zerlegung(A)
        return True
    except:
        return False


def solve_upper_triangular(R, b):
    n = len(b)
    x = np.zeros_like(b)

    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]

    return x


def solve_lower_triangular(L, b):
    n = len(b)
    x = np.zeros_like(b)

    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]

    return x


def solve_least_squares_normal_equations(A, b):
    A_T_A = A.T @ A
    A_T_b = A.T @ b

    if not ist_positiv_definit(A_T_A):
        raise ValueError("Matrix A^T * A ist nicht positiv definit.")

    L = cholesky_zerlegung(A_T_A)

    y = solve_lower_triangular(L, A_T_b)
    x = solve_upper_triangular(L.T, y)

    return x


def householder_transformation(A):
    (m, n) = A.shape
    Q = np.eye(m)
    R = A.copy()

    for k in range(n):
        x = R[k:, k]
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x)
        u = x - e1
        v = u / np.linalg.norm(u)

        Q_k = np.eye(m)
        Q_k[k:, k:] -= 2.0 * np.outer(v, v)

        R = Q_k @ R
        Q = Q @ Q_k.T

    return Q, R


def solve_least_squares_householder(A, b):
    Q, R = householder_transformation(A)
    Q_T_b = Q.T @ b
    n = A.shape[1]
    b1 = Q_T_b[:n]

    x = solve_upper_triangular(R[:n, :], b1)
    return x


def givens_rotation(A):
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


def solve_least_squares_givens(A, b):
    Q, R = givens_rotation(A)
    Q_T_b = Q.T @ b
    n = A.shape[1]
    b1 = Q_T_b[:n]

    x = solve_upper_triangular(R[:n, :], b1)
    return x


def test_least_squares_methods():
    examples = [
        (np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float),
         np.array([1, -2, 7], dtype=float)),
        (np.array([[4, 2, 3], [3, -1, 2], [1, 2, 0]], dtype=float),
         np.array([10, 5, 7], dtype=float)),
        (np.array([[1, 2, 1], [2, 2, 3], [1, 3, 4]], dtype=float),
         np.array([4, 7, 8], dtype=float))
    ]

    for i, (A, b) in enumerate(examples):
        print(f"Beispiel {i + 1}:")

        try:
            x_normal = solve_least_squares_normal_equations(A, b)
            residuum_normal = np.linalg.norm(A @ x_normal - b)
            print(f"Lösung mit Normalengleichungen: {x_normal}")
            print(f"Residuum: {residuum_normal}")

            x_householder = solve_least_squares_householder(A, b)
            residuum_householder = np.linalg.norm(A @ x_householder - b)
            print(f"Lösung mit Householder-Orthogonalisierung: {x_householder}")
            print(f"Residuum: {residuum_householder}")

            x_givens = solve_least_squares_givens(A, b)
            residuum_givens = np.linalg.norm(A @ x_givens - b)
            print(f"Lösung mit Givens-Rotation: {x_givens}")
            print(f"Residuum: {residuum_givens}")

        except ValueError as e:
            print(e)

        print()


test_least_squares_methods()
