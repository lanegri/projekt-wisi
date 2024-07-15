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

# Beispiel für die Verwendung
A = np.array([[4, 12, -16],
              [12, 37, -43],
              [-16, -43, 98]])

# Überprüfung der Symmetrie
if ist_symmetrisch(A):
    print("Die Matrix ist symmetrisch.")
else:
    print("Die Matrix ist nicht symmetrisch.")

# Überprüfung der positiven Definitheit und Cholesky-Zerlegung
if ist_positiv_definit(A):
    print("Die Matrix ist positiv definit.")
    L = cholesky_zerlegung(A)
    print("Cholesky-Zerlegung L:")
    print(np.array(L))
    print("Überprüfung: A = L * L.T")
    print(np.dot(L, L.T))
else:
    print("Die Matrix ist nicht positiv definit.")
