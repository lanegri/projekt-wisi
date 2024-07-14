import numpy as np
import math

def is_symmetric(A, tol=1e-8):
    """Überprüft, ob die Matrix A symmetrisch ist."""
    return np.allclose(A, A.T, atol=tol)

def is_positive_definite(A):
    """Überprüft, ob die Matrix A positiv definit ist."""
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

def choleski(A):
    """Berechnet die Cholesky-Zerlegung der Matrix A, wenn sie symmetrisch und positiv definit ist."""
    if not is_symmetric(A):
        raise ValueError("Die Matrix ist nicht symmetrisch.")
    
    if not is_positive_definite(A):
        raise ValueError("Die Matrix ist nicht positiv definit.")
    
    n = len(A)
    L = np.zeros_like(A)

    for k in range(n):
        sum_k = sum(L[k][j] * L[k][j] for j in range(k))
        L[k][k] = math.sqrt(A[k][k] - sum_k)
        for i in range(k + 1, n):
            sum_ik = sum(L[i][j] * L[k][j] for j in range(k))
            L[i][k] = (A[i][k] - sum_ik) / L[k][k]
    
    return L

def main():
    # Beispielmatrix
    A = np.array([[4.0, -2.0, 2.0],
                  [-2.0, 2.0, 4.0],
                  [2.0, -4.0, 11.0]])

    print("Eingabematrix A:")
    print(A)

    if is_symmetric(A):
        print("Die Matrix ist symmetrisch.")
    else:
        print("Die Matrix ist nicht symmetrisch.")

    if is_positive_definite(A):
        print("Die Matrix ist positiv definit.")
    else:
        print("Die Matrix ist nicht positiv definit.")
    
    try:
        L = choleski(A)
        print("Cholesky-Zerlegung L:")
        print(L)
        print("Überprüfung: L * L.T = A")
        print(np.dot(L, L.T))
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
