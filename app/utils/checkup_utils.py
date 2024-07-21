import numpy as np


def check_is_symmetric(in_matrix) -> bool:
    """
    Überprüft, ob die Matrix A symmetrisch ist.
    NumPy: return np.allclose(A, A.T)
    """
    n = len(in_matrix)
    for i in range(n):
        for j in range(n):
            if in_matrix[i][j] != in_matrix[j][i]:
                return False
    return True


def check_definiteness(in_matrix) -> str:
    # Berechne die Eigenwerte der Matrix
    eigenvalues = np.linalg.eigvals(in_matrix)

    # Überprüfe die Definitheit anhand der Eigenwerte
    if np.all(eigenvalues > 0):
        return "positiv definit"
    elif np.all(eigenvalues < 0):
        return "negativ definit"
    elif np.all(eigenvalues >= 0):
        return "positiv semidefinit"
    elif np.all(eigenvalues <= 0):
        return "negativ semidefinit"
    else:
        return "indefinit"


def check_positiv_definit(in_matrix) -> bool:
    # Berechne die Eigenwerte der Matrix
    eigenvalues = np.linalg.eigvals(in_matrix)

    # Überprüfe die Positiv Definitheit anhand der Eigenwerte
    return True if np.all(eigenvalues > 0) else False

