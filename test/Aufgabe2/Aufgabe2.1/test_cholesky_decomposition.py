import numpy as np

from app.Aufgabe2.Aufgabe_2_1.cholesky_decomposition import cholesky_decomposition


def test_cholesky_decomposition(test_matrix):
    # Beispiel für die Verwendung
    # A = np.array([[4, 12, -16],
    #               [12, 37, -43],
    #               [-16, -43, 98]])

    L = cholesky_decomposition(test_matrix)
    B = np.dot(L, L.T)
    print("\n Cholesky-Zerlegung L:")
    print(np.array(L))
    print("Überprüfung: A = L * L.T")
    print(B)

    equal = np.array_equal(test_matrix, B)
    if equal:
        print("Matrices are identical")
    else:
        print("Matrices are not identical")
    assert equal is True
