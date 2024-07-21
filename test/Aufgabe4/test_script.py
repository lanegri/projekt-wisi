import numpy as np

from app.Aufgabe2.Aufgabe_2_1.cholesky_decomposition import cholesky_decomposition
from app.Aufgabe2.Aufgabe_2_2.gauss_normal_equations import solve_least_squares_normal_equations
from app.Aufgabe3.Aufgabe_3_1.givens_rotation import solve_least_squares_givens
from app.utils.checkup_utils import check_is_symmetric, check_positiv_definit

A = np.array([[4, 12, -16],
              [12, 37, -43],
              [-16, -43, 98]])

"""
    Test ob die Matrix A symmetrisch ist
"""


def test_check_is_symmetrisch():
    is_symmetric = check_is_symmetric(A)
    if is_symmetric:
        print("\n Die Matrix ist symmetrisch.")
    else:
        print("\n Die Matrix ist nicht symmetrisch.")
    # Überprüfung der Symmetrie
    assert check_is_symmetric(A) is True


"""
    Test ob die Matrix A positiv definiert ist
"""


def test_check_is_positiv_definit():
    positiv_definit = check_positiv_definit(A)

    if positiv_definit:
        print("\n Die Matrix ist positiv definit.")
    else:
        print("\n Die Matrix A ist nicht positiv definit.")
    assert positiv_definit is True


"""
    Test Cholesky Zerlegung der Matrix A
"""


def test_cholesky_decomposition():
    L = cholesky_decomposition(A)
    B = np.dot(L, L.T)
    print("\n Cholesky-Zerlegung L:")
    print(np.array(L))
    print("Überprüfung: A = L * L.T")
    print(B)

    equal = np.array_equal(A, B)
    if equal:
        print("Matrices are identical")
    else:
        print("Matrices are not identical")
    assert equal is True


"""
    Test Gauß linearen Ausgleichung der Matrix A
"""

print("\n Lösung des linearen Ausgleichungsproblems:")


def test_solve_least_squares_normal_equations():
    # Beispiel
    _A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    b = np.array([1, -2, 7], dtype=float)

    x = solve_least_squares_normal_equations(_A, b)
    print("\n Lösung mit Gauß:", np.array(x))
    assert str(x) == '[-0.3125  3.1875  1.5625]'


def test_least_squares_givens():
    _A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    b = np.array([1, -2, 7], dtype=float)

    x = solve_least_squares_givens(_A, b)
    print("\n Lösung mit Givens-Rotation:", x)
    assert str(x) == '[-0.3125  3.1875  1.5625]'
