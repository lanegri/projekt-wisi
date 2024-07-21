import numpy as np

from app.Aufgabe3.Aufgabe_3_1.givens_rotation import solve_least_squares_givens


def test_least_squares_givens():
    A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    b = np.array([1, -2, 7], dtype=float)

    x = solve_least_squares_givens(A, b)
    print("\n Lösung mit Givens-Rotation:", x)
    assert str(x) == '[-0.3125  3.1875  1.5625]'
