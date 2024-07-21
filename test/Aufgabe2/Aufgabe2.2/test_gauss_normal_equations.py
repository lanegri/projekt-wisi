import numpy as np

from app.Aufgabe2.Aufgabe_2_2.gauss_normal_equations import solve_least_squares_normal_equations


def test_solve_least_squares_normal_equations():
    # Beispiel
    A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    b = np.array([1, -2, 7], dtype=float)

    x = solve_least_squares_normal_equations(A, b)
    print("\n ‚Lösung des linearen Ausgleichsproblems:", np.array(x))
    assert str(x) == '[-0.3125  3.1875  1.5625]'
