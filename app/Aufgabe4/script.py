import numpy as np

from app.Aufgabe2.Aufgabe_2_2.gauss_normal_equations import solve_with_gauss_normal_equations
from app.Aufgabe3.Aufgabe_3_2.equation_system import solve_least_squares_householder, solve_least_squares_givens

# np.set_printoptions(precision=1, suppress=True)
# np.seterr(divide='ignore', invalid='ignore')


def out_least_squares_methods():
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
            A_T_A, A_T_b, L, y, x_gauss_normal = solve_with_gauss_normal_equations(A, b)
            residuum_normal = np.linalg.norm(A @ x_gauss_normal - b)
            print(f"Lösung mit Normalengleichungen: {x_gauss_normal}")
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


out_least_squares_methods()
