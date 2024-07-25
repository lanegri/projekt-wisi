import numpy as np

from app.Aufgabe2.Aufgabe_2_1.cholesky_decomposition import cholesky_decomposition
from app.Aufgabe2.Aufgabe_2_2.gauss_normal_equations import solve_with_gauss_normal_equations
from app.Aufgabe3.Aufgabe_3_1.transform_system_of_equations import q_r_with_householder_transformation, q_r_with_givens_rotation
from app.Aufgabe3.Aufgabe_3_2.equation_system import solve_least_squares_householder, solve_least_squares_givens
from app.utils.checkup_utils import check_is_symmetric, check_positiv_definit

np.set_printoptions(precision=1, suppress=True)
np.seterr(divide='ignore', invalid='ignore')


def out_check_is_symmetric():
    """
        Test ob die Matrix A symmetrisch ist
    """
    A = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]])
    print(f"\n Matrix A = \n {A}")
    is_symmetric = check_is_symmetric(A)
    if is_symmetric:
        print("\n Die Matrix ist symmetrisch.")
    else:
        print("\n Die Matrix ist nicht symmetrisch.")


def out_check_is_positiv_definit():
    """
        Test ob die Matrix A positiv definiert ist
    """
    A = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]])
    positiv_definit = check_positiv_definit(A)

    if positiv_definit:
        print("\n Die Matrix ist positiv definit.")
    else:
        print("\n Die Matrix A ist nicht positiv definit.")


def out_cholesky_decomposition():
    """
        Test Cholesky Zerlegung der Matrix A
    """
    A = np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]])
    L = cholesky_decomposition(A)
    B = np.dot(L, L.T)
    print(f"\n Cholesky-Zerlegung L: \n {np.array(L)} \n")
    print(f"Überprüfung: A = L * L.T: \n {B}")

    if np.array_equal(A, B):
        print("Matrizen A und L * L.T sind identisch")
    else:
        print("Matrizen A und L * L.T sind nicht identisch")


def out_solve_with_gauss_normal_equations():
    """
        Test Gauß linearen Ausgleichung der Matrix A
    """
    A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    b = np.array([1, -2, 7], dtype=float)

    L, y, x = solve_with_gauss_normal_equations(A, b)
    print("\n Lösung des linearen Ausgleichungsproblems mit Gauß:")
    print("\n Lösung mit Gauß:", x)


def out_householder_transformation():
    """
        Test Householder-Transformation der Matrix A
    """
    A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    out_Q, out_R = q_r_with_householder_transformation(A)
    print(f"\n Lösung Householder-Transformation: \n Q: \n {out_Q} \n R: \n {out_R}")


def out_householder_transformation_2():
    """
        Test Householder-Transformation der Matrix A
    """
    A = [[2, 4, -4],
         [1, 1, 2],
         [2, -3, 0]]
    matrix = np.array(A, dtype=int)
    out_Q, out_R = q_r_with_householder_transformation(matrix)
    print(f"\n Lösung Householder-Transformation 2: \n Q: \n {out_Q} \n R: \n {out_R}")


def out_givens_rotation():
    """
        Test Givens-Rotation der Matrix A
    """
    B = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    out_Q, out_R = q_r_with_givens_rotation(B)
    print(f"\n Lösung Givens-Rotation: \n Q: \n {out_Q} \n R: \n {out_R}")


def out_givens_rotation2():
    """
        Test Givens-Rotation der Matrix A
    """
    B = [[1, 0, 0, 0],
         [2, 2, 0, 0],
         [3, 3, 3, 0],
         [4, 4, 4, 4]]
    matrix = np.array(B, dtype=float)
    out_Q, out_R = q_r_with_givens_rotation(matrix)
    print(f"\n Lösung Givens-Rotation 2: \n Q: \n {out_Q} \n R: \n {out_R}")


def out_equation_with_householder_transformation():
    """
        Test Lösen der Gleichungssystem  durch Householder-Transformation
    """
    A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    b = np.array([1, -2, 7], dtype=float)

    x_householder = solve_least_squares_householder(A, b)
    print("\n Lösung mit Householder-Orthogonalisierung:", x_householder)


def out_equation_with_givens_rotation():
    """
        Test Test Lösen der Gleichungssystem  durch Givens-Rotation
    """
    A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    b = np.array([1, -2, 7], dtype=float)

    x_givens = solve_least_squares_givens(A, b)
    print("\n Lösung mit Givens-Rotation:", x_givens)


######################

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
            L, y, x_gauss_normal = solve_with_gauss_normal_equations(A, b)
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
