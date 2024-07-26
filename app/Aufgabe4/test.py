import numpy as np
import pytest

from app.Aufgabe2.Aufgabe_2_1.cholesky_decomposition import cholesky_decomposition
from app.Aufgabe2.Aufgabe_2_2.gauss_normal_equations import solve_with_gauss_normal_equations
from app.Aufgabe3.Aufgabe_3_1.transform_system_of_equations import q_r_with_householder_transformation, q_r_with_givens_rotation
from app.Aufgabe3.Aufgabe_3_2.equation_system import solve_least_squares_householder, solve_least_squares_givens
from app.utils.checkup_utils import check_is_symmetric, check_positiv_definit

np.set_printoptions(precision=5, suppress=True)

np.seterr(divide='ignore', invalid='ignore')


@pytest.mark.dependency(name="symmetric")
def test_check_is_symmetric():
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
    # Überprüfung der Symmetrie
    assert is_symmetric is True


@pytest.mark.dependency(name="positiv")
def test_check_is_positiv_definit():
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
    assert positiv_definit is True


@pytest.mark.dependency(name="cholesky", depends=["symmetric", "positiv"])
def test_cholesky_decomposition():
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

    equal = np.array_equal(A, B)
    if equal:
        print("Matrizen A und L * L.T sind identisch")
    else:
        print("Matrizen A und L * L.T sind nicht identisch")
    assert equal is True


@pytest.mark.dependency(name="gauss_normal", depends=["cholesky"])
def test_solve_with_gauss_normal_equations():
    """
        Test Gauß linearen Ausgleichung der Matrix A
    """
    A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    b = np.array([1, -2, 7], dtype=float)

    _, _, L, y, x = solve_with_gauss_normal_equations(A, b)
    print("\n Lösung des linearen Ausgleichungsproblems mit Gauß:")
    print("\n Lösung mit Gauß:", x)
    assert str(x) == '[-0.3125  3.1875  1.5625]'


@pytest.mark.dependency(name="householder", depends=[])
def test_householder_transformation():
    """
        Test Householder-Transformation der Matrix A
    """
    A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    out_Q, out_R = q_r_with_householder_transformation(A)
    print(f"\n Lösung Householder-Transformation: \n Q: \n {out_Q} \n R: \n {out_R}")


@pytest.mark.dependency(name="householder2", depends=[])
def test_householder_transformation_2():
    """
        Test Householder-Transformation der Matrix A
    """
    A = [[2, 4, -4],
         [1, 1, 2],
         [2, -3, 0]]
    matrix = np.array(A, dtype=int)
    out_Q, out_R = q_r_with_householder_transformation(matrix)
    print(f"\n Lösung Householder-Transformation 2: \n Q: \n {out_Q} \n R: \n {out_R}")


@pytest.mark.dependency(name="givens", depends=[])
def test_givens_rotation():
    """
        Test Givens-Rotation der Matrix A
    """
    B = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    out_Q, out_R = q_r_with_givens_rotation(B)
    print(f"\n Lösung Givens-Rotation: \n Q: \n {out_Q} \n R: \n {out_R}")


@pytest.mark.dependency(name="givens2", depends=[])
def test_givens_rotation2():
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


@pytest.mark.dependency(name="givens3", depends=[])
def test_givens_rotation2():
    """
        Test Givens-Rotation der Matrix A
    """
    B = [[2, -1, -2],
         [-4, 6, 3],
         [-4, -3, 8]]
    matrix = np.array(B, dtype=float)
    out_Q, out_R = q_r_with_givens_rotation(matrix)
    print(f"\n Lösung Givens-Rotation 3: \n Q: \n {out_Q} \n R: \n {out_R}")


@pytest.mark.dependency(name="equation_with_householder", depends=["householder"])
def test_equation_with_householder_transformation():
    """
        Test Lösen der Gleichungssystem  durch Householder-Transformation
    """
    A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    b = np.array([1, -2, 7], dtype=float)

    x_householder = solve_least_squares_householder(A, b)
    print("\n Lösung mit Householder-Orthogonalisierung:", x_householder)


@pytest.mark.dependency(name="equation_with_givens", depends=["givens"])
def test_equation_with_givens_rotation():
    """
        Test Test Lösen der Gleichungssystem  durch Givens-Rotation
    """
    A = np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float)
    b = np.array([1, -2, 7], dtype=float)

    x_givens = solve_least_squares_givens(A, b)
    print("\n Lösung mit Givens-Rotation:", x_givens)
