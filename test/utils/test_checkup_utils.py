import numpy as np
from app.utils.checkup_utils import check_definiteness, check_is_symmetric


def test_check_is_symmetrisch():
    matrix = np.array([[4, 12, -16],
                       [12, 37, -43],
                       [-16, -43, 98]])

    # Überprüfung der Symmetrie
    assert check_is_symmetric(matrix) is True


def test_check_definiteness():
    matrix = np.array([[2, -1],
                       [-1, 2]])
    definiteness = check_definiteness(matrix)
    assert definiteness == 'positiv definit'


def test_check_definiteness_2():
    matrix = np.array([[4, 12, -16],
                       [12, 37, -43],
                       [-16, -43, 98]])
    definiteness = check_definiteness(matrix)
    assert definiteness == 'positiv definit'
