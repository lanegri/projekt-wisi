import pytest
import numpy as np


@pytest.fixture()
def test_matrix():
    # return np.array([[4, 12, -16],
    #               [12, 37, -43],
    #               [-16, -43, 98]])

    return [
        (np.array([[2, 1, -1], [1, -2, 3], [3, 2, 1]], dtype=float),
         np.array([1, -2, 7], dtype=float)),
        (np.array([[4, 2, 3], [3, -1, 2], [1, 2, 0]], dtype=float),
         np.array([10, 5, 7], dtype=float)),
        (np.array([[1, 2, 1], [2, 2, 3], [1, 3, 4]], dtype=float),
         np.array([4, 7, 8], dtype=float))
    ]
