import pytest
import numpy as np


@pytest.fixture()
def test_matrix():
    return np.array([[4, 12, -16],
                  [12, 37, -43],
                  [-16, -43, 98]])
