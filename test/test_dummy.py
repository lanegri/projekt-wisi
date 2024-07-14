from app.dummy import *


def test_greater():
    number = 1001
    assert func_number_greater_100(number)


def test_greater_equal():
    number = 100
    assert func_number_greater_equal_100(number)


def test_less():
    number = 100
    assert func_number_less_200(number)
