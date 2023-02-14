
import pytest
import numpy

import modin.numpy as np

from .utils import (
    generate_mats,
    generate_mat_with_nan,
    generate_vecs,
    arr_equals,
)


@pytest.mark.parametrize("f_name", ["any", "all"])
@pytest.mark.parametrize("axis", [None, 0], ids=["axis=None", "axis=0"])
def test_unary_vec(f_name, axis):
    x1 = generate_vecs()[0]
    result_numpy = getattr(numpy, f_name)(x1, axis=axis)
    x1 = np.array(x1)
    result = getattr(np, f_name)(x1, axis=axis)
    arr_equals(result, result_numpy)


@pytest.mark.parametrize("f_name", ["any", "all"])
@pytest.mark.parametrize("axis", [None, 0, 1], ids=["axis=None", "axis=0", "axis=1"])
def test_unary_mat(f_name, axis):
    x1 = generate_mats()[0]
    result_numpy = getattr(numpy, f_name)(x1, axis=axis)
    x1 = np.array(x1)
    result = getattr(np, f_name)(x1, axis=axis)
    arr_equals(result, result_numpy)


@pytest.mark.parametrize("data", [generate_mat_with_nan(), generate_vecs()[0]], ids=["mat", "vec"])
def test_isnan(data):
    x1 = data
    result_numpy = numpy.isnan(x1)
    x1 = np.array(x1)
    result = np.isnan(x1)
    arr_equals(result, result_numpy)


@pytest.mark.parametrize("data", [generate_mats()[0], generate_vecs()[0]], ids=["mat", "vec"])
def test_logical_not(data):
    x1 = data
    result_numpy = numpy.logical_not(x1)
    x1 = np.array(x1)
    result = np.logical_not(x1)
    arr_equals(result, result_numpy)


@pytest.mark.parametrize("data", [generate_mats(), generate_vecs()], ids=["mat", "vec"])
@pytest.mark.parametrize("f_name", ["logical_and", "logical_or", "logical_xor"])
def test_logical_binops(data, f_name):
    x1, x2 = data
    result_numpy = getattr(numpy, f_name)(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    result = getattr(np, f_name)(x1, x2)
    arr_equals(result, result_numpy)


def test_array_equal_vec():
    x1 = numpy.array([1, 2, 3])
    x2 = numpy.array([1, 2, 6])
    result_numpy = numpy.array_equal(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    result = np.array_equal(x1, x2)
    arr_equals(result, result_numpy)
    x1 = numpy.array([1, 2, 3])
    x2 = numpy.array([1, 2, 3])
    result_numpy = numpy.array_equal(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    result = np.array_equal(x1, x2)
    arr_equals(result, result_numpy)
    x1 = numpy.array([1, 2, 3])
    x2 = numpy.array([1, 2])
    result_numpy = numpy.array_equal(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    result = np.array_equal(x1, x2)
    arr_equals(result, result_numpy)


def test_array_equal_mat():
    x1 = numpy.array([[1, 2, 6], [1, 2, 3]])
    x2 = numpy.array([[1, 2, 6], [1, 1, 1]])
    result_numpy = numpy.array_equal(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    result = np.array_equal(x1, x2)
    arr_equals(result, result_numpy)
    x1 = numpy.array([[1, 2, 6], [1, 2, 3]])
    x2 = numpy.array([[1, 2, 6], [1, 1, 3]])
    result_numpy = numpy.array_equal(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    result = np.array_equal(x1, x2)
    arr_equals(result, result_numpy)
    x1 = numpy.array([[1, 2, 6], [1, 2, 3]])
    x2 = numpy.array([[1, 2], [1, 1]])
    result_numpy = numpy.array_equal(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    result = np.array_equal(x1, x2)
    arr_equals(result, result_numpy)
