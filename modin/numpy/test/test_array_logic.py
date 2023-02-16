
import pytest
import numpy

import modin.numpy as np


small_arr_c_2d = numpy.array([
    [1j, 1, 0, -numpy.inf, numpy.inf, 0.5],
    [1 + 1.1j, numpy.nan, 0, numpy.nan, 2, 0.3]
])
small_arr_c_1d = numpy.array([numpy.nan, 0, -numpy.inf, numpy.inf, 5, -0.1, 1 + 1.1j])

small_arr_r_2d = numpy.array([
    [1, 0, -numpy.inf, numpy.inf, 0.5],
    [numpy.nan, 0, numpy.nan, 2, 0.3]
])
small_arr_r_1d = numpy.array([numpy.nan, 0, -numpy.inf, numpy.inf, 5, -0.1])


@pytest.mark.parametrize("operand_shape", [100, (3, 100)])
@pytest.mark.parametrize("operator", ["any", "all"])
@pytest.mark.parametrize("axis", [None, 0, 1], ids=["axis=None", "axis=0", "axis=1"])
def test_unary_with_axis(operand_shape, operator, axis):
    if isinstance(operand_shape, int) and axis == 1:
        pytest.skip("cannot use axis=1 on 1D arrays")
    x1 = numpy.random.randint(-100, 100, size=operand_shape)
    numpy_result = getattr(numpy, operator)(x1, axis=axis)
    x1 = np.array(x1)
    modin_result = getattr(np, operator)(x1, axis=axis)
    # Result may be a scalar
    if isinstance(modin_result, np.array):
        modin_result = modin_result._to_numpy()
    numpy.testing.assert_array_equal(modin_result, numpy_result, err_msg=f"Unary operator {operator} failed.")


@pytest.mark.parametrize("data", [small_arr_c_2d, small_arr_c_1d], ids=["2D", "1D"])
@pytest.mark.parametrize(
    "operator", ["isfinite", "isinf", "isnan", "iscomplex", "isreal"]
)
def test_unary_with_complex(data, operator):
    x1 = data
    numpy_result = getattr(numpy, operator)(x1)
    x1 = np.array(x1)
    modin_result = getattr(np, operator)(x1)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)


def test_isnat():
    x1 = numpy.array([numpy.datetime64("2016-01-01"), numpy.datetime64("NaT")])
    numpy_result = numpy.isnat(x1)
    x1 = np.array(x1)
    modin_result = np.isnat(x1)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)


@pytest.mark.parametrize("data", [small_arr_r_2d, small_arr_r_1d], ids=["2D", "1D"])
@pytest.mark.parametrize("operator", ["isneginf", "isposinf"])
def test_unary_without_complex(data, operator):
    x1 = data
    numpy_result = getattr(numpy, operator)(x1)
    x1 = np.array(x1)
    modin_result = getattr(np, operator)(x1)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)


@pytest.mark.parametrize("data", [small_arr_r_2d, small_arr_r_1d], ids=["2D", "1D"])
def test_logical_not(data):
    x1 = data
    numpy_result = numpy.logical_not(x1)
    x1 = np.array(x1)
    modin_result = np.logical_not(x1)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)


@pytest.mark.parametrize("operand1_shape", [100, (3, 100)])
@pytest.mark.parametrize("operand2_shape", [100, (3, 100)])
@pytest.mark.parametrize("operator", ["logical_and", "logical_or", "logical_xor"])
def test_logical_binops(operand1_shape, operand2_shape, operator):
    if operand1_shape != operand2_shape:
        pytest.xfail("TODO fix broadcasting behavior for binary logic operators")
    x1 = numpy.random.randint(-100, 100, size=operand1_shape)
    x2 = numpy.random.randint(-100, 100, size=operand2_shape)
    numpy_result = getattr(numpy, operator)(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    modin_result = getattr(np, operator)(x1, x2)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result, err_msg=f"Logic binary operator {operator} failed.")
