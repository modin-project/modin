# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import numpy
import pytest

import modin.numpy as np


@pytest.mark.parametrize("operand1_shape", [100, (3, 100)])
@pytest.mark.parametrize("operand2_shape", [100, (3, 100)])
@pytest.mark.parametrize(
    "operator",
    [
        "__add__",
        "__sub__",
        "__truediv__",
        "__mul__",
        "__rtruediv__",
        "__rmul__",
        "__radd__",
        "__rsub__",
        "__ge__",
        "__gt__",
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
    ],
)
def test_basic_arithmetic_with_broadcast(operand1_shape, operand2_shape, operator):
    """Test of operators that support broadcasting."""
    operand1 = numpy.random.randint(-100, 100, size=operand1_shape)
    operand2 = numpy.random.randint(-100, 100, size=operand2_shape)
    modin_result = getattr(np.array(operand1), operator)(np.array(operand2))
    numpy_result = getattr(operand1, operator)(operand2)
    if operator not in ["__truediv__", "__rtruediv__"]:
        numpy.testing.assert_array_equal(
            modin_result._to_numpy(),
            numpy_result,
            err_msg=f"Binary Op {operator} failed.",
        )
    else:
        # Truediv can have precision issues, where thanks to floating point error, the numbers
        # aren't exactly the same across both, but are functionally equivalent, since the difference
        # is less than 1e-12.
        numpy.testing.assert_array_almost_equal(
            modin_result._to_numpy(),
            numpy_result,
            decimal=12,
            err_msg="Binary Op __truediv__ failed.",
        )


@pytest.mark.parametrize("operator", ["__pow__", "__floordiv__", "__mod__"])
def test_arithmetic(operator):
    """Test of operators that do not yet support broadcasting."""
    for size, textdim in ((100, "1D"), ((10, 10), "2D")):
        operand1 = numpy.random.randint(-100, 100, size=size)
        lower_bound = -100 if operator != "__pow__" else 0
        operand2 = numpy.random.randint(lower_bound, 100, size=size)
        modin_result = getattr(np.array(operand1), operator)(np.array(operand2))
        numpy_result = getattr(operand1, operator)(operand2)
        numpy.testing.assert_array_almost_equal(
            modin_result._to_numpy(),
            numpy_result,
            decimal=12,
            err_msg=f"Binary Op {operator} failed on {textdim} arrays.",
        )


def test_arithmetic_nans_and_zeros():
    numpy_arr1 = numpy.array([[1, 0, 3], [numpy.nan, 0, numpy.nan]])
    numpy_arr2 = numpy.array([1, 0, 0])
    numpy.testing.assert_array_equal(
        numpy_arr1 // numpy_arr2,
        (np.array(numpy_arr1) // np.array(numpy_arr2))._to_numpy(),
    )
    numpy.testing.assert_array_equal(
        numpy.array([0]) // 0, (np.array([0]) // 0)._to_numpy()
    )
    numpy.testing.assert_array_equal(
        numpy.array([0], dtype=numpy.float64) // 0,
        (np.array([0], dtype=numpy.float64) // 0)._to_numpy(),
    )


@pytest.mark.parametrize("size", [100, (2, 100), (100, 2), (1, 100), (100, 1)])
def test_scalar_arithmetic(size):
    numpy_arr = numpy.random.randint(-100, 100, size=size)
    modin_arr = np.array(numpy_arr)
    scalar = numpy.random.randint(1, 100)
    numpy.testing.assert_array_equal(
        (scalar * modin_arr)._to_numpy(), scalar * numpy_arr, err_msg="__mul__ failed."
    )
    numpy.testing.assert_array_equal(
        (modin_arr * scalar)._to_numpy(),
        scalar * numpy_arr,
        err_msg="__rmul__ failed.",
    )
    numpy.testing.assert_array_equal(
        (scalar / modin_arr)._to_numpy(),
        scalar / numpy_arr,
        err_msg="__rtruediv__ failed.",
    )
    numpy.testing.assert_array_equal(
        (modin_arr / scalar)._to_numpy(),
        numpy_arr / scalar,
        err_msg="__truediv__ failed.",
    )
    numpy.testing.assert_array_equal(
        (scalar + modin_arr)._to_numpy(),
        scalar + numpy_arr,
        err_msg="__radd__ failed.",
    )
    numpy.testing.assert_array_equal(
        (modin_arr + scalar)._to_numpy(), scalar + numpy_arr, err_msg="__add__ failed."
    )
    numpy.testing.assert_array_equal(
        (scalar - modin_arr)._to_numpy(),
        scalar - numpy_arr,
        err_msg="__rsub__ failed.",
    )
    numpy.testing.assert_array_equal(
        (modin_arr - scalar)._to_numpy(), numpy_arr - scalar, err_msg="__sub__ failed."
    )


@pytest.mark.parametrize("op_name", ["abs", "exp", "sqrt", "tanh"])
def test_unary_arithmetic(op_name):
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    numpy.testing.assert_array_equal(
        getattr(numpy, op_name)(numpy_flat_arr),
        getattr(np, op_name)(modin_flat_arr)._to_numpy(),
    )
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    numpy.testing.assert_array_equal(
        getattr(numpy, op_name)(numpy_arr), getattr(np, op_name)(modin_arr)._to_numpy()
    )


def test_invert():
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    numpy.testing.assert_array_equal(~numpy_flat_arr, (~modin_flat_arr)._to_numpy())
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    numpy.testing.assert_array_equal(~numpy_arr, (~modin_arr)._to_numpy())
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100) < 0
    modin_flat_arr = np.array(numpy_flat_arr)
    numpy.testing.assert_array_equal(~numpy_flat_arr, (~modin_flat_arr)._to_numpy())
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    numpy.testing.assert_array_equal(~numpy_arr, (~modin_arr)._to_numpy())
