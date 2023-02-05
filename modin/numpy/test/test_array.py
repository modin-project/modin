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
import warnings
import modin.numpy as np


@pytest.mark.parametrize("size", [100, (2, 100), (100, 2), (1, 100), (100, 1)])
def test_repr(size):
    numpy_arr = numpy.random.randint(-100, 100, size=size)
    modin_arr = np.array(numpy_arr)
    assert repr(modin_arr) == repr(numpy_arr)


@pytest.mark.parametrize("size", [100, (2, 100), (100, 2), (1, 100), (100, 1)])
def test_shape(size):
    numpy_arr = numpy.random.randint(-100, 100, size=size)
    modin_arr = np.array(numpy_arr)
    assert modin_arr.shape == numpy_arr.shape


@pytest.mark.parametrize("operand1shape", [100, (3, 100)])
@pytest.mark.parametrize("operand2shape", [100, (3, 100)])
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
    ],
)
def test_basic_arithmetic_with_broadcast(operand1shape, operand2shape, operator):
    """Test of operators that support broadcasting."""
    operand1 = numpy.random.randint(-100, 100, size=operand1shape)
    operand2 = numpy.random.randint(-100, 100, size=operand2shape)
    modin_result = getattr(np.array(operand1), operator)(np.array(operand2))
    numpy_result = getattr(operand1, operator)(operand2)
    if operator not in ["__truediv__", "__rtruediv__"]:
        numpy.testing.assert_array_equal(
            modin_result._to_numpy(),
            numpy_result,
            err_msg=f"Binary Op {operator} failed.",
        )
    else:
        # Truediv can have precision issues.
        numpy.testing.assert_array_almost_equal(
            modin_result._to_numpy(),
            numpy_result,
            decimal=12,
            err_msg="Binary Op __truediv__ failed.",
        )


@pytest.mark.parametrize("operator", ["__pow__", "__floordiv__", "__mod__"])
def test_arithmetic(operator):
    """Test of operators that do not yet support broadcasting"""
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

@pytest.mark.parametrize("size", [100, (2, 100), (100, 2), (1, 100), (100, 1)])
def test_array_ufunc(size):
    # Test ufunc.__call__
    numpy_arr = numpy.random.randint(-100, 100, size=size)
    modin_arr = np.array(numpy_arr)
    modin_result = numpy.sign(modin_arr)._to_numpy()
    numpy_result = numpy.sign(numpy_arr)
    numpy.testing.assert_array_equal(modin_result, numpy_result)
    # Test ufunc that we have support for.
    modin_result = numpy.add(modin_arr, modin_arr)._to_numpy()
    numpy_result = numpy.add(numpy_arr, numpy_arr)
    numpy.testing.assert_array_equal(modin_result, numpy_result)
    # Test ufunc that we have support for, but method that we do not implement.
    modin_result = numpy.add.reduce(modin_arr)
    numpy_result = numpy.add.reduce(numpy_arr)
    assert numpy_result == modin_result
    # We do not test ufunc.reduce and ufunc.accumulate, since these require a binary reduce
    # operation that Modin does not currently support.

@pytest.mark.parametrize("size", [100, (2, 100), (100, 2), (1, 100), (100, 1)])
def test_array_function(size):
    numpy_arr = numpy.random.randint(-100, 100, size=size)
    modin_arr = np.array(numpy_arr)
    # Test from array shaping
    modin_result = numpy.ravel(modin_arr)._to_numpy()
    numpy_result = numpy.ravel(numpy_arr)
    numpy.testing.assert_array_equal(modin_result, numpy_result)
    # Test from array creation
    modin_result = numpy.zeros_like(modin_arr)._to_numpy()
    numpy_result = numpy.zeros_like(numpy_arr)
    numpy.testing.assert_array_equal(modin_result, numpy_result)
    # Test from math
    modin_result = numpy.sum(modin_arr)
    numpy_result = numpy.sum(numpy_arr)
    assert numpy_result == modin_result

# def test_array_where():
#     numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
#     modin_flat_arr = np.array(numpy_flat_arr)
