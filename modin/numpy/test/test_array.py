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


def test_dtype():
    numpy_arr = numpy.array([[1, "2"], [3, "4"]])
    modin_arr = np.array([[1, "2"], [3, "4"]])
    assert modin_arr.dtype == numpy_arr.dtype
    modin_arr = modin_arr == modin_arr.T
    numpy_arr = numpy_arr == numpy_arr.T
    assert modin_arr.dtype == numpy_arr.dtype


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
        "__ge__",
        "__gt__",
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
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


def test_array_where():
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    with pytest.warns(
        UserWarning, match="np.where method with only condition specified"
    ):
        warnings.filterwarnings("ignore", message="Distributing")
        modin_flat_arr.where()
    with pytest.raises(ValueError, match="np.where requires x and y"):
        modin_flat_arr.where(x=["Should Fail."])
    with pytest.warns(UserWarning, match="np.where not supported when both x and y"):
        warnings.filterwarnings("ignore", message="Distributing")
        modin_result = modin_flat_arr.where(x=4, y=5)
    numpy_result = numpy.where(numpy_flat_arr, 4, 5)
    numpy.testing.assert_array_equal(numpy_result, modin_result._to_numpy())
    modin_flat_bool_arr = modin_flat_arr <= 0
    numpy_flat_bool_arr = numpy_flat_arr <= 0
    modin_result = modin_flat_bool_arr.where(x=5, y=modin_flat_arr)
    numpy_result = numpy.where(numpy_flat_bool_arr, 5, numpy_flat_arr)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_flat_bool_arr.where(x=modin_flat_arr, y=5)
    numpy_result = numpy.where(numpy_flat_bool_arr, numpy_flat_arr, 5)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_flat_bool_arr.where(x=modin_flat_arr, y=(-1 * modin_flat_arr))
    numpy_result = numpy.where(
        numpy_flat_bool_arr, numpy_flat_arr, (-1 * numpy_flat_arr)
    )
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    modin_bool_arr = modin_arr > 0
    numpy_bool_arr = numpy_arr > 0
    modin_result = modin_bool_arr.where(modin_arr, 10 * modin_arr)
    numpy_result = numpy.where(numpy_bool_arr, numpy_arr, 10 * numpy_arr)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)


def test_max():
    # Test 1D
    numpy_arr = numpy.random.randint(-100, 100, size=100)
    modin_arr = np.array(numpy_arr)
    assert modin_arr.max() == numpy_arr.max()
    modin_result = modin_arr.max(axis=0)
    numpy_result = modin_arr.max(axis=0)
    assert modin_result == numpy_result
    modin_result = modin_arr.max(initial=200)
    numpy_result = numpy_arr.max(initial=200)
    assert modin_result == numpy_result
    modin_result = modin_arr.max(initial=0, where=False)
    numpy_result = numpy_arr.max(initial=0, where=False)
    assert modin_result == numpy_result
    modin_result = modin_arr.max(keepdims=True)
    numpy_result = numpy_arr.max(keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    # Test 2D
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    assert modin_arr.max() == numpy_arr.max()
    modin_result = modin_arr.max(axis=0)
    numpy_result = numpy_arr.max(axis=0)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.max(axis=0, keepdims=True)
    numpy_result = numpy_arr.max(axis=0, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.max(axis=1)
    numpy_result = numpy_arr.max(axis=1)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.max(axis=1, keepdims=True)
    numpy_result = numpy_arr.max(axis=1, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.max(initial=200)
    numpy_result = numpy_arr.max(initial=200)
    assert modin_result == numpy_result
    modin_result = modin_arr.max(initial=0, where=False)
    numpy_result = numpy_arr.max(initial=0, where=False)
    assert modin_result == numpy_result
    with pytest.raises(ValueError):
        modin_result = modin_arr.max(out=modin_arr, keepdims=True)
    modin_out = np.array([[1]])
    numpy_out = modin_out._to_numpy()
    modin_result = modin_arr.max(out=modin_out, keepdims=True)
    numpy_result = numpy_arr.max(out=numpy_out, keepdims=True)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    modin_result = modin_arr.max(axis=0, where=False, initial=4)
    numpy_result = numpy_arr.max(axis=0, where=False, initial=4)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.max(axis=0, where=False, initial=4, out=modin_out)
    numpy_result = numpy_arr.max(axis=0, where=False, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.max(axis=0, initial=4, out=modin_out)
    numpy_result = numpy_arr.max(axis=0, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.max(axis=1, initial=4, out=modin_out)
    numpy_result = numpy_arr.max(axis=1, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    numpy_where = numpy.full(20, False)
    numpy_where[:10] = True
    numpy.random.shuffle(numpy_where)
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.max(axis=0, initial=4, out=modin_out, where=modin_where)
    numpy_result = numpy_arr.max(axis=0, initial=4, out=numpy_out, where=numpy_where)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)


def test_min():
    # Test 1D
    numpy_arr = numpy.random.randint(-100, 100, size=100)
    modin_arr = np.array(numpy_arr)
    assert modin_arr.min() == numpy_arr.min()
    modin_result = modin_arr.min(axis=0)
    numpy_result = modin_arr.min(axis=0)
    assert modin_result == numpy_result
    modin_result = modin_arr.min(initial=-200)
    numpy_result = numpy_arr.min(initial=-200)
    assert modin_result == numpy_result
    modin_result = modin_arr.min(initial=0, where=False)
    numpy_result = numpy_arr.min(initial=0, where=False)
    assert modin_result == numpy_result
    modin_result = modin_arr.min(keepdims=True)
    numpy_result = numpy_arr.min(keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    # Test 2D
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    assert modin_arr.min() == numpy_arr.min()
    modin_result = modin_arr.min(axis=0)
    numpy_result = numpy_arr.min(axis=0)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.min(axis=0, keepdims=True)
    numpy_result = numpy_arr.min(axis=0, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.min(axis=1)
    numpy_result = numpy_arr.min(axis=1)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.min(axis=1, keepdims=True)
    numpy_result = numpy_arr.min(axis=1, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.min(initial=-200)
    numpy_result = numpy_arr.min(initial=-200)
    assert modin_result == numpy_result
    modin_result = modin_arr.min(initial=0, where=False)
    numpy_result = numpy_arr.min(initial=0, where=False)
    assert modin_result == numpy_result
    with pytest.raises(ValueError):
        modin_result = modin_arr.min(out=modin_arr, keepdims=True)
    modin_out = np.array([[1]])
    numpy_out = modin_out._to_numpy()
    modin_result = modin_arr.min(out=modin_out, keepdims=True)
    numpy_result = numpy_arr.min(out=numpy_out, keepdims=True)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    modin_result = modin_arr.min(axis=0, where=False, initial=4)
    numpy_result = numpy_arr.min(axis=0, where=False, initial=4)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.min(axis=0, where=False, initial=4, out=modin_out)
    numpy_result = numpy_arr.min(axis=0, where=False, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.min(axis=0, initial=4, out=modin_out)
    numpy_result = numpy_arr.min(axis=0, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.min(axis=1, initial=4, out=modin_out)
    numpy_result = numpy_arr.min(axis=1, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    numpy_where = numpy.full(20, False)
    numpy_where[:10] = True
    numpy.random.shuffle(numpy_where)
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.min(axis=0, initial=4, out=modin_out, where=modin_where)
    numpy_result = numpy_arr.min(axis=0, initial=4, out=numpy_out, where=numpy_where)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)


def test_sum():
    # Test 1D
    numpy_arr = numpy.random.randint(-100, 100, size=100)
    modin_arr = np.array(numpy_arr)
    assert modin_arr.sum() == numpy_arr.sum()
    modin_result = modin_arr.sum(axis=0)
    numpy_result = modin_arr.sum(axis=0)
    assert modin_result == numpy_result
    modin_result = modin_arr.sum(initial=-200)
    numpy_result = numpy_arr.sum(initial=-200)
    assert modin_result == numpy_result
    modin_result = modin_arr.sum(initial=0, where=False)
    numpy_result = numpy_arr.sum(initial=0, where=False)
    assert modin_result == numpy_result
    modin_result = modin_arr.sum(keepdims=True)
    numpy_result = numpy_arr.sum(keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    # Test 2D
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    assert modin_arr.sum() == numpy_arr.sum()
    modin_result = modin_arr.sum(axis=0)
    numpy_result = numpy_arr.sum(axis=0)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.sum(axis=0, keepdims=True)
    numpy_result = numpy_arr.sum(axis=0, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.sum(axis=1)
    numpy_result = numpy_arr.sum(axis=1)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.sum(axis=1, keepdims=True)
    numpy_result = numpy_arr.sum(axis=1, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.sum(initial=-200)
    numpy_result = numpy_arr.sum(initial=-200)
    assert modin_result == numpy_result
    modin_result = modin_arr.sum(initial=0, where=False)
    numpy_result = numpy_arr.sum(initial=0, where=False)
    assert modin_result == numpy_result
    with pytest.raises(ValueError):
        modin_result = modin_arr.sum(out=modin_arr, keepdims=True)
    modin_out = np.array([[1]])
    numpy_out = modin_out._to_numpy()
    modin_result = modin_arr.sum(out=modin_out, keepdims=True)
    numpy_result = numpy_arr.sum(out=numpy_out, keepdims=True)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    modin_result = modin_arr.sum(axis=0, where=False, initial=4)
    numpy_result = numpy_arr.sum(axis=0, where=False, initial=4)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.sum(axis=0, where=False, initial=4, out=modin_out)
    numpy_result = numpy_arr.sum(axis=0, where=False, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.sum(axis=0, initial=4, out=modin_out)
    numpy_result = numpy_arr.sum(axis=0, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.sum(axis=1, initial=4, out=modin_out)
    numpy_result = numpy_arr.sum(axis=1, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    numpy_where = numpy.full(20, False)
    numpy_where[:10] = True
    numpy.random.shuffle(numpy_where)
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.sum(axis=0, initial=4, out=modin_out, where=modin_where)
    numpy_result = numpy_arr.sum(axis=0, initial=4, out=numpy_out, where=numpy_where)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)


def test_mean():
    # Test 1D
    numpy_arr = numpy.random.randint(-100, 100, size=100)
    modin_arr = np.array(numpy_arr)
    assert modin_arr.mean() == numpy_arr.mean()
    modin_result = modin_arr.mean(axis=0)
    numpy_result = modin_arr.mean(axis=0)
    assert modin_result == numpy_result
    modin_result = modin_arr.mean()
    numpy_result = numpy_arr.mean()
    assert modin_result == numpy_result
    modin_result = modin_arr.mean(keepdims=True)
    numpy_result = numpy_arr.mean(keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    # Test 2D
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    assert modin_arr.mean() == numpy_arr.mean()
    modin_result = modin_arr.mean(axis=0)
    numpy_result = numpy_arr.mean(axis=0)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.mean(axis=0, keepdims=True)
    numpy_result = numpy_arr.mean(axis=0, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.mean(axis=1)
    numpy_result = numpy_arr.mean(axis=1)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.mean(axis=1, keepdims=True)
    numpy_result = numpy_arr.mean(axis=1, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.mean()
    numpy_result = numpy_arr.mean()
    assert modin_result == numpy_result
    with pytest.raises(ValueError):
        modin_result = modin_arr.mean(out=modin_arr, keepdims=True)
    modin_out = np.array([[1]])
    numpy_out = modin_out._to_numpy()
    modin_result = modin_arr.mean(out=modin_out, keepdims=True)
    numpy_result = numpy_arr.mean(out=numpy_out, keepdims=True)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.mean(axis=0, where=False, out=modin_out)
    numpy_result = numpy_arr.mean(axis=0, where=False, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.mean(axis=0, out=modin_out)
    numpy_result = numpy_arr.mean(axis=0, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.mean(axis=1, out=modin_out)
    numpy_result = numpy_arr.mean(axis=1, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    numpy_where = numpy.full(20, False)
    numpy_where[:10] = True
    numpy.random.shuffle(numpy_where)
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.mean(axis=0, out=modin_out, where=modin_where)
    numpy_result = numpy_arr.mean(axis=0, out=numpy_out, where=numpy_where)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)


def test_prod():
    # Test 1D
    numpy_arr = numpy.random.randint(-100, 100, size=100)
    modin_arr = np.array(numpy_arr)
    assert modin_arr.prod() == numpy_arr.prod()
    modin_result = modin_arr.prod(axis=0)
    numpy_result = modin_arr.prod(axis=0)
    assert modin_result == numpy_result
    modin_result = modin_arr.prod(initial=-200)
    numpy_result = numpy_arr.prod(initial=-200)
    assert modin_result == numpy_result
    modin_result = modin_arr.prod(initial=0, where=False)
    numpy_result = numpy_arr.prod(initial=0, where=False)
    assert modin_result == numpy_result
    modin_result = modin_arr.prod(keepdims=True)
    numpy_result = numpy_arr.prod(keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    # Test 2D
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    assert modin_arr.prod() == numpy_arr.prod()
    modin_result = modin_arr.prod(axis=0)
    numpy_result = numpy_arr.prod(axis=0)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.prod(axis=0, keepdims=True)
    numpy_result = numpy_arr.prod(axis=0, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.prod(axis=1)
    numpy_result = numpy_arr.prod(axis=1)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.prod(axis=1, keepdims=True)
    numpy_result = numpy_arr.prod(axis=1, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.prod(initial=-200)
    numpy_result = numpy_arr.prod(initial=-200)
    assert modin_result == numpy_result
    modin_result = modin_arr.prod(initial=0, where=False)
    numpy_result = numpy_arr.prod(initial=0, where=False)
    assert modin_result == numpy_result
    with pytest.raises(ValueError):
        modin_result = modin_arr.prod(out=modin_arr, keepdims=True)
    modin_out = np.array([[1]])
    numpy_out = modin_out._to_numpy()
    modin_result = modin_arr.prod(out=modin_out, keepdims=True)
    numpy_result = numpy_arr.prod(out=numpy_out, keepdims=True)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    modin_result = modin_arr.prod(axis=0, where=False, initial=4)
    numpy_result = numpy_arr.prod(axis=0, where=False, initial=4)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.prod(axis=0, where=False, initial=4, out=modin_out)
    numpy_result = numpy_arr.prod(axis=0, where=False, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_arr = numpy.random.randint(-5, 5, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.prod(axis=0, initial=4, out=modin_out)
    numpy_result = numpy_arr.prod(axis=0, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.prod(axis=1, initial=4, out=modin_out)
    numpy_result = numpy_arr.prod(axis=1, initial=4, out=numpy_out)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    numpy_where = numpy.full(20, False)
    numpy_where[:10] = True
    numpy.random.shuffle(numpy_where)
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.prod(axis=0, initial=4, out=modin_out, where=modin_where)
    numpy_result = numpy_arr.prod(axis=0, initial=4, out=numpy_out, where=numpy_where)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_out._to_numpy(), numpy_out)


def test_abs():
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    numpy.testing.assert_array_equal(
        numpy.abs(numpy_flat_arr), np.abs(modin_flat_arr)._to_numpy()
    )
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    numpy.testing.assert_array_equal(
        numpy.abs(numpy_arr), np.abs(modin_arr)._to_numpy()
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


def test_flatten():
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    numpy.testing.assert_array_equal(
        numpy_flat_arr.flatten(), modin_flat_arr.flatten()._to_numpy()
    )
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    numpy.testing.assert_array_equal(
        numpy_arr.flatten(), modin_arr.flatten()._to_numpy()
    )


def test_transpose():
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    numpy.testing.assert_array_equal(
        numpy_flat_arr.transpose(), modin_flat_arr.transpose()._to_numpy()
    )
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    numpy.testing.assert_array_equal(
        numpy_arr.transpose(), modin_arr.transpose()._to_numpy()
    )
    numpy.testing.assert_array_equal(numpy_arr.T, modin_arr.T._to_numpy())


def test_astype():
    numpy_arr = numpy.array([[1, 2], [3, 4]])
    modin_arr = np.array([[1, 2], [3, 4]])
    modin_result = modin_arr.astype(numpy.float64)
    numpy_result = numpy_arr.astype(numpy.float64)
    assert modin_result.dtype == numpy_result.dtype
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.astype(str)
    numpy_result = numpy_arr.astype(str)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_arr._to_numpy(), numpy_arr)
    modin_result = modin_arr.astype(str, copy=False)
    numpy_result = numpy_arr.astype(str, copy=False)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_arr._to_numpy(), numpy_arr)
    modin_result = modin_arr.astype(numpy.float64, copy=False)
    numpy_result = numpy_arr.astype(numpy.float64, copy=False)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_arr._to_numpy(), numpy_arr)


def test_zeros_like():
    modin_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    numpy_arr = modin_arr._to_numpy()
    numpy.testing.assert_array_equal(
        numpy.zeros_like(numpy_arr), np.zeros_like(modin_arr)._to_numpy()
    )
    numpy.testing.assert_array_equal(
        numpy.zeros_like(numpy_arr, dtype=numpy.int8),
        np.zeros_like(modin_arr, dtype=numpy.int8)._to_numpy(),
    )
    numpy.testing.assert_array_equal(
        numpy.zeros_like(numpy_arr, shape=(10, 10)),
        np.zeros_like(modin_arr, shape=(10, 10))._to_numpy(),
    )
    modin_arr = np.array([[1, 2], [3, 4]])
    numpy_arr = modin_arr._to_numpy()
    numpy.testing.assert_array_equal(
        numpy.zeros_like(numpy_arr), np.zeros_like(modin_arr)._to_numpy()
    )


def test_ones_like():
    modin_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    numpy_arr = modin_arr._to_numpy()
    numpy.testing.assert_array_equal(
        numpy.ones_like(numpy_arr), np.ones_like(modin_arr)._to_numpy()
    )
    numpy.testing.assert_array_equal(
        numpy.ones_like(numpy_arr, dtype=numpy.int8),
        np.ones_like(modin_arr, dtype=numpy.int8)._to_numpy(),
    )
    numpy.testing.assert_array_equal(
        numpy.ones_like(numpy_arr, shape=(10, 10)),
        np.ones_like(modin_arr, shape=(10, 10))._to_numpy(),
    )
    modin_arr = np.array([[1, 2], [3, 4]])
    numpy_arr = modin_arr._to_numpy()
    numpy.testing.assert_array_equal(
        numpy.ones_like(numpy_arr), np.ones_like(modin_arr)._to_numpy()
    )
