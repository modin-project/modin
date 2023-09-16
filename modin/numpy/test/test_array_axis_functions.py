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

from .utils import assert_scalar_or_array_equal


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
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_arr = numpy.array([1, 10000, 2, 3, 4, 5])
    modin_arr = np.array(numpy_arr)
    numpy_mask = numpy.array([True, False, True, True, True, True])
    modin_mask = np.array(numpy_mask)
    assert numpy_arr.max(where=numpy_mask, initial=5) == modin_arr.max(
        where=modin_mask, initial=5
    )
    # Test 2D
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    assert modin_arr.max() == numpy_arr.max()
    modin_result = modin_arr.max(axis=0)
    numpy_result = numpy_arr.max(axis=0)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.max(axis=0, keepdims=True)
    numpy_result = numpy_arr.max(axis=0, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.max(axis=1)
    numpy_result = numpy_arr.max(axis=1)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.max(axis=1, keepdims=True)
    numpy_result = numpy_arr.max(axis=1, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.max(initial=200)
    numpy_result = numpy_arr.max(initial=200)
    assert modin_result == numpy_result
    modin_result = modin_arr.max(initial=0, where=False)
    numpy_result = numpy_arr.max(initial=0, where=False)
    assert modin_result == numpy_result
    with pytest.raises(ValueError):
        modin_arr.max(out=modin_arr, keepdims=True)
    modin_out = np.array([[1]])
    numpy_out = modin_out._to_numpy()
    modin_result = modin_arr.max(out=modin_out, keepdims=True)
    numpy_result = numpy_arr.max(out=numpy_out, keepdims=True)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    modin_result = modin_arr.max(axis=0, where=False, initial=4)
    numpy_result = numpy_arr.max(axis=0, where=False, initial=4)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.max(axis=0, where=False, initial=4, out=modin_out)
    numpy_result = numpy_arr.max(axis=0, where=False, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.max(axis=0, initial=4, out=modin_out)
    numpy_result = numpy_arr.max(axis=0, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.max(axis=1, initial=4, out=modin_out)
    numpy_result = numpy_arr.max(axis=1, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    numpy_where = numpy.full(20, False)
    numpy_where[:10] = True
    numpy.random.shuffle(numpy_where)
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.max(axis=0, initial=4, out=modin_out, where=modin_where)
    numpy_result = numpy_arr.max(axis=0, initial=4, out=numpy_out, where=numpy_where)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_arr = numpy.array([[1, 10000, 2], [3, 4, 5]])
    modin_arr = np.array(numpy_arr)
    numpy_mask = numpy.array([[True, False, True], [True, True, True]])
    modin_mask = np.array(numpy_mask)
    assert_scalar_or_array_equal(
        modin_arr.max(where=modin_mask, initial=5),
        numpy_arr.max(where=numpy_mask, initial=5),
    )


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
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_arr = numpy.array([1, -10000, 2, 3, 4, 5])
    modin_arr = np.array(numpy_arr)
    numpy_mask = numpy.array([True, False, True, True, True, True])
    modin_mask = np.array(numpy_mask)
    assert numpy_arr.min(where=numpy_mask, initial=5) == modin_arr.min(
        where=modin_mask, initial=5
    )
    # Test 2D
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    assert modin_arr.min() == numpy_arr.min()
    modin_result = modin_arr.min(axis=0)
    numpy_result = numpy_arr.min(axis=0)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.min(axis=0, keepdims=True)
    numpy_result = numpy_arr.min(axis=0, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.min(axis=1)
    numpy_result = numpy_arr.min(axis=1)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.min(axis=1, keepdims=True)
    numpy_result = numpy_arr.min(axis=1, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.min(initial=-200)
    numpy_result = numpy_arr.min(initial=-200)
    assert modin_result == numpy_result
    modin_result = modin_arr.min(initial=0, where=False)
    numpy_result = numpy_arr.min(initial=0, where=False)
    assert modin_result == numpy_result
    with pytest.raises(ValueError):
        modin_arr.min(out=modin_arr, keepdims=True)
    modin_out = np.array([[1]])
    numpy_out = modin_out._to_numpy()
    modin_result = modin_arr.min(out=modin_out, keepdims=True)
    numpy_result = numpy_arr.min(out=numpy_out, keepdims=True)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    modin_result = modin_arr.min(axis=0, where=False, initial=4)
    numpy_result = numpy_arr.min(axis=0, where=False, initial=4)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.min(axis=0, where=False, initial=4, out=modin_out)
    numpy_result = numpy_arr.min(axis=0, where=False, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.min(axis=0, initial=4, out=modin_out)
    numpy_result = numpy_arr.min(axis=0, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.min(axis=1, initial=4, out=modin_out)
    numpy_result = numpy_arr.min(axis=1, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    numpy_where = numpy.full(20, False)
    numpy_where[:10] = True
    numpy.random.shuffle(numpy_where)
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.min(axis=0, initial=4, out=modin_out, where=modin_where)
    numpy_result = numpy_arr.min(axis=0, initial=4, out=numpy_out, where=numpy_where)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_arr = numpy.array([[1, -10000, 2], [3, 4, 5]])
    modin_arr = np.array(numpy_arr)
    numpy_mask = numpy.array([[True, False, True], [True, True, True]])
    modin_mask = np.array(numpy_mask)
    assert_scalar_or_array_equal(
        modin_arr.min(where=modin_mask, initial=5),
        numpy_arr.min(where=numpy_mask, initial=5),
    )


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
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_arr = numpy.array([1, 10000, 2, 3, 4, 5])
    modin_arr = np.array(numpy_arr)
    numpy_mask = numpy.array([True, False, True, True, True, True])
    modin_mask = np.array(numpy_mask)
    assert numpy_arr.sum(where=numpy_mask) == modin_arr.sum(where=modin_mask)
    # Test 2D
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    assert modin_arr.sum() == numpy_arr.sum()
    modin_result = modin_arr.sum(axis=0)
    numpy_result = numpy_arr.sum(axis=0)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.sum(axis=0, keepdims=True)
    numpy_result = numpy_arr.sum(axis=0, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.sum(axis=1)
    numpy_result = numpy_arr.sum(axis=1)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.sum(axis=1, keepdims=True)
    numpy_result = numpy_arr.sum(axis=1, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.sum(initial=-200)
    numpy_result = numpy_arr.sum(initial=-200)
    assert modin_result == numpy_result
    modin_result = modin_arr.sum(initial=0, where=False)
    numpy_result = numpy_arr.sum(initial=0, where=False)
    assert modin_result == numpy_result
    with pytest.raises(ValueError):
        modin_arr.sum(out=modin_arr, keepdims=True)
    modin_out = np.array([[1]])
    numpy_out = modin_out._to_numpy()
    modin_result = modin_arr.sum(out=modin_out, keepdims=True)
    numpy_result = numpy_arr.sum(out=numpy_out, keepdims=True)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    modin_result = modin_arr.sum(axis=0, where=False, initial=4)
    numpy_result = numpy_arr.sum(axis=0, where=False, initial=4)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.sum(axis=0, where=False, initial=4, out=modin_out)
    numpy_result = numpy_arr.sum(axis=0, where=False, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.sum(axis=0, initial=4, out=modin_out)
    numpy_result = numpy_arr.sum(axis=0, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.sum(axis=1, initial=4, out=modin_out)
    numpy_result = numpy_arr.sum(axis=1, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    numpy_where = numpy.full(20, False)
    numpy_where[:10] = True
    numpy.random.shuffle(numpy_where)
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.sum(axis=0, initial=4, out=modin_out, where=modin_where)
    numpy_result = numpy_arr.sum(axis=0, initial=4, out=numpy_out, where=numpy_where)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_where = numpy.full(400, False)
    numpy_where[:200] = True
    numpy.random.shuffle(numpy_where)
    numpy_where = numpy_where.reshape((20, 20))
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.sum(where=modin_where)
    numpy_result = numpy_arr.sum(where=numpy_where)
    assert modin_result == numpy_result
    # Test NA propagation
    numpy_arr = numpy.array([[1, 2], [3, 4], [5, numpy.nan]])
    modin_arr = np.array([[1, 2], [3, 4], [5, np.nan]])
    assert numpy.isnan(modin_arr.sum())
    assert_scalar_or_array_equal(
        modin_arr.sum(axis=1),
        numpy_arr.sum(axis=1),
    )
    assert_scalar_or_array_equal(
        modin_arr.sum(axis=0),
        numpy_arr.sum(axis=0),
    )


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
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_arr = numpy.array([1, 10000, 2, 3, 4, 5])
    modin_arr = np.array(numpy_arr)
    numpy_mask = numpy.array([True, False, True, True, True, True])
    modin_mask = np.array(numpy_mask)
    assert numpy_arr.mean(where=numpy_mask) == modin_arr.mean(where=modin_mask)
    # Test 2D
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    assert modin_arr.mean() == numpy_arr.mean()
    modin_result = modin_arr.mean(axis=0)
    numpy_result = numpy_arr.mean(axis=0)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.mean(axis=0, keepdims=True)
    numpy_result = numpy_arr.mean(axis=0, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.mean(axis=1)
    numpy_result = numpy_arr.mean(axis=1)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.mean(axis=1, keepdims=True)
    numpy_result = numpy_arr.mean(axis=1, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.mean()
    numpy_result = numpy_arr.mean()
    assert modin_result == numpy_result
    with pytest.raises(ValueError):
        modin_arr.mean(out=modin_arr, keepdims=True)
    modin_out = np.array([[1]])
    numpy_out = modin_out._to_numpy()
    modin_result = modin_arr.mean(out=modin_out, keepdims=True)
    numpy_result = numpy_arr.mean(out=numpy_out, keepdims=True)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.mean(axis=0, where=False, out=modin_out)
    numpy_result = numpy_arr.mean(axis=0, where=False, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.mean(axis=0, out=modin_out)
    numpy_result = numpy_arr.mean(axis=0, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.mean(axis=1, out=modin_out)
    numpy_result = numpy_arr.mean(axis=1, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    numpy_where = numpy.full(20, False)
    numpy_where[:10] = True
    numpy.random.shuffle(numpy_where)
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.mean(axis=0, out=modin_out, where=modin_where)
    numpy_result = numpy_arr.mean(axis=0, out=numpy_out, where=numpy_where)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_where = numpy.full(400, False)
    numpy_where[:200] = True
    numpy.random.shuffle(numpy_where)
    numpy_where = numpy_where.reshape((20, 20))
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.mean(where=modin_where)
    numpy_result = numpy_arr.mean(where=numpy_where)
    assert modin_result == numpy_result
    # Test NA propagation
    numpy_arr = numpy.array([[1, 2], [3, 4], [5, numpy.nan]])
    modin_arr = np.array([[1, 2], [3, 4], [5, np.nan]])
    assert numpy.isnan(modin_arr.mean())
    assert_scalar_or_array_equal(
        modin_arr.mean(axis=1),
        numpy_arr.mean(axis=1),
    )
    assert_scalar_or_array_equal(
        modin_arr.mean(axis=0),
        numpy_arr.mean(axis=0),
    )
    numpy_where = numpy.array([[True, True], [True, True], [True, False]])
    modin_where = np.array(numpy_where)
    assert modin_arr.mean(where=modin_where) == numpy_arr.mean(where=numpy_where)


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
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_arr = numpy.array([1, 10000, 2, 3, 4, 5])
    modin_arr = np.array(numpy_arr)
    numpy_mask = numpy.array([True, False, True, True, True, True])
    modin_mask = np.array(numpy_mask)
    assert numpy_arr.prod(where=numpy_mask) == modin_arr.prod(where=modin_mask)
    # Test 2D
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    assert modin_arr.prod() == numpy_arr.prod()
    modin_result = modin_arr.prod(axis=0)
    numpy_result = numpy_arr.prod(axis=0)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.prod(axis=0, keepdims=True)
    numpy_result = numpy_arr.prod(axis=0, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.prod(axis=1)
    numpy_result = numpy_arr.prod(axis=1)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.prod(axis=1, keepdims=True)
    numpy_result = numpy_arr.prod(axis=1, keepdims=True)
    assert modin_result.shape == numpy_result.shape
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.prod(initial=-200)
    numpy_result = numpy_arr.prod(initial=-200)
    assert modin_result == numpy_result
    modin_result = modin_arr.prod(initial=0, where=False)
    numpy_result = numpy_arr.prod(initial=0, where=False)
    assert modin_result == numpy_result
    with pytest.raises(ValueError):
        modin_arr.prod(out=modin_arr, keepdims=True)
    modin_out = np.array([[1]])
    numpy_out = modin_out._to_numpy()
    modin_result = modin_arr.prod(out=modin_out, keepdims=True)
    numpy_result = numpy_arr.prod(out=numpy_out, keepdims=True)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    modin_result = modin_arr.prod(axis=0, where=False, initial=4)
    numpy_result = numpy_arr.prod(axis=0, where=False, initial=4)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.prod(axis=0, where=False, initial=4, out=modin_out)
    numpy_result = numpy_arr.prod(axis=0, where=False, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_arr = numpy.random.randint(-100, 100, size=(20, 20))
    modin_arr = np.array(numpy_arr)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.prod(axis=0, initial=4, out=modin_out)
    numpy_result = numpy_arr.prod(axis=0, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    modin_result = modin_arr.prod(axis=1, initial=4, out=modin_out)
    numpy_result = numpy_arr.prod(axis=1, initial=4, out=numpy_out)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_out = numpy.ones(20)
    modin_out = np.array(numpy_out)
    numpy_where = numpy.full(20, False)
    numpy_where[:10] = True
    numpy.random.shuffle(numpy_where)
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.prod(axis=0, initial=4, out=modin_out, where=modin_where)
    numpy_result = numpy_arr.prod(axis=0, initial=4, out=numpy_out, where=numpy_where)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_out, numpy_out)
    numpy_where = numpy.full(400, False)
    numpy_where[:200] = True
    numpy.random.shuffle(numpy_where)
    numpy_where = numpy_where.reshape((20, 20))
    modin_where = np.array(numpy_where)
    modin_result = modin_arr.prod(where=modin_where)
    numpy_result = numpy_arr.prod(where=numpy_where)
    assert modin_result == numpy_result
    # Test NA propagation
    numpy_arr = numpy.array([[1, 2], [3, 4], [5, numpy.nan]])
    modin_arr = np.array([[1, 2], [3, 4], [5, np.nan]])
    assert numpy.isnan(modin_arr.prod())
    assert_scalar_or_array_equal(
        modin_arr.prod(axis=1),
        numpy_arr.prod(axis=1),
    )
    assert_scalar_or_array_equal(
        modin_arr.prod(axis=0),
        numpy_arr.prod(axis=0),
    )
