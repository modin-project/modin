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


@pytest.mark.parametrize("operand_shape", [100, (100, 3), (3, 100)])
def test_ravel(operand_shape):
    x = numpy.random.randint(-100, 100, size=operand_shape)
    numpy_result = numpy.ravel(x)
    modin_result = np.ravel(np.array(x))
    assert_scalar_or_array_equal(modin_result, numpy_result)


@pytest.mark.parametrize("operand_shape", [100, (100, 3), (3, 100)])
def test_shape(operand_shape):
    x = numpy.random.randint(-100, 100, size=operand_shape)
    numpy_result = numpy.shape(x)
    modin_result = np.shape(np.array(x))
    assert modin_result == numpy_result


@pytest.mark.parametrize("operand_shape", [100, (100, 3), (3, 100)])
def test_transpose(operand_shape):
    x = numpy.random.randint(-100, 100, size=operand_shape)
    numpy_result = numpy.transpose(x)
    modin_result = np.transpose(np.array(x))
    assert_scalar_or_array_equal(modin_result, numpy_result)


@pytest.mark.parametrize("axis", [0, 1])
def test_split_2d(axis):
    x = numpy.random.randint(-100, 100, size=(6, 4))
    # Integer argument: split into N equal arrays along axis
    numpy_result = numpy.split(x, 2, axis=axis)
    modin_result = np.split(np.array(x), 2, axis=axis)
    for modin_entry, numpy_entry in zip(modin_result, numpy_result):
        assert_scalar_or_array_equal(modin_entry, numpy_entry)
    # List argument: split at specified indices
    idxs = [2, 3]
    numpy_result = numpy.split(x, idxs, axis=axis)
    modin_result = np.split(np.array(x), idxs, axis=axis)
    for modin_entry, numpy_entry in zip(modin_result, numpy_result):
        assert_scalar_or_array_equal(modin_entry, numpy_entry)


def test_split_2d_oob():
    # Supplying an index out of bounds results in an empty sub-array, for which modin
    # would return a numpy array by default
    x = numpy.random.randint(-100, 100, size=(6, 4))
    idxs = [2, 3, 6]
    numpy_result = numpy.split(x, idxs)
    modin_result = np.split(np.array(x), idxs)
    for modin_entry, numpy_entry in zip(modin_result, numpy_result):
        assert_scalar_or_array_equal(modin_entry, numpy_entry)


def test_split_2d_uneven():
    x = np.array(numpy.random.randint(-100, 100, size=(3, 2)))
    with pytest.raises(
        ValueError, match="array split does not result in an equal division"
    ):
        np.split(x, 2)


def test_hstack():
    # 2D arrays
    a = numpy.random.randint(-100, 100, size=(5, 3))
    b = numpy.random.randint(-100, 100, size=(5, 2))
    numpy_result = numpy.hstack((a, b))
    modin_result = np.hstack((np.array(a), np.array(b)))
    assert_scalar_or_array_equal(modin_result, numpy_result)
    # 1D arrays
    a = numpy.random.randint(-100, 100, size=(5,))
    b = numpy.random.randint(-100, 100, size=(3,))
    numpy_result = numpy.hstack((a, b))
    modin_result = np.hstack((np.array(a), np.array(b)))
    assert_scalar_or_array_equal(modin_result, numpy_result)


def test_append():
    # Examples taken from numpy docs
    xs = [[1, 2, 3], [[4, 5, 6], [7, 8, 9]]]
    numpy_result = numpy.append(*xs)
    modin_result = np.append(*[np.array(x) for x in xs])
    assert_scalar_or_array_equal(modin_result, numpy_result)

    numpy_result = numpy.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
    modin_result = np.append(np.array([[1, 2, 3], [4, 5, 6]]), [[7, 8, 9]], axis=0)
    assert_scalar_or_array_equal(modin_result, numpy_result)


@pytest.mark.xfail(reason="append error checking is incorrect: see GH#5896")
def test_append_error():
    with pytest.raises(ValueError):
        np.append(np.array([[1, 2, 3], [4, 5, 6]]), np.array([7, 8, 9]), axis=0)
