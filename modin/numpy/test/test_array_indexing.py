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
from pandas.core.dtypes.common import is_list_like

import modin.numpy as np


@pytest.mark.parametrize(
    "index",
    (
        0,
        1,
        -1,  # Scalar indices
        slice(0, 1, 1),
        slice(1, -1, 1),  # Slices
        [0, 2],
        [1, -1],  # Lists
    ),
    ids=lambda i: f"index={i}",
)
def test_getitem_1d(index):
    data = [1, 2, 3, 4, 5]
    numpy_result = numpy.array(data)[index]
    modin_result = np.array(data)[index]
    if is_list_like(numpy_result):
        numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
        assert modin_result.shape == numpy_result.shape
    else:
        assert modin_result == numpy_result


@pytest.mark.parametrize(
    "index",
    (
        0,
        1,
        -1,  # Scalar indices
        slice(0, 1, 1),
        slice(1, -1, 1),  # Slices
        slice(None, None, None),
        slice(None, 1, None),
        slice(0, 1, None),
        slice(0, None, None),
        [0, 2],
        [2, 0],
        [1, -1],  # Lists
    ),
    ids=lambda i: f"index={i}",
)
def test_getitem_2d(index):
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    numpy_result = numpy.array(data)[index]
    modin_result = np.array(data)[index]
    if is_list_like(numpy_result):
        numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
        assert modin_result.shape == numpy_result.shape
    else:
        assert modin_result == numpy_result


def test_getitem_nested():
    # Index into the result of slicing a 1D array
    data = [1, 2, 3, 4, 5]
    numpy_result = numpy.array(data)[1:3][1]
    modin_result = np.array(data)[1:3][1]
    if is_list_like(numpy_result):
        numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
        assert modin_result.shape == numpy_result.shape
    else:
        assert (
            modin_result == numpy_result
        )  # Index into the result of indexing a 2D array
    data = [[1, 2, 3], [4, 5, 6]]
    numpy_result = numpy.array(data)[1][1]
    modin_result = np.array(data)[1][1]
    if is_list_like(numpy_result):
        numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
        assert modin_result.shape == numpy_result.shape
    else:
        assert modin_result == numpy_result
