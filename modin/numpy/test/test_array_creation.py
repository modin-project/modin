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

import modin.numpy as np

from .utils import assert_scalar_or_array_equal


def test_zeros_like():
    modin_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    numpy_arr = modin_arr._to_numpy()
    assert_scalar_or_array_equal(np.zeros_like(modin_arr), numpy.zeros_like(numpy_arr))
    assert_scalar_or_array_equal(
        np.zeros_like(modin_arr, dtype=numpy.int8),
        numpy.zeros_like(numpy_arr, dtype=numpy.int8),
    )
    assert_scalar_or_array_equal(
        np.zeros_like(modin_arr, shape=(10, 10)),
        numpy.zeros_like(numpy_arr, shape=(10, 10)),
    )
    modin_arr = np.array([[1, 2], [3, 4]])
    numpy_arr = modin_arr._to_numpy()
    assert_scalar_or_array_equal(
        np.zeros_like(modin_arr),
        numpy.zeros_like(numpy_arr),
    )


def test_ones_like():
    modin_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    numpy_arr = modin_arr._to_numpy()
    assert_scalar_or_array_equal(
        np.ones_like(modin_arr),
        numpy.ones_like(numpy_arr),
    )
    assert_scalar_or_array_equal(
        np.ones_like(modin_arr, dtype=numpy.int8),
        numpy.ones_like(numpy_arr, dtype=numpy.int8),
    )
    assert_scalar_or_array_equal(
        np.ones_like(modin_arr, shape=(10, 10)),
        numpy.ones_like(numpy_arr, shape=(10, 10)),
    )
    modin_arr = np.array([[1, 2], [3, 4]])
    numpy_arr = modin_arr._to_numpy()
    assert_scalar_or_array_equal(
        np.ones_like(modin_arr),
        numpy.ones_like(numpy_arr),
    )
