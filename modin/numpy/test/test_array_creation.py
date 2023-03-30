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
