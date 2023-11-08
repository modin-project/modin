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


@pytest.mark.parametrize(
    "data",
    [
        [3, 2, 1, 1],
        [-87.434, -90.908, -87.152, -84.903],
        [-87.434, -90.908, np.nan, -87.152, -84.903],
    ],
    ids=["ints", "floats", "floats with nan"],
)
@pytest.mark.parametrize("op", ["argmin", "argmax"])
def test_argmax_argmin(data, op):
    numpy_result = getattr(numpy, op)(numpy.array(data))
    modin_result = getattr(np, op)(np.array(data))
    assert_scalar_or_array_equal(modin_result, numpy_result)


def test_rem_mod():
    """Tests remainder and mod, which, unlike the C/matlab equivalents, are identical in numpy."""
    a = numpy.array([[2, -1], [10, -3]])
    b = numpy.array(([-3, 3], [3, -7]))
    numpy_result = numpy.remainder(a, b)
    modin_result = np.remainder(np.array(a), np.array(b))
    assert_scalar_or_array_equal(modin_result, numpy_result)

    numpy_result = numpy.mod(a, b)
    modin_result = np.mod(np.array(a), np.array(b))
    assert_scalar_or_array_equal(modin_result, numpy_result)
