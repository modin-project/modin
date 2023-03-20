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


@pytest.mark.parametrize(
    "data",
    [
        [3, 2, 1, 1],
        [-87.434, -90.908, -87.152, -84.903],
        [-87.434, -90.908, np.nan, -87.152, -84.903],
    ],
    ids=["ints", "floats", "floats with nan"],
)
def test_argmax_argmin(data):
    numpy_result = numpy.argmax(numpy.array(data))
    modin_result = np.argmax(np.array(data))
    numpy.testing.assert_array_equal(modin_result, numpy_result)

    numpy_result = numpy.argmin(numpy.array(data))
    modin_result = np.argmin(np.array(data))
    numpy.testing.assert_array_equal(modin_result, numpy_result)
