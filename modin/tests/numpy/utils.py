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


def assert_scalar_or_array_equal(x1, x2, err_msg=""):
    """
    Assert whether the result of the numpy and modin computations are the same.

    If either argument is a modin array object, then `_to_numpy()` is called on it.
    The arguments are compared with `numpy.testing.assert_array_equals`.
    """
    if isinstance(x1, np.array):
        x1 = x1._to_numpy()
    if isinstance(x2, np.array):
        x2 = x2._to_numpy()
    numpy.testing.assert_array_equal(x1, x2, err_msg=err_msg)
