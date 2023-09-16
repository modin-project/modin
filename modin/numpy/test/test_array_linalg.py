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
import numpy.linalg as NLA
import pytest

import modin.numpy as np
import modin.numpy.linalg as LA
import modin.pandas as pd

from .utils import assert_scalar_or_array_equal


def test_dot_from_pandas_reindex():
    # Reindexing the dataframe does not change the output of dot
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dot.html
    df = pd.DataFrame([[0, 1, -2, -1], [1, 1, 1, 1]])
    s = pd.Series([1, 1, 2, 1])
    result1 = np.dot(df, s)
    s2 = s.reindex([1, 0, 2, 3])
    result2 = np.dot(df, s2)
    assert_scalar_or_array_equal(result1, result2)


def test_dot_1d():
    x1 = numpy.random.randint(-100, 100, size=100)
    x2 = numpy.random.randint(-100, 100, size=100)
    numpy_result = numpy.dot(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    modin_result = np.dot(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result)


def test_dot_2d():
    x1 = numpy.random.randint(-100, 100, size=(100, 3))
    x2 = numpy.random.randint(-100, 100, size=(3, 50))
    numpy_result = numpy.dot(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    modin_result = np.dot(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result)


def test_dot_scalar():
    x1 = numpy.random.randint(-100, 100, size=(100, 3))
    x2 = numpy.random.randint(-100, 100)
    numpy_result = numpy.dot(x1, x2)
    x1 = np.array(x1)
    modin_result = np.dot(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result)


def test_matmul_scalar():
    x1 = numpy.random.randint(-100, 100, size=(100, 3))
    x2 = numpy.random.randint(-100, 100)
    x1 = np.array(x1)
    # Modin error message differs from numpy for readability; the original numpy error is:
    # ValueError: matmul: Input operand 1 does not have enough dimensions (has 0, gufunc
    # core with signature (n?,k),(k,m?)->(n?,m?) requires 1)
    with pytest.raises(ValueError):
        x1 @ x2


def test_dot_broadcast():
    # 2D @ 1D
    x1 = numpy.random.randint(-100, 100, size=(100, 3))
    x2 = numpy.random.randint(-100, 100, size=(3,))
    numpy_result = numpy.dot(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    modin_result = np.dot(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result)

    # 1D @ 2D
    x1 = numpy.random.randint(-100, 100, size=(100,))
    x2 = numpy.random.randint(-100, 100, size=(100, 3))
    numpy_result = numpy.dot(x1, x2)
    x1, x2 = np.array(x1), np.array(x2)
    modin_result = np.dot(x1, x2)
    assert_scalar_or_array_equal(modin_result, numpy_result)


@pytest.mark.parametrize("axis", [None, 0, 1], ids=["axis=None", "axis=0", "axis=1"])
def test_norm_fro_2d(axis):
    x1 = numpy.random.randint(-10, 10, size=(100, 3))
    numpy_result = NLA.norm(x1, axis=axis)
    x1 = np.array(x1)
    modin_result = LA.norm(x1, axis=axis)
    # Result may be a scalar
    if isinstance(modin_result, np.array):
        modin_result = modin_result._to_numpy()
    numpy.testing.assert_allclose(modin_result, numpy_result, rtol=1e-12)


def test_norm_fro_1d():
    x1 = numpy.random.randint(-10, 10, size=100)
    numpy_result = NLA.norm(x1)
    x1 = np.array(x1)
    modin_result = LA.norm(x1)
    numpy.testing.assert_allclose(modin_result, numpy_result, rtol=1e-12)
