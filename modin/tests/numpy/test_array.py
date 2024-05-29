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

import warnings

import numpy
import pytest

import modin.numpy as np

from .utils import assert_scalar_or_array_equal


@pytest.fixture
def change_numpy_print_threshold():
    prev_threshold = numpy.get_printoptions()["threshold"]
    numpy.set_printoptions(threshold=50)
    yield prev_threshold
    numpy.set_printoptions(threshold=prev_threshold)


@pytest.mark.parametrize(
    "size",
    [
        100,
        (2, 100),
        (100, 2),
        (1, 100),
        (100, 1),
        (100, 100),
        (6, 100),
        (100, 6),
        (100, 7),
        (7, 100),
    ],
)
def test_repr(size, change_numpy_print_threshold):
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


def test_conversion():
    import modin.pandas as pd
    from modin.numpy.utils import try_convert_from_interoperable_type

    df = pd.DataFrame(numpy.random.randint(0, 100, size=(100, 100)))
    series = df.iloc[0]
    df_converted = try_convert_from_interoperable_type(df)
    assert isinstance(df_converted, np.array)
    series_converted = try_convert_from_interoperable_type(series)
    assert isinstance(series_converted, np.array)
    assert_scalar_or_array_equal(df_converted, df)
    assert_scalar_or_array_equal(series_converted, series)
    pandas_df = df._to_pandas()
    pandas_series = series._to_pandas()
    pandas_converted = try_convert_from_interoperable_type(pandas_df)
    assert isinstance(pandas_converted, type(pandas_df))
    assert pandas_converted.equals(pandas_df)
    pandas_converted = try_convert_from_interoperable_type(pandas_series)
    assert isinstance(pandas_converted, type(pandas_series))
    assert pandas_converted.equals(pandas_series)


def test_to_df():
    import pandas

    import modin.pandas as pd
    from modin.tests.pandas.utils import df_equals

    modin_df = pd.DataFrame(np.array([1, 2, 3]))
    pandas_df = pandas.DataFrame(numpy.array([1, 2, 3]))
    df_equals(pandas_df, modin_df)
    modin_df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]))
    pandas_df = pandas.DataFrame(numpy.array([[1, 2, 3], [4, 5, 6]]))
    df_equals(pandas_df, modin_df)
    for kw in [{}, {"dtype": str}]:
        modin_df, pandas_df = [
            lib[0].DataFrame(
                lib[1].array([[1, 2, 3], [4, 5, 6]]),
                columns=["col 0", "col 1", "col 2"],
                index=pd.Index([4, 6]),
                **kw
            )
            for lib in ((pd, np), (pandas, numpy))
        ]
        df_equals(pandas_df, modin_df)
    df_equals(pandas_df, modin_df)


def test_to_series():
    import pandas

    import modin.pandas as pd
    from modin.tests.pandas.utils import df_equals

    with pytest.raises(ValueError, match="Data must be 1-dimensional"):
        pd.Series(np.array([[1, 2, 3], [4, 5, 6]]))
    modin_series = pd.Series(np.array([1, 2, 3]), index=pd.Index([-1, -2, -3]))
    pandas_series = pandas.Series(
        numpy.array([1, 2, 3]), index=pandas.Index([-1, -2, -3])
    )
    df_equals(modin_series, pandas_series)
    modin_series = pd.Series(
        np.array([1, 2, 3]), index=pd.Index([-1, -2, -3]), dtype=str
    )
    pandas_series = pandas.Series(
        numpy.array([1, 2, 3]), index=pandas.Index([-1, -2, -3]), dtype=str
    )
    df_equals(modin_series, pandas_series)


def test_update_inplace():
    out = np.array([1, 2, 3])
    arr1 = np.array([1, 2, 3])
    arr2 = np.array(out, copy=False)
    np.add(arr1, arr1, out=out)
    assert_scalar_or_array_equal(out, arr2)
    out = np.array([1, 2, 3])
    arr2 = np.array(out, copy=False)
    np.add(arr1, arr1, out=out, where=False)
    assert_scalar_or_array_equal(out, arr2)


@pytest.mark.parametrize(
    "data_out",
    [
        numpy.zeros((1, 3)),
        numpy.zeros((2, 3)),
    ],
)
def test_out_broadcast(data_out):
    if data_out.shape == (2, 3):
        pytest.xfail("broadcasting would require duplicating row: see GH#5819")
    data1 = [[1, 2, 3]]
    data2 = [7, 8, 9]
    modin_out, numpy_out = np.array(data_out), numpy.array(data_out)
    numpy.add(numpy.array(data1), numpy.array(data2), out=numpy_out)
    np.add(np.array(data1), np.array(data2), out=modin_out)
    assert_scalar_or_array_equal(modin_out, numpy_out)


def test_out_broadcast_error():
    with pytest.raises(ValueError):
        # Incompatible dimensions between inputs
        np.add(np.array([1, 2, 3]), np.array([[1, 2], [3, 4]]))

    with pytest.raises(ValueError):
        # Compatible input broadcast dimensions, but output array dimensions are wrong
        out = np.array([0])
        np.add(np.array([[1, 2], [3, 4]]), np.array([1, 2]), out=out)

    with pytest.raises(ValueError):
        # Compatible input broadcast dimensions, but output array dimensions are wrong
        # (cannot broadcast a 2x2 result into a 1x2 array)
        out = np.array([0, 0])
        np.add(np.array([[1, 2], [3, 4]]), np.array([1, 2]), out=out)

    with pytest.raises(ValueError):
        # Compatible input broadcast dimensions, but output array dimensions are wrong
        # (cannot broadcast 1x2 into 1D 2-element array)
        out = np.array([0, 0])
        np.add(np.array([[1, 2]]), np.array([1, 2]), out=out)

    with pytest.raises(ValueError):
        # Compatible input broadcast dimensions, but output array dimensions are wrong
        # (cannot broadcast a 2x2 result into a 3x2 array)
        # Technically, our error message here does not match numpy's exactly, as the
        # numpy message will specify both input shapes, whereas we only specify the
        # shape of the default broadcast between the two inputs
        out = np.array([[0, 0], [0, 0], [0, 0]])
        np.add(np.array([[1, 2], [3, 4]]), np.array([1, 2]), out=out)


@pytest.mark.parametrize("size", [100, (2, 100), (100, 2), (1, 100), (100, 1)])
def test_array_ufunc(size):
    # Test ufunc.__call__
    numpy_arr = numpy.random.randint(-100, 100, size=size)
    modin_arr = np.array(numpy_arr)
    modin_result = numpy.sign(modin_arr)
    numpy_result = numpy.sign(numpy_arr)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    # Test ufunc that we have support for.
    modin_result = numpy.add(modin_arr, modin_arr)
    numpy_result = numpy.add(numpy_arr, numpy_arr)
    assert_scalar_or_array_equal(modin_result, numpy_result)
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
    modin_result = numpy.ravel(modin_arr)
    numpy_result = numpy.ravel(numpy_arr)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    # Test from array creation
    modin_result = numpy.zeros_like(modin_arr)
    numpy_result = numpy.zeros_like(numpy_arr)
    assert_scalar_or_array_equal(modin_result, numpy_result)
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
        (modin_flat_arr <= 0).where()
    with pytest.raises(ValueError, match="np.where requires x and y"):
        (modin_flat_arr <= 0).where(x=["Should Fail."])
    with pytest.warns(UserWarning, match="np.where not supported when both x and y"):
        warnings.filterwarnings("ignore", message="Distributing")
        modin_result = (modin_flat_arr <= 0).where(x=4, y=5)
    numpy_result = numpy.where(numpy_flat_arr <= 0, 4, 5)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_flat_bool_arr = modin_flat_arr <= 0
    numpy_flat_bool_arr = numpy_flat_arr <= 0
    modin_result = modin_flat_bool_arr.where(x=5, y=modin_flat_arr)
    numpy_result = numpy.where(numpy_flat_bool_arr, 5, numpy_flat_arr)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_flat_bool_arr.where(x=modin_flat_arr, y=5)
    numpy_result = numpy.where(numpy_flat_bool_arr, numpy_flat_arr, 5)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_flat_bool_arr.where(x=modin_flat_arr, y=(-1 * modin_flat_arr))
    numpy_result = numpy.where(
        numpy_flat_bool_arr, numpy_flat_arr, (-1 * numpy_flat_arr)
    )
    assert_scalar_or_array_equal(modin_result, numpy_result)
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    modin_bool_arr = modin_arr > 0
    numpy_bool_arr = numpy_arr > 0
    modin_result = modin_bool_arr.where(modin_arr, 10 * modin_arr)
    numpy_result = numpy.where(numpy_bool_arr, numpy_arr, 10 * numpy_arr)
    assert_scalar_or_array_equal(modin_result, numpy_result)


@pytest.mark.parametrize("method", ["argmax", "argmin"])
def test_argmax_argmin(method):
    numpy_arr = numpy.array([[1, 2, 3], [4, 5, np.nan]])
    modin_arr = np.array(numpy_arr)
    assert_scalar_or_array_equal(
        getattr(np, method)(modin_arr, axis=1),
        getattr(numpy, method)(numpy_arr, axis=1),
    )


def test_flatten():
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    assert_scalar_or_array_equal(modin_flat_arr.flatten(), numpy_flat_arr.flatten())
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    assert_scalar_or_array_equal(modin_arr.flatten(), numpy_arr.flatten())


def test_transpose():
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    assert_scalar_or_array_equal(modin_flat_arr.transpose(), numpy_flat_arr.transpose())
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    assert_scalar_or_array_equal(modin_arr.transpose(), numpy_arr.transpose())
    assert_scalar_or_array_equal(modin_arr.T, numpy_arr.T)


def test_astype():
    numpy_arr = numpy.array([[1, 2], [3, 4]])
    modin_arr = np.array([[1, 2], [3, 4]])
    modin_result = modin_arr.astype(numpy.float64)
    numpy_result = numpy_arr.astype(numpy.float64)
    assert modin_result.dtype == numpy_result.dtype
    assert_scalar_or_array_equal(modin_result, numpy_result)
    modin_result = modin_arr.astype(str)
    numpy_result = numpy_arr.astype(str)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_arr, numpy_arr)
    modin_result = modin_arr.astype(str, copy=False)
    numpy_result = numpy_arr.astype(str, copy=False)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_arr, numpy_arr)
    modin_result = modin_arr.astype(numpy.float64, copy=False)
    numpy_result = numpy_arr.astype(numpy.float64, copy=False)
    assert_scalar_or_array_equal(modin_result, numpy_result)
    assert_scalar_or_array_equal(modin_arr, numpy_arr)


def test_set_shape():
    numpy_arr = numpy.array([[1, 2, 3], [4, 5, 6]])
    numpy_arr.shape = (6,)
    modin_arr = np.array([[1, 2, 3], [4, 5, 6]])
    modin_arr.shape = (6,)
    assert_scalar_or_array_equal(modin_arr, numpy_arr)
    modin_arr.shape = 6  # Same as using (6,)
    assert_scalar_or_array_equal(modin_arr, numpy_arr)
    with pytest.raises(ValueError, match="cannot reshape"):
        modin_arr.shape = (4,)


def test__array__():
    numpy_arr = numpy.array([[1, 2, 3], [4, 5, 6]])
    modin_arr = np.array(numpy_arr)
    # this implicitly calls `__array__`
    converted_array = numpy.array(modin_arr)
    assert type(converted_array) is type(numpy_arr)
    assert_scalar_or_array_equal(converted_array, numpy_arr)
