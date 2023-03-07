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
import warnings

import modin.numpy as np


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
    numpy.testing.assert_array_equal(df_converted._to_numpy(), df.to_numpy())
    numpy.testing.assert_array_equal(series_converted._to_numpy(), series.to_numpy())
    pandas_df = df._to_pandas()
    pandas_series = series._to_pandas()
    pandas_converted = try_convert_from_interoperable_type(pandas_df)
    assert isinstance(pandas_converted, type(pandas_df))
    assert pandas_converted.equals(pandas_df)
    pandas_converted = try_convert_from_interoperable_type(pandas_series)
    assert isinstance(pandas_converted, type(pandas_series))
    assert pandas_converted.equals(pandas_series)


def test_to_df():
    import modin.pandas as pd
    from modin.pandas.test.utils import df_equals

    import pandas

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
    import modin.pandas as pd
    from modin.pandas.test.utils import df_equals

    import pandas

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
    numpy.testing.assert_array_equal(out._to_numpy(), arr2._to_numpy())
    out = np.array([1, 2, 3])
    arr2 = np.array(out, copy=False)
    np.add(arr1, arr1, out=out, where=False)
    numpy.testing.assert_array_equal(out._to_numpy(), arr2._to_numpy())


@pytest.mark.parametrize("size", [100, (2, 100), (100, 2), (1, 100), (100, 1)])
def test_array_ufunc(size):
    # Test ufunc.__call__
    numpy_arr = numpy.random.randint(-100, 100, size=size)
    modin_arr = np.array(numpy_arr)
    modin_result = numpy.sign(modin_arr)._to_numpy()
    numpy_result = numpy.sign(numpy_arr)
    numpy.testing.assert_array_equal(modin_result, numpy_result)
    # Test ufunc that we have support for.
    modin_result = numpy.add(modin_arr, modin_arr)._to_numpy()
    numpy_result = numpy.add(numpy_arr, numpy_arr)
    numpy.testing.assert_array_equal(modin_result, numpy_result)
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
    modin_result = numpy.ravel(modin_arr)._to_numpy()
    numpy_result = numpy.ravel(numpy_arr)
    numpy.testing.assert_array_equal(modin_result, numpy_result)
    # Test from array creation
    modin_result = numpy.zeros_like(modin_arr)._to_numpy()
    numpy_result = numpy.zeros_like(numpy_arr)
    numpy.testing.assert_array_equal(modin_result, numpy_result)
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
    numpy.testing.assert_array_equal(numpy_result, modin_result._to_numpy())
    modin_flat_bool_arr = modin_flat_arr <= 0
    numpy_flat_bool_arr = numpy_flat_arr <= 0
    modin_result = modin_flat_bool_arr.where(x=5, y=modin_flat_arr)
    numpy_result = numpy.where(numpy_flat_bool_arr, 5, numpy_flat_arr)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_flat_bool_arr.where(x=modin_flat_arr, y=5)
    numpy_result = numpy.where(numpy_flat_bool_arr, numpy_flat_arr, 5)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_flat_bool_arr.where(x=modin_flat_arr, y=(-1 * modin_flat_arr))
    numpy_result = numpy.where(
        numpy_flat_bool_arr, numpy_flat_arr, (-1 * numpy_flat_arr)
    )
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    modin_bool_arr = modin_arr > 0
    numpy_bool_arr = numpy_arr > 0
    modin_result = modin_bool_arr.where(modin_arr, 10 * modin_arr)
    numpy_result = numpy.where(numpy_bool_arr, numpy_arr, 10 * numpy_arr)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)


def test_flatten():
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    numpy.testing.assert_array_equal(
        numpy_flat_arr.flatten(), modin_flat_arr.flatten()._to_numpy()
    )
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    numpy.testing.assert_array_equal(
        numpy_arr.flatten(), modin_arr.flatten()._to_numpy()
    )


def test_transpose():
    numpy_flat_arr = numpy.random.randint(-100, 100, size=100)
    modin_flat_arr = np.array(numpy_flat_arr)
    numpy.testing.assert_array_equal(
        numpy_flat_arr.transpose(), modin_flat_arr.transpose()._to_numpy()
    )
    numpy_arr = numpy_flat_arr.reshape((10, 10))
    modin_arr = np.array(numpy_arr)
    numpy.testing.assert_array_equal(
        numpy_arr.transpose(), modin_arr.transpose()._to_numpy()
    )
    numpy.testing.assert_array_equal(numpy_arr.T, modin_arr.T._to_numpy())


def test_astype():
    numpy_arr = numpy.array([[1, 2], [3, 4]])
    modin_arr = np.array([[1, 2], [3, 4]])
    modin_result = modin_arr.astype(numpy.float64)
    numpy_result = numpy_arr.astype(numpy.float64)
    assert modin_result.dtype == numpy_result.dtype
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    modin_result = modin_arr.astype(str)
    numpy_result = numpy_arr.astype(str)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_arr._to_numpy(), numpy_arr)
    modin_result = modin_arr.astype(str, copy=False)
    numpy_result = numpy_arr.astype(str, copy=False)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_arr._to_numpy(), numpy_arr)
    modin_result = modin_arr.astype(numpy.float64, copy=False)
    numpy_result = numpy_arr.astype(numpy.float64, copy=False)
    numpy.testing.assert_array_equal(modin_result._to_numpy(), numpy_result)
    numpy.testing.assert_array_equal(modin_arr._to_numpy(), numpy_arr)
