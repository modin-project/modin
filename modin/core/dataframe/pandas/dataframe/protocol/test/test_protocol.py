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

"""
Tests related to the dataframe exchange protocol implementation correctness.

See more in https://data-apis.org/dataframe-protocol/latest/index.html.
"""


import pandas
import pandas.testing as tm
import numpy as np
import pytest
from typing import Any, Tuple

from ..dataframe import Column, Buffer, DTypeKind, from_dataframe, DataFrameObject
import modin.pandas as pd


# Roundtrip testing
# -----------------


def assert_buffer_equal(buffer_dtype: Tuple[Buffer, Any], pdcol: pandas.Series):
    buf, dtype = buffer_dtype
    pytest.raises(NotImplementedError, buf.__dlpack__)
    assert buf.__dlpack_device__() == (1, None)
    # It seems that `bitwidth` is handled differently for `int` and `category`
    # assert dtype[1] == pdcol.dtype.itemsize * 8, f"{dtype[1]} is not {pdcol.dtype.itemsize}"
    # print(pdcol)
    # if isinstance(pdcol, pandas.CategoricalDtype):
    #     col = pdcol.values.codes
    # else:
    #     col = pdcol

    # assert dtype[1] == col.dtype.itemsize * 8, f"{dtype[1]} is not {col.dtype.itemsize * 8}"
    # assert dtype[2] == col.dtype.str, f"{dtype[2]} is not {col.dtype.str}"


def assert_column_equal(col: Column, pdcol: pandas.Series):
    assert col.size == pdcol.size
    assert col.offset == 0
    assert col.null_count == pdcol.isnull().sum()
    assert col.num_chunks() == 1
    if col.dtype[0] != DTypeKind.STRING:
        pytest.raises(RuntimeError, col._get_validity_buffer)
    assert_buffer_equal(col._get_data_buffer(), pdcol)


def assert_dataframe_equal(dfo: DataFrameObject, df: pandas.DataFrame):
    assert dfo.num_columns() == len(df.columns)
    assert dfo.num_rows() == len(df)
    assert dfo.num_chunks() == 1
    assert dfo.column_names() == list(df.columns)
    for col in df.columns:
        assert_column_equal(dfo.get_column_by_name(col), df[col])


def test_float_only():
    df = pandas.DataFrame(data=dict(a=[1.5, 2.5, 3.5], b=[9.2, 10.5, 11.8]))
    df2 = from_dataframe(df)
    assert_dataframe_equal(df.__dataframe__(), df)
    tm.assert_frame_equal(df, df2)


def test_mixed_intfloat():
    df = pandas.DataFrame(
        data=dict(a=[1, 2, 3], b=[3, 4, 5], c=[1.5, 2.5, 3.5], d=[9, 10, 11])
    )
    df2 = from_dataframe(df)
    assert_dataframe_equal(df.__dataframe__(), df)
    tm.assert_frame_equal(df, df2)


def test_noncontiguous_columns():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df = pandas.DataFrame(arr, columns=["a", "b", "c"])
    assert df["a"].to_numpy().strides == (24,)
    df2 = from_dataframe(df)  # uses default of allow_copy=True
    assert_dataframe_equal(df.__dataframe__(), df)
    tm.assert_frame_equal(df, df2)

    with pytest.raises(RuntimeError):
        from_dataframe(df, allow_copy=False)


def test_categorical_dtype():
    pandas_df = pandas.DataFrame({"A": [1, 2, 5, 1]})
    modin_df = pd.DataFrame(pandas_df)
    modin_df["B"] = modin_df["A"].astype("category")
    modin_df.at[1, "B"] = np.nan  # Set one item to null

    # Some detailed testing for correctness of dtype and null handling:
    df_impl_protocol = modin_df.__dataframe__()
    col = df_impl_protocol.get_column_by_name("B")
    assert col.dtype[0] == DTypeKind.CATEGORICAL
    assert col.null_count == 1
    assert col.describe_null == (2, -1)  # sentinel value -1
    assert col.num_chunks() == 1
    assert col.describe_categorical == (False, True, {0: 1, 1: 2, 2: 5})

    df2 = from_dataframe(modin_df)
    assert_dataframe_equal(df_impl_protocol, modin_df)
    tm.assert_frame_equal(modin_df, df2)


def test_string_dtype():
    pandas_df = pandas.DataFrame({"A": ["a", "b", "cdef", "", "g"]})
    modin_df = pd.DataFrame(pandas_df)
    modin_df["B"] = modin_df["A"].astype("object")
    modin_df.at[1, "B"] = np.nan  # Set one item to null

    # Test for correctness and null handling:
    df_impl_protocol = modin_df.__dataframe__()
    col = df_impl_protocol.get_column_by_name("B")
    assert col.dtype[0] == DTypeKind.STRING
    assert col.null_count == 1
    assert col.describe_null == (4, 0)
    assert col.num_chunks() == 1

    assert_dataframe_equal(df_impl_protocol, df)


def test_metadata():
    pandas_df = pandas.DataFrame({"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]})
    modin_df = pd.DataFrame(pandas_df)

    # Check the metadata from the dataframe
    df_impl_protocol = modin_df.__dataframe__()
    df_metadata = df_impl_protocol.metadata
    expected = {"pandas.index": modin_df.index}
    for key in df_metadata:
        assert all(df_metadata[key] == expected[key])

    # Check the metadata from the column
    col_metadata = df_impl_protocol.get_column(0).metadata
    expected = {}
    for key in col_metadata:
        assert col_metadata[key] == expected[key]

    df2 = from_dataframe(modin_df)
    assert_dataframe_equal(modin_df.__dataframe__(), modin_df)
    tm.assert_frame_equal(modin_df, df2)
