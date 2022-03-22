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

import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.core.dataframe.base.exchange.dataframe_protocol.utils import (
    DTypeKind,
    ColumnNullType,
)
from modin.pandas.test.utils import df_equals, NCOLS, NROWS, test_bool_data, test_data, test_string_data
from modin.pandas.utils import from_dataframe


test_data_categorical = {
    "ordered": pandas.Categorical(list("testdata")*30, ordered=True),
    "unordered": pandas.Categorical(list("testdata")*30, ordered=False),
}


@pytest.mark.parametrize("num_partitions", [2, 4, 6])
@pytest.mark.parametrize("data", [("ordered", True), ("unordered", False)])
def test_categorical_dtype_two_columns(data, num_partitions):
    NPartitions.put(num_partitions)
    pd_df = pd.DataFrame({"A": (test_data_categorical[data[0]])})
    df = pd.DataFrame(pd_df)

    col = df.__dataframe__().get_column_by_name("A")
    assert col.dtype[0] == DTypeKind.CATEGORICAL
    assert col.null_count == 0
    if StorageFormat.get() == "Pandas":
        assert col.describe_null == (ColumnNullType.USE_SENTINEL, -1)
    elif StorageFormat.get() == "Omnisci":
        assert col.describe_null == (ColumnNullType.USE_BYTEMASK, -1)
    if StorageFormat.get() == "Pandas":
        assert col.num_chunks() == num_partitions
    assert tuple(col.describe_categorical.values()) == (data[1], True, {4: 's', 2: 'd', 3: 'e', 1: 't'})

    df_equals(df, from_dataframe(df.__dataframe__()))


def test_chunks():
    df = pd.DataFrame({"x": np.arange(10)})
    df2 = df.__dataframe__()
    chunk_iter = iter(df2.get_chunks(3))
    chunk = next(chunk_iter)
    assert chunk.num_rows() == 4
    chunk = next(chunk_iter)
    assert chunk.num_rows() == 4
    chunk = next(chunk_iter)
    assert chunk.num_rows() == 2
    with pytest.raises(StopIteration):
        chunk = next(chunk_iter)


@pytest.mark.parametrize("num_partitions", [2, 4, 6])
@pytest.mark.parametrize("data", [test_data["int_data"], test_data["float_data"], test_bool_data])
def test_dataframe(data, num_partitions):
    NPartitions.put(num_partitions)
    df = pd.DataFrame(data)

    df2 = df.__dataframe__()

    assert df2._allow_copy is True
    assert df2.num_columns() == NCOLS
    assert df2.num_rows() == NROWS
    if StorageFormat.get() == "Pandas":
        assert df2.num_chunks() == num_partitions

    assert list(df2.column_names()) == list(data.keys())

    assert from_dataframe(df2.select_columns((0, 2))) == from_dataframe(
        df2.select_columns_by_name(("col33", "col35"))
    )
    assert from_dataframe(df2.select_columns((0, 2))) == from_dataframe(
        df2.select_columns_by_name(("col33", "col35"))
    )


def test_float_only():
    df = pd.DataFrame(test_data["float_data"])
    df_equals(df, from_dataframe(df.__dataframe__()))


def test_get_chunks():
    df = pd.DataFrame({"x": np.arange(10)})
    df2 = df.__dataframe__()
    chunk_iter = iter(df2.get_chunks())


def test_metadata():
    df = pd.DataFrame(test_data["int_data"])

    df_metadata = df.__dataframe__().metadata
    expected = {"modin.index": df.index}
    for key in df_metadata:
        assert all(df_metadata[key] == expected[key])
    col_metadata = df.__dataframe__().get_column(0).metadata
    for key in col_metadata:
        assert all(col_metadata[key] == expected[key])
    df_equals(df, from_dataframe(df.__dataframe__()))


def test_missing_from_masked():
    df = pd.DataFrame(
        {
            "x": np.array([1, 2, 3, 4, 0]),
            "y": np.array([1.5, 2.5, 3.5, 4.5, 0]),
            "z": np.array([True, False, True, True, True]),
        }
    )

    df2 = df.__dataframe__()

    # for col_name in df.columns:
        # assert convert_column_to_array(df2.get_column_by_name(col_name) == df[col_name].tolist()
        # assert df[col_name].dtype == convert_column_to_array(df2.get_column_by_name(col_name)).dtype

    rng = np.random.RandomState(42)
    dict_null = {col: rng.randint(low=0, high=len(df)) for col in df.columns}
    for col, num_nulls in dict_null.items():
        null_idx = df.index[
            rng.choice(np.arange(len(df)), size=num_nulls, replace=False)
        ]
        df.loc[null_idx, col] = None

    df2 = df.__dataframe__()

    assert df2.get_column_by_name("x").null_count == dict_null["x"]
    assert df2.get_column_by_name("y").null_count == dict_null["y"]
    assert df2.get_column_by_name("z").null_count == dict_null["z"]


@pytest.mark.parametrize(
    "data",
    [
        {"x": [1.5, 2.5, 3.5], "y": [9.2, 10.5, 11.8]},
        {"x": [1, 2, 0], "y": [9.2, 10.5, 11.8]},
        {
            "x": np.array([True, True, False]),
            "y": np.array([1, 2, 0]),
            "z": np.array([9.2, 10.5, 11.8]),
        },
    ],
)
def test_mixed_data(data):
    df = pd.DataFrame(data)
    df2 = df.__dataframe__()

    for col_name in df.columns:
        # assert convert_column_to_array(df2.get_column_by_name(col_name) == df[col_name].tolist()
        assert df2.get_column_by_name(col_name).null_count == 0


def test_mixed_missing():
    df = pd.DataFrame(
        {
            "x": np.array([True, None, False, None, True]),
            "y": np.array([None, 2, None, 1, 2]),
            "z": np.array([9.2, 10.5, None, 11.8, None]),
        }
    )

    df2 = df.__dataframe__()

    for col_name in df.columns:
        # assert convert_column_to_array(df2.get_column_by_name(col_name) == df[col_name].tolist()
        assert df2.get_column_by_name(col_name).null_count == 2
        # assert df[col_name].dtype == convert_column_to_array(df2.get_column_by_name(col_name)).dtype


def test_object():
    df = pd.DataFrame(test_bool_data)
    col = df.__dataframe__().get_column_by_name("x")

    assert col.size == NROWS

    with pytest.raises(NotImplementedError):
        assert col.dtype
    with pytest.raises(NotImplementedError):
        assert col.describe_null


def test_select_columns_error():
    df = pd.DataFrame(test_data["int_data"])

    df2 = df.__dataframe__()
    
    with pytest.raises(ValueError):
        assert from_dataframe(df2.select_columns(np.array([0, 2]))) == from_dataframe(df2.select_columns_by_name(("col33", "col35")))


def test_select_columns_by_name_error():
    df = pd.DataFrame(test_data["int_data"])

    df2 = df.__dataframe__()
    
    with pytest.raises(ValueError):
        assert from_dataframe(df2.select_columns_by_name(np.array(["col33", "col35"]))) == from_dataframe(df2.select_columns((0, 2)))
    

def test_string():
    test_str_data = test_string_data["separator data"]
    test_str_data.append("")
    df = pd.DataFrame({"A": test_str_data})
    col = df.__dataframe__().get_column_by_name("A")

    assert col.size == 6
    assert col.null_count == 1
    assert col.dtype[0] == DTypeKind.STRING
    if StorageFormat.get() == "Pandas":
        assert col.describe_null == (ColumnNullType.USE_BYTEMASK, 0)
    elif StorageFormat.get() == "Omnisci":
        assert col.describe_null == (ColumnNullType.USE_BYTEMASK, 0)

    df2 = df.__dataframe__()
    #assert convert_column_to_array(df2.get_column_by_name("A") == df["A"].tolist()
    assert df2.get_column_by_name("A").null_count == 1
    if StorageFormat.get() == "Pandas" or StorageFormat.get() == "Omnisci":
        assert df2.get_column_by_name("A").describe_null == (
            ColumnNullType.USE_BYTEMASK,
            0,
        )
    assert df2.get_column_by_name("A").dtype[0] == DTypeKind.STRING

    df_sliced = df[1:]
    col = df_sliced.__dataframe__().get_column_by_name("A")
    assert col.size == 5
    assert col.null_count == 1
    assert col.dtype[0] == DTypeKind.STRING
    if StorageFormat.get() == "Pandas" or StorageFormat.get() == "Omnisci":
        assert col.describe_null == (ColumnNullType.USE_BYTEMASK, 0)

    df2 = df_sliced.__dataframe__()
    #assert convert_column_to_array(df2.get_column_by_name("A") == df_sliced["A"].tolist()
    assert df2.get_column_by_name("A").null_count == 1
    if StorageFormat.get() == "Pandas" or StorageFormat.get() == "Omnisci":
        assert df2.get_column_by_name("A").describe_null == (
            ColumnNullType.USE_BYTEMASK,
            0,
        )
    assert df2.get_column_by_name("A").dtype[0] == DTypeKind.STRING


@pytest.mark.parametrize("num_partitions", [2, 4, 6])
def test_string_dtype(num_partitions):
    NPartitions.put(num_partitions)
    df = pd.DataFrame({"A": (test_string_data["separator data"])*30})

    col = df.__dataframe__().get_column_by_name("A")
    assert col.dtype[0] == DTypeKind.STRING
    if StorageFormat.get() == "Pandas" or StorageFormat.get() == "Omnisci":
        assert col.describe_null == (ColumnNullType.USE_BYTEMASK, 0)
    if StorageFormat.get() == "Pandas":
        assert col.num_chunks() == num_partitions
