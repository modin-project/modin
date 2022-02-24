import enum

import modin.pandas as pd
import numpy as np
import pytest
from modin.pandas.test.utils import df_equals


class _DtypeKind(enum.IntEnum):
    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21
    DATETIME = 22
    CATEGORICAL = 23


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1.5, 2.5, 3.5], "b": [9.2, 10.5, 11.8]},
        {"a": [1, 2, 3], "b": [3, 4, 5], "c": [1.5, 2.5, 3.5], "d": [9, 10, 11]},
    ],
)
def test_float_only(data):
    df = pd.DataFrame(data)
    df_equals(df, df.__dataframe__())


def test_noncontiguous_columns():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df = pd.DataFrame(arr, columns=["a", "b", "c"])
    assert df["a"].to_numpy().strides == (24,)
    df_equals(df, df.__dataframe__())


def test_categorical_dtype_two_columns():
    df = pd.DataFrame({"A": [1, 2, 5, 1]})
    df["B"] = df["A"].astype("category")

    col = df.__dataframe__().get_column_by_name("B")
    assert col.dtype[0] == _DtypeKind.CATEGORICAL
    assert col.null_count == 1
    assert col.describe_null == (2, -1)
    assert col.num_chunks() == 1
    assert col.describe_categorical == (False, True, {0: 1, 1: 2, 2: 5})

    df_equals(df, df.__dataframe__())


def test_string_dtype():
    df = pd.DataFrame({"A": ["a", "b", "cdef", "", "g"]})
    df["B"] = df["A"].astype("object")

    col = df.__dataframe__().get_column_by_name("B")
    assert col.dtype[0] == _DtypeKind.STRING
    assert col.null_count == 1
    assert col.describe_null == (4, 0)
    assert col.num_chunks() == 1


def test_metadata():
    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]})

    df_metadata = df.__dataframe__().metadata
    expected = {"pandas.index": df.index}
    for key in df_metadata:
        assert all(df_metadata[key] == expected[key])
    col_metadata = df.__dataframe__().get_column(0).metadata
    expected = {}
    for key in col_metadata:
        assert col_metadata[key] == expected[key]
    df_equals(df, df.__dataframe__())


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
def test_float_data(data):
    df = pd.DataFrame(data)
    df2 = df.__dataframe__()

    for col_name in df.columns:
        assert df2[col_name].tolist() == df[col_name].tolist()
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
        assert df[col_name].tolist() == df2[col_name].tolist()
        assert not df2[col_name].is_masked
        assert df2.get_column_by_name(col_name).null_count == 2
        assert df[col_name].dtype == df2[col_name].dtype


def test_missing_from_masked():
    df = pd.DataFrame(
        {
            "x": np.array([1, 2, 3, 4, 0]),
            "y": np.array([1.5, 2.5, 3.5, 4.5, 0]),
            "z": np.array([True, False, True, True, True]),
        }
    )

    df2 = df.__dataframe__()

    for col_name in df.columns:
        assert df[col_name].tolist() == df2[col_name].tolist()
        assert not df2[col_name].is_masked
        assert df[col_name].dtype == df2[col_name].dtype

    assert df2.get_column_by_name("x").null_count == 2
    assert df2.get_column_by_name("y").null_count == 3
    assert df2.get_column_by_name("z").null_count == 2


def test_string():
    df = pd.DataFrame({"A": ["a", None, "cdef", "", "g"]})
    col = df.__dataframe__().get_column_by_name("A")

    assert col._col.tolist() == df.A.tolist()
    assert col.size == 5
    assert col.null_count == 1
    assert col.dtype[0] == _DtypeKind.STRING
    assert col.describe_null == (3, 0)

    df2 = df.__dataframe__()
    assert df2.A.tolist() == df.A.tolist()
    assert df2.get_column_by_name("A").null_count == 1
    assert df2.get_column_by_name("A").describe_null == (3, 0)
    assert df2.get_column_by_name("A").dtype[0] == _DtypeKind.STRING

    df_sliced = df[1:]
    col = df_sliced.__dataframe__().get_column_by_name("A")
    assert col.size == 4
    assert col.null_count == 1
    assert col.dtype[0] == _DtypeKind.STRING
    assert col.describe_null == (3, 0)

    df2 = df_sliced.__dataframe__()
    assert df2.A.tolist() == df_sliced.A.tolist()
    assert df2.get_column_by_name("A").null_count == 1
    assert df2.get_column_by_name("A").describe_null == (3, 0)
    assert df2.get_column_by_name("A").dtype[0] == _DtypeKind.STRING


def test_object():
    df = pd.DataFrame({"x": np.array([None, True, False])})
    col = df.__dataframe__().get_column_by_name("x")

    assert col._col.tolist() == df.x.tolist()
    assert col.size == 3

    with pytest.raises(ValueError):
        assert col.dtype
    with pytest.raises(ValueError):
        assert col.describe_null


def test_DataFrame():
    df = pd.DataFrame(
        {
            "x": np.array([True, True, False]),
            "y": np.array([1, 2, 0]),
            "z": np.array([9.2, 10.5, 11.8]),
        }
    )

    df2 = df.__dataframe__()

    assert df2._allow_copy is True
    assert df2.num_columns() == 3
    assert df2.num_rows() == 3
    assert df2.num_chunks() == 1

    assert df2.column_names() == ["x", "y", "z"]
    assert df2.get_column(0)._col.tolist() == df.x.tolist()
    assert df2.get_column_by_name("y")._col.tolist() == df.y.tolist()

    for col in df2.get_columns():
        assert col._col.tolist() == df[col._col.expression].tolist()

    assert (
        df2.select_columns((0, 2))._df[:, 0].tolist()
        == df2.select_columns_by_name(("x", "z"))._df[:, 0].tolist()
    )
    assert (
        df2.select_columns((0, 2))._df[:, 1].tolist()
        == df2.select_columns_by_name(("x", "z"))._df[:, 1].tolist()
    )


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


def test_categorical_dtype():
    df = pd.DataFrame({"A": [1, 2, 5, 1]})
    df["A"].astype("category")
    col = df.__dataframe__().get_column_by_name("A")
    assert col.dtype[0] == _DtypeKind.CATEGORICAL
    assert col.describe_categorical == (False, True, {0: 1, 1: 2, 2: 5})


def test_NA_categorical_dtype():
    df = pd.DataFrame({"A": [1, 2, 5, 1]})
    df["B"] = df["A"]

    col = df.__dataframe__().get_column_by_name("B")
    assert col.dtype[0] == _DtypeKind.CATEGORICAL
    assert col.null_count == 0
    assert col.describe_null == (3, 0)
    assert col.num_chunks() == 1
    assert col.describe_categorical == (False, True, {0: 1, 1: 2, 2: 5})


def test_NA_string_dtype():
    df = pd.DataFrame({"A": ["a", "b", "cdef", "", "g"]})
    df["B"] = df["A"].astype("object")

    col = df.__dataframe__().get_column_by_name("B")
    assert col.dtype[0] == _DtypeKind.STRING
    assert col.null_count == 0
    assert col.describe_null == (3, 0)
    assert col.num_chunks() == 1
