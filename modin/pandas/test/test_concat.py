import numpy as np
import pytest
import pandas

import modin.pandas as pd
from modin.pandas.utils import from_pandas
from .utils import df_equals

pd.DEFAULT_NPARTITIONS = 4


def generate_dfs():
    df = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 10, 11],
            "col4": [12, 13, 14, 15],
            "col5": [0, 0, 0, 0],
        }
    )

    df2 = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 10, 11],
            "col6": [12, 13, 14, 15],
            "col7": [0, 0, 0, 0],
        }
    )
    return df, df2


def generate_none_dfs():
    df = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, None, 7],
            "col3": [8, 9, 10, 11],
            "col4": [12, 13, 14, 15],
            "col5": [None, None, None, None],
        }
    )

    df2 = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 10, 11],
            "col6": [12, 13, 14, 15],
            "col7": [0, 0, 0, 0],
        }
    )
    return df, df2


def test_df_concat():
    df, df2 = generate_dfs()

    df_equals(pd.concat([df, df2]), pandas.concat([df, df2]))


def test_ray_concat():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = from_pandas(df), from_pandas(df2)

    df_equals(pd.concat([modin_df, modin_df2]), pandas.concat([df, df2]))


def test_ray_concat_with_series():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = from_pandas(df), from_pandas(df2)
    pandas_series = pandas.Series([1, 2, 3, 4], name="new_col")

    df_equals(
        pd.concat([modin_df, modin_df2, pandas_series], axis=0),
        pandas.concat([df, df2, pandas_series], axis=0),
    )

    df_equals(
        pd.concat([modin_df, modin_df2, pandas_series], axis=1),
        pandas.concat([df, df2, pandas_series], axis=1),
    )


def test_ray_concat_on_index():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = from_pandas(df), from_pandas(df2)

    df_equals(
        pd.concat([modin_df, modin_df2], axis="index"),
        pandas.concat([df, df2], axis="index"),
    )

    df_equals(
        pd.concat([modin_df, modin_df2], axis="rows"),
        pandas.concat([df, df2], axis="rows"),
    )

    df_equals(
        pd.concat([modin_df, modin_df2], axis=0), pandas.concat([df, df2], axis=0)
    )


def test_ray_concat_on_column():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = from_pandas(df), from_pandas(df2)

    df_equals(
        pd.concat([modin_df, modin_df2], axis=1), pandas.concat([df, df2], axis=1)
    )

    df_equals(
        pd.concat([modin_df, modin_df2], axis="columns"),
        pandas.concat([df, df2], axis="columns"),
    )


def test_invalid_axis_errors():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = from_pandas(df), from_pandas(df2)

    with pytest.raises(ValueError):
        pd.concat([modin_df, modin_df2], axis=2)


def test_mixed_concat():
    df, df2 = generate_dfs()
    df3 = df.copy()

    mixed_dfs = [from_pandas(df), from_pandas(df2), df3]

    df_equals(pd.concat(mixed_dfs), pandas.concat([df, df2, df3]))


def test_mixed_inner_concat():
    df, df2 = generate_dfs()
    df3 = df.copy()

    mixed_dfs = [from_pandas(df), from_pandas(df2), df3]

    df_equals(
        pd.concat(mixed_dfs, join="inner"), pandas.concat([df, df2, df3], join="inner")
    )


def test_mixed_none_concat():
    df, df2 = generate_none_dfs()
    df3 = df.copy()

    mixed_dfs = [from_pandas(df), from_pandas(df2), df3]

    df_equals(pd.concat(mixed_dfs), pandas.concat([df, df2, df3]))


def test_ignore_index_concat():
    df, df2 = generate_dfs()

    df_equals(
        pd.concat([df, df2], ignore_index=True),
        pandas.concat([df, df2], ignore_index=True),
    )


def test_concat_non_subscriptable_keys():
    frame_data = np.random.randint(0, 100, size=(2 ** 10, 2 ** 6))
    df = pd.DataFrame(frame_data).add_prefix("col")
    pdf = pandas.DataFrame(frame_data).add_prefix("col")

    modin_dict = {"c": df.copy(), "b": df.copy()}
    pandas_dict = {"c": pdf.copy(), "b": pdf.copy()}
    modin_result = pd.concat(modin_dict.values(), keys=modin_dict.keys())
    pandas_result = pandas.concat(pandas_dict.values(), keys=pandas_dict.keys())
    df_equals(modin_result, pandas_result)


def test_concat_series_only():
    modin_series = pd.Series(list(range(1000)))
    pandas_series = pandas.Series(list(range(1000)))

    df_equals(
        pd.concat([modin_series, modin_series]),
        pandas.concat([pandas_series, pandas_series]),
    )


def test_concat_with_empty_frame():
    modin_empty_df = pd.DataFrame()
    pandas_empty_df = pandas.DataFrame()
    modin_row = pd.Series({0: "a", 1: "b"})
    pandas_row = pandas.Series({0: "a", 1: "b"})
    df_equals(
        pd.concat([modin_empty_df, modin_row]),
        pandas.concat([pandas_empty_df, pandas_row]),
    )
