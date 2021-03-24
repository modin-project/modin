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

import modin.pandas as pd
from modin.pandas.test.utils import df_equals, df_is_empty
from modin.engines.base.frame.dataframe import DimensionError
import pandas
import numpy as np

import pytest


class TestMap:
    def test_map_index_axis(self):
        values = np.random.rand(2 ** 10, 2 ** 8)
        for i in range(0, 2 ** 8, 2 ** 3):
            values[:, i] = np.nan
        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )
        pandas_df = pandas.DataFrame(values).add_prefix("col")
        replace_vals = {}
        for i in range(0, 2 ** 8, 2 ** 3):
            replace_vals["col{}".format(i)] = pandas_df["col{}".format(i + 1)].mean()

        def fillna(df):
            replace_v = {}
            for c in df.columns:
                if int(c[3:]) % 2 ** 3 == 0:
                    replace_v[c] = df["col{}".format(int(c[3:]) + 1)].mean()
            return df.fillna(replace_v)

        modin_df = modin_frame.map(fillna, axis=0).to_pandas()
        pandas_df.fillna(replace_vals, axis=0, inplace=True)
        df_equals(modin_df, pandas_df)
        modin_arr = modin_frame.map(fillna, axis=1).to_numpy()
        pandas_arr = pandas_df.to_numpy()
        assert not np.array_equal(modin_arr, pandas_arr)

    def test_map_col_axis(self):
        values = np.random.rand(2 ** 10, 2 ** 8)
        for i in range(0, 2 ** 10, 128):
            values[i] = np.nan
        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )

        def fillna(df):
            return (
                df.fillna(0) if df.shape[0] < 2 ** 10 and df.shape[1] == 2 ** 8 else df
            )

        modin_df = modin_frame.map(fillna, axis=1).to_pandas()
        assert not modin_df.isna().values.any()
        modin_df = modin_frame.map(fillna, axis=0).to_pandas()
        assert modin_df.isna().values.any()

    def test_map_no_axis(self):
        values = np.random.rand(2 ** 10, 2 ** 8)
        for i in range(0, 2 ** 10, 128):
            values[i] = np.nan
        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )

        def fillna(df):
            return (
                df.fillna(0) if df.shape[0] < 2 ** 10 and df.shape[1] < 2 ** 8 else df
            )

        modin_df = modin_frame.map(fillna, axis=None).to_pandas()
        assert not modin_df.isna().values.any()
        values = np.random.rand(2 ** 10, 2 ** 8)

        def sub(df):
            return df - 1

        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )
        modin_arr = modin_frame.map(sub, axis=None).to_numpy()
        np.testing.assert_equal(modin_arr, values - 1)

    def test_map_result_schema(self):
        values = np.random.rand(2 ** 10, 2 ** 8)

        def sub(df):
            return df - 1

        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )
        result_schema = {n: values.dtype for n in modin_frame.columns}
        modin_arr = modin_frame.map(
            sub, axis=None, result_schema=result_schema
        ).to_numpy()
        np.testing.assert_equal(modin_arr, values - 1)

        def to_int(df):
            return df.astype(np.int64)

        values *= 100
        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )
        with pytest.raises(TypeError):
            modin_frame.map(to_int, result_schema=result_schema).to_numpy()
        result_schema = {n: np.dtype(np.int64) for n in modin_frame.columns}
        modin_arr = modin_frame.map(
            to_int, axis=None, result_schema=result_schema
        ).to_numpy()
        np.testing.assert_equal(modin_arr, values.astype(np.int64))

    def test_map_shape(self):
        values = np.random.rand(2 ** 10, 2 ** 8)

        def add_col(df):
            df["new_col"] = df[df.columns[0]]
            return df

        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )
        with pytest.raises(DimensionError):
            modin_frame.map(add_col, axis=None).to_numpy()

        def add_row(df):
            return df.append(df)

        with pytest.raises(DimensionError):
            modin_frame.map(add_row, axis=None).to_numpy()

        def del_col(df):
            return df.drop(columns=[df.columns[0]])

        with pytest.raises(DimensionError):
            modin_frame.map(del_col, axis=None).to_numpy()

        def del_row(df):
            return df.drop(df.index[0])

        with pytest.raises(DimensionError):
            modin_frame.map(del_row, axis=None).to_numpy()


class TestFilterByTypes:
    def test_filter_by_types(self):
        values = np.random.rand(2 ** 10, 2 ** 8)
        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )
        new_df = modin_frame.filter_by_types([values.dtype]).to_pandas()
        df_equals(new_df, modin_frame.to_pandas())
        new_df = modin_frame.filter_by_types([np.dtype(np.int64)]).to_pandas()
        df_is_empty(new_df)
        modin_frame = pd.DataFrame(
            [["hello", 32], ["goodbye", 64]]
        )._query_compiler._modin_frame
        new_df = modin_frame.filter_by_types(
            [np.dtype(int), np.dtype(object)]
        ).to_pandas()
        df_equals(new_df, modin_frame.to_pandas())
        modin_df = pd.DataFrame([["hello"], ["goodbye"]])
        new_df = modin_frame.filter_by_types([np.dtype(object)]).to_pandas()
        df_equals(modin_df, new_df)
        modin_df = modin_frame.to_pandas().drop(columns=[0])
        new_df = modin_frame.filter_by_types([np.dtype(int)]).to_pandas()
        df_equals(modin_df, new_df)
        modin_frame = pd.DataFrame(
            [["hello", 32, 32.9], ["goodbye", 64, 64.5]]
        )._query_compiler._modin_frame
        new_df = modin_frame.filter_by_types(
            [np.dtype(float), np.dtype(object)]
        ).to_pandas()
        modin_df = modin_frame.to_pandas().drop(columns=[1])
        df_equals(new_df, modin_df)


class TestFilter:
    def test_filter(self):
        values = np.random.rand(2 ** 10, 2 ** 8)
        pandas_df = pandas.DataFrame(values).add_prefix("col")
        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )
        filtered_df = modin_frame.filter(0, lambda df: df)
        df_equals(modin_frame.to_pandas(), filtered_df.to_pandas())

        filtered_df = modin_frame.filter(1, lambda df: df)
        df_equals(modin_frame.to_pandas(), filtered_df.to_pandas())

        null_df = modin_frame.filter(
            1, lambda df: df.drop(columns=df.columns)
        ).to_pandas()
        df_is_empty(null_df)

        null_df = modin_frame.filter(0, lambda df: df.drop(index=df.index)).to_pandas()
        df_is_empty(null_df)

        even_row_df = modin_frame.filter(
            1, lambda df: df[df.index % 2 == 0]
        ).to_pandas()
        df_equals(even_row_df, pandas_df[pandas_df.index % 2 == 0])

        drop_cols = [c for c in pandas_df.columns if int(c[3:]) % 2 != 0]
        odd_col_df = modin_frame.filter(
            0,
            lambda df: df.drop(columns=[c for c in df.columns if int(c[3:]) % 2 == 1]),
        ).to_pandas()
        df_equals(odd_col_df, pandas_df.drop(columns=drop_cols))


# TODO[Todd]:
# Filtering where all is true - returns the same value
# Filtering where none is true - returns nothing
# Try proper filters
# Try filtering axiswise


class TestReduce:
    def test_reduce_axes(self):
        values = np.random.rand(2 ** 10, 2 ** 8)
        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )
        pandas_df = pandas.DataFrame(values).add_prefix("col")
        with pytest.raises(Exception):
            modin_frame.reduce(0, lambda x: x).to_pandas()
        with pytest.raises(Exception):
            modin_frame.reduce(1, lambda x: x).to_pandas()
        arr = modin_frame.reduce(0, lambda x: x.mean()).to_numpy().flatten()
        np.testing.assert_equal(arr, pandas_df.apply(lambda x: x.mean()).values)
        arr = modin_frame.reduce(1, lambda x: x.mean(axis=1)).to_numpy().flatten()
        np.testing.assert_array_almost_equal(
            arr, pandas_df.apply(lambda x: x.mean(), axis=1).values, decimal=12
        )

    def test_reduce_result_schema(self):
        values = np.random.rand(2 ** 10, 2 ** 8)
        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )
        result_schema = {n: values.dtype for n in modin_frame.columns}
        int_result_schema = {n: np.dtype(np.int64) for n in modin_frame.columns}
        pandas_df = pandas.DataFrame(values).add_prefix("col")
        arr = (
            modin_frame.reduce(0, lambda x: x.sum(), result_schema=result_schema)
            .to_numpy()
            .flatten()
        )
        np.testing.assert_equal(arr, pandas_df.apply(lambda x: x.sum()).values)
        arr = (
            modin_frame.reduce(
                0, lambda x: x.sum().astype(np.int64), result_schema=int_result_schema
            )
            .to_numpy()
            .flatten()
        )
        np.testing.assert_equal(
            arr, pandas_df.apply(lambda x: x.sum().astype(np.int64)).values
        )
        arr = (
            modin_frame.reduce(
                1,
                lambda x: x.sum(axis=1),
                result_schema={"__reduced__": np.dtype(np.float64)},
            )
            .to_numpy()
            .flatten()
        )
        np.testing.assert_equal(arr, pandas_df.apply(sum, axis=1).values)
        arr = (
            modin_frame.reduce(
                1,
                lambda x: x.sum(axis=1).astype(np.int64),
                result_schema={"__reduced__": np.dtype(np.int64)},
            )
            .to_numpy()
            .flatten()
        )
        np.testing.assert_equal(
            arr, pandas_df.apply(lambda x: x.sum().astype(np.int64), axis=1).values
        )
        with pytest.raises(TypeError):
            modin_frame.reduce(0, np.sum, result_schema=int_result_schema).to_numpy()
        with pytest.raises(TypeError):
            modin_frame.reduce(
                0, lambda x: x.sum().astype(np.int64), result_schema=result_schema
            ).to_numpy()
        with pytest.raises(TypeError):
            modin_frame.reduce(
                1,
                lambda x: x.sum(axis=1).astype(np.int64),
                result_schema={"__reduced__": np.dtype(np.float64)},
            ).to_numpy()
        with pytest.raises(TypeError):
            modin_frame.reduce(
                1,
                lambda x: x.sum(axis=1),
                result_schema={"__reduced__": np.dtype(np.int64)},
            ).to_numpy()


class TestTreeReduce:
    def test_tree_reduce(self):
        values = np.random.rand(2 ** 10, 2 ** 8)
        modin_frame = (
            pd.DataFrame(values).add_prefix("col")._query_compiler._modin_frame
        )
        pandas_df = pandas.DataFrame(values).add_prefix("col")
        arr = modin_frame.reduce(0, lambda x: x.sum()).to_numpy().flatten()
        np.testing.assert_equal(arr, pandas_df.apply(lambda x: x.sum()).values)
        arr_tree = modin_frame.tree_reduce(0, lambda x: x.sum()).to_numpy().flatten()
        np.testing.assert_equal(arr, arr_tree)
        arr = modin_frame.reduce(1, lambda x: x.sum(axis=1)).to_numpy().flatten()
        np.testing.assert_array_almost_equal(
            arr, pandas_df.apply(lambda x: x.sum(), axis=1).values, decimal=12
        )
        arr_tree = (
            modin_frame.tree_reduce(1, lambda x: x.sum(axis=1)).to_numpy().flatten()
        )
        np.testing.assert_array_almost_equal(arr, arr_tree, decimal=12)
        arr = modin_frame.reduce(0, lambda x: x.median()).to_numpy().flatten()
        np.testing.assert_equal(arr, pandas_df.apply(lambda x: x.median()).values)
        arr_tree = modin_frame.tree_reduce(0, lambda x: x.median()).to_numpy().flatten()
        assert not np.array_equal(arr, arr_tree)
        arr = modin_frame.reduce(1, lambda x: x.median(axis=1)).to_numpy().flatten()
        np.testing.assert_equal(
            arr, pandas_df.apply(lambda x: x.median(), axis=1).values
        )
        arr_tree = (
            modin_frame.tree_reduce(1, lambda x: x.median(axis=1)).to_numpy().flatten()
        )
        assert not np.array_equal(arr, arr_tree)
