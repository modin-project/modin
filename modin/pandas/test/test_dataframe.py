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

import pytest
import numpy as np
import pandas
import pandas.util.testing as tm
import os
import matplotlib
import modin.pandas as pd
from modin.pandas.utils import to_pandas
from numpy.testing import assert_array_equal
import io
import sys

from .utils import (
    random_state,
    RAND_LOW,
    RAND_HIGH,
    df_equals,
    df_is_empty,
    arg_keys,
    name_contains,
    test_data_values,
    test_data_keys,
    test_data_with_duplicates_values,
    test_data_with_duplicates_keys,
    numeric_dfs,
    no_numeric_dfs,
    test_func_keys,
    test_func_values,
    query_func_keys,
    query_func_values,
    agg_func_keys,
    agg_func_values,
    numeric_agg_funcs,
    quantiles_keys,
    quantiles_values,
    indices_keys,
    indices_values,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    int_arg_keys,
    int_arg_values,
    eval_general,
    create_test_dfs,
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


def eval_insert(modin_df, pandas_df, **kwargs):
    _kwargs = {"loc": 0, "col": "New column"}
    _kwargs.update(kwargs)

    eval_general(
        modin_df,
        pandas_df,
        operation=lambda df, **kwargs: df.insert(**kwargs),
        **_kwargs,
    )


class TestDataFrameBinary:
    def inter_df_math_helper(self, modin_df, pandas_df, op):
        # Test dataframe to dataframe
        try:
            pandas_result = getattr(pandas_df, op)(pandas_df)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(modin_df)
        else:
            modin_result = getattr(modin_df, op)(modin_df)
            df_equals(modin_result, pandas_result)

        # Test dataframe to int
        try:
            pandas_result = getattr(pandas_df, op)(4)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(4)
        else:
            modin_result = getattr(modin_df, op)(4)
            df_equals(modin_result, pandas_result)

        # Test dataframe to float
        try:
            pandas_result = getattr(pandas_df, op)(4.0)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(4.0)
        else:
            modin_result = getattr(modin_df, op)(4.0)
            df_equals(modin_result, pandas_result)

        # Test transposed dataframes to float
        try:
            pandas_result = getattr(pandas_df.T, op)(4.0)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df.T, op)(4.0)
        else:
            modin_result = getattr(modin_df.T, op)(4.0)
            df_equals(modin_result, pandas_result)

        frame_data = {
            "{}_other".format(modin_df.columns[0]): [0, 2],
            modin_df.columns[0]: [0, 19],
            modin_df.columns[1]: [1, 1],
        }
        modin_df2 = pd.DataFrame(frame_data)
        pandas_df2 = pandas.DataFrame(frame_data)

        # Test dataframe to different dataframe shape
        try:
            pandas_result = getattr(pandas_df, op)(pandas_df2)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(modin_df2)
        else:
            modin_result = getattr(modin_df, op)(modin_df2)
            df_equals(modin_result, pandas_result)

        # Test dataframe fill value
        try:
            pandas_result = getattr(pandas_df, op)(pandas_df2, fill_value=0)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(modin_df2, fill_value=0)
        else:
            modin_result = getattr(modin_df, op)(modin_df2, fill_value=0)
            df_equals(modin_result, pandas_result)

        # Test dataframe to list
        list_test = random_state.randint(RAND_LOW, RAND_HIGH, size=(modin_df.shape[1]))
        try:
            pandas_result = getattr(pandas_df, op)(list_test, axis=1)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(list_test, axis=1)
        else:
            modin_result = getattr(modin_df, op)(list_test, axis=1)
            df_equals(modin_result, pandas_result)

        # Test dataframe to series axis=0
        series_test_modin = modin_df[modin_df.columns[0]]
        series_test_pandas = pandas_df[pandas_df.columns[0]]
        try:
            pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=0)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(series_test_modin, axis=0)
        else:
            modin_result = getattr(modin_df, op)(series_test_modin, axis=0)
            df_equals(modin_result, pandas_result)

        # Test dataframe to series axis=1
        series_test_modin = modin_df.iloc[0]
        series_test_pandas = pandas_df.iloc[0]
        try:
            pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=1)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(series_test_modin, axis=1)
        else:
            modin_result = getattr(modin_df, op)(series_test_modin, axis=1)
            df_equals(modin_result, pandas_result)

        # Test dataframe to list axis=1
        series_test_modin = series_test_pandas = list(pandas_df.iloc[0])
        try:
            pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=1)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(series_test_modin, axis=1)
        else:
            modin_result = getattr(modin_df, op)(series_test_modin, axis=1)
            df_equals(modin_result, pandas_result)

        # Test dataframe to list axis=0
        series_test_modin = series_test_pandas = list(pandas_df[pandas_df.columns[0]])
        try:
            pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=0)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(series_test_modin, axis=0)
        else:
            modin_result = getattr(modin_df, op)(series_test_modin, axis=0)
            df_equals(modin_result, pandas_result)

        # Test dataframe to series missing values
        series_test_modin = modin_df.iloc[0, :-2]
        series_test_pandas = pandas_df.iloc[0, :-2]
        try:
            pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=1)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(series_test_modin, axis=1)
        else:
            modin_result = getattr(modin_df, op)(series_test_modin, axis=1)
            df_equals(modin_result, pandas_result)

        # Test dataframe to series with different index
        series_test_modin = modin_df[modin_df.columns[0]].reset_index(drop=True)
        series_test_pandas = pandas_df[pandas_df.columns[0]].reset_index(drop=True)
        try:
            pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=0)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(series_test_modin, axis=0)
        else:
            modin_result = getattr(modin_df, op)(series_test_modin, axis=0)
            df_equals(modin_result, pandas_result)

        # Level test
        new_idx = pandas.MultiIndex.from_tuples(
            [(i // 4, i // 2, i) for i in modin_df.index]
        )
        modin_df_multi_level = modin_df.copy()
        modin_df_multi_level.index = new_idx
        # Defaults to pandas
        with pytest.warns(UserWarning):
            # Operation against self for sanity check
            getattr(modin_df_multi_level, op)(modin_df_multi_level, axis=0, level=1)

    @pytest.mark.parametrize(
        "function",
        [
            "add",
            "div",
            "divide",
            "floordiv",
            "mod",
            "mul",
            "multiply",
            "pow",
            "sub",
            "subtract",
            "truediv",
            "__div__",
            "__add__",
            "__radd__",
            "__mul__",
            "__rmul__",
            "__pow__",
            "__rpow__",
            "__sub__",
            "__floordiv__",
            "__rfloordiv__",
            "__truediv__",
            "__rtruediv__",
            "__mod__",
            "__rmod__",
            "__rdiv__",
        ],
    )
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_math_functions(self, data, function):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        self.inter_df_math_helper(modin_df, pandas_df, function)

    @pytest.mark.parametrize("other", ["as_left", 4, 4.0, "a"])
    @pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_comparison(self, data, op, other):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        eval_general(
            modin_df,
            pandas_df,
            operation=lambda df, **kwargs: getattr(df, op)(
                df if other == "as_left" else other
            ),
        )

    @pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_multi_level_comparison(self, data, op):
        modin_df_multi_level = pd.DataFrame(data)

        new_idx = pandas.MultiIndex.from_tuples(
            [(i // 4, i // 2, i) for i in modin_df_multi_level.index]
        )
        modin_df_multi_level.index = new_idx

        # Defaults to pandas
        with pytest.warns(UserWarning):
            # Operation against self for sanity check
            getattr(modin_df_multi_level, op)(modin_df_multi_level, axis=0, level=1)

    # Test dataframe right operations
    def inter_df_math_right_ops_helper(self, modin_df, pandas_df, op):
        try:
            pandas_result = getattr(pandas_df, op)(4)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(4)
        else:
            modin_result = getattr(modin_df, op)(4)
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = getattr(pandas_df, op)(4.0)
        except Exception as e:
            with pytest.raises(type(e)):
                getattr(modin_df, op)(4.0)
        else:
            modin_result = getattr(modin_df, op)(4.0)
            df_equals(modin_result, pandas_result)

        new_idx = pandas.MultiIndex.from_tuples(
            [(i // 4, i // 2, i) for i in modin_df.index]
        )
        modin_df_multi_level = modin_df.copy()
        modin_df_multi_level.index = new_idx

        # Defaults to pandas
        with pytest.warns(UserWarning):
            # Operation against self for sanity check
            getattr(modin_df_multi_level, op)(modin_df_multi_level, axis=0, level=1)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_radd(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        self.inter_df_math_right_ops_helper(modin_df, pandas_df, "radd")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_rdiv(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        self.inter_df_math_right_ops_helper(modin_df, pandas_df, "rdiv")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_rfloordiv(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        self.inter_df_math_right_ops_helper(modin_df, pandas_df, "rfloordiv")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_rmod(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        self.inter_df_math_right_ops_helper(modin_df, pandas_df, "rmod")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_rmul(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        self.inter_df_math_right_ops_helper(modin_df, pandas_df, "rmul")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_rpow(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        # TODO: Revert to others once we have an efficient way of preprocessing for positive values
        # We need to check that negative integers are not used efficiently
        if "100x100" not in request.node.name:
            self.inter_df_math_right_ops_helper(modin_df, pandas_df, "rpow")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_rsub(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        self.inter_df_math_right_ops_helper(modin_df, pandas_df, "rsub")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_rtruediv(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        self.inter_df_math_right_ops_helper(modin_df, pandas_df, "rtruediv")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___rsub__(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        self.inter_df_math_right_ops_helper(modin_df, pandas_df, "__rsub__")

    # END test dataframe right operations

    def test_equals(self):
        frame_data = {"col1": [2.9, 3, 3, 3], "col2": [2, 3, 4, 1]}
        modin_df1 = pd.DataFrame(frame_data)
        modin_df2 = pd.DataFrame(frame_data)

        assert modin_df1.equals(modin_df2)

        df_equals(modin_df1, modin_df2)
        df_equals(modin_df1, pd.DataFrame(modin_df1))

        frame_data = {"col1": [2.9, 3, 3, 3], "col2": [2, 3, 5, 1]}
        modin_df3 = pd.DataFrame(frame_data, index=list("abcd"))

        assert not modin_df1.equals(modin_df3)

        with pytest.raises(AssertionError):
            df_equals(modin_df3, modin_df1)

        with pytest.raises(AssertionError):
            df_equals(modin_df3, modin_df2)

        assert modin_df1.equals(modin_df2._query_compiler.to_pandas())


class TestDataFrameMapMetadata:
    def test_indexing(self):
        modin_df = pd.DataFrame(
            dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9]), index=["a", "b", "c"]
        )
        pandas_df = pandas.DataFrame(
            dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9]), index=["a", "b", "c"]
        )

        modin_result = modin_df
        pandas_result = pandas_df
        df_equals(modin_result, pandas_result)

        modin_result = modin_df["b"]
        pandas_result = pandas_df["b"]
        df_equals(modin_result, pandas_result)

        modin_result = modin_df[["b"]]
        pandas_result = pandas_df[["b"]]
        df_equals(modin_result, pandas_result)

        modin_result = modin_df[["b", "a"]]
        pandas_result = pandas_df[["b", "a"]]
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.loc["b"]
        pandas_result = pandas_df.loc["b"]
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.loc[["b"]]
        pandas_result = pandas_df.loc[["b"]]
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.loc[["b", "a"]]
        pandas_result = pandas_df.loc[["b", "a"]]
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.loc[["b", "a"], ["a", "c"]]
        pandas_result = pandas_df.loc[["b", "a"], ["a", "c"]]
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.loc[:, ["a", "c"]]
        pandas_result = pandas_df.loc[:, ["a", "c"]]
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.loc[:, ["c"]]
        pandas_result = pandas_df.loc[:, ["c"]]
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.loc[[]]
        pandas_result = pandas_df.loc[[]]
        df_equals(modin_result, pandas_result)

    def test_empty_df(self):
        df = pd.DataFrame(index=["a", "b"])
        df_is_empty(df)
        tm.assert_index_equal(df.index, pd.Index(["a", "b"]))
        assert len(df.columns) == 0

        df = pd.DataFrame(columns=["a", "b"])
        df_is_empty(df)
        assert len(df.index) == 0
        tm.assert_index_equal(df.columns, pd.Index(["a", "b"]))

        df = pd.DataFrame()
        df_is_empty(df)
        assert len(df.index) == 0
        assert len(df.columns) == 0

        df = pd.DataFrame(index=["a", "b"])
        df_is_empty(df)
        tm.assert_index_equal(df.index, pd.Index(["a", "b"]))
        assert len(df.columns) == 0

        df = pd.DataFrame(columns=["a", "b"])
        df_is_empty(df)
        assert len(df.index) == 0
        tm.assert_index_equal(df.columns, pd.Index(["a", "b"]))

        df = pd.DataFrame()
        df_is_empty(df)
        assert len(df.index) == 0
        assert len(df.columns) == 0

        df = pd.DataFrame()
        pd_df = pandas.DataFrame()
        df["a"] = [1, 2, 3, 4, 5]
        pd_df["a"] = [1, 2, 3, 4, 5]
        df_equals(df, pd_df)

        df = pd.DataFrame()
        pd_df = pandas.DataFrame()
        df["a"] = list("ABCDEF")
        pd_df["a"] = list("ABCDEF")
        df_equals(df, pd_df)

        df = pd.DataFrame()
        pd_df = pandas.DataFrame()
        df["a"] = pd.Series([1, 2, 3, 4, 5])
        pd_df["a"] = pandas.Series([1, 2, 3, 4, 5])
        df_equals(df, pd_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_abs(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.abs()
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.abs()
        else:
            modin_result = modin_df.abs()
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_add_prefix(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        test_prefix = "TEST"
        new_modin_df = modin_df.add_prefix(test_prefix)
        new_pandas_df = pandas_df.add_prefix(test_prefix)
        df_equals(new_modin_df.columns, new_pandas_df.columns)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("testfunc", test_func_values, ids=test_func_keys)
    def test_applymap(self, request, data, testfunc):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        with pytest.raises(ValueError):
            x = 2
            modin_df.applymap(x)

        try:
            pandas_result = pandas_df.applymap(testfunc)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.applymap(testfunc)
        else:
            modin_result = modin_df.applymap(testfunc)
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("testfunc", test_func_values, ids=test_func_keys)
    def test_applymap_numeric(self, request, data, testfunc):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if name_contains(request.node.name, numeric_dfs):
            try:
                pandas_result = pandas_df.applymap(testfunc)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.applymap(testfunc)
            else:
                modin_result = modin_df.applymap(testfunc)
                df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_add_suffix(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        test_suffix = "TEST"
        new_modin_df = modin_df.add_suffix(test_suffix)
        new_pandas_df = pandas_df.add_suffix(test_suffix)

        df_equals(new_modin_df.columns, new_pandas_df.columns)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_at(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        # We skip nan datasets because nan != nan
        if "nan" not in request.node.name:
            key1 = modin_df.columns[0]
            # Scaler
            assert modin_df.at[0, key1] == pandas_df.at[0, key1]

            # Series
            df_equals(modin_df.loc[0].at[key1], pandas_df.loc[0].at[key1])

            # Write Item
            modin_df_copy = modin_df.copy()
            pandas_df_copy = pandas_df.copy()
            modin_df_copy.at[1, key1] = modin_df.at[0, key1]
            pandas_df_copy.at[1, key1] = pandas_df.at[0, key1]
            df_equals(modin_df_copy, pandas_df_copy)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_axes(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        for modin_axis, pd_axis in zip(modin_df.axes, pandas_df.axes):
            assert np.array_equal(modin_axis, pd_axis)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_copy(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)  # noqa F841

        # pandas_df is unused but there so there won't be confusing list comprehension
        # stuff in the pytest.mark.parametrize
        new_modin_df = modin_df.copy()

        assert new_modin_df is not modin_df
        assert np.array_equal(
            new_modin_df._query_compiler._modin_frame._partitions,
            modin_df._query_compiler._modin_frame._partitions,
        )
        assert new_modin_df is not modin_df
        df_equals(new_modin_df, modin_df)

        # Shallow copy tests
        modin_df = pd.DataFrame(data)
        modin_df_cp = modin_df.copy(False)

        modin_df[modin_df.columns[0]] = 0
        df_equals(modin_df, modin_df_cp)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_dtypes(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(modin_df.dtypes, pandas_df.dtypes)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("key", indices_values, ids=indices_keys)
    def test_get(self, data, key):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(modin_df.get(key), pandas_df.get(key))
        df_equals(
            modin_df.get(key, default="default"), pandas_df.get(key, default="default")
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize(
        "dummy_na", bool_arg_values, ids=arg_keys("dummy_na", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "drop_first", bool_arg_values, ids=arg_keys("drop_first", bool_arg_keys)
    )
    def test_get_dummies(self, request, data, dummy_na, drop_first):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas.get_dummies(
                pandas_df, dummy_na=dummy_na, drop_first=drop_first
            )
        except Exception as e:
            with pytest.raises(type(e)):
                pd.get_dummies(modin_df, dummy_na=dummy_na, drop_first=drop_first)
        else:
            modin_result = pd.get_dummies(
                modin_df, dummy_na=dummy_na, drop_first=drop_first
            )
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_isna(self, data):
        pandas_df = pandas.DataFrame(data)
        modin_df = pd.DataFrame(data)

        pandas_result = pandas_df.isna()
        modin_result = modin_df.isna()

        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_isnull(self, data):
        pandas_df = pandas.DataFrame(data)
        modin_df = pd.DataFrame(data)

        pandas_result = pandas_df.isnull()
        modin_result = modin_df.isnull()

        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_append(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        data_to_append = {"append_a": 2, "append_b": 1000}

        ignore_idx_values = [True, False]

        for ignore in ignore_idx_values:
            try:
                pandas_result = pandas_df.append(data_to_append, ignore_index=ignore)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.append(data_to_append, ignore_index=ignore)
            else:
                modin_result = modin_df.append(data_to_append, ignore_index=ignore)
                df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.append(pandas_df.iloc[-1])
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.append(modin_df.iloc[-1])
        else:
            modin_result = modin_df.append(modin_df.iloc[-1])
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.append(list(pandas_df.iloc[-1]))
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.append(list(modin_df.iloc[-1]))
        else:
            modin_result = modin_df.append(list(modin_df.iloc[-1]))
            df_equals(modin_result, pandas_result)

        verify_integrity_values = [True, False]

        for verify_integrity in verify_integrity_values:
            try:
                pandas_result = pandas_df.append(
                    [pandas_df, pandas_df], verify_integrity=verify_integrity
                )
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.append(
                        [modin_df, modin_df], verify_integrity=verify_integrity
                    )
            else:
                modin_result = modin_df.append(
                    [modin_df, modin_df], verify_integrity=verify_integrity
                )
                df_equals(modin_result, pandas_result)

            try:
                pandas_result = pandas_df.append(
                    pandas_df, verify_integrity=verify_integrity
                )
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.append(modin_df, verify_integrity=verify_integrity)
            else:
                modin_result = modin_df.append(
                    modin_df, verify_integrity=verify_integrity
                )
                df_equals(modin_result, pandas_result)

    def test_astype(self):
        td = pandas.DataFrame(tm.getSeriesData())
        modin_df = pd.DataFrame(td.values, index=td.index, columns=td.columns)
        expected_df = pandas.DataFrame(td.values, index=td.index, columns=td.columns)

        modin_df_casted = modin_df.astype(np.int32)
        expected_df_casted = expected_df.astype(np.int32)
        df_equals(modin_df_casted, expected_df_casted)

        modin_df_casted = modin_df.astype(np.float64)
        expected_df_casted = expected_df.astype(np.float64)
        df_equals(modin_df_casted, expected_df_casted)

        modin_df_casted = modin_df.astype(str)
        expected_df_casted = expected_df.astype(str)
        df_equals(modin_df_casted, expected_df_casted)

        modin_df_casted = modin_df.astype("category")
        expected_df_casted = expected_df.astype("category")
        df_equals(modin_df_casted, expected_df_casted)

        dtype_dict = {"A": np.int32, "B": np.int64, "C": str}
        modin_df_casted = modin_df.astype(dtype_dict)
        expected_df_casted = expected_df.astype(dtype_dict)
        df_equals(modin_df_casted, expected_df_casted)

        # Ignore lint because this is testing bad input
        bad_dtype_dict = {"B": np.int32, "B": np.int64, "B": str}  # noqa F601
        modin_df_casted = modin_df.astype(bad_dtype_dict)
        expected_df_casted = expected_df.astype(bad_dtype_dict)
        df_equals(modin_df_casted, expected_df_casted)

        modin_df = pd.DataFrame(index=["row1"], columns=["col1"])
        modin_df["col1"]["row1"] = 11
        modin_df_casted = modin_df.astype(int)
        expected_df = pandas.DataFrame(index=["row1"], columns=["col1"])
        expected_df["col1"]["row1"] = 11
        expected_df_casted = expected_df.astype(int)
        df_equals(modin_df_casted, expected_df_casted)

        with pytest.raises(KeyError):
            modin_df.astype({"not_exists": np.uint8})

    def test_astype_category(self):
        modin_df = pd.DataFrame(
            {"col1": ["A", "A", "B", "B", "A"], "col2": [1, 2, 3, 4, 5]}
        )
        pandas_df = pandas.DataFrame(
            {"col1": ["A", "A", "B", "B", "A"], "col2": [1, 2, 3, 4, 5]}
        )

        modin_result = modin_df.astype({"col1": "category"})
        pandas_result = pandas_df.astype({"col1": "category"})
        df_equals(modin_result, pandas_result)
        assert modin_result.dtypes.equals(pandas_result.dtypes)

        modin_result = modin_df.astype("category")
        pandas_result = pandas_df.astype("category")
        df_equals(modin_result, pandas_result)
        assert modin_result.dtypes.equals(pandas_result.dtypes)

    @pytest.mark.xfail(
        reason="Categorical dataframe created in memory don't work yet and categorical dtype is lost"
    )
    def test_astype_category_large(self):
        series_length = 10_000
        modin_df = pd.DataFrame(
            {
                "col1": ["str{0}".format(i) for i in range(0, series_length)],
                "col2": [i for i in range(0, series_length)],
            }
        )
        pandas_df = pandas.DataFrame(
            {
                "col1": ["str{0}".format(i) for i in range(0, series_length)],
                "col2": [i for i in range(0, series_length)],
            }
        )

        modin_result = modin_df.astype({"col1": "category"})
        pandas_result = pandas_df.astype({"col1": "category"})
        df_equals(modin_result, pandas_result)
        assert modin_result.dtypes.equals(pandas_result.dtypes)

        modin_result = modin_df.astype("category")
        pandas_result = pandas_df.astype("category")
        df_equals(modin_result, pandas_result)
        assert modin_result.dtypes.equals(pandas_result.dtypes)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    def test_clip(self, request, data, axis):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if name_contains(request.node.name, numeric_dfs):
            ind_len = (
                len(modin_df.index)
                if not pandas.DataFrame()._get_axis_number(axis)
                else len(modin_df.columns)
            )
            # set bounds
            lower, upper = np.sort(random_state.random_integers(RAND_LOW, RAND_HIGH, 2))
            lower_list = random_state.random_integers(RAND_LOW, RAND_HIGH, ind_len)
            upper_list = random_state.random_integers(RAND_LOW, RAND_HIGH, ind_len)

            # test only upper scalar bound
            modin_result = modin_df.clip(None, upper, axis=axis)
            pandas_result = pandas_df.clip(None, upper, axis=axis)
            df_equals(modin_result, pandas_result)

            # test lower and upper scalar bound
            modin_result = modin_df.clip(lower, upper, axis=axis)
            pandas_result = pandas_df.clip(lower, upper, axis=axis)
            df_equals(modin_result, pandas_result)

            # test lower and upper list bound on each column
            modin_result = modin_df.clip(lower_list, upper_list, axis=axis)
            pandas_result = pandas_df.clip(lower_list, upper_list, axis=axis)
            df_equals(modin_result, pandas_result)

            # test only upper list bound on each column
            modin_result = modin_df.clip(np.nan, upper_list, axis=axis)
            pandas_result = pandas_df.clip(np.nan, upper_list, axis=axis)
            df_equals(modin_result, pandas_result)

            with pytest.raises(ValueError):
                modin_df.clip(lower=[1, 2, 3], axis=None)

    def test_drop(self):
        frame_data = {"A": [1, 2, 3, 4], "B": [0, 1, 2, 3]}
        simple = pandas.DataFrame(frame_data)
        modin_simple = pd.DataFrame(frame_data)
        df_equals(modin_simple.drop("A", axis=1), simple[["B"]])
        df_equals(modin_simple.drop(["A", "B"], axis="columns"), simple[[]])
        df_equals(modin_simple.drop([0, 1, 3], axis=0), simple.loc[[2], :])
        df_equals(modin_simple.drop([0, 3], axis="index"), simple.loc[[1, 2], :])

        pytest.raises(ValueError, modin_simple.drop, 5)
        pytest.raises(ValueError, modin_simple.drop, "C", 1)
        pytest.raises(ValueError, modin_simple.drop, [1, 5])
        pytest.raises(ValueError, modin_simple.drop, ["A", "C"], 1)

        # errors = 'ignore'
        df_equals(modin_simple.drop(5, errors="ignore"), simple)
        df_equals(modin_simple.drop([0, 5], errors="ignore"), simple.loc[[1, 2, 3], :])
        df_equals(modin_simple.drop("C", axis=1, errors="ignore"), simple)
        df_equals(modin_simple.drop(["A", "C"], axis=1, errors="ignore"), simple[["B"]])

        # non-unique
        nu_df = pandas.DataFrame(
            zip(range(3), range(-3, 1), list("abc")), columns=["a", "a", "b"]
        )
        modin_nu_df = pd.DataFrame(nu_df)
        df_equals(modin_nu_df.drop("a", axis=1), nu_df[["b"]])
        df_equals(modin_nu_df.drop("b", axis="columns"), nu_df["a"])
        df_equals(modin_nu_df.drop([]), nu_df)

        nu_df = nu_df.set_index(pandas.Index(["X", "Y", "X"]))
        nu_df.columns = list("abc")
        modin_nu_df = pd.DataFrame(nu_df)
        df_equals(modin_nu_df.drop("X", axis="rows"), nu_df.loc[["Y"], :])
        df_equals(modin_nu_df.drop(["X", "Y"], axis=0), nu_df.loc[[], :])

        # inplace cache issue
        frame_data = random_state.randn(10, 3)
        df = pandas.DataFrame(frame_data, columns=list("abc"))
        modin_df = pd.DataFrame(frame_data, columns=list("abc"))
        expected = df[~(df.b > 0)]
        modin_df.drop(labels=df[df.b > 0].index, inplace=True)
        df_equals(modin_df, expected)

        midx = pd.MultiIndex(
            levels=[["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        df = pd.DataFrame(
            index=midx,
            columns=["big", "small"],
            data=[
                [45, 30],
                [200, 100],
                [1.5, 1],
                [30, 20],
                [250, 150],
                [1.5, 0.8],
                [320, 250],
                [1, 0.8],
                [0.3, 0.2],
            ],
        )
        with pytest.warns(UserWarning):
            df.drop(index="length", level=1)

    def test_drop_api_equivalence(self):
        # equivalence of the labels/axis and index/columns API's
        frame_data = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]

        modin_df = pd.DataFrame(
            frame_data, index=["a", "b", "c"], columns=["d", "e", "f"]
        )

        modin_df1 = modin_df.drop("a")
        modin_df2 = modin_df.drop(index="a")
        df_equals(modin_df1, modin_df2)

        modin_df1 = modin_df.drop("d", 1)
        modin_df2 = modin_df.drop(columns="d")
        df_equals(modin_df1, modin_df2)

        modin_df1 = modin_df.drop(labels="e", axis=1)
        modin_df2 = modin_df.drop(columns="e")
        df_equals(modin_df1, modin_df2)

        modin_df1 = modin_df.drop(["a"], axis=0)
        modin_df2 = modin_df.drop(index=["a"])
        df_equals(modin_df1, modin_df2)

        modin_df1 = modin_df.drop(["a"], axis=0).drop(["d"], axis=1)
        modin_df2 = modin_df.drop(index=["a"], columns=["d"])
        df_equals(modin_df1, modin_df2)

        with pytest.raises(ValueError):
            modin_df.drop(labels="a", index="b")

        with pytest.raises(ValueError):
            modin_df.drop(labels="a", columns="b")

        with pytest.raises(ValueError):
            modin_df.drop(axis=1)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_drop_transpose(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        modin_result = modin_df.T.drop(columns=[0, 1, 2])
        pandas_result = pandas_df.T.drop(columns=[0, 1, 2])
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.T.drop(index=["col3", "col1"])
        pandas_result = pandas_df.T.drop(index=["col3", "col1"])
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.T.drop(columns=[0, 1, 2], index=["col3", "col1"])
        pandas_result = pandas_df.T.drop(columns=[0, 1, 2], index=["col3", "col1"])
        df_equals(modin_result, pandas_result)

    def test_droplevel(self):
        df = (
            pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
            .set_index([0, 1])
            .rename_axis(["a", "b"])
        )
        df.columns = pd.MultiIndex.from_tuples(
            [("c", "e"), ("d", "f")], names=["level_1", "level_2"]
        )
        df.droplevel("a")
        df.droplevel("level_2", axis=1)

    @pytest.mark.parametrize(
        "data", test_data_with_duplicates_values, ids=test_data_with_duplicates_keys
    )
    @pytest.mark.parametrize(
        "keep", ["last", "first", False], ids=["last", "first", "False"]
    )
    @pytest.mark.parametrize(
        "subset",
        [None, "col1", "name", ("col1", "col3"), ["col1", "col3", "col7"]],
        ids=["None", "string", "name", "tuple", "list"],
    )
    def test_drop_duplicates(self, data, keep, subset):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_df.drop_duplicates(keep=keep, inplace=False, subset=subset)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.drop_duplicates(keep=keep, inplace=False, subset=subset)
        else:
            df_equals(
                pandas_df.drop_duplicates(keep=keep, inplace=False, subset=subset),
                modin_df.drop_duplicates(keep=keep, inplace=False, subset=subset),
            )

        try:
            pandas_results = pandas_df.drop_duplicates(
                keep=keep, inplace=True, subset=subset
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.drop_duplicates(keep=keep, inplace=True, subset=subset)
        else:
            modin_results = modin_df.drop_duplicates(
                keep=keep, inplace=True, subset=subset
            )
            df_equals(modin_results, pandas_results)

    def test_drop_duplicates_with_missing_index_values(self):
        data = {
            "columns": ["value", "time", "id"],
            "index": [
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
            ],
            "data": [
                ["3", 1279213398000.0, 88.0],
                ["3", 1279204682000.0, 88.0],
                ["0", 1245772835000.0, 448.0],
                ["0", 1270564258000.0, 32.0],
                ["0", 1267106669000.0, 118.0],
                ["7", 1300621123000.0, 5.0],
                ["0", 1251130752000.0, 957.0],
                ["0", 1311683506000.0, 62.0],
                ["9", 1283692698000.0, 89.0],
                ["9", 1270234253000.0, 64.0],
                ["0", 1285088818000.0, 50.0],
                ["0", 1218212725000.0, 695.0],
                ["2", 1383933968000.0, 348.0],
                ["0", 1368227625000.0, 257.0],
                ["1", 1454514093000.0, 446.0],
                ["1", 1428497427000.0, 134.0],
                ["1", 1459184936000.0, 568.0],
                ["1", 1502293302000.0, 599.0],
                ["1", 1491833358000.0, 829.0],
                ["1", 1485431534000.0, 806.0],
                ["8", 1351800505000.0, 101.0],
                ["0", 1357247721000.0, 916.0],
                ["0", 1335804423000.0, 370.0],
                ["24", 1327547726000.0, 720.0],
                ["0", 1332334140000.0, 415.0],
                ["0", 1309543100000.0, 30.0],
                ["18", 1309541141000.0, 30.0],
                ["0", 1298979435000.0, 48.0],
                ["14", 1276098160000.0, 59.0],
                ["0", 1233936302000.0, 109.0],
            ],
        }

        pandas_df = pandas.DataFrame(
            data["data"], index=data["index"], columns=data["columns"]
        )
        modin_df = pd.DataFrame(
            data["data"], index=data["index"], columns=data["columns"]
        )
        modin_result = modin_df.sort_values(["id", "time"]).drop_duplicates(["id"])
        pandas_result = pandas_df.sort_values(["id", "time"]).drop_duplicates(["id"])
        df_equals(modin_result, pandas_result)

    def test_drop_duplicates_after_sort(self):
        data = [
            {"value": 1, "time": 2},
            {"value": 1, "time": 1},
            {"value": 2, "time": 1},
            {"value": 2, "time": 2},
        ]
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_result = modin_df.sort_values(["value", "time"]).drop_duplicates(
            ["value"]
        )
        pandas_result = pandas_df.sort_values(["value", "time"]).drop_duplicates(
            ["value"]
        )
        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize("how", ["any", "all"], ids=["any", "all"])
    def test_dropna(self, data, axis, how):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        with pytest.raises(ValueError):
            modin_df.dropna(axis=axis, how="invalid")

        with pytest.raises(TypeError):
            modin_df.dropna(axis=axis, how=None, thresh=None)

        with pytest.raises(KeyError):
            modin_df.dropna(axis=axis, subset=["NotExists"], how=how)

        modin_result = modin_df.dropna(axis=axis, how=how)
        pandas_result = pandas_df.dropna(axis=axis, how=how)
        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_dropna_inplace(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        pandas_result = pandas_df.dropna()
        modin_df.dropna(inplace=True)
        df_equals(modin_df, pandas_result)

        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        pandas_df.dropna(thresh=2, inplace=True)
        modin_df.dropna(thresh=2, inplace=True)
        df_equals(modin_df, pandas_df)

        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        pandas_df.dropna(axis=1, how="any", inplace=True)
        modin_df.dropna(axis=1, how="any", inplace=True)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_dropna_multiple_axes(self, data):
        modin_df = pd.DataFrame(data)

        with pytest.raises(TypeError):
            modin_df.dropna(how="all", axis=[0, 1])
        with pytest.raises(TypeError):
            modin_df.dropna(how="all", axis=(0, 1))

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_dropna_subset(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if "empty_data" not in request.node.name:
            column_subset = modin_df.columns[0:2]
            df_equals(
                modin_df.dropna(how="all", subset=column_subset),
                pandas_df.dropna(how="all", subset=column_subset),
            )
            df_equals(
                modin_df.dropna(how="any", subset=column_subset),
                pandas_df.dropna(how="any", subset=column_subset),
            )

            row_subset = modin_df.index[0:2]
            df_equals(
                modin_df.dropna(how="all", axis=1, subset=row_subset),
                pandas_df.dropna(how="all", axis=1, subset=row_subset),
            )
            df_equals(
                modin_df.dropna(how="any", axis=1, subset=row_subset),
                pandas_df.dropna(how="any", axis=1, subset=row_subset),
            )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_dropna_subset_error(self, data):
        modin_df = pd.DataFrame(data)

        # pandas_df is unused so there won't be confusing list comprehension
        # stuff in the pytest.mark.parametrize
        with pytest.raises(KeyError):
            modin_df.dropna(subset=list("EF"))

        if len(modin_df.columns) < 5:
            with pytest.raises(KeyError):
                modin_df.dropna(axis=1, subset=[4, 5])

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize(
        "astype",
        [
            "category",
            pytest.param(
                "int32",
                marks=pytest.mark.xfail(
                    reason="Modin astype() does not raises ValueError at non-numeric argument when Pandas does."
                ),
            ),
            "float",
        ],
    )
    def test_insert_dtypes(self, data, astype):
        modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

        # categories with NaN works incorrect for now
        if astype == "category" and pandas_df.iloc[:, 0].isnull().any():
            return

        eval_insert(
            modin_df,
            pandas_df,
            col="TypeSaver",
            value=lambda df: df.iloc[:, 0].astype(astype),
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("loc", int_arg_values, ids=arg_keys("loc", int_arg_keys))
    def test_insert_loc(self, data, loc):
        modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
        value = modin_df.iloc[:, 0]

        eval_insert(modin_df, pandas_df, loc=loc, value=value)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_insert(self, data):
        modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

        eval_insert(
            modin_df, pandas_df, col="Duplicate", value=lambda df: df[df.columns[0]]
        )
        eval_insert(modin_df, pandas_df, col="Scalar", value=100)
        eval_insert(
            pd.DataFrame(columns=list("ab")),
            pandas.DataFrame(columns=list("ab")),
            col=lambda df: df.columns[0],
            value=lambda df: df[df.columns[0]],
        )
        eval_insert(
            pd.DataFrame(index=modin_df.index),
            pandas.DataFrame(index=pandas_df.index),
            col=lambda df: df.columns[0],
            value=lambda df: df[df.columns[0]],
        )
        eval_insert(
            modin_df,
            pandas_df,
            col="DataFrame insert",
            value=lambda df: df[[df.columns[0]]],
        )

        # Bad inserts
        eval_insert(modin_df, pandas_df, col="Bad Column", value=lambda df: df)
        eval_insert(
            modin_df,
            pandas_df,
            col="Too Short",
            value=lambda df: list(df[df.columns[0]])[:-1],
        )
        eval_insert(
            modin_df,
            pandas_df,
            col=lambda df: df.columns[0],
            value=lambda df: df[df.columns[0]],
        )
        eval_insert(
            modin_df,
            pandas_df,
            loc=lambda df: len(df.columns) + 100,
            col="Bad Loc",
            value=100,
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_ndim(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        assert modin_df.ndim == pandas_df.ndim

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_notna(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(modin_df.notna(), pandas_df.notna())

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_notnull(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(modin_df.notnull(), pandas_df.notnull())

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_round(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(modin_df.round(), pandas_df.round())
        df_equals(modin_df.round(1), pandas_df.round(1))

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    def test_set_axis(self, data, axis):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        x = pandas.DataFrame()._get_axis_number(axis)
        index = modin_df.columns if x else modin_df.index
        labels = ["{0}_{1}".format(index[i], i) for i in range(modin_df.shape[x])]

        modin_result = modin_df.set_axis(labels, axis=axis, inplace=False)
        pandas_result = pandas_df.set_axis(labels, axis=axis, inplace=False)
        df_equals(modin_result, pandas_result)

        modin_df_copy = modin_df.copy()
        modin_df.set_axis(labels, axis=axis, inplace=True)

        # Check that the copy and original are different
        try:
            df_equals(modin_df, modin_df_copy)
        except AssertionError:
            assert True
        else:
            assert False

        pandas_df.set_axis(labels, axis=axis, inplace=True)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize(
        "drop", bool_arg_values, ids=arg_keys("drop", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "append", bool_arg_values, ids=arg_keys("append", bool_arg_keys)
    )
    def test_set_index(self, request, data, drop, append):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if "empty_data" not in request.node.name:
            key = modin_df.columns[0]
            modin_result = modin_df.set_index(
                key, drop=drop, append=append, inplace=False
            )
            pandas_result = pandas_df.set_index(
                key, drop=drop, append=append, inplace=False
            )
            df_equals(modin_result, pandas_result)

            modin_df_copy = modin_df.copy()
            modin_df.set_index(key, drop=drop, append=append, inplace=True)

            # Check that the copy and original are different
            try:
                df_equals(modin_df, modin_df_copy)
            except AssertionError:
                assert True
            else:
                assert False

            pandas_df.set_index(key, drop=drop, append=append, inplace=True)
            df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_shape(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        assert modin_df.shape == pandas_df.shape

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_size(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        assert modin_df.size == pandas_df.size

    def test_squeeze(self):
        frame_data = {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 10, 11],
            "col4": [12, 13, 14, 15],
            "col5": [0, 0, 0, 0],
        }
        frame_data_2 = {"col1": [0, 1, 2, 3]}
        frame_data_3 = {
            "col1": [0],
            "col2": [4],
            "col3": [8],
            "col4": [12],
            "col5": [0],
        }
        frame_data_4 = {"col1": [2]}
        frame_data_5 = {"col1": ["string"]}
        # Different data for different cases
        pandas_df = pandas.DataFrame(frame_data).squeeze()
        modin_df = pd.DataFrame(frame_data).squeeze()
        df_equals(modin_df, pandas_df)

        pandas_df_2 = pandas.DataFrame(frame_data_2).squeeze()
        modin_df_2 = pd.DataFrame(frame_data_2).squeeze()
        df_equals(modin_df_2, pandas_df_2)

        pandas_df_3 = pandas.DataFrame(frame_data_3).squeeze()
        modin_df_3 = pd.DataFrame(frame_data_3).squeeze()
        df_equals(modin_df_3, pandas_df_3)

        pandas_df_4 = pandas.DataFrame(frame_data_4).squeeze()
        modin_df_4 = pd.DataFrame(frame_data_4).squeeze()
        df_equals(modin_df_4, pandas_df_4)

        pandas_df_5 = pandas.DataFrame(frame_data_5).squeeze()
        modin_df_5 = pd.DataFrame(frame_data_5).squeeze()
        df_equals(modin_df_5, pandas_df_5)

        data = [
            [
                pd.Timestamp("2019-01-02"),
                pd.Timestamp("2019-01-03"),
                pd.Timestamp("2019-01-04"),
                pd.Timestamp("2019-01-05"),
            ],
            [1, 1, 1, 2],
        ]
        df = pd.DataFrame(data, index=["date", "value"]).T
        pf = pandas.DataFrame(data, index=["date", "value"]).T
        df.set_index("date", inplace=True)
        pf.set_index("date", inplace=True)
        df_equals(df.iloc[0], pf.iloc[0])

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_transpose(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(modin_df.T, pandas_df.T)
        df_equals(modin_df.transpose(), pandas_df.transpose())

        # Test for map across full axis for select indices
        df_equals(modin_df.T.dropna(), pandas_df.T.dropna())
        # Test for map across full axis
        df_equals(modin_df.T.nunique(), pandas_df.T.nunique())
        # Test for map across blocks
        df_equals(modin_df.T.notna(), pandas_df.T.notna())

    @pytest.mark.parametrize(
        "data, other_data",
        [
            ({"A": [1, 2, 3], "B": [400, 500, 600]}, {"B": [4, 5, 6], "C": [7, 8, 9]}),
            (
                {"A": ["a", "b", "c"], "B": ["x", "y", "z"]},
                {"B": ["d", "e", "f", "g", "h", "i"]},
            ),
            ({"A": [1, 2, 3], "B": [400, 500, 600]}, {"B": [4, np.nan, 6]}),
        ],
    )
    def test_update(self, data, other_data):
        modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
        other_modin_df, other_pandas_df = (
            pd.DataFrame(other_data),
            pandas.DataFrame(other_data),
        )
        modin_df.update(other_modin_df)
        pandas_df.update(other_pandas_df)
        df_equals(modin_df, pandas_df)

        with pytest.raises(ValueError):
            modin_df.update(other_modin_df, errors="raise")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___neg__(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.__neg__()
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.__neg__()
        else:
            modin_result = modin_df.__neg__()
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___invert__(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        try:
            pandas_result = ~pandas_df
        except Exception as e:
            with pytest.raises(type(e)):
                repr(~modin_df)
        else:
            modin_result = ~modin_df
            df_equals(modin_result, pandas_result)

    def test___hash__(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            try:
                pd.DataFrame(data).__hash__()
            except TypeError:
                pass

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___delitem__(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if "empty_data" not in request.node.name:
            key = pandas_df.columns[0]

            modin_df = modin_df.copy()
            pandas_df = pandas_df.copy()
            modin_df.__delitem__(key)
            pandas_df.__delitem__(key)
            df_equals(modin_df, pandas_df)

            # Issue 2027
            last_label = pandas_df.iloc[:, -1].name
            modin_df.__delitem__(last_label)
            pandas_df.__delitem__(last_label)
            df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___nonzero__(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)  # noqa F841

        with pytest.raises(ValueError):
            # Always raises ValueError
            modin_df.__nonzero__()

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___abs__(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = abs(pandas_df)
        except Exception as e:
            with pytest.raises(type(e)):
                abs(modin_df)
        else:
            modin_result = abs(modin_df)
            df_equals(modin_result, pandas_result)

    def test___round__(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).__round__()


class TestDataFrameUDF:
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
    def test_agg(self, data, axis, func):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.agg(func, axis)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.agg(func, axis)
        else:
            modin_result = modin_df.agg(func, axis)
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
    def test_agg_numeric(self, request, data, axis, func):
        if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
            request.node.name, numeric_dfs
        ):
            modin_df = pd.DataFrame(data)
            pandas_df = pandas.DataFrame(data)

            try:
                pandas_result = pandas_df.agg(func, axis)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.agg(func, axis)
            else:
                modin_result = modin_df.agg(func, axis)
                df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
    def test_aggregate(self, request, data, func, axis):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.aggregate(func, axis)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.aggregate(func, axis)
        else:
            modin_result = modin_df.aggregate(func, axis)
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
    def test_aggregate_numeric(self, request, data, axis, func):
        if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
            request.node.name, numeric_dfs
        ):
            modin_df = pd.DataFrame(data)
            pandas_df = pandas.DataFrame(data)

            try:
                pandas_result = pandas_df.agg(func, axis)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.agg(func, axis)
            else:
                modin_result = modin_df.agg(func, axis)
                df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_aggregate_error_checking(self, data):
        modin_df = pd.DataFrame(data)

        assert modin_df.aggregate("ndim") == 2

        with pytest.warns(UserWarning):
            modin_df.aggregate(
                {modin_df.columns[0]: "sum", modin_df.columns[1]: "mean"}
            )

        with pytest.warns(UserWarning):
            modin_df.aggregate("cumproduct")

        with pytest.raises(ValueError):
            modin_df.aggregate("NOT_EXISTS")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
    def test_apply(self, request, data, func, axis):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        with pytest.raises(TypeError):
            modin_df.apply({"row": func}, axis=1)

        try:
            pandas_result = pandas_df.apply(func, axis)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.apply(func, axis)
        else:
            modin_result = modin_df.apply(func, axis)
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("level", [None, -1, 0, 1])
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("func", ["count", "sum", "mean", "all", "kurt"])
    def test_apply_text_func_with_level(self, level, data, func, axis):
        func_kwargs = {"level": level, "axis": axis}
        rows_number = len(next(iter(data.values())))  # length of the first data column
        level_0 = np.random.choice([0, 1, 2], rows_number)
        level_1 = np.random.choice([3, 4, 5], rows_number)
        index = pd.MultiIndex.from_arrays([level_0, level_1])

        eval_general(
            pd.DataFrame(data, index=index),
            pandas.DataFrame(data, index=index),
            lambda df, *args, **kwargs: df.apply(func, *args, **kwargs),
            **func_kwargs,
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    def test_apply_args(self, data, axis):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        def apply_func(series, y):
            try:
                return series + y
            except TypeError:
                return series.map(str) + str(y)

        modin_result = modin_df.apply(apply_func, axis=axis, args=(1,))
        pandas_result = pandas_df.apply(apply_func, axis=axis, args=(1,))
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.apply(apply_func, axis=axis, args=("_A",))
        pandas_result = pandas_df.apply(apply_func, axis=axis, args=("_A",))
        df_equals(modin_result, pandas_result)

    def test_apply_metadata(self):
        def add(a, b, c):
            return a + b + c

        data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}

        modin_df = pd.DataFrame(data)
        modin_df["add"] = modin_df.apply(
            lambda row: add(row["A"], row["B"], row["C"]), axis=1
        )

        pandas_df = pandas.DataFrame(data)
        pandas_df["add"] = pandas_df.apply(
            lambda row: add(row["A"], row["B"], row["C"]), axis=1
        )
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    def test_apply_numeric(self, request, data, func, axis):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if name_contains(request.node.name, numeric_dfs):
            try:
                pandas_result = pandas_df.apply(func, axis)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.apply(func, axis)
            else:
                modin_result = modin_df.apply(func, axis)
                df_equals(modin_result, pandas_result)

        if "empty_data" not in request.node.name:
            key = modin_df.columns[0]
            modin_result = modin_df.apply(lambda df: df.drop(key), axis=1)
            pandas_result = pandas_df.apply(lambda df: df.drop(key), axis=1)
            df_equals(modin_result, pandas_result)

    def test_eval_df_use_case(self):
        frame_data = {"a": random_state.randn(10), "b": random_state.randn(10)}
        df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)

        # test eval for series results
        tmp_pandas = df.eval("arctan2(sin(a), b)", engine="python", parser="pandas")
        tmp_modin = modin_df.eval(
            "arctan2(sin(a), b)", engine="python", parser="pandas"
        )

        assert isinstance(tmp_modin, pd.Series)
        df_equals(tmp_modin, tmp_pandas)

        # Test not inplace assignments
        tmp_pandas = df.eval("e = arctan2(sin(a), b)", engine="python", parser="pandas")
        tmp_modin = modin_df.eval(
            "e = arctan2(sin(a), b)", engine="python", parser="pandas"
        )
        df_equals(tmp_modin, tmp_pandas)

        # Test inplace assignments
        df.eval(
            "e = arctan2(sin(a), b)", engine="python", parser="pandas", inplace=True
        )
        modin_df.eval(
            "e = arctan2(sin(a), b)", engine="python", parser="pandas", inplace=True
        )
        # TODO: Use a series equality validator.
        df_equals(modin_df, df)

    def test_eval_df_arithmetic_subexpression(self):
        frame_data = {"a": random_state.randn(10), "b": random_state.randn(10)}
        df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)
        df.eval("not_e = sin(a + b)", engine="python", parser="pandas", inplace=True)
        modin_df.eval(
            "not_e = sin(a + b)", engine="python", parser="pandas", inplace=True
        )
        # TODO: Use a series equality validator.
        df_equals(modin_df, df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_filter(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        by = {"items": ["col1", "col5"], "regex": "4$|3$", "like": "col"}
        df_equals(
            modin_df.filter(items=by["items"]), pandas_df.filter(items=by["items"])
        )

        df_equals(
            modin_df.filter(regex=by["regex"], axis=0),
            pandas_df.filter(regex=by["regex"], axis=0),
        )
        df_equals(
            modin_df.filter(regex=by["regex"], axis=1),
            pandas_df.filter(regex=by["regex"], axis=1),
        )

        df_equals(modin_df.filter(like=by["like"]), pandas_df.filter(like=by["like"]))

        with pytest.raises(TypeError):
            modin_df.filter(items=by["items"], regex=by["regex"])

        with pytest.raises(TypeError):
            modin_df.filter()

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_pipe(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        n = len(modin_df.index)
        a, b, c = 2 % n, 0, 3 % n
        col = modin_df.columns[3 % len(modin_df.columns)]

        def h(x):
            return x.drop(columns=[col])

        def g(x, arg1=0):
            for _ in range(arg1):
                x = x.append(x)
            return x

        def f(x, arg2=0, arg3=0):
            return x.drop([arg2, arg3])

        df_equals(
            f(g(h(modin_df), arg1=a), arg2=b, arg3=c),
            (modin_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
        )
        df_equals(
            (modin_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
            (pandas_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("funcs", query_func_values, ids=query_func_keys)
    def test_query(self, data, funcs):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        with pytest.raises(ValueError):
            modin_df.query("")
        with pytest.raises(NotImplementedError):
            x = 2  # noqa F841
            modin_df.query("col1 < @x")

        try:
            pandas_result = pandas_df.query(funcs)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.query(funcs)
        else:
            modin_result = modin_df.query(funcs)
            df_equals(modin_result, pandas_result)

    def test_query_after_insert(self):
        modin_df = pd.DataFrame({"x": [-1, 0, 1, None], "y": [1, 2, None, 3]})
        modin_df["z"] = modin_df.eval("x / y")
        modin_df = modin_df.query("z >= 0")
        modin_result = modin_df.reset_index(drop=True)
        modin_result.columns = ["a", "b", "c"]

        pandas_df = pd.DataFrame({"x": [-1, 0, 1, None], "y": [1, 2, None, 3]})
        pandas_df["z"] = pandas_df.eval("x / y")
        pandas_df = pandas_df.query("z >= 0")
        pandas_result = pandas_df.reset_index(drop=True)
        pandas_result.columns = ["a", "b", "c"]

        df_equals(modin_result, pandas_result)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
    def test_transform(self, request, data, func):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.transform(func)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.transform(func)
        else:
            modin_result = modin_df.transform(func)
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
    def test_transform_numeric(self, request, data, func):
        if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
            request.node.name, numeric_dfs
        ):
            modin_df = pd.DataFrame(data)
            pandas_df = pandas.DataFrame(data)

            try:
                pandas_result = pandas_df.transform(func)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.transform(func)
            else:
                modin_result = modin_df.transform(func)
                df_equals(modin_result, pandas_result)


class TestDataFrameDefault:
    def test_align(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).align(pd.DataFrame(data))

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_to_numpy(self, data):
        modin_frame = pd.DataFrame(data)
        pandas_frame = pandas.DataFrame(data)
        assert_array_equal(modin_frame.values, pandas_frame.values)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_partition_to_numpy(self, data):
        frame = pd.DataFrame(data)
        for (
            partition
        ) in frame._query_compiler._modin_frame._partitions.flatten().tolist():
            assert_array_equal(partition.to_pandas().values, partition.to_numpy())

    def test_asfreq(self):
        index = pd.date_range("1/1/2000", periods=4, freq="T")
        series = pd.Series([0.0, None, 2.0, 3.0], index=index)
        df = pd.DataFrame({"s": series})
        with pytest.warns(UserWarning):
            # We are only testing that this defaults to pandas, so we will just check for
            # the warning
            df.asfreq(freq="30S")

    def test_asof(self):
        df = pd.DataFrame(
            {"a": [10, 20, 30, 40, 50], "b": [None, None, None, None, 500]},
            index=pd.DatetimeIndex(
                [
                    "2018-02-27 09:01:00",
                    "2018-02-27 09:02:00",
                    "2018-02-27 09:03:00",
                    "2018-02-27 09:04:00",
                    "2018-02-27 09:05:00",
                ]
            ),
        )
        with pytest.warns(UserWarning):
            df.asof(pd.DatetimeIndex(["2018-02-27 09:03:30", "2018-02-27 09:04:30"]))

    def test_assign(self):
        data = test_data_values[0]
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        modin_result = modin_df.assign(new_column=pd.Series(modin_df.iloc[:, 0]))
        pandas_result = pandas_df.assign(new_column=pandas.Series(pandas_df.iloc[:, 0]))
        df_equals(modin_result, pandas_result)
        modin_result = modin_df.assign(
            new_column=pd.Series(modin_df.iloc[:, 0]),
            new_column2=pd.Series(modin_df.iloc[:, 1]),
        )
        pandas_result = pandas_df.assign(
            new_column=pandas.Series(pandas_df.iloc[:, 0]),
            new_column2=pandas.Series(pandas_df.iloc[:, 1]),
        )
        df_equals(modin_result, pandas_result)

    def test_at_time(self):
        i = pd.date_range("2008-01-01", periods=1000, freq="12H")
        modin_df = pd.DataFrame(
            {"A": list(range(1000)), "B": list(range(1000))}, index=i
        )
        pandas_df = pandas.DataFrame(
            {"A": list(range(1000)), "B": list(range(1000))}, index=i
        )
        df_equals(modin_df.at_time("12:00"), pandas_df.at_time("12:00"))
        df_equals(modin_df.at_time("3:00"), pandas_df.at_time("3:00"))
        df_equals(
            modin_df.T.at_time("12:00", axis=1), pandas_df.T.at_time("12:00", axis=1)
        )

    def test_between_time(self):
        i = pd.date_range("2008-01-01", periods=1000, freq="12H")
        modin_df = pd.DataFrame(
            {"A": list(range(1000)), "B": list(range(1000))}, index=i
        )
        pandas_df = pandas.DataFrame(
            {"A": list(range(1000)), "B": list(range(1000))}, index=i
        )
        df_equals(
            modin_df.between_time("12:00", "17:00"),
            pandas_df.between_time("12:00", "17:00"),
        )
        df_equals(
            modin_df.between_time("3:00", "4:00"),
            pandas_df.between_time("3:00", "4:00"),
        )
        df_equals(
            modin_df.T.between_time("12:00", "17:00", axis=1),
            pandas_df.T.between_time("12:00", "17:00", axis=1),
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_bfill(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        df_equals(modin_df.bfill(), pandas_df.bfill())

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_bool(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)  # noqa F841

        with pytest.raises(ValueError):
            modin_df.bool()
            modin_df.__bool__()

        single_bool_pandas_df = pandas.DataFrame([True])
        single_bool_modin_df = pd.DataFrame([True])

        assert single_bool_pandas_df.bool() == single_bool_modin_df.bool()

        with pytest.raises(ValueError):
            # __bool__ always raises this error for DataFrames
            single_bool_modin_df.__bool__()

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_boxplot(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)  # noqa F841

        assert modin_df.boxplot() == to_pandas(modin_df).boxplot()

    def test_combine_first(self):
        data1 = {"A": [None, 0], "B": [None, 4]}
        modin_df1 = pd.DataFrame(data1)
        pandas_df1 = pandas.DataFrame(data1)
        data2 = {"A": [1, 1], "B": [3, 3]}
        modin_df2 = pd.DataFrame(data2)
        pandas_df2 = pandas.DataFrame(data2)
        df_equals(
            modin_df1.combine_first(modin_df2), pandas_df1.combine_first(pandas_df2)
        )

    def test_corr(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).corr()

    def test_corrwith(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).corrwith(pd.DataFrame(data))

    def test_cov(self):
        data = test_data_values[0]
        modin_result = pd.DataFrame(data).cov()
        pandas_result = pandas.DataFrame(data).cov()
        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_dot(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        col_len = len(modin_df.columns)

        # Test list input
        arr = np.arange(col_len)
        modin_result = modin_df.dot(arr)
        pandas_result = pandas_df.dot(arr)
        df_equals(modin_result, pandas_result)

        # Test bad dimensions
        with pytest.raises(ValueError):
            modin_result = modin_df.dot(np.arange(col_len + 10))

        # Test series input
        modin_series = pd.Series(np.arange(col_len), index=modin_df.columns)
        pandas_series = pandas.Series(np.arange(col_len), index=pandas_df.columns)
        modin_result = modin_df.dot(modin_series)
        pandas_result = pandas_df.dot(pandas_series)
        df_equals(modin_result, pandas_result)

        # Test dataframe input
        modin_result = modin_df.dot(modin_df.T)
        pandas_result = pandas_df.dot(pandas_df.T)
        df_equals(modin_result, pandas_result)

        # Test when input series index doesn't line up with columns
        with pytest.raises(ValueError):
            modin_result = modin_df.dot(pd.Series(np.arange(col_len)))

        # Test case when left dataframe has size (n x 1)
        # and right dataframe has size (1 x n)
        modin_df = pd.DataFrame(modin_series)
        pandas_df = pandas.DataFrame(pandas_series)
        modin_result = modin_df.dot(modin_df.T)
        pandas_result = pandas_df.dot(pandas_df.T)
        df_equals(modin_result, pandas_result)

        # Test case when left dataframe has size (1 x 1)
        # and right dataframe has size (1 x n)
        modin_result = pd.DataFrame([1]).dot(modin_df.T)
        pandas_result = pandas.DataFrame([1]).dot(pandas_df.T)
        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_matmul(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        col_len = len(modin_df.columns)

        # Test list input
        arr = np.arange(col_len)
        modin_result = modin_df @ arr
        pandas_result = pandas_df @ arr
        df_equals(modin_result, pandas_result)

        # Test bad dimensions
        with pytest.raises(ValueError):
            modin_result = modin_df @ np.arange(col_len + 10)

        # Test series input
        modin_series = pd.Series(np.arange(col_len), index=modin_df.columns)
        pandas_series = pandas.Series(np.arange(col_len), index=pandas_df.columns)
        modin_result = modin_df @ modin_series
        pandas_result = pandas_df @ pandas_series
        df_equals(modin_result, pandas_result)

        # Test dataframe input
        modin_result = modin_df @ modin_df.T
        pandas_result = pandas_df @ pandas_df.T
        df_equals(modin_result, pandas_result)

        # Test when input series index doesn't line up with columns
        with pytest.raises(ValueError):
            modin_result = modin_df @ pd.Series(np.arange(col_len))

    def test_ewm(self):
        df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
        with pytest.warns(UserWarning):
            df.ewm(com=0.5).mean()

    def test_expanding(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).expanding()

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_explode(self, data):
        modin_df = pd.DataFrame(data)
        with pytest.warns(UserWarning):
            modin_df.explode(modin_df.columns[0])

    def test_first(self):
        i = pd.date_range("2010-04-09", periods=400, freq="2D")
        modin_df = pd.DataFrame({"A": list(range(400)), "B": list(range(400))}, index=i)
        pandas_df = pandas.DataFrame(
            {"A": list(range(400)), "B": list(range(400))}, index=i
        )
        df_equals(modin_df.first("3D"), pandas_df.first("3D"))
        df_equals(modin_df.first("20D"), pandas_df.first("20D"))

    @pytest.mark.skip(reason="Defaulting to Pandas")
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_from_dict(self, data):
        modin_df = pd.DataFrame(data)  # noqa F841
        pandas_df = pandas.DataFrame(data)  # noqa F841

        with pytest.raises(NotImplementedError):
            pd.DataFrame.from_dict(None)

    @pytest.mark.skip(reason="Defaulting to Pandas")
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_from_items(self, data):
        modin_df = pd.DataFrame(data)  # noqa F841
        pandas_df = pandas.DataFrame(data)  # noqa F841

        with pytest.raises(NotImplementedError):
            pd.DataFrame.from_items(None)

    @pytest.mark.skip(reason="Defaulting to Pandas")
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_from_records(self, data):
        modin_df = pd.DataFrame(data)  # noqa F841
        pandas_df = pandas.DataFrame(data)  # noqa F841

        with pytest.raises(NotImplementedError):
            pd.DataFrame.from_records(None)

    def test_hist(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).hist(None)

    def test_infer_objects(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).infer_objects()

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("verbose", [None, True, False])
    @pytest.mark.parametrize("max_cols", [None, 10, 99999999])
    @pytest.mark.parametrize("memory_usage", [None, True, False, "deep"])
    @pytest.mark.parametrize("null_counts", [None, True, False])
    def test_info(self, data, verbose, max_cols, memory_usage, null_counts):
        with io.StringIO() as first, io.StringIO() as second:
            eval_general(
                pd.DataFrame(data),
                pandas.DataFrame(data),
                operation=lambda df, **kwargs: df.info(**kwargs),
                verbose=verbose,
                max_cols=max_cols,
                memory_usage=memory_usage,
                null_counts=null_counts,
                buf=lambda df: second if isinstance(df, pandas.DataFrame) else first,
            )
            modin_info = first.getvalue().splitlines()
            pandas_info = second.getvalue().splitlines()

            assert modin_info[0] == str(pd.DataFrame)
            assert pandas_info[0] == str(pandas.DataFrame)
            assert modin_info[1:] == pandas_info[1:]

    def test_interpolate(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).interpolate()

    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize("skipna", bool_arg_values, ids=bool_arg_keys)
    @pytest.mark.parametrize("level", [None, -1, 0, 1])
    @pytest.mark.parametrize("numeric_only", bool_arg_values, ids=bool_arg_keys)
    def test_kurt_kurtosis(self, axis, skipna, level, numeric_only):
        func_kwargs = {
            "axis": axis,
            "skipna": skipna,
            "level": level,
            "numeric_only": numeric_only,
        }
        data = test_data_values[0]
        df_modin = pd.DataFrame(data)
        df_pandas = pandas.DataFrame(data)

        eval_general(
            df_modin, df_pandas, lambda df: df.kurtosis(**func_kwargs),
        )

        if level is not None:
            cols_number = len(data.keys())
            arrays = [
                np.random.choice(["bar", "baz", "foo", "qux"], cols_number),
                np.random.choice(["one", "two"], cols_number),
            ]
            index = pd.MultiIndex.from_tuples(
                list(zip(*arrays)), names=["first", "second"]
            )
            df_modin.columns = index
            df_pandas.columns = index
            eval_general(
                df_modin, df_pandas, lambda df: df.kurtosis(**func_kwargs),
            )

    def test_last(self):
        modin_index = pd.date_range("2010-04-09", periods=400, freq="2D")
        pandas_index = pandas.date_range("2010-04-09", periods=400, freq="2D")
        modin_df = pd.DataFrame(
            {"A": list(range(400)), "B": list(range(400))}, index=modin_index
        )
        pandas_df = pandas.DataFrame(
            {"A": list(range(400)), "B": list(range(400))}, index=pandas_index
        )
        df_equals(modin_df.last("3D"), pandas_df.last("3D"))
        df_equals(modin_df.last("20D"), pandas_df.last("20D"))

    def test_lookup(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).lookup([0, 1], ["col1", "col2"])

    @pytest.mark.parametrize("data", test_data_values)
    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("skipna", [None, True, False])
    @pytest.mark.parametrize("level", [0, -1, None])
    def test_mad(self, level, data, axis, skipna):
        modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
        df_equals(
            modin_df.mad(axis=axis, skipna=skipna, level=level),
            pandas_df.mad(axis=axis, skipna=skipna, level=level),
        )

    def test_mask(self):
        df = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"])
        m = df % 3 == 0
        with pytest.warns(UserWarning):
            try:
                df.mask(~m, -df)
            except ValueError:
                pass

    def test_melt(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).melt()

    def test_pct_change(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).pct_change()

    def test_pivot(self):
        df = pd.DataFrame(
            {
                "foo": ["one", "one", "one", "two", "two", "two"],
                "bar": ["A", "B", "C", "A", "B", "C"],
                "baz": [1, 2, 3, 4, 5, 6],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )
        with pytest.warns(UserWarning):
            df.pivot(index="foo", columns="bar", values="baz")

    def test_pivot_table(self):
        df = pd.DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )
        with pytest.warns(UserWarning):
            df.pivot_table(values="D", index=["A", "B"], columns=["C"], aggfunc=np.sum)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_plot(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if name_contains(request.node.name, numeric_dfs):
            # We have to test this way because equality in plots means same object.
            zipped_plot_lines = zip(modin_df.plot().lines, pandas_df.plot().lines)
            for left, right in zipped_plot_lines:
                if isinstance(left.get_xdata(), np.ma.core.MaskedArray) and isinstance(
                    right.get_xdata(), np.ma.core.MaskedArray
                ):
                    assert all((left.get_xdata() == right.get_xdata()).data)
                else:
                    assert np.array_equal(left.get_xdata(), right.get_xdata())
                if isinstance(left.get_ydata(), np.ma.core.MaskedArray) and isinstance(
                    right.get_ydata(), np.ma.core.MaskedArray
                ):
                    assert all((left.get_ydata() == right.get_ydata()).data)
                else:
                    assert np.array_equal(left.get_xdata(), right.get_xdata())

    def test_replace(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).replace()

    def test_resample(self):
        d = dict(
            {
                "price": [10, 11, 9, 13, 14, 18, 17, 19],
                "volume": [50, 60, 40, 100, 50, 100, 40, 50],
            }
        )
        df = pd.DataFrame(d)
        df["week_starting"] = pd.date_range("01/01/2018", periods=8, freq="W")
        with pytest.warns(UserWarning):
            df.resample("M", on="week_starting")

    def test_rolling(self):
        df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
        with pytest.warns(UserWarning):
            df.rolling(2, win_type="triang")

    def test_sem(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).sem()

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("index", ["default", "ndarray"])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("periods", [0, 1, -1, 10, -10, 1000000000, -1000000000])
    def test_shift(self, data, index, axis, periods):
        if index == "default":
            modin_df = pd.DataFrame(data)
            pandas_df = pandas.DataFrame(data)
        elif index == "ndarray":
            data_column_length = len(data[next(iter(data))])
            index_data = np.arange(2, data_column_length + 2)
            modin_df = pd.DataFrame(data, index=index_data)
            pandas_df = pandas.DataFrame(data, index=index_data)

        df_equals(
            modin_df.shift(periods=periods, axis=axis),
            pandas_df.shift(periods=periods, axis=axis),
        )
        df_equals(
            modin_df.shift(periods=periods, axis=axis, fill_value=777),
            pandas_df.shift(periods=periods, axis=axis, fill_value=777),
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("index", ["default", "ndarray"])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("periods", [0, 1, -1, 10, -10, 1000000000, -1000000000])
    def test_slice_shift(self, data, index, axis, periods):
        if index == "default":
            modin_df = pd.DataFrame(data)
            pandas_df = pandas.DataFrame(data)
        elif index == "ndarray":
            data_column_length = len(data[next(iter(data))])
            index_data = np.arange(2, data_column_length + 2)
            modin_df = pd.DataFrame(data, index=index_data)
            pandas_df = pandas.DataFrame(data, index=index_data)

        df_equals(
            modin_df.slice_shift(periods=periods, axis=axis),
            pandas_df.slice_shift(periods=periods, axis=axis),
        )

    def test_stack(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).stack()

    def test_style(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).style

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis1", [0, 1, "columns", "index"])
    @pytest.mark.parametrize("axis2", [0, 1, "columns", "index"])
    def test_swapaxes(self, data, axis1, axis2):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        try:
            pandas_result = pandas_df.swapaxes(axis1, axis2)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.swapaxes(axis1, axis2)
        else:
            modin_result = modin_df.swapaxes(axis1, axis2)
            df_equals(modin_result, pandas_result)

    def test_swaplevel(self):
        data = np.random.randint(1, 100, 12)
        modin_df = pd.DataFrame(
            data,
            index=pd.MultiIndex.from_tuples(
                [
                    (num, letter, color)
                    for num in range(1, 3)
                    for letter in ["a", "b", "c"]
                    for color in ["Red", "Green"]
                ],
                names=["Number", "Letter", "Color"],
            ),
        )
        pandas_df = pandas.DataFrame(
            data,
            index=pandas.MultiIndex.from_tuples(
                [
                    (num, letter, color)
                    for num in range(1, 3)
                    for letter in ["a", "b", "c"]
                    for color in ["Red", "Green"]
                ],
                names=["Number", "Letter", "Color"],
            ),
        )
        df_equals(
            modin_df.swaplevel("Number", "Color"),
            pandas_df.swaplevel("Number", "Color"),
        )
        df_equals(modin_df.swaplevel(), pandas_df.swaplevel())
        df_equals(modin_df.swaplevel(0, 1), pandas_df.swaplevel(0, 1))

    def test_take(self):
        modin_df = pd.DataFrame(
            [
                ("falcon", "bird", 389.0),
                ("parrot", "bird", 24.0),
                ("lion", "mammal", 80.5),
                ("monkey", "mammal", np.nan),
            ],
            columns=["name", "class", "max_speed"],
            index=[0, 2, 3, 1],
        )
        pandas_df = pandas.DataFrame(
            [
                ("falcon", "bird", 389.0),
                ("parrot", "bird", 24.0),
                ("lion", "mammal", 80.5),
                ("monkey", "mammal", np.nan),
            ],
            columns=["name", "class", "max_speed"],
            index=[0, 2, 3, 1],
        )
        df_equals(modin_df.take([0, 3]), pandas_df.take([0, 3]))
        df_equals(modin_df.take([2], axis=1), pandas_df.take([2], axis=1))

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_to_records(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        # Skips nan because only difference is nan instead of NaN
        if not name_contains(request.node.name, ["nan"]):
            try:
                pandas_result = pandas_df.to_records()
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.to_records()
            else:
                modin_result = modin_df.to_records()
                assert np.array_equal(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_to_string(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)  # noqa F841

        # Skips nan because only difference is nan instead of NaN
        if not name_contains(request.node.name, ["nan"]):
            assert modin_df.to_string() == to_pandas(modin_df).to_string()

    def test_to_timestamp(self):
        idx = pd.date_range("1/1/2012", periods=5, freq="M")
        df = pd.DataFrame(np.random.randint(0, 100, size=(len(idx), 4)), index=idx)

        with pytest.warns(UserWarning):
            df.to_period().to_timestamp()

    def test_to_xarray(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).to_xarray()

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_truncate(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        before = 1
        after = len(modin_df - 3)
        df_equals(modin_df.truncate(before, after), pandas_df.truncate(before, after))

        before = 1
        after = 3
        df_equals(modin_df.truncate(before, after), pandas_df.truncate(before, after))

        before = modin_df.columns[1]
        after = modin_df.columns[-3]
        try:
            pandas_result = pandas_df.truncate(before, after, axis=1)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.truncate(before, after, axis=1)
        else:
            modin_result = modin_df.truncate(before, after, axis=1)
            df_equals(modin_result, pandas_result)

        before = modin_df.columns[1]
        after = modin_df.columns[3]
        try:
            pandas_result = pandas_df.truncate(before, after, axis=1)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.truncate(before, after, axis=1)
        else:
            modin_result = modin_df.truncate(before, after, axis=1)
            df_equals(modin_result, pandas_result)

        before = None
        after = None
        df_equals(modin_df.truncate(before, after), pandas_df.truncate(before, after))
        try:
            pandas_result = pandas_df.truncate(before, after, axis=1)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.truncate(before, after, axis=1)
        else:
            modin_result = modin_df.truncate(before, after, axis=1)
            df_equals(modin_result, pandas_result)

    def test_tshift(self):
        idx = pd.date_range("1/1/2012", periods=5, freq="M")
        data = np.random.randint(0, 100, size=(len(idx), 4))
        modin_df = pd.DataFrame(data, index=idx)
        pandas_df = pandas.DataFrame(data, index=idx)
        df_equals(modin_df.tshift(4), pandas_df.tshift(4))

    def test_tz_convert(self):
        modin_idx = pd.date_range(
            "1/1/2012", periods=500, freq="2D", tz="America/Los_Angeles"
        )
        pandas_idx = pandas.date_range(
            "1/1/2012", periods=500, freq="2D", tz="America/Los_Angeles"
        )
        data = np.random.randint(0, 100, size=(len(modin_idx), 4))
        modin_df = pd.DataFrame(data, index=modin_idx)
        pandas_df = pandas.DataFrame(data, index=pandas_idx)
        modin_result = modin_df.tz_convert("UTC", axis=0)
        pandas_result = pandas_df.tz_convert("UTC", axis=0)
        df_equals(modin_result, pandas_result)

        modin_multi = pd.MultiIndex.from_arrays([modin_idx, range(len(modin_idx))])
        pandas_multi = pandas.MultiIndex.from_arrays(
            [pandas_idx, range(len(modin_idx))]
        )
        modin_series = pd.DataFrame(data, index=modin_multi)
        pandas_series = pandas.DataFrame(data, index=pandas_multi)
        df_equals(
            modin_series.tz_convert("UTC", axis=0, level=0),
            pandas_series.tz_convert("UTC", axis=0, level=0),
        )

    def test_tz_localize(self):
        idx = pd.date_range("1/1/2012", periods=400, freq="2D")
        data = np.random.randint(0, 100, size=(len(idx), 4))
        modin_df = pd.DataFrame(data, index=idx)
        pandas_df = pandas.DataFrame(data, index=idx)
        df_equals(
            modin_df.tz_localize("UTC", axis=0), pandas_df.tz_localize("UTC", axis=0)
        )
        df_equals(
            modin_df.tz_localize("America/Los_Angeles", axis=0),
            pandas_df.tz_localize("America/Los_Angeles", axis=0),
        )

    def test_unstack(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).unstack()

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___array__(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        assert_array_equal(modin_df.__array__(), pandas_df.__array__())

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___bool__(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.__bool__()
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.__bool__()
        else:
            modin_result = modin_df.__bool__()
            df_equals(modin_result, pandas_result)

    def test___getstate__(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).__getstate__()

    def test___setstate__(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            try:
                pd.DataFrame(data).__setstate__(None)
            except TypeError:
                pass

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_hasattr_sparse(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        try:
            pandas_result = hasattr(pandas_df, "sparse")
        except Exception as e:
            with pytest.raises(type(e)):
                hasattr(modin_df, "sparse")
        else:
            modin_result = hasattr(modin_df, "sparse")
            assert modin_result == pandas_result


class TestDataFrameReduction_A:
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "bool_only", bool_arg_values, ids=arg_keys("bool_only", bool_arg_keys)
    )
    def test_all(self, data, axis, skipna, bool_only):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
        else:
            modin_result = modin_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
            df_equals(modin_result, pandas_result)

        # Test when axis is None. This will get repeated but easier than using list in parameterize decorator
        try:
            pandas_result = pandas_df.all(axis=None, skipna=skipna, bool_only=bool_only)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.all(axis=None, skipna=skipna, bool_only=bool_only)
        else:
            modin_result = modin_df.all(axis=None, skipna=skipna, bool_only=bool_only)
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.all(
                axis=axis, skipna=skipna, bool_only=bool_only
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.all(axis=axis, skipna=skipna, bool_only=bool_only)
        else:
            modin_result = modin_df.T.all(axis=axis, skipna=skipna, bool_only=bool_only)
            df_equals(modin_result, pandas_result)

        # Test when axis is None. This will get repeated but easier than using list in parameterize decorator
        try:
            pandas_result = pandas_df.T.all(
                axis=None, skipna=skipna, bool_only=bool_only
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.all(axis=None, skipna=skipna, bool_only=bool_only)
        else:
            modin_result = modin_df.T.all(axis=None, skipna=skipna, bool_only=bool_only)
            df_equals(modin_result, pandas_result)

        # test level
        modin_df_multi_level = modin_df.copy()
        pandas_df_multi_level = pandas_df.copy()
        axis = modin_df._get_axis_number(axis) if axis is not None else 0
        levels = 3
        axis_names_list = [["a", "b", "c"], None]
        for axis_names in axis_names_list:
            if axis == 0:
                new_idx = pandas.MultiIndex.from_tuples(
                    [(i // 4, i // 2, i) for i in range(len(modin_df.index))],
                    names=axis_names,
                )
                modin_df_multi_level.index = new_idx
                pandas_df_multi_level.index = new_idx
            else:
                new_col = pandas.MultiIndex.from_tuples(
                    [(i // 4, i // 2, i) for i in range(len(modin_df.columns))],
                    names=axis_names,
                )
                modin_df_multi_level.columns = new_col
                pandas_df_multi_level.columns = new_col

            for level in list(range(levels)) + (axis_names if axis_names else []):
                try:
                    pandas_multi_level_result = pandas_df_multi_level.all(
                        axis=axis, bool_only=bool_only, level=level, skipna=skipna
                    )

                except Exception as e:
                    with pytest.raises(type(e)):
                        modin_df_multi_level.all(
                            axis=axis, bool_only=bool_only, level=level, skipna=skipna
                        )
                else:
                    modin_multi_level_result = modin_df_multi_level.all(
                        axis=axis, bool_only=bool_only, level=level, skipna=skipna
                    )

                    df_equals(modin_multi_level_result, pandas_multi_level_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "bool_only", bool_arg_values, ids=arg_keys("bool_only", bool_arg_keys)
    )
    def test_any(self, data, axis, skipna, bool_only):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
        else:
            modin_result = modin_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.any(axis=None, skipna=skipna, bool_only=bool_only)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.any(axis=None, skipna=skipna, bool_only=bool_only)
        else:
            modin_result = modin_df.any(axis=None, skipna=skipna, bool_only=bool_only)
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.any(
                axis=axis, skipna=skipna, bool_only=bool_only
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.any(axis=axis, skipna=skipna, bool_only=bool_only)
        else:
            modin_result = modin_df.T.any(axis=axis, skipna=skipna, bool_only=bool_only)
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.any(
                axis=None, skipna=skipna, bool_only=bool_only
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.any(axis=None, skipna=skipna, bool_only=bool_only)
        else:
            modin_result = modin_df.T.any(axis=None, skipna=skipna, bool_only=bool_only)
            df_equals(modin_result, pandas_result)

        # test level
        modin_df_multi_level = modin_df.copy()
        pandas_df_multi_level = pandas_df.copy()
        axis = modin_df._get_axis_number(axis) if axis is not None else 0
        levels = 3
        axis_names_list = [["a", "b", "c"], None]
        for axis_names in axis_names_list:
            if axis == 0:
                new_idx = pandas.MultiIndex.from_tuples(
                    [(i // 4, i // 2, i) for i in range(len(modin_df.index))],
                    names=axis_names,
                )
                modin_df_multi_level.index = new_idx
                pandas_df_multi_level.index = new_idx
            else:
                new_col = pandas.MultiIndex.from_tuples(
                    [(i // 4, i // 2, i) for i in range(len(modin_df.columns))],
                    names=axis_names,
                )
                modin_df_multi_level.columns = new_col
                pandas_df_multi_level.columns = new_col

            for level in list(range(levels)) + (axis_names if axis_names else []):
                try:
                    pandas_multi_level_result = pandas_df_multi_level.any(
                        axis=axis, bool_only=bool_only, level=level, skipna=skipna
                    )

                except Exception as e:
                    with pytest.raises(type(e)):
                        modin_df_multi_level.any(
                            axis=axis, bool_only=bool_only, level=level, skipna=skipna
                        )
                else:
                    modin_multi_level_result = modin_df_multi_level.any(
                        axis=axis, bool_only=bool_only, level=level, skipna=skipna
                    )

                    df_equals(modin_multi_level_result, pandas_multi_level_result)


class TestDataFrameReduction_B:
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    def test_count(self, request, data, axis, numeric_only):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_result = modin_df.count(axis=axis, numeric_only=numeric_only)
        pandas_result = pandas_df.count(axis=axis, numeric_only=numeric_only)
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.T.count(axis=axis, numeric_only=numeric_only)
        pandas_result = pandas_df.T.count(axis=axis, numeric_only=numeric_only)
        df_equals(modin_result, pandas_result)

        # test level
        modin_df_multi_level = modin_df.copy()
        pandas_df_multi_level = pandas_df.copy()
        axis = modin_df._get_axis_number(axis) if axis is not None else 0
        levels = 3
        axis_names_list = [["a", "b", "c"], None]
        for axis_names in axis_names_list:
            if axis == 0:
                new_idx = pandas.MultiIndex.from_tuples(
                    [(i // 4, i // 2, i) for i in range(len(modin_df.index))],
                    names=axis_names,
                )
                modin_df_multi_level.index = new_idx
                pandas_df_multi_level.index = new_idx
                try:  # test error
                    pandas_df_multi_level.count(
                        axis=1, numeric_only=numeric_only, level=0
                    )
                except Exception as e:
                    with pytest.raises(type(e)):
                        modin_df_multi_level.count(
                            axis=1, numeric_only=numeric_only, level=0
                        )
            else:
                new_col = pandas.MultiIndex.from_tuples(
                    [(i // 4, i // 2, i) for i in range(len(modin_df.columns))],
                    names=axis_names,
                )
                modin_df_multi_level.columns = new_col
                pandas_df_multi_level.columns = new_col
                try:  # test error
                    pandas_df_multi_level.count(
                        axis=0, numeric_only=numeric_only, level=0
                    )
                except Exception as e:
                    with pytest.raises(type(e)):
                        modin_df_multi_level.count(
                            axis=0, numeric_only=numeric_only, level=0
                        )

            for level in list(range(levels)) + (axis_names if axis_names else []):
                modin_multi_level_result = modin_df_multi_level.count(
                    axis=axis, numeric_only=numeric_only, level=level
                )
                pandas_multi_level_result = pandas_df_multi_level.count(
                    axis=axis, numeric_only=numeric_only, level=level
                )
                df_equals(modin_multi_level_result, pandas_multi_level_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_describe(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(modin_df.describe(), pandas_df.describe())
        percentiles = [0.10, 0.11, 0.44, 0.78, 0.99]
        df_equals(
            modin_df.describe(percentiles=percentiles),
            pandas_df.describe(percentiles=percentiles),
        )

        try:
            pandas_result = pandas_df.describe(exclude=[np.float64])
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.describe(exclude=[np.float64])
        else:
            modin_result = modin_df.describe(exclude=[np.float64])
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.describe(exclude=np.float64)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.describe(exclude=np.float64)
        else:
            modin_result = modin_df.describe(exclude=np.float64)
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.describe(
                include=[np.timedelta64, np.datetime64, np.object, np.bool]
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.describe(
                    include=[np.timedelta64, np.datetime64, np.object, np.bool]
                )
        else:
            modin_result = modin_df.describe(
                include=[np.timedelta64, np.datetime64, np.object, np.bool]
            )
            df_equals(modin_result, pandas_result)

        modin_result = modin_df.describe(include=str(modin_df.dtypes.values[0]))
        pandas_result = pandas_df.describe(include=str(pandas_df.dtypes.values[0]))
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.describe(include=[np.number])
        pandas_result = pandas_df.describe(include=[np.number])
        df_equals(modin_result, pandas_result)

        df_equals(modin_df.describe(include="all"), pandas_df.describe(include="all"))

        modin_df = pd.DataFrame(data).applymap(str)
        pandas_df = pandas.DataFrame(data).applymap(str)

        try:
            df_equals(modin_df.describe(), pandas_df.describe())
        except AssertionError:
            # We have to do this because we choose the highest count slightly differently
            # than pandas. Because there is no true guarantee which one will be first,
            # If they don't match, make sure that the `freq` is the same at least.
            df_equals(
                modin_df.describe().loc[["count", "unique", "freq"]],
                pandas_df.describe().loc[["count", "unique", "freq"]],
            )

    def test_describe_dtypes(self):
        modin_df = pd.DataFrame(
            {
                "col1": list("abc"),
                "col2": list("abc"),
                "col3": list("abc"),
                "col4": [1, 2, 3],
            }
        )
        pandas_df = pandas.DataFrame(
            {
                "col1": list("abc"),
                "col2": list("abc"),
                "col3": list("abc"),
                "col4": [1, 2, 3],
            }
        )

        modin_result = modin_df.describe()
        pandas_result = pandas_df.describe()

        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    def test_idxmax(self, data, axis, skipna):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        pandas_result = pandas_df.idxmax(axis=axis, skipna=skipna)
        modin_result = modin_df.idxmax(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)

        pandas_result = pandas_df.T.idxmax(axis=axis, skipna=skipna)
        modin_result = modin_df.T.idxmax(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    def test_idxmin(self, data, axis, skipna):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_result = modin_df.idxmin(axis=axis, skipna=skipna)
        pandas_result = pandas_df.idxmin(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.T.idxmin(axis=axis, skipna=skipna)
        pandas_result = pandas_df.T.idxmin(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_last_valid_index(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        assert modin_df.last_valid_index() == (pandas_df.last_valid_index())

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    def test_max(self, request, data, axis, skipna, numeric_only):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.max(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
        else:
            modin_result = modin_df.max(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.max(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.T.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
        else:
            modin_result = modin_df.T.max(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    def test_mean(self, request, data, axis, skipna, numeric_only):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.mean(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.mean(axis=axis, skipna=skipna, numeric_only=numeric_only)
        else:
            modin_result = modin_df.mean(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.mean(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.mean(axis=axis, skipna=skipna, numeric_only=numeric_only)
        else:
            modin_result = modin_df.T.mean(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize(
        "index", bool_arg_values, ids=arg_keys("index", bool_arg_keys)
    )
    def test_memory_usage(self, data, index):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)  # noqa F841

        modin_result = modin_df.memory_usage(index=index)
        pandas_result = pandas_df.memory_usage(index=index)
        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    def test_min(self, data, axis, skipna, numeric_only):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.min(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.min(axis=axis, skipna=skipna, numeric_only=numeric_only)
        else:
            modin_result = modin_df.min(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.min(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.T.min(axis=axis, skipna=skipna, numeric_only=numeric_only)
        else:
            modin_result = modin_df.T.min(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
            df_equals(modin_result, pandas_result)

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Windows has a memory issue for large numbers on this test",
    )
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
    )
    def test_prod(self, request, data, axis, skipna, numeric_only, min_count):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.prod(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.prod(
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only,
                    min_count=min_count,
                )
        else:
            modin_result = modin_df.prod(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.prod(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.T.prod(
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only,
                    min_count=min_count,
                )
        else:
            modin_result = modin_df.T.prod(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
            df_equals(modin_result, pandas_result)

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Windows has a memory issue for large numbers on this test",
    )
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
    )
    def test_product(self, request, data, axis, skipna, numeric_only, min_count):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.product(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.product(
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only,
                    min_count=min_count,
                )
        else:
            modin_result = modin_df.product(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
    )
    def test_sum(self, request, data, axis, skipna, numeric_only, min_count):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.sum(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.sum(
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only,
                    min_count=min_count,
                )
        else:
            modin_result = modin_df.sum(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
            df_equals(modin_result, pandas_result)
        try:
            pandas_result = pandas_df.T.sum(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.T.sum(
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only,
                    min_count=min_count,
                )
        else:
            modin_result = modin_df.T.sum(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_sum_single_column(self, data):
        modin_df = pd.DataFrame(data).iloc[:, [0]]
        pandas_df = pandas.DataFrame(data).iloc[:, [0]]
        df_equals(modin_df.sum(), pandas_df.sum())
        df_equals(modin_df.sum(axis=1), pandas_df.sum(axis=1))


class TestDataFrameWindow:
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    def test_cummax(self, request, data, axis, skipna):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.cummax(axis=axis, skipna=skipna)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.cummax(axis=axis, skipna=skipna)
        else:
            modin_result = modin_df.cummax(axis=axis, skipna=skipna)
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.cummax(axis=axis, skipna=skipna)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.cummax(axis=axis, skipna=skipna)
        else:
            modin_result = modin_df.T.cummax(axis=axis, skipna=skipna)
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    def test_cummin(self, request, data, axis, skipna):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.cummin(axis=axis, skipna=skipna)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.cummin(axis=axis, skipna=skipna)
        else:
            modin_result = modin_df.cummin(axis=axis, skipna=skipna)
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.cummin(axis=axis, skipna=skipna)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.cummin(axis=axis, skipna=skipna)
        else:
            modin_result = modin_df.T.cummin(axis=axis, skipna=skipna)
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    def test_cumprod(self, request, data, axis, skipna):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.cumprod(axis=axis, skipna=skipna)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.cumprod(axis=axis, skipna=skipna)
        else:
            modin_result = modin_df.cumprod(axis=axis, skipna=skipna)
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.cumprod(axis=axis, skipna=skipna)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.cumprod(axis=axis, skipna=skipna)
        else:
            modin_result = modin_df.T.cumprod(axis=axis, skipna=skipna)
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    def test_cumsum(self, request, data, axis, skipna):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        # pandas exhibits weird behavior for this case
        # Remove this case when we can pull the error messages from backend
        if name_contains(request.node.name, ["datetime_timedelta_data"]) and (
            axis == 0 or axis == "rows"
        ):
            with pytest.raises(TypeError):
                modin_df.cumsum(axis=axis, skipna=skipna)
        else:
            try:
                pandas_result = pandas_df.cumsum(axis=axis, skipna=skipna)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.cumsum(axis=axis, skipna=skipna)
            else:
                modin_result = modin_df.cumsum(axis=axis, skipna=skipna)
                df_equals(modin_result, pandas_result)

        if name_contains(request.node.name, ["datetime_timedelta_data"]) and (
            axis == 0 or axis == "rows"
        ):
            with pytest.raises(TypeError):
                modin_df.T.cumsum(axis=axis, skipna=skipna)
        else:
            try:
                pandas_result = pandas_df.T.cumsum(axis=axis, skipna=skipna)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.T.cumsum(axis=axis, skipna=skipna)
            else:
                modin_result = modin_df.T.cumsum(axis=axis, skipna=skipna)
                df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "periods", int_arg_values, ids=arg_keys("periods", int_arg_keys)
    )
    def test_diff(self, request, data, axis, periods):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.diff(axis=axis, periods=periods)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.diff(axis=axis, periods=periods)
        else:
            modin_result = modin_df.diff(axis=axis, periods=periods)
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.diff(axis=axis, periods=periods)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.diff(axis=axis, periods=periods)
        else:
            modin_result = modin_df.T.diff(axis=axis, periods=periods)
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize(
        "keep", ["last", "first", False], ids=["last", "first", "False"]
    )
    def test_duplicated(self, data, keep):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        pandas_result = pandas_df.duplicated(keep=keep)
        modin_result = modin_df.duplicated(keep=keep)
        df_equals(modin_result, pandas_result)

        import random

        subset = random.sample(
            list(pandas_df.columns), random.randint(1, len(pandas_df.columns))
        )
        pandas_result = pandas_df.duplicated(keep=keep, subset=subset)
        modin_result = modin_df.duplicated(keep=keep, subset=subset)

        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_ffill(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        df_equals(modin_df.ffill(), pandas_df.ffill())

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize(
        "method",
        ["backfill", "bfill", "pad", "ffill", None],
        ids=["backfill", "bfill", "pad", "ffill", "None"],
    )
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize("limit", int_arg_values, ids=int_arg_keys)
    def test_fillna(self, data, method, axis, limit):
        # We are not testing when limit is not positive until pandas-27042 gets fixed.
        # We are not testing when axis is over rows until pandas-17399 gets fixed.
        if limit > 0 and axis != 1 and axis != "columns":
            modin_df = pd.DataFrame(data)
            pandas_df = pandas.DataFrame(data)

            try:
                pandas_result = pandas_df.fillna(
                    0, method=method, axis=axis, limit=limit
                )
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.fillna(0, method=method, axis=axis, limit=limit)
            else:
                modin_result = modin_df.fillna(0, method=method, axis=axis, limit=limit)
                df_equals(modin_result, pandas_result)

    def test_fillna_sanity(self):
        # with different dtype
        frame_data = [
            ["a", "a", np.nan, "a"],
            ["b", "b", np.nan, "b"],
            ["c", "c", np.nan, "c"],
        ]
        df = pandas.DataFrame(frame_data)

        result = df.fillna({2: "foo"})
        modin_df = pd.DataFrame(frame_data).fillna({2: "foo"})

        df_equals(modin_df, result)

        modin_df = pd.DataFrame(df)
        df.fillna({2: "foo"}, inplace=True)
        modin_df.fillna({2: "foo"}, inplace=True)
        df_equals(modin_df, result)

        frame_data = {
            "Date": [pandas.NaT, pandas.Timestamp("2014-1-1")],
            "Date2": [pandas.Timestamp("2013-1-1"), pandas.NaT],
        }
        df = pandas.DataFrame(frame_data)
        result = df.fillna(value={"Date": df["Date2"]})
        modin_df = pd.DataFrame(frame_data).fillna(value={"Date": df["Date2"]})
        df_equals(modin_df, result)

        frame_data = {"A": [pandas.Timestamp("2012-11-11 00:00:00+01:00"), pandas.NaT]}
        df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)
        df_equals(modin_df.fillna(method="pad"), df.fillna(method="pad"))

        frame_data = {"A": [pandas.NaT, pandas.Timestamp("2012-11-11 00:00:00+01:00")]}
        df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data).fillna(method="bfill")
        df_equals(modin_df, df.fillna(method="bfill"))

    def test_fillna_downcast(self):
        # infer int64 from float64
        frame_data = {"a": [1.0, np.nan]}
        df = pandas.DataFrame(frame_data)
        result = df.fillna(0, downcast="infer")
        modin_df = pd.DataFrame(frame_data).fillna(0, downcast="infer")
        df_equals(modin_df, result)

        # infer int64 from float64 when fillna value is a dict
        df = pandas.DataFrame(frame_data)
        result = df.fillna({"a": 0}, downcast="infer")
        modin_df = pd.DataFrame(frame_data).fillna({"a": 0}, downcast="infer")
        df_equals(modin_df, result)

    def test_fillna_inplace(self):
        frame_data = random_state.randn(10, 4)
        df = pandas.DataFrame(frame_data)
        df[1][:4] = np.nan
        df[3][-4:] = np.nan

        modin_df = pd.DataFrame(df)
        df.fillna(value=0, inplace=True)
        try:
            df_equals(modin_df, df)
        except AssertionError:
            pass
        else:
            assert False

        modin_df.fillna(value=0, inplace=True)
        df_equals(modin_df, df)

        modin_df = pd.DataFrame(df).fillna(value={0: 0}, inplace=True)
        assert modin_df is None

        df[1][:4] = np.nan
        df[3][-4:] = np.nan
        modin_df = pd.DataFrame(df)
        df.fillna(method="ffill", inplace=True)
        try:
            df_equals(modin_df, df)
        except AssertionError:
            pass
        else:
            assert False

        modin_df.fillna(method="ffill", inplace=True)
        df_equals(modin_df, df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_frame_fillna_limit(self, data):
        pandas_df = pandas.DataFrame(data)

        index = pandas_df.index

        result = pandas_df[:2].reindex(index)
        modin_df = pd.DataFrame(result)
        df_equals(
            modin_df.fillna(method="pad", limit=2), result.fillna(method="pad", limit=2)
        )

        result = pandas_df[-2:].reindex(index)
        modin_df = pd.DataFrame(result)
        df_equals(
            modin_df.fillna(method="backfill", limit=2),
            result.fillna(method="backfill", limit=2),
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_frame_pad_backfill_limit(self, data):
        pandas_df = pandas.DataFrame(data)

        index = pandas_df.index

        result = pandas_df[:2].reindex(index)
        modin_df = pd.DataFrame(result)
        df_equals(
            modin_df.fillna(method="pad", limit=2), result.fillna(method="pad", limit=2)
        )

        result = pandas_df[-2:].reindex(index)
        modin_df = pd.DataFrame(result)
        df_equals(
            modin_df.fillna(method="backfill", limit=2),
            result.fillna(method="backfill", limit=2),
        )

    def test_fillna_dtype_conversion(self):
        # make sure that fillna on an empty frame works
        df = pandas.DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
        modin_df = pd.DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
        df_equals(modin_df.fillna("nan"), df.fillna("nan"))

        frame_data = {"A": [1, np.nan], "B": [1.0, 2.0]}
        df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)
        for v in ["", 1, np.nan, 1.0]:
            df_equals(modin_df.fillna(v), df.fillna(v))

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_fillna_skip_certain_blocks(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        # don't try to fill boolean, int blocks
        df_equals(modin_df.fillna(np.nan), pandas_df.fillna(np.nan))

    def test_fillna_dict_series(self):
        frame_data = {
            "a": [np.nan, 1, 2, np.nan, np.nan],
            "b": [1, 2, 3, np.nan, np.nan],
            "c": [np.nan, 1, 2, 3, 4],
        }
        df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)

        df_equals(modin_df.fillna({"a": 0, "b": 5}), df.fillna({"a": 0, "b": 5}))

        df_equals(
            modin_df.fillna({"a": 0, "b": 5, "d": 7}),
            df.fillna({"a": 0, "b": 5, "d": 7}),
        )

        # Series treated same as dict
        df_equals(modin_df.fillna(modin_df.max()), df.fillna(df.max()))

    def test_fillna_dataframe(self):
        frame_data = {
            "a": [np.nan, 1, 2, np.nan, np.nan],
            "b": [1, 2, 3, np.nan, np.nan],
            "c": [np.nan, 1, 2, 3, 4],
        }
        df = pandas.DataFrame(frame_data, index=list("VWXYZ"))
        modin_df = pd.DataFrame(frame_data, index=list("VWXYZ"))

        # df2 may have different index and columns
        df2 = pandas.DataFrame(
            {
                "a": [np.nan, 10, 20, 30, 40],
                "b": [50, 60, 70, 80, 90],
                "foo": ["bar"] * 5,
            },
            index=list("VWXuZ"),
        )
        modin_df2 = pd.DataFrame(df2)

        # only those columns and indices which are shared get filled
        df_equals(modin_df.fillna(modin_df2), df.fillna(df2))

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_fillna_columns(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(
            modin_df.fillna(method="ffill", axis=1),
            pandas_df.fillna(method="ffill", axis=1),
        )

        df_equals(
            modin_df.fillna(method="ffill", axis=1),
            pandas_df.fillna(method="ffill", axis=1),
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_fillna_invalid_method(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)  # noqa F841

        with pytest.raises(ValueError):
            modin_df.fillna(method="ffil")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_fillna_invalid_value(self, data):
        modin_df = pd.DataFrame(data)
        # list
        pytest.raises(TypeError, modin_df.fillna, [1, 2])
        # tuple
        pytest.raises(TypeError, modin_df.fillna, (1, 2))
        # frame with series
        pytest.raises(TypeError, modin_df.iloc[:, 0].fillna, modin_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_fillna_col_reordering(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(modin_df.fillna(method="ffill"), pandas_df.fillna(method="ffill"))

    def test_fillna_datetime_columns(self):
        frame_data = {
            "A": [-1, -2, np.nan],
            "B": pd.date_range("20130101", periods=3),
            "C": ["foo", "bar", None],
            "D": ["foo2", "bar2", None],
        }
        df = pandas.DataFrame(frame_data, index=pd.date_range("20130110", periods=3))
        modin_df = pd.DataFrame(frame_data, index=pd.date_range("20130110", periods=3))
        df_equals(modin_df.fillna("?"), df.fillna("?"))

        frame_data = {
            "A": [-1, -2, np.nan],
            "B": [
                pandas.Timestamp("2013-01-01"),
                pandas.Timestamp("2013-01-02"),
                pandas.NaT,
            ],
            "C": ["foo", "bar", None],
            "D": ["foo2", "bar2", None],
        }
        df = pandas.DataFrame(frame_data, index=pd.date_range("20130110", periods=3))
        modin_df = pd.DataFrame(frame_data, index=pd.date_range("20130110", periods=3))
        df_equals(modin_df.fillna("?"), df.fillna("?"))

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    def test_median(self, request, data, axis, skipna, numeric_only):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.median(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.median(axis=axis, skipna=skipna, numeric_only=numeric_only)
        else:
            modin_result = modin_df.median(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.median(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.T.median(axis=axis, skipna=skipna, numeric_only=numeric_only)
        else:
            modin_result = modin_df.T.median(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    def test_mode(self, request, data, axis, numeric_only):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.mode(axis=axis, numeric_only=numeric_only)
        except Exception:
            with pytest.raises(TypeError):
                modin_df.mode(axis=axis, numeric_only=numeric_only)
        else:
            modin_result = modin_df.mode(axis=axis, numeric_only=numeric_only)
            df_equals(modin_result, pandas_result)

    def test_nlargest(self):
        data = {
            "population": [
                59000000,
                65000000,
                434000,
                434000,
                434000,
                337000,
                11300,
                11300,
                11300,
            ],
            "GDP": [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311],
            "alpha-2": ["IT", "FR", "MT", "MV", "BN", "IS", "NR", "TV", "AI"],
        }
        index = [
            "Italy",
            "France",
            "Malta",
            "Maldives",
            "Brunei",
            "Iceland",
            "Nauru",
            "Tuvalu",
            "Anguilla",
        ]
        modin_df = pd.DataFrame(data=data, index=index)
        pandas_df = pandas.DataFrame(data=data, index=index)
        df_equals(
            modin_df.nlargest(3, "population"), pandas_df.nlargest(3, "population")
        )

    def test_nsmallest(self):
        data = {
            "population": [
                59000000,
                65000000,
                434000,
                434000,
                434000,
                337000,
                11300,
                11300,
                11300,
            ],
            "GDP": [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311],
            "alpha-2": ["IT", "FR", "MT", "MV", "BN", "IS", "NR", "TV", "AI"],
        }
        index = [
            "Italy",
            "France",
            "Malta",
            "Maldives",
            "Brunei",
            "Iceland",
            "Nauru",
            "Tuvalu",
            "Anguilla",
        ]
        modin_df = pd.DataFrame(data=data, index=index)
        pandas_df = pandas.DataFrame(data=data, index=index)
        df_equals(
            modin_df.nsmallest(n=3, columns="population"),
            pandas_df.nsmallest(n=3, columns="population"),
        )
        df_equals(
            modin_df.nsmallest(n=2, columns=["population", "GDP"], keep="all"),
            pandas_df.nsmallest(n=2, columns=["population", "GDP"], keep="all"),
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "dropna", bool_arg_values, ids=arg_keys("dropna", bool_arg_keys)
    )
    def test_nunique(self, data, axis, dropna):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_result = modin_df.nunique(axis=axis, dropna=dropna)
        pandas_result = pandas_df.nunique(axis=axis, dropna=dropna)
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.T.nunique(axis=axis, dropna=dropna)
        pandas_result = pandas_df.T.nunique(axis=axis, dropna=dropna)
        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("q", quantiles_values, ids=quantiles_keys)
    def test_quantile(self, request, data, q):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if not name_contains(request.node.name, no_numeric_dfs):
            df_equals(modin_df.quantile(q), pandas_df.quantile(q))
            df_equals(modin_df.quantile(q, axis=1), pandas_df.quantile(q, axis=1))

            try:
                pandas_result = pandas_df.quantile(q, axis=1, numeric_only=False)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.quantile(q, axis=1, numeric_only=False)
            else:
                modin_result = modin_df.quantile(q, axis=1, numeric_only=False)
                df_equals(modin_result, pandas_result)
        else:
            with pytest.raises(ValueError):
                modin_df.quantile(q)

        if not name_contains(request.node.name, no_numeric_dfs):
            df_equals(modin_df.T.quantile(q), pandas_df.T.quantile(q))
            df_equals(modin_df.T.quantile(q, axis=1), pandas_df.T.quantile(q, axis=1))

            try:
                pandas_result = pandas_df.T.quantile(q, axis=1, numeric_only=False)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df.T.quantile(q, axis=1, numeric_only=False)
            else:
                modin_result = modin_df.T.quantile(q, axis=1, numeric_only=False)
                df_equals(modin_result, pandas_result)
        else:
            with pytest.raises(ValueError):
                modin_df.T.quantile(q)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "na_option", ["keep", "top", "bottom"], ids=["keep", "top", "bottom"]
    )
    def test_rank(self, data, axis, numeric_only, na_option):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.rank(
                axis=axis, numeric_only=numeric_only, na_option=na_option
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.rank(axis=axis, numeric_only=numeric_only, na_option=na_option)
        else:
            modin_result = modin_df.rank(
                axis=axis, numeric_only=numeric_only, na_option=na_option
            )
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    def test_skew(self, request, data, axis, skipna, numeric_only):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.skew(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.skew(axis=axis, skipna=skipna, numeric_only=numeric_only)
        else:
            modin_result = modin_df.skew(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.skew(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.T.skew(axis=axis, skipna=skipna, numeric_only=numeric_only)
        else:
            modin_result = modin_df.T.skew(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    @pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
    def test_std(self, request, data, axis, skipna, numeric_only, ddof):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.std(
                axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.std(
                    axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
                )
        else:
            modin_result = modin_df.std(
                axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
            )
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.std(
                axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.std(
                    axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
                )
        else:
            modin_result = modin_df.T.std(
                axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
            )
            df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_values(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        np.testing.assert_equal(modin_df.values, pandas_df.values)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
    )
    @pytest.mark.parametrize(
        "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
    )
    @pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
    def test_var(self, request, data, axis, skipna, numeric_only, ddof):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.var(
                axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.var(
                    axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
                )
        else:
            modin_result = modin_df.var(
                axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
            )
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.T.var(
                axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
            )
        except Exception:
            with pytest.raises(TypeError):
                modin_df.T.var(
                    axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
                )
        else:
            modin_result = modin_df.T.var(
                axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
            )
            df_equals(modin_result, pandas_result)


class TestDataFrameIndexing:
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_first_valid_index(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        assert modin_df.first_valid_index() == (pandas_df.first_valid_index())

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
    def test_head(self, data, n):
        # Test normal dataframe head
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        df_equals(modin_df.head(n), pandas_df.head(n))
        df_equals(modin_df.head(len(modin_df) + 1), pandas_df.head(len(pandas_df) + 1))

        # Test head when we call it from a QueryCompilerView
        modin_result = modin_df.loc[:, ["col1", "col3", "col3"]].head(n)
        pandas_result = pandas_df.loc[:, ["col1", "col3", "col3"]].head(n)
        df_equals(modin_result, pandas_result)

    @pytest.mark.skip(reason="Defaulting to Pandas")
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_iat(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)  # noqa F841

        with pytest.raises(NotImplementedError):
            modin_df.iat()

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_iloc(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if not name_contains(request.node.name, ["empty_data"]):
            # Scaler
            np.testing.assert_equal(modin_df.iloc[0, 1], pandas_df.iloc[0, 1])

            # Series
            df_equals(modin_df.iloc[0], pandas_df.iloc[0])
            df_equals(modin_df.iloc[1:, 0], pandas_df.iloc[1:, 0])
            df_equals(modin_df.iloc[1:2, 0], pandas_df.iloc[1:2, 0])

            # DataFrame
            df_equals(modin_df.iloc[[1, 2]], pandas_df.iloc[[1, 2]])
            # See issue #80
            # df_equals(modin_df.iloc[[1, 2], [1, 0]], pandas_df.iloc[[1, 2], [1, 0]])
            df_equals(modin_df.iloc[1:2, 0:2], pandas_df.iloc[1:2, 0:2])

            # Issue #43
            modin_df.iloc[0:3, :]

            # Write Item
            modin_df.iloc[[1, 2]] = 42
            pandas_df.iloc[[1, 2]] = 42
            df_equals(modin_df, pandas_df)

            modin_df = pd.DataFrame(data)
            pandas_df = pandas.DataFrame(data)
            modin_df.iloc[0] = modin_df.iloc[1]
            pandas_df.iloc[0] = pandas_df.iloc[1]
            df_equals(modin_df, pandas_df)

            modin_df = pd.DataFrame(data)
            pandas_df = pandas.DataFrame(data)
            modin_df.iloc[:, 0] = modin_df.iloc[:, 1]
            pandas_df.iloc[:, 0] = pandas_df.iloc[:, 1]
            df_equals(modin_df, pandas_df)
        else:
            with pytest.raises(IndexError):
                modin_df.iloc[0, 1]

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_index(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(modin_df.index, pandas_df.index)
        modin_df_cp = modin_df.copy()
        pandas_df_cp = pandas_df.copy()

        modin_df_cp.index = [str(i) for i in modin_df_cp.index]
        pandas_df_cp.index = [str(i) for i in pandas_df_cp.index]
        df_equals(modin_df_cp.index, pandas_df_cp.index)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_indexing_duplicate_axis(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        modin_df.index = pandas_df.index = [i // 3 for i in range(len(modin_df))]
        assert any(modin_df.index.duplicated())
        assert any(pandas_df.index.duplicated())

        df_equals(modin_df.iloc[0], pandas_df.iloc[0])
        df_equals(modin_df.loc[0], pandas_df.loc[0])
        df_equals(modin_df.iloc[0, 0:4], pandas_df.iloc[0, 0:4])
        df_equals(
            modin_df.loc[0, modin_df.columns[0:4]],
            pandas_df.loc[0, pandas_df.columns[0:4]],
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_keys(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(modin_df.keys(), pandas_df.keys())

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_loc(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        # We skip nan datasets because nan != nan
        if "nan" not in request.node.name:
            key1 = modin_df.columns[0]
            key2 = modin_df.columns[1]
            # Scaler
            assert modin_df.loc[0, key1] == pandas_df.loc[0, key1]

            # Series
            df_equals(modin_df.loc[0], pandas_df.loc[0])
            df_equals(modin_df.loc[1:, key1], pandas_df.loc[1:, key1])
            df_equals(modin_df.loc[1:2, key1], pandas_df.loc[1:2, key1])

            # DataFrame
            df_equals(modin_df.loc[[1, 2]], pandas_df.loc[[1, 2]])

            # List-like of booleans
            indices = [i % 3 == 0 for i in range(len(modin_df.index))]
            columns = [i % 5 == 0 for i in range(len(modin_df.columns))]
            modin_result = modin_df.loc[indices, columns]
            pandas_result = pandas_df.loc[indices, columns]
            df_equals(modin_result, pandas_result)

            modin_result = modin_df.loc[:, columns]
            pandas_result = pandas_df.loc[:, columns]
            df_equals(modin_result, pandas_result)

            modin_result = modin_df.loc[indices]
            pandas_result = pandas_df.loc[indices]
            df_equals(modin_result, pandas_result)

            # See issue #80
            # df_equals(modin_df.loc[[1, 2], ['col1']], pandas_df.loc[[1, 2], ['col1']])
            df_equals(modin_df.loc[1:2, key1:key2], pandas_df.loc[1:2, key1:key2])

            # From issue #421
            df_equals(modin_df.loc[:, [key2, key1]], pandas_df.loc[:, [key2, key1]])
            df_equals(modin_df.loc[[2, 1], :], pandas_df.loc[[2, 1], :])

            # From issue #1023
            key1 = modin_df.columns[0]
            key2 = modin_df.columns[-2]
            df_equals(modin_df.loc[:, key1:key2], pandas_df.loc[:, key1:key2])

            # Write Item
            modin_df_copy = modin_df.copy()
            pandas_df_copy = pandas_df.copy()
            modin_df_copy.loc[[1, 2]] = 42
            pandas_df_copy.loc[[1, 2]] = 42
            df_equals(modin_df_copy, pandas_df_copy)

        # From issue #1374
        with pytest.raises(KeyError):
            modin_df.loc["NO_EXIST"]

    def test_loc_multi_index(self):
        modin_df = pd.read_csv(
            "modin/pandas/test/data/blah.csv", header=[0, 1, 2, 3], index_col=0
        )
        pandas_df = pandas.read_csv(
            "modin/pandas/test/data/blah.csv", header=[0, 1, 2, 3], index_col=0
        )

        df_equals(modin_df.loc[1], pandas_df.loc[1])
        df_equals(modin_df.loc[1, "Presidents"], pandas_df.loc[1, "Presidents"])
        df_equals(
            modin_df.loc[1, ("Presidents", "Pure mentions")],
            pandas_df.loc[1, ("Presidents", "Pure mentions")],
        )
        assert (
            modin_df.loc[1, ("Presidents", "Pure mentions", "IND", "all")]
            == pandas_df.loc[1, ("Presidents", "Pure mentions", "IND", "all")]
        )
        df_equals(
            modin_df.loc[(1, 2), "Presidents"], pandas_df.loc[(1, 2), "Presidents"]
        )

        tuples = [
            ("bar", "one"),
            ("bar", "two"),
            ("bar", "three"),
            ("bar", "four"),
            ("baz", "one"),
            ("baz", "two"),
            ("baz", "three"),
            ("baz", "four"),
            ("foo", "one"),
            ("foo", "two"),
            ("foo", "three"),
            ("foo", "four"),
            ("qux", "one"),
            ("qux", "two"),
            ("qux", "three"),
            ("qux", "four"),
        ]

        modin_index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
        pandas_index = pandas.MultiIndex.from_tuples(tuples, names=["first", "second"])
        frame_data = np.random.randint(0, 100, size=(16, 100))
        modin_df = pd.DataFrame(
            frame_data,
            index=modin_index,
            columns=["col{}".format(i) for i in range(100)],
        )
        pandas_df = pandas.DataFrame(
            frame_data,
            index=pandas_index,
            columns=["col{}".format(i) for i in range(100)],
        )
        df_equals(modin_df.loc["bar", "col1"], pandas_df.loc["bar", "col1"])
        assert (
            modin_df.loc[("bar", "one"), "col1"]
            == pandas_df.loc[("bar", "one"), "col1"]
        )
        df_equals(
            modin_df.loc["bar", ("col1", "col2")],
            pandas_df.loc["bar", ("col1", "col2")],
        )

        # From issue #1456
        transposed_modin = modin_df.T
        transposed_pandas = pandas_df.T
        df_equals(
            transposed_modin.loc[transposed_modin.index[:-2], :],
            transposed_pandas.loc[transposed_pandas.index[:-2], :],
        )

    def test_loc_assignment(self):
        modin_df = pd.DataFrame(
            index=["row1", "row2", "row3"], columns=["col1", "col2"]
        )
        pandas_df = pandas.DataFrame(
            index=["row1", "row2", "row3"], columns=["col1", "col2"]
        )
        modin_df.loc["row1"]["col1"] = 11
        modin_df.loc["row2"]["col1"] = 21
        modin_df.loc["row3"]["col1"] = 31
        modin_df.loc["row1"]["col2"] = 12
        modin_df.loc["row2"]["col2"] = 22
        modin_df.loc["row3"]["col2"] = 32
        pandas_df.loc["row1"]["col1"] = 11
        pandas_df.loc["row2"]["col1"] = 21
        pandas_df.loc["row3"]["col1"] = 31
        pandas_df.loc["row1"]["col2"] = 12
        pandas_df.loc["row2"]["col2"] = 22
        pandas_df.loc["row3"]["col2"] = 32
        df_equals(modin_df, pandas_df)

    def test_iloc_assignment(self):
        modin_df = pd.DataFrame(
            index=["row1", "row2", "row3"], columns=["col1", "col2"]
        )
        pandas_df = pandas.DataFrame(
            index=["row1", "row2", "row3"], columns=["col1", "col2"]
        )
        modin_df.iloc[0]["col1"] = 11
        modin_df.iloc[1]["col1"] = 21
        modin_df.iloc[2]["col1"] = 31
        modin_df.iloc[0]["col2"] = 12
        modin_df.iloc[1]["col2"] = 22
        modin_df.iloc[2]["col2"] = 32
        pandas_df.iloc[0]["col1"] = 11
        pandas_df.iloc[1]["col1"] = 21
        pandas_df.iloc[2]["col1"] = 31
        pandas_df.iloc[0]["col2"] = 12
        pandas_df.iloc[1]["col2"] = 22
        pandas_df.iloc[2]["col2"] = 32
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_pop(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if "empty_data" not in request.node.name:
            key = modin_df.columns[0]
            temp_modin_df = modin_df.copy()
            temp_pandas_df = pandas_df.copy()
            modin_popped = temp_modin_df.pop(key)
            pandas_popped = temp_pandas_df.pop(key)
            df_equals(modin_popped, pandas_popped)
            df_equals(temp_modin_df, temp_pandas_df)

    def test_reindex(self):
        frame_data = {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 10, 11],
            "col4": [12, 13, 14, 15],
            "col5": [0, 0, 0, 0],
        }
        pandas_df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)

        df_equals(modin_df.reindex([0, 3, 2, 1]), pandas_df.reindex([0, 3, 2, 1]))
        df_equals(modin_df.reindex([0, 6, 2]), pandas_df.reindex([0, 6, 2]))
        df_equals(
            modin_df.reindex(["col1", "col3", "col4", "col2"], axis=1),
            pandas_df.reindex(["col1", "col3", "col4", "col2"], axis=1),
        )
        df_equals(
            modin_df.reindex(["col1", "col7", "col4", "col8"], axis=1),
            pandas_df.reindex(["col1", "col7", "col4", "col8"], axis=1),
        )
        df_equals(
            modin_df.reindex(index=[0, 1, 5], columns=["col1", "col7", "col4", "col8"]),
            pandas_df.reindex(
                index=[0, 1, 5], columns=["col1", "col7", "col4", "col8"]
            ),
        )
        df_equals(
            modin_df.T.reindex(["col1", "col7", "col4", "col8"], axis=0),
            pandas_df.T.reindex(["col1", "col7", "col4", "col8"], axis=0),
        )

    def test_reindex_like(self):
        df1 = pd.DataFrame(
            [
                [24.3, 75.7, "high"],
                [31, 87.8, "high"],
                [22, 71.6, "medium"],
                [35, 95, "medium"],
            ],
            columns=["temp_celsius", "temp_fahrenheit", "windspeed"],
            index=pd.date_range(start="2014-02-12", end="2014-02-15", freq="D"),
        )
        df2 = pd.DataFrame(
            [[28, "low"], [30, "low"], [35.1, "medium"]],
            columns=["temp_celsius", "windspeed"],
            index=pd.DatetimeIndex(["2014-02-12", "2014-02-13", "2014-02-15"]),
        )
        with pytest.warns(UserWarning):
            df2.reindex_like(df1)

    def test_rename_sanity(self):
        test_data = pandas.DataFrame(tm.getSeriesData())
        mapping = {"A": "a", "B": "b", "C": "c", "D": "d"}

        modin_df = pd.DataFrame(test_data)
        df_equals(modin_df.rename(columns=mapping), test_data.rename(columns=mapping))

        renamed2 = test_data.rename(columns=str.lower)
        df_equals(modin_df.rename(columns=str.lower), renamed2)

        modin_df = pd.DataFrame(renamed2)
        df_equals(
            modin_df.rename(columns=str.upper), renamed2.rename(columns=str.upper)
        )

        # index
        data = {"A": {"foo": 0, "bar": 1}}

        # gets sorted alphabetical
        df = pandas.DataFrame(data)
        modin_df = pd.DataFrame(data)
        tm.assert_index_equal(
            modin_df.rename(index={"foo": "bar", "bar": "foo"}).index,
            df.rename(index={"foo": "bar", "bar": "foo"}).index,
        )

        tm.assert_index_equal(
            modin_df.rename(index=str.upper).index, df.rename(index=str.upper).index
        )

        # Using the `mapper` functionality with `axis`
        tm.assert_index_equal(
            modin_df.rename(str.upper, axis=0).index, df.rename(str.upper, axis=0).index
        )
        tm.assert_index_equal(
            modin_df.rename(str.upper, axis=1).columns,
            df.rename(str.upper, axis=1).columns,
        )

        # have to pass something
        with pytest.raises(TypeError):
            modin_df.rename()

        # partial columns
        renamed = test_data.rename(columns={"C": "foo", "D": "bar"})
        modin_df = pd.DataFrame(test_data)
        tm.assert_index_equal(
            modin_df.rename(columns={"C": "foo", "D": "bar"}).index,
            test_data.rename(columns={"C": "foo", "D": "bar"}).index,
        )

        # other axis
        renamed = test_data.T.rename(index={"C": "foo", "D": "bar"})
        tm.assert_index_equal(
            test_data.T.rename(index={"C": "foo", "D": "bar"}).index,
            modin_df.T.rename(index={"C": "foo", "D": "bar"}).index,
        )

        # index with name
        index = pandas.Index(["foo", "bar"], name="name")
        renamer = pandas.DataFrame(data, index=index)
        modin_df = pd.DataFrame(data, index=index)

        renamed = renamer.rename(index={"foo": "bar", "bar": "foo"})
        modin_renamed = modin_df.rename(index={"foo": "bar", "bar": "foo"})
        tm.assert_index_equal(renamed.index, modin_renamed.index)

        assert renamed.index.name == modin_renamed.index.name

    def test_rename_multiindex(self):
        tuples_index = [("foo1", "bar1"), ("foo2", "bar2")]
        tuples_columns = [("fizz1", "buzz1"), ("fizz2", "buzz2")]
        index = pandas.MultiIndex.from_tuples(tuples_index, names=["foo", "bar"])
        columns = pandas.MultiIndex.from_tuples(tuples_columns, names=["fizz", "buzz"])

        frame_data = [(0, 0), (1, 1)]
        df = pandas.DataFrame(frame_data, index=index, columns=columns)
        modin_df = pd.DataFrame(frame_data, index=index, columns=columns)

        #
        # without specifying level -> accross all levels
        renamed = df.rename(
            index={"foo1": "foo3", "bar2": "bar3"},
            columns={"fizz1": "fizz3", "buzz2": "buzz3"},
        )
        modin_renamed = modin_df.rename(
            index={"foo1": "foo3", "bar2": "bar3"},
            columns={"fizz1": "fizz3", "buzz2": "buzz3"},
        )
        tm.assert_index_equal(renamed.index, modin_renamed.index)

        renamed = df.rename(
            index={"foo1": "foo3", "bar2": "bar3"},
            columns={"fizz1": "fizz3", "buzz2": "buzz3"},
        )
        tm.assert_index_equal(renamed.columns, modin_renamed.columns)
        assert renamed.index.names == modin_renamed.index.names
        assert renamed.columns.names == modin_renamed.columns.names

        #
        # with specifying a level

        # dict
        renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=0)
        modin_renamed = modin_df.rename(
            columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=0
        )
        tm.assert_index_equal(renamed.columns, modin_renamed.columns)
        renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="fizz")
        modin_renamed = modin_df.rename(
            columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="fizz"
        )
        tm.assert_index_equal(renamed.columns, modin_renamed.columns)

        renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=1)
        modin_renamed = modin_df.rename(
            columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=1
        )
        tm.assert_index_equal(renamed.columns, modin_renamed.columns)
        renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="buzz")
        modin_renamed = modin_df.rename(
            columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="buzz"
        )
        tm.assert_index_equal(renamed.columns, modin_renamed.columns)

        # function
        func = str.upper
        renamed = df.rename(columns=func, level=0)
        modin_renamed = modin_df.rename(columns=func, level=0)
        tm.assert_index_equal(renamed.columns, modin_renamed.columns)
        renamed = df.rename(columns=func, level="fizz")
        modin_renamed = modin_df.rename(columns=func, level="fizz")
        tm.assert_index_equal(renamed.columns, modin_renamed.columns)

        renamed = df.rename(columns=func, level=1)
        modin_renamed = modin_df.rename(columns=func, level=1)
        tm.assert_index_equal(renamed.columns, modin_renamed.columns)
        renamed = df.rename(columns=func, level="buzz")
        modin_renamed = modin_df.rename(columns=func, level="buzz")
        tm.assert_index_equal(renamed.columns, modin_renamed.columns)

        # index
        renamed = df.rename(index={"foo1": "foo3", "bar2": "bar3"}, level=0)
        modin_renamed = modin_df.rename(index={"foo1": "foo3", "bar2": "bar3"}, level=0)
        tm.assert_index_equal(modin_renamed.index, renamed.index)

    @pytest.mark.skip(reason="Pandas does not pass this test")
    def test_rename_nocopy(self):
        test_data = pandas.DataFrame(tm.getSeriesData())
        modin_df = pd.DataFrame(test_data)
        modin_renamed = modin_df.rename(columns={"C": "foo"}, copy=False)
        modin_renamed["foo"] = 1
        assert (modin_df["C"] == 1).all()

    def test_rename_inplace(self):
        test_data = pandas.DataFrame(tm.getSeriesData())
        modin_df = pd.DataFrame(test_data)

        df_equals(
            modin_df.rename(columns={"C": "foo"}),
            test_data.rename(columns={"C": "foo"}),
        )

        frame = test_data.copy()
        modin_frame = modin_df.copy()
        frame.rename(columns={"C": "foo"}, inplace=True)
        modin_frame.rename(columns={"C": "foo"}, inplace=True)

        df_equals(modin_frame, frame)

    def test_rename_bug(self):
        # rename set ref_locs, and set_index was not resetting
        frame_data = {0: ["foo", "bar"], 1: ["bah", "bas"], 2: [1, 2]}
        df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)
        df = df.rename(columns={0: "a"})
        df = df.rename(columns={1: "b"})
        df = df.set_index(["a", "b"])
        df.columns = ["2001-01-01"]

        modin_df = modin_df.rename(columns={0: "a"})
        modin_df = modin_df.rename(columns={1: "b"})
        modin_df = modin_df.set_index(["a", "b"])
        modin_df.columns = ["2001-01-01"]

        df_equals(modin_df, df)

    def test_rename_axis(self):
        data = {"num_legs": [4, 4, 2], "num_arms": [0, 0, 2]}
        index = ["dog", "cat", "monkey"]
        modin_df = pd.DataFrame(data, index)
        pandas_df = pandas.DataFrame(data, index)
        df_equals(modin_df.rename_axis("animal"), pandas_df.rename_axis("animal"))
        df_equals(
            modin_df.rename_axis("limbs", axis="columns"),
            pandas_df.rename_axis("limbs", axis="columns"),
        )

        modin_df.rename_axis("limbs", axis="columns", inplace=True)
        pandas_df.rename_axis("limbs", axis="columns", inplace=True)
        df_equals(modin_df, pandas_df)

        new_index = pd.MultiIndex.from_product(
            [["mammal"], ["dog", "cat", "monkey"]], names=["type", "name"]
        )
        modin_df.index = new_index
        pandas_df.index = new_index

        df_equals(
            modin_df.rename_axis(index={"type": "class"}),
            pandas_df.rename_axis(index={"type": "class"}),
        )
        df_equals(
            modin_df.rename_axis(columns=str.upper),
            pandas_df.rename_axis(columns=str.upper),
        )
        df_equals(
            modin_df.rename_axis(
                columns=[str.upper(o) for o in modin_df.columns.names]
            ),
            pandas_df.rename_axis(
                columns=[str.upper(o) for o in pandas_df.columns.names]
            ),
        )

        with pytest.raises(ValueError):
            df_equals(
                modin_df.rename_axis(str.upper, axis=1),
                pandas_df.rename_axis(str.upper, axis=1),
            )

    def test_rename_axis_inplace(self):
        test_frame = pandas.DataFrame(tm.getSeriesData())
        modin_df = pd.DataFrame(test_frame)

        result = test_frame.copy()
        modin_result = modin_df.copy()
        no_return = result.rename_axis("foo", inplace=True)
        modin_no_return = modin_result.rename_axis("foo", inplace=True)

        assert no_return is modin_no_return
        df_equals(modin_result, result)

        result = test_frame.copy()
        modin_result = modin_df.copy()
        no_return = result.rename_axis("bar", axis=1, inplace=True)
        modin_no_return = modin_result.rename_axis("bar", axis=1, inplace=True)

        assert no_return is modin_no_return
        df_equals(modin_result, result)

    def test_reorder_levels(self):
        data = np.random.randint(1, 100, 12)
        modin_df = pd.DataFrame(
            data,
            index=pd.MultiIndex.from_tuples(
                [
                    (num, letter, color)
                    for num in range(1, 3)
                    for letter in ["a", "b", "c"]
                    for color in ["Red", "Green"]
                ],
                names=["Number", "Letter", "Color"],
            ),
        )
        pandas_df = pandas.DataFrame(
            data,
            index=pandas.MultiIndex.from_tuples(
                [
                    (num, letter, color)
                    for num in range(1, 3)
                    for letter in ["a", "b", "c"]
                    for color in ["Red", "Green"]
                ],
                names=["Number", "Letter", "Color"],
            ),
        )
        df_equals(
            modin_df.reorder_levels(["Letter", "Color", "Number"]),
            pandas_df.reorder_levels(["Letter", "Color", "Number"]),
        )

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_reset_index(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_result = modin_df.reset_index(inplace=False)
        pandas_result = pandas_df.reset_index(inplace=False)
        df_equals(modin_result, pandas_result)

        modin_df_cp = modin_df.copy()
        pd_df_cp = pandas_df.copy()
        modin_df_cp.reset_index(inplace=True)
        pd_df_cp.reset_index(inplace=True)
        df_equals(modin_df_cp, pd_df_cp)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    def test_sample(self, data, axis):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        with pytest.raises(ValueError):
            modin_df.sample(n=3, frac=0.4, axis=axis)

        with pytest.raises(KeyError):
            modin_df.sample(frac=0.5, weights="CoLuMn_No_ExIsT", axis=0)

        with pytest.raises(ValueError):
            modin_df.sample(frac=0.5, weights=modin_df.columns[0], axis=1)

        with pytest.raises(ValueError):
            modin_df.sample(
                frac=0.5, weights=[0.5 for _ in range(len(modin_df.index[:-1]))], axis=0
            )

        with pytest.raises(ValueError):
            modin_df.sample(
                frac=0.5,
                weights=[0.5 for _ in range(len(modin_df.columns[:-1]))],
                axis=1,
            )

        with pytest.raises(ValueError):
            modin_df.sample(n=-3, axis=axis)

        with pytest.raises(ValueError):
            modin_df.sample(frac=0.2, weights=pandas.Series(), axis=axis)

        if isinstance(axis, str):
            num_axis = pandas.DataFrame()._get_axis_number(axis)
        else:
            num_axis = axis

        # weights that sum to 1
        sums = sum(i % 2 for i in range(len(modin_df.axes[num_axis])))
        weights = [i % 2 / sums for i in range(len(modin_df.axes[num_axis]))]

        modin_result = modin_df.sample(
            frac=0.5, random_state=42, weights=weights, axis=axis
        )
        pandas_result = pandas_df.sample(
            frac=0.5, random_state=42, weights=weights, axis=axis
        )
        df_equals(modin_result, pandas_result)

        # weights that don't sum to 1
        weights = [i % 2 for i in range(len(modin_df.axes[num_axis]))]
        modin_result = modin_df.sample(
            frac=0.5, random_state=42, weights=weights, axis=axis
        )
        pandas_result = pandas_df.sample(
            frac=0.5, random_state=42, weights=weights, axis=axis
        )
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.sample(n=0, axis=axis)
        pandas_result = pandas_df.sample(n=0, axis=axis)
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.sample(frac=0.5, random_state=42, axis=axis)
        pandas_result = pandas_df.sample(frac=0.5, random_state=42, axis=axis)
        df_equals(modin_result, pandas_result)

        modin_result = modin_df.sample(n=2, random_state=42, axis=axis)
        pandas_result = pandas_df.sample(n=2, random_state=42, axis=axis)
        df_equals(modin_result, pandas_result)

    def test_select_dtypes(self):
        frame_data = {
            "test1": list("abc"),
            "test2": np.arange(3, 6).astype("u1"),
            "test3": np.arange(8.0, 11.0, dtype="float64"),
            "test4": [True, False, True],
            "test5": pandas.date_range("now", periods=3).values,
            "test6": list(range(5, 8)),
        }
        df = pandas.DataFrame(frame_data)
        rd = pd.DataFrame(frame_data)

        include = np.float, "integer"
        exclude = (np.bool_,)
        r = rd.select_dtypes(include=include, exclude=exclude)

        e = df[["test2", "test3", "test6"]]
        df_equals(r, e)

        r = rd.select_dtypes(include=np.bool_)
        e = df[["test4"]]
        df_equals(r, e)

        r = rd.select_dtypes(exclude=np.bool_)
        e = df[["test1", "test2", "test3", "test5", "test6"]]
        df_equals(r, e)

        try:
            pd.DataFrame().select_dtypes()
            assert False
        except ValueError:
            assert True

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
    def test_tail(self, data, n):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        df_equals(modin_df.tail(n), pandas_df.tail(n))
        df_equals(modin_df.tail(len(modin_df)), pandas_df.tail(len(pandas_df)))

    def test_xs(self):
        d = {
            "num_legs": [4, 4, 2, 2],
            "num_wings": [0, 0, 2, 2],
            "class": ["mammal", "mammal", "mammal", "bird"],
            "animal": ["cat", "dog", "bat", "penguin"],
            "locomotion": ["walks", "walks", "flies", "walks"],
        }
        df = pd.DataFrame(data=d)
        df = df.set_index(["class", "animal", "locomotion"])
        with pytest.warns(UserWarning):
            df.xs("mammal")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___getitem__(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        key = modin_df.columns[0]
        modin_col = modin_df.__getitem__(key)
        assert isinstance(modin_col, pd.Series)

        pd_col = pandas_df[key]
        df_equals(pd_col, modin_col)

        slices = [
            (None, -1),
            (-1, None),
            (1, 2),
            (1, None),
            (None, 1),
            (1, -1),
            (-3, -1),
            (1, -1, 2),
        ]

        # slice test
        for slice_param in slices:
            s = slice(*slice_param)
            df_equals(modin_df[s], pandas_df[s])

        # Test empty
        df_equals(pd.DataFrame([])[:10], pandas.DataFrame([])[:10])

    def test_getitem_empty_mask(self):
        # modin-project/modin#517
        modin_frames = []
        pandas_frames = []
        data1 = np.random.randint(0, 100, size=(100, 4))
        mdf1 = pd.DataFrame(data1, columns=list("ABCD"))
        pdf1 = pandas.DataFrame(data1, columns=list("ABCD"))
        modin_frames.append(mdf1)
        pandas_frames.append(pdf1)

        data2 = np.random.randint(0, 100, size=(100, 4))
        mdf2 = pd.DataFrame(data2, columns=list("ABCD"))
        pdf2 = pandas.DataFrame(data2, columns=list("ABCD"))
        modin_frames.append(mdf2)
        pandas_frames.append(pdf2)

        data3 = np.random.randint(0, 100, size=(100, 4))
        mdf3 = pd.DataFrame(data3, columns=list("ABCD"))
        pdf3 = pandas.DataFrame(data3, columns=list("ABCD"))
        modin_frames.append(mdf3)
        pandas_frames.append(pdf3)

        modin_data = pd.concat(modin_frames)
        pandas_data = pandas.concat(pandas_frames)
        df_equals(
            modin_data[[False for _ in modin_data.index]],
            pandas_data[[False for _ in modin_data.index]],
        )

    def test_getitem_datetime_slice(self):
        data = {"data": range(1000)}
        index = pd.date_range("2017/1/4", periods=1000)
        modin_df = pd.DataFrame(data=data, index=index)
        pandas_df = pandas.DataFrame(data=data, index=index)

        s = slice("2017-01-06", "2017-01-09")
        df_equals(modin_df[s], pandas_df[s])

    def test_getitem_same_name(self):
        data = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ]
        columns = ["c1", "c2", "c1", "c3"]
        modin_df = pd.DataFrame(data, columns=columns)
        pandas_df = pandas.DataFrame(data, columns=columns)
        df_equals(modin_df["c1"], pandas_df["c1"])
        df_equals(modin_df["c2"], pandas_df["c2"])
        df_equals(modin_df[["c1", "c2"]], pandas_df[["c1", "c2"]])
        df_equals(modin_df["c3"], pandas_df["c3"])

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___getattr__(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)  # noqa F841

        if "empty_data" not in request.node.name:
            key = modin_df.columns[0]
            col = modin_df.__getattr__(key)

            col = modin_df.__getattr__("col1")
            assert isinstance(col, pd.Series)

            col = getattr(modin_df, "col1")
            assert isinstance(col, pd.Series)

            # Check that lookup in column doesn't override other attributes
            df2 = modin_df.rename(index=str, columns={key: "columns"})
            assert isinstance(df2.columns, pandas.Index)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___setitem__(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_df.__setitem__(modin_df.columns[-1], 1)
        pandas_df.__setitem__(pandas_df.columns[-1], 1)
        df_equals(modin_df, pandas_df)

        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_df[modin_df.columns[-1]] = pd.DataFrame(modin_df[modin_df.columns[0]])
        pandas_df[pandas_df.columns[-1]] = pandas.DataFrame(
            pandas_df[pandas_df.columns[0]]
        )
        df_equals(modin_df, pandas_df)

        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        rows = len(modin_df)
        arr = np.arange(rows * 2).reshape(-1, 2)
        modin_df[modin_df.columns[-1]] = arr
        pandas_df[pandas_df.columns[-1]] = arr
        df_equals(pandas_df, modin_df)

        with pytest.raises(ValueError, match=r"Wrong number of items passed"):
            modin_df["___NON EXISTENT COLUMN"] = arr

        modin_df[modin_df.columns[0]] = np.arange(len(modin_df))
        pandas_df[pandas_df.columns[0]] = np.arange(len(pandas_df))
        df_equals(modin_df, pandas_df)

        modin_df = pd.DataFrame(columns=modin_df.columns)
        pandas_df = pandas.DataFrame(columns=pandas_df.columns)

        for col in modin_df.columns:
            modin_df[col] = np.arange(1000)

        for col in pandas_df.columns:
            pandas_df[col] = np.arange(1000)

        df_equals(modin_df, pandas_df)

        # Test series assignment to column
        modin_df = pd.DataFrame(columns=modin_df.columns)
        pandas_df = pandas.DataFrame(columns=pandas_df.columns)
        modin_df[modin_df.columns[-1]] = modin_df[modin_df.columns[0]]
        pandas_df[pandas_df.columns[-1]] = pandas_df[pandas_df.columns[0]]
        df_equals(modin_df, pandas_df)

        if not sys.version_info.major == 3 and sys.version_info.minor > 6:
            # This test doesn't work correctly on Python 3.6
            # Test 2d ndarray assignment to column
            modin_df = pd.DataFrame(data)
            pandas_df = pandas.DataFrame(data)
            modin_df["new_col"] = modin_df[[modin_df.columns[0]]].values
            pandas_df["new_col"] = pandas_df[[pandas_df.columns[0]]].values
            df_equals(modin_df, pandas_df)
            assert isinstance(modin_df["new_col"][0], type(pandas_df["new_col"][0]))

        # Transpose test
        modin_df = pd.DataFrame(data).T
        pandas_df = pandas.DataFrame(data).T

        # We default to pandas on non-string column names
        if not all(isinstance(c, str) for c in modin_df.columns):
            with pytest.warns(UserWarning):
                modin_df[modin_df.columns[0]] = 0
        else:
            modin_df[modin_df.columns[0]] = 0

        pandas_df[pandas_df.columns[0]] = 0

        df_equals(modin_df, pandas_df)

        modin_df.columns = [str(i) for i in modin_df.columns]
        pandas_df.columns = [str(i) for i in pandas_df.columns]

        modin_df[modin_df.columns[0]] = 0
        pandas_df[pandas_df.columns[0]] = 0

        df_equals(modin_df, pandas_df)

        modin_df[modin_df.columns[0]][modin_df.index[0]] = 12345
        pandas_df[pandas_df.columns[0]][pandas_df.index[0]] = 12345

        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize(
        "data",
        [
            {},
            pytest.param(
                {"id": [], "max_speed": [], "health": []},
                marks=pytest.mark.xfail(
                    reason="Throws an exception because generally assigning Series or other objects of length different from DataFrame does not work right now"
                ),
            ),
        ],
        ids=["empty", "empty_columns"],
    )
    @pytest.mark.parametrize(
        "value", [np.array(["one", "two"]), [11, 22]], ids=["ndarray", "list"],
    )
    @pytest.mark.parametrize("convert_to_series", [False, True])
    @pytest.mark.parametrize("new_col_id", [123, "new_col"], ids=["integer", "string"])
    def test_setitem_on_empty_df(self, data, value, convert_to_series, new_col_id):
        pandas_df = pandas.DataFrame(data)
        modin_df = pd.DataFrame(data)

        pandas_df[new_col_id] = pandas.Series(value) if convert_to_series else value
        modin_df[new_col_id] = pd.Series(value) if convert_to_series else value
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___len__(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        assert len(modin_df) == len(pandas_df)


class TestDataFrameIter:
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_items(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_items = modin_df.items()
        pandas_items = pandas_df.items()
        for modin_item, pandas_item in zip(modin_items, pandas_items):
            modin_index, modin_series = modin_item
            pandas_index, pandas_series = pandas_item
            df_equals(pandas_series, modin_series)
            assert pandas_index == modin_index

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_iteritems(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_items = modin_df.iteritems()
        pandas_items = pandas_df.iteritems()
        for modin_item, pandas_item in zip(modin_items, pandas_items):
            modin_index, modin_series = modin_item
            pandas_index, pandas_series = pandas_item
            df_equals(pandas_series, modin_series)
            assert pandas_index == modin_index

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_iterrows(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_iterrows = modin_df.iterrows()
        pandas_iterrows = pandas_df.iterrows()
        for modin_row, pandas_row in zip(modin_iterrows, pandas_iterrows):
            modin_index, modin_series = modin_row
            pandas_index, pandas_series = pandas_row
            df_equals(pandas_series, modin_series)
            assert pandas_index == modin_index

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_itertuples(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        # test default
        modin_it_default = modin_df.itertuples()
        pandas_it_default = pandas_df.itertuples()
        for modin_row, pandas_row in zip(modin_it_default, pandas_it_default):
            np.testing.assert_equal(modin_row, pandas_row)

        # test all combinations of custom params
        indices = [True, False]
        names = [None, "NotPandas", "Pandas"]

        for index in indices:
            for name in names:
                modin_it_custom = modin_df.itertuples(index=index, name=name)
                pandas_it_custom = pandas_df.itertuples(index=index, name=name)
                for modin_row, pandas_row in zip(modin_it_custom, pandas_it_custom):
                    np.testing.assert_equal(modin_row, pandas_row)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___iter__(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_iterator = modin_df.__iter__()

        # Check that modin_iterator implements the iterator interface
        assert hasattr(modin_iterator, "__iter__")
        assert hasattr(modin_iterator, "next") or hasattr(modin_iterator, "__next__")

        pd_iterator = pandas_df.__iter__()
        assert list(modin_iterator) == list(pd_iterator)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___contains__(self, request, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        result = False
        key = "Not Exist"
        assert result == modin_df.__contains__(key)
        assert result == (key in modin_df)

        if "empty_data" not in request.node.name:
            result = True
            key = pandas_df.columns[0]
            assert result == modin_df.__contains__(key)
            assert result == (key in modin_df)

    def test__options_display(self):
        frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 102))
        pandas_df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)

        pandas.options.display.max_rows = 10
        pandas.options.display.max_columns = 10
        x = repr(pandas_df)
        pd.options.display.max_rows = 5
        pd.options.display.max_columns = 5
        y = repr(modin_df)
        assert x != y
        pd.options.display.max_rows = 10
        pd.options.display.max_columns = 10
        y = repr(modin_df)
        assert x == y

        # test for old fixed max values
        pandas.options.display.max_rows = 75
        pandas.options.display.max_columns = 75
        x = repr(pandas_df)
        pd.options.display.max_rows = 75
        pd.options.display.max_columns = 75
        y = repr(modin_df)
        assert x == y

    def test___finalize__(self):
        data = test_data_values[0]
        with pytest.warns(UserWarning):
            pd.DataFrame(data).__finalize__(None)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___copy__(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_df_copy, pandas_df_copy = modin_df.__copy__(), pandas_df.__copy__()
        df_equals(modin_df_copy, pandas_df_copy)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test___deepcopy__(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        modin_df_copy, pandas_df_copy = (
            modin_df.__deepcopy__(),
            pandas_df.__deepcopy__(),
        )
        df_equals(modin_df_copy, pandas_df_copy)

    def test___repr__(self):
        frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 100))
        pandas_df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)
        assert repr(pandas_df) == repr(modin_df)

        frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 99))
        pandas_df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)
        assert repr(pandas_df) == repr(modin_df)

        frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 101))
        pandas_df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)
        assert repr(pandas_df) == repr(modin_df)

        frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 102))
        pandas_df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)
        assert repr(pandas_df) == repr(modin_df)

        # ___repr___ method has a different code path depending on
        # whether the number of rows is >60; and a different code path
        # depending on the number of columns is >20.
        # Previous test cases already check the case when cols>20
        # and rows>60. The cases that follow exercise the other three
        # combinations.
        # rows <= 60, cols > 20
        frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(10, 100))
        pandas_df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)

        assert repr(pandas_df) == repr(modin_df)

        # rows <= 60, cols <= 20
        frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(10, 10))
        pandas_df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)

        assert repr(pandas_df) == repr(modin_df)

        # rows > 60, cols <= 20
        frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(100, 10))
        pandas_df = pandas.DataFrame(frame_data)
        modin_df = pd.DataFrame(frame_data)

        assert repr(pandas_df) == repr(modin_df)

        # Empty
        pandas_df = pandas.DataFrame(columns=["col{}".format(i) for i in range(100)])
        modin_df = pd.DataFrame(columns=["col{}".format(i) for i in range(100)])

        assert repr(pandas_df) == repr(modin_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_reset_index_with_multi_index(self, data):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if len(modin_df.columns) > len(pandas_df.columns):
            col0 = modin_df.columns[0]
            col1 = modin_df.columns[1]
            modin_cols = modin_df.groupby([col0, col1]).count().reset_index().columns
            pandas_cols = pandas_df.groupby([col0, col1]).count().reset_index().columns

            assert modin_cols.equals(pandas_cols)

    def test_reset_index_with_named_index(self):
        modin_df = pd.DataFrame(test_data_values[0])
        pandas_df = pandas.DataFrame(test_data_values[0])

        modin_df.index.name = pandas_df.index.name = "NAME_OF_INDEX"
        df_equals(modin_df, pandas_df)
        df_equals(modin_df.reset_index(drop=False), pandas_df.reset_index(drop=False))

        modin_df.reset_index(drop=True, inplace=True)
        pandas_df.reset_index(drop=True, inplace=True)
        df_equals(modin_df, pandas_df)

        modin_df = pd.DataFrame(test_data_values[0])
        pandas_df = pandas.DataFrame(test_data_values[0])
        modin_df.index.name = pandas_df.index.name = "NEW_NAME"
        df_equals(modin_df.reset_index(drop=False), pandas_df.reset_index(drop=False))

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_inplace_series_ops(self, data):
        pandas_df = pandas.DataFrame(data)
        modin_df = pd.DataFrame(data)

        if len(modin_df.columns) > len(pandas_df.columns):
            col0 = modin_df.columns[0]
            col1 = modin_df.columns[1]
            pandas_df[col1].dropna(inplace=True)
            modin_df[col1].dropna(inplace=True)
            df_equals(modin_df, pandas_df)

            pandas_df[col0].fillna(0, inplace=True)
            modin_df[col0].fillna(0, inplace=True)
            df_equals(modin_df, pandas_df)

    def test___setattr__(self,):
        pandas_df = pandas.DataFrame([1, 2, 3])
        modin_df = pd.DataFrame([1, 2, 3])

        pandas_df.new_col = [4, 5, 6]
        modin_df.new_col = [4, 5, 6]

        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_isin(self, data):
        pandas_df = pandas.DataFrame(data)
        modin_df = pd.DataFrame(data)

        val = [1, 2, 3, 4]
        pandas_result = pandas_df.isin(val)
        modin_result = modin_df.isin(val)

        df_equals(modin_result, pandas_result)

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_constructor(self, data):
        pandas_df = pandas.DataFrame(data)
        modin_df = pd.DataFrame(data)
        df_equals(pandas_df, modin_df)

        pandas_df = pandas.DataFrame({k: pandas.Series(v) for k, v in data.items()})
        modin_df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})
        df_equals(pandas_df, modin_df)

    @pytest.mark.parametrize(
        "data",
        [
            np.arange(1, 10000, dtype=np.float32),
            [
                pd.Series([1, 2, 3], dtype="int32"),
                pandas.Series([4, 5, 6], dtype="int64"),
                np.array([7, 8, 9], dtype=np.float32),
            ],
            pandas.Categorical([1, 2, 3, 4, 5]),
        ],
    )
    def test_constructor_dtypes(self, data):
        md_df, pd_df = create_test_dfs(data)
        df_equals(md_df, pd_df)

    def test_constructor_columns_and_index(self):
        modin_df = pd.DataFrame(
            [[1, 1, 10], [2, 4, 20], [3, 7, 30]],
            index=[1, 2, 3],
            columns=["id", "max_speed", "health"],
        )
        pandas_df = pandas.DataFrame(
            [[1, 1, 10], [2, 4, 20], [3, 7, 30]],
            index=[1, 2, 3],
            columns=["id", "max_speed", "health"],
        )
        df_equals(modin_df, pandas_df)
        df_equals(pd.DataFrame(modin_df), pandas.DataFrame(pandas_df))
        df_equals(
            pd.DataFrame(modin_df, columns=["max_speed", "health"]),
            pandas.DataFrame(pandas_df, columns=["max_speed", "health"]),
        )
        df_equals(
            pd.DataFrame(modin_df, index=[1, 2]),
            pandas.DataFrame(pandas_df, index=[1, 2]),
        )
        df_equals(
            pd.DataFrame(modin_df, index=[1, 2], columns=["health"]),
            pandas.DataFrame(pandas_df, index=[1, 2], columns=["health"]),
        )
        df_equals(
            pd.DataFrame(modin_df.iloc[:, 0], index=[1, 2, 3]),
            pandas.DataFrame(pandas_df.iloc[:, 0], index=[1, 2, 3]),
        )
        df_equals(
            pd.DataFrame(modin_df.iloc[:, 0], columns=["NO_EXIST"]),
            pandas.DataFrame(pandas_df.iloc[:, 0], columns=["NO_EXIST"]),
        )
        with pytest.raises(NotImplementedError):
            pd.DataFrame(modin_df, index=[1, 2, 99999])
        with pytest.raises(NotImplementedError):
            pd.DataFrame(modin_df, columns=["NO_EXIST"])


class TestDataFrameJoinSort:
    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    def test_combine(self, data):
        pandas_df = pandas.DataFrame(data)
        modin_df = pd.DataFrame(data)

        modin_df.combine(
            modin_df + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2
        )
        pandas_df.combine(
            pandas_df + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2
        )

    def test_join(self):
        frame_data = {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 0, 1],
            "col4": [2, 4, 5, 6],
        }

        modin_df = pd.DataFrame(frame_data)
        pandas_df = pandas.DataFrame(frame_data)

        frame_data2 = {"col5": [0], "col6": [1]}
        modin_df2 = pd.DataFrame(frame_data2)
        pandas_df2 = pandas.DataFrame(frame_data2)

        join_types = ["left", "right", "outer", "inner"]
        for how in join_types:
            modin_join = modin_df.join(modin_df2, how=how)
            pandas_join = pandas_df.join(pandas_df2, how=how)
            df_equals(modin_join, pandas_join)

        frame_data3 = {"col7": [1, 2, 3, 5, 6, 7, 8]}

        modin_df3 = pd.DataFrame(frame_data3)
        pandas_df3 = pandas.DataFrame(frame_data3)

        join_types = ["left", "outer", "inner"]
        for how in join_types:
            modin_join = modin_df.join([modin_df2, modin_df3], how=how)
            pandas_join = pandas_df.join([pandas_df2, pandas_df3], how=how)
            df_equals(modin_join, pandas_join)

    def test_merge(self):
        frame_data = {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 0, 1],
            "col4": [2, 4, 5, 6],
        }

        modin_df = pd.DataFrame(frame_data)
        pandas_df = pandas.DataFrame(frame_data)

        frame_data2 = {"col1": [0, 1, 2], "col2": [1, 5, 6]}
        modin_df2 = pd.DataFrame(frame_data2)
        pandas_df2 = pandas.DataFrame(frame_data2)

        join_types = ["outer", "inner"]
        for how in join_types:
            # Defaults
            modin_result = modin_df.merge(modin_df2, how=how)
            pandas_result = pandas_df.merge(pandas_df2, how=how)
            df_equals(modin_result, pandas_result)

            # left_on and right_index
            modin_result = modin_df.merge(
                modin_df2, how=how, left_on="col1", right_index=True
            )
            pandas_result = pandas_df.merge(
                pandas_df2, how=how, left_on="col1", right_index=True
            )
            df_equals(modin_result, pandas_result)

            # left_index and right_on
            modin_result = modin_df.merge(
                modin_df2, how=how, left_index=True, right_on="col1"
            )
            pandas_result = pandas_df.merge(
                pandas_df2, how=how, left_index=True, right_on="col1"
            )
            df_equals(modin_result, pandas_result)

            # left_on and right_on col1
            modin_result = modin_df.merge(
                modin_df2, how=how, left_on="col1", right_on="col1"
            )
            pandas_result = pandas_df.merge(
                pandas_df2, how=how, left_on="col1", right_on="col1"
            )
            df_equals(modin_result, pandas_result)

            # left_on and right_on col2
            modin_result = modin_df.merge(
                modin_df2, how=how, left_on="col2", right_on="col2"
            )
            pandas_result = pandas_df.merge(
                pandas_df2, how=how, left_on="col2", right_on="col2"
            )
            df_equals(modin_result, pandas_result)

            # left_index and right_index
            modin_result = modin_df.merge(
                modin_df2, how=how, left_index=True, right_index=True
            )
            pandas_result = pandas_df.merge(
                pandas_df2, how=how, left_index=True, right_index=True
            )
            df_equals(modin_result, pandas_result)

        # Named Series promoted to DF
        s = pd.Series(frame_data2.get("col1"))
        with pytest.raises(ValueError):
            modin_df.merge(s)

        s = pd.Series(frame_data2.get("col1"), name="col1")
        df_equals(modin_df.merge(s), modin_df.merge(modin_df2[["col1"]]))

        with pytest.raises(ValueError):
            modin_df.merge("Non-valid type")

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys)
    )
    @pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
    @pytest.mark.parametrize(
        "sort_remaining", bool_arg_values, ids=arg_keys("sort_remaining", bool_arg_keys)
    )
    def test_sort_index(self, data, axis, ascending, na_position, sort_remaining):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        # Change index value so sorting will actually make a difference
        if axis == "rows" or axis == 0:
            length = len(modin_df.index)
            modin_df.index = [(i - length / 2) % length for i in range(length)]
            pandas_df.index = [(i - length / 2) % length for i in range(length)]
        # Add NaNs to sorted index
        if axis == "rows" or axis == 0:
            length = len(modin_df.index)
            modin_df.index = [
                np.nan if i % 2 == 0 else modin_df.index[i] for i in range(length)
            ]
            pandas_df.index = [
                np.nan if i % 2 == 0 else pandas_df.index[i] for i in range(length)
            ]
        else:
            length = len(modin_df.columns)
            modin_df.columns = [
                np.nan if i % 2 == 0 else modin_df.columns[i] for i in range(length)
            ]
            pandas_df.columns = [
                np.nan if i % 2 == 0 else pandas_df.columns[i] for i in range(length)
            ]

        modin_result = modin_df.sort_index(
            axis=axis, ascending=ascending, na_position=na_position, inplace=False
        )
        pandas_result = pandas_df.sort_index(
            axis=axis, ascending=ascending, na_position=na_position, inplace=False
        )
        df_equals(modin_result, pandas_result)

        modin_df_cp = modin_df.copy()
        pandas_df_cp = pandas_df.copy()
        modin_df_cp.sort_index(
            axis=axis, ascending=ascending, na_position=na_position, inplace=True
        )
        pandas_df_cp.sort_index(
            axis=axis, ascending=ascending, na_position=na_position, inplace=True
        )
        df_equals(modin_df_cp, pandas_df_cp)

        # MultiIndex
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        modin_df.index = pd.MultiIndex.from_tuples(
            [(i // 10, i // 5, i) for i in range(len(modin_df))]
        )
        pandas_df.index = pandas.MultiIndex.from_tuples(
            [(i // 10, i // 5, i) for i in range(len(pandas_df))]
        )

        with pytest.warns(UserWarning):
            df_equals(modin_df.sort_index(level=0), pandas_df.sort_index(level=0))
        with pytest.warns(UserWarning):
            df_equals(modin_df.sort_index(axis=0), pandas_df.sort_index(axis=0))

    @pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
    @pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
    @pytest.mark.parametrize(
        "ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys)
    )
    @pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
    def test_sort_values(self, request, data, axis, ascending, na_position):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        if "empty_data" not in request.node.name and (
            (axis == 0 or axis == "over rows")
            or name_contains(request.node.name, numeric_dfs)
        ):
            index = (
                modin_df.index if axis == 1 or axis == "columns" else modin_df.columns
            )
            key = index[0]
            modin_result = modin_df.sort_values(
                key,
                axis=axis,
                ascending=ascending,
                na_position=na_position,
                inplace=False,
            )
            pandas_result = pandas_df.sort_values(
                key,
                axis=axis,
                ascending=ascending,
                na_position=na_position,
                inplace=False,
            )
            df_equals(modin_result, pandas_result)

            modin_df_cp = modin_df.copy()
            pandas_df_cp = pandas_df.copy()
            modin_df_cp.sort_values(
                key,
                axis=axis,
                ascending=ascending,
                na_position=na_position,
                inplace=True,
            )
            pandas_df_cp.sort_values(
                key,
                axis=axis,
                ascending=ascending,
                na_position=na_position,
                inplace=True,
            )
            df_equals(modin_df_cp, pandas_df_cp)

            keys = [key, index[-1]]
            modin_result = modin_df.sort_values(
                keys,
                axis=axis,
                ascending=ascending,
                na_position=na_position,
                inplace=False,
            )
            pandas_result = pandas_df.sort_values(
                keys,
                axis=axis,
                ascending=ascending,
                na_position=na_position,
                inplace=False,
            )
            df_equals(modin_result, pandas_result)

            modin_df_cp = modin_df.copy()
            pandas_df_cp = pandas_df.copy()
            modin_df_cp.sort_values(
                keys,
                axis=axis,
                ascending=ascending,
                na_position=na_position,
                inplace=True,
            )
            pandas_df_cp.sort_values(
                keys,
                axis=axis,
                ascending=ascending,
                na_position=na_position,
                inplace=True,
            )
            df_equals(modin_df_cp, pandas_df_cp)

    def test_sort_values_with_duplicates(self):
        modin_df = pd.DataFrame({"col": [2, 1, 1]}, index=[1, 1, 0])
        pandas_df = pandas.DataFrame({"col": [2, 1, 1]}, index=[1, 1, 0])

        key = modin_df.columns[0]
        modin_result = modin_df.sort_values(key, inplace=False,)
        pandas_result = pandas_df.sort_values(key, inplace=False,)
        df_equals(modin_result, pandas_result)

    def test_where(self):
        frame_data = random_state.randn(100, 10)
        pandas_df = pandas.DataFrame(frame_data, columns=list("abcdefghij"))
        modin_df = pd.DataFrame(frame_data, columns=list("abcdefghij"))
        pandas_cond_df = pandas_df % 5 < 2
        modin_cond_df = modin_df % 5 < 2

        pandas_result = pandas_df.where(pandas_cond_df, -pandas_df)
        modin_result = modin_df.where(modin_cond_df, -modin_df)
        assert all((to_pandas(modin_result) == pandas_result).all())

        other = pandas_df.loc[3]
        pandas_result = pandas_df.where(pandas_cond_df, other, axis=1)
        modin_result = modin_df.where(modin_cond_df, other, axis=1)
        assert all((to_pandas(modin_result) == pandas_result).all())

        other = pandas_df["e"]
        pandas_result = pandas_df.where(pandas_cond_df, other, axis=0)
        modin_result = modin_df.where(modin_cond_df, other, axis=0)
        assert all((to_pandas(modin_result) == pandas_result).all())

        pandas_result = pandas_df.where(pandas_df < 2, True)
        modin_result = modin_df.where(modin_df < 2, True)
        assert all((to_pandas(modin_result) == pandas_result).all())
