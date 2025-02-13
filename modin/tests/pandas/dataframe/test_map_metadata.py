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

from decimal import Decimal

import matplotlib
import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import MinRowPartitionSize, NPartitions, StorageFormat
from modin.core.dataframe.pandas.metadata import LazyProxyCategoricalDtype
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.pandas.testing import assert_index_equal, assert_series_equal
from modin.tests.pandas.utils import (
    RAND_HIGH,
    RAND_LOW,
    arg_keys,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    create_test_dfs,
    default_to_pandas_ignore_string,
    df_equals,
    df_is_empty,
    eval_general,
    indices_keys,
    indices_values,
    name_contains,
    numeric_dfs,
    random_state,
    sort_if_range_partitioning,
    test_data,
    test_data_keys,
    test_data_values,
    test_data_with_duplicates_keys,
    test_data_with_duplicates_values,
    test_func_keys,
    test_func_values,
)
from modin.tests.test_utils import (
    current_execution_is_native,
    df_or_series_using_native_execution,
    warns_that_defaulting_to_pandas_if,
)
from modin.utils import get_current_execution

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


def eval_insert(modin_df, pandas_df, **kwargs):
    if "col" in kwargs and "column" not in kwargs:
        kwargs["column"] = kwargs.pop("col")
    _kwargs = {"loc": 0, "column": "New column"}
    _kwargs.update(kwargs)

    eval_general(
        modin_df,
        pandas_df,
        operation=lambda df, **kwargs: df.insert(**kwargs),
        __inplace__=True,
        **_kwargs,
    )


def test_indexing():
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


def test_empty_df():
    df = pd.DataFrame(index=["a", "b"])
    df_is_empty(df)
    assert_index_equal(df.index, pd.Index(["a", "b"]))
    assert len(df.columns) == 0

    df = pd.DataFrame(columns=["a", "b"])
    df_is_empty(df)
    assert len(df.index) == 0
    assert_index_equal(df.columns, pd.Index(["a", "b"]))

    df = pd.DataFrame()
    df_is_empty(df)
    assert len(df.index) == 0
    assert len(df.columns) == 0

    df = pd.DataFrame(index=["a", "b"])
    df_is_empty(df)
    assert_index_equal(df.index, pd.Index(["a", "b"]))
    assert len(df.columns) == 0

    df = pd.DataFrame(columns=["a", "b"])
    df_is_empty(df)
    assert len(df.index) == 0
    assert_index_equal(df.columns, pd.Index(["a", "b"]))

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
def test_abs(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.abs()
    except Exception as err:
        with pytest.raises(type(err)):
            modin_df.abs()
    else:
        modin_result = modin_df.abs()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add_prefix(data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    test_prefix = "TEST"
    new_modin_df = modin_df.add_prefix(test_prefix, axis=axis)
    new_pandas_df = pandas_df.add_prefix(test_prefix, axis=axis)
    df_equals(new_modin_df.columns, new_pandas_df.columns)
    # TODO(https://github.com/modin-project/modin/issues/3804):
    # make df_equals always check dtypes.
    df_equals(new_modin_df.dtypes, new_pandas_df.dtypes)


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add_suffix(data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    test_suffix = "TEST"
    new_modin_df = modin_df.add_suffix(test_suffix, axis=axis)
    new_pandas_df = pandas_df.add_suffix(test_suffix, axis=axis)

    df_equals(new_modin_df.columns, new_pandas_df.columns)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("testfunc", test_func_values, ids=test_func_keys)
@pytest.mark.parametrize(
    "na_action", [None, "ignore"], ids=["no_na_action", "ignore_na"]
)
def test_applymap(data, testfunc, na_action):
    modin_df, pandas_df = create_test_dfs(data)

    with pytest.raises(ValueError):
        x = 2
        modin_df.applymap(x)

    eval_general(modin_df, pandas_df, lambda df: df.applymap(testfunc, na_action))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("testfunc", test_func_values, ids=test_func_keys)
def test_applymap_numeric(request, data, testfunc):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        try:
            pandas_result = pandas_df.applymap(testfunc)
        except Exception as err:
            with pytest.raises(type(err)):
                modin_df.applymap(testfunc)
        else:
            modin_result = modin_df.applymap(testfunc)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_at(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    key1 = modin_df.columns[0]
    # Scalar
    df_equals(modin_df.at[0, key1], pandas_df.at[0, key1])

    # Series
    df_equals(modin_df.loc[0].at[key1], pandas_df.loc[0].at[key1])

    # Write Item
    modin_df_copy = modin_df.copy()
    pandas_df_copy = pandas_df.copy()
    modin_df_copy.at[1, key1] = modin_df.at[0, key1]
    pandas_df_copy.at[1, key1] = pandas_df.at[0, key1]
    df_equals(modin_df_copy, pandas_df_copy)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_axes(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    for modin_axis, pd_axis in zip(modin_df.axes, pandas_df.axes):
        assert np.array_equal(modin_axis, pd_axis)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_copy(data):
    modin_df = pd.DataFrame(data)

    # pandas_df is unused but there so there won't be confusing list comprehension
    # stuff in the pytest.mark.parametrize
    new_modin_df = modin_df.copy(deep=True)

    assert new_modin_df is not modin_df
    assert new_modin_df.index is not modin_df.index
    assert new_modin_df.columns is not modin_df.columns
    assert new_modin_df.dtypes is not modin_df.dtypes

    if get_current_execution() != "BaseOnPython" and not current_execution_is_native():
        assert np.array_equal(
            new_modin_df._query_compiler._modin_frame._partitions,
            modin_df._query_compiler._modin_frame._partitions,
        )
    df_equals(new_modin_df, modin_df)

    # Shallow copy tests
    modin_df = pd.DataFrame(data)
    modin_df_cp = modin_df.copy(deep=False)

    assert modin_df_cp is not modin_df
    assert modin_df_cp.index is modin_df.index
    assert modin_df_cp.columns is modin_df.columns
    # FIXME: we're different from pandas here as modin doesn't copy dtypes for a shallow copy
    # https://github.com/modin-project/modin/issues/5602
    # assert modin_df_cp.dtypes is not modin_df.dtypes

    modin_df[modin_df.columns[0]] = 0
    df_equals(modin_df, modin_df_cp)


def test_copy_empty_dataframe():
    df = pd.DataFrame(range(3))
    res = df[:0].copy()
    assert res.dtypes.equals(df.dtypes)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dtypes(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.dtypes, pandas_df.dtypes)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("key", indices_values, ids=indices_keys)
def test_get(data, key):
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
def test_get_dummies(request, data, dummy_na, drop_first):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas.get_dummies(
            pandas_df, dummy_na=dummy_na, drop_first=drop_first
        )
    except Exception as err:
        with pytest.raises(type(err)):
            pd.get_dummies(modin_df, dummy_na=dummy_na, drop_first=drop_first)
    else:
        modin_result = pd.get_dummies(
            modin_df, dummy_na=dummy_na, drop_first=drop_first
        )
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isna(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    pandas_result = pandas_df.isna()
    modin_result = modin_df.isna()

    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isnull(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    pandas_result = pandas_df.isnull()
    modin_result = modin_df.isnull()

    df_equals(modin_result, pandas_result)


def test_astype():
    td = pandas.DataFrame(test_data["int_data"])[["col1", "index", "col3", "col4"]]
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

    # pandas nullable dtype
    modin_df_casted = modin_df.astype("Float64")
    expected_df_casted = expected_df.astype("Float64")
    df_equals(modin_df_casted, expected_df_casted)

    modin_df_casted = modin_df.astype("category")
    expected_df_casted = expected_df.astype("category")
    df_equals(modin_df_casted, expected_df_casted)

    dtype_dict = {"col1": np.int32, "index": np.int64, "col3": str}
    modin_df_casted = modin_df.astype(dtype_dict)
    expected_df_casted = expected_df.astype(dtype_dict)
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

    # The dtypes series must have a unique index.
    eval_general(
        modin_df,
        expected_df,
        lambda df: df.astype(
            pd.Series([str, str], index=["col1", "col1"])
            if isinstance(df, pd.DataFrame)
            else pandas.Series([str, str], index=["col1", "col1"])
        ),
        expected_exception=ValueError(
            "cannot reindex on an axis with duplicate labels"
        ),
    )


@pytest.mark.parametrize("errors", ["raise", "ignore"])
def test_astype_errors(errors):
    data = {"a": ["a", 2, -1]}
    modin_df, pandas_df = create_test_dfs(data)
    expected_exception = None
    if errors == "raise":
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/7025")
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.astype("int", errors=errors),
        # https://github.com/modin-project/modin/issues/5962
        comparator_kwargs={"check_dtypes": errors != "ignore"},
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("has_dtypes", [False, True])
def test_astype_copy(has_dtypes):
    data = [1]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    if not has_dtypes:
        modin_df._query_compiler.set_frame_dtypes_cache(None)
    eval_general(modin_df, pandas_df, lambda df: df.astype(str, copy=False))

    # trivial case where copying can be avoided, behavior should match pandas
    s1 = pd.Series([1, 2])
    if not has_dtypes:
        modin_df._query_compiler.set_frame_dtypes_cache(None)
    s2 = s1.astype("int64", copy=False)
    s2[0] = 10
    df_equals(s1, s2)


@pytest.mark.parametrize("dtypes_are_dict", [True, False])
def test_astype_dict_or_series_multiple_column_partitions(dtypes_are_dict):
    # Test astype with a dtypes dict that is complex in that:
    # - It applies to columns spanning multiple column partitions
    # - Within a partition frame df:
    #   - dtypes.index is not a subset of df.columns
    #   - df.columns is not a subset of dtypes.index

    modin_df, pandas_df = create_test_dfs(test_data["int_data"])
    if dtypes_are_dict:
        new_dtypes = {}
    else:
        new_dtypes = pandas.Series()
    for i, column in enumerate(pandas_df.columns):
        if i % 3 == 1:
            new_dtypes[column] = "string"
        elif i % 3 == 2:
            new_dtypes[column] = float
    eval_general(modin_df, pandas_df, lambda df: df.astype(new_dtypes))


def test_astype_category():
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


def test_astype_category_large():
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


def test_astype_int64_to_astype_category_github_issue_6259():
    eval_general(
        *create_test_dfs(
            {"c0": [0, 1, 2, 3, 4], "par": ["foo", "boo", "bar", "foo", "boo"]},
            index=["a", "b", "c", "d", "e"],
        ),
        lambda df: df["c0"].astype("Int64").astype("category"),
    )


@pytest.mark.skipif(
    get_current_execution() == "BaseOnPython" or current_execution_is_native(),
    reason="BaseOnPython and NativeQueryCompiler don't have proxy categories",
)
class TestCategoricalProxyDtype:
    """This class contains test and test usilities for the ``LazyProxyCategoricalDtype`` class."""

    @staticmethod
    def _get_lazy_proxy():
        """
        Build a dataframe containing a column that has a proxy type and return
        this proxy together with an original dtype that this proxy is emulating.

        Returns
        -------
        (LazyProxyCategoricalDtype, pandas.CategoricalDtype, modin.pandas.DataFrame)
        """
        nchunks = 3
        pandas_df = pandas.DataFrame({"a": [1, 1, 2, 2, 3, 2], "b": [1, 2, 3, 4, 5, 6]})
        original_dtype = pandas_df.astype({"a": "category"}).dtypes["a"]

        chunks = split_result_of_axis_func_pandas(
            axis=0,
            num_splits=nchunks,
            result=pandas_df,
            min_block_size=MinRowPartitionSize.get(),
            length_list=[2, 2, 2],
        )

        if StorageFormat.get() == "Pandas":
            df = pd.concat([pd.DataFrame(chunk) for chunk in chunks])
            assert df._query_compiler._modin_frame._partitions.shape == (nchunks, 1)

            df = df.astype({"a": "category"})
            return df.dtypes["a"], original_dtype, df
        else:
            raise NotImplementedError()

    def test_update_proxy(self):
        """Verify that ``LazyProxyCategoricalDtype._update_proxy`` method works as expected."""
        lazy_proxy, _, _ = self._get_lazy_proxy()
        new_parent = pd.DataFrame({"a": [10, 20, 30]})._query_compiler._modin_frame

        assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)
        # When we try to create a new proxy from the same arguments it should return itself
        assert (
            lazy_proxy._update_proxy(lazy_proxy._parent, lazy_proxy._column_name)
            is lazy_proxy
        )

        # When any of the arguments is changing we should create a new proxy
        proxy_with_new_column = lazy_proxy._update_proxy(
            lazy_proxy._parent, "other_column"
        )
        assert proxy_with_new_column is not lazy_proxy and isinstance(
            proxy_with_new_column, LazyProxyCategoricalDtype
        )

        # When any of the arguments is changing we should create a new proxy
        proxy_with_new_parent = lazy_proxy._update_proxy(
            new_parent, lazy_proxy._column_name
        )
        assert proxy_with_new_parent is not lazy_proxy and isinstance(
            proxy_with_new_parent, LazyProxyCategoricalDtype
        )

        lazy_proxy.categories  # trigger materialization
        # `._update_proxy` now should produce pandas Categoricals instead of a proxy as it already has materialized data
        assert (
            type(lazy_proxy._update_proxy(lazy_proxy._parent, lazy_proxy._column_name))
            == pandas.CategoricalDtype
        )

    def test_update_proxy_implicit(self):
        """
        Verify that a lazy proxy correctly updates its parent when passed from one parent to another.
        """
        lazy_proxy, _, parent = self._get_lazy_proxy()
        parent_frame = parent._query_compiler._modin_frame

        if StorageFormat.get() == "Pandas":
            assert lazy_proxy._parent is parent_frame
        else:
            raise NotImplementedError(
                f"The test is not implemented for {StorageFormat.get()} storage format"
            )

        # Making a copy of the dataframe, the new proxy should now start pointing to the new parent
        new_parent = parent.copy()
        new_parent_frame = new_parent._query_compiler._modin_frame
        new_lazy_proxy = new_parent_frame.dtypes[lazy_proxy._column_name]

        if StorageFormat.get() == "Pandas":
            # Make sure that the old proxy still pointing to the old parent
            assert lazy_proxy._parent is parent_frame
            assert new_lazy_proxy._parent is new_parent_frame
        else:
            raise NotImplementedError(
                f"The test is not implemented for {StorageFormat.get()} storage format"
            )

    def test_if_proxy_lazy(self):
        """Verify that proxy is able to pass simple comparison checks without triggering materialization."""
        lazy_proxy, actual_dtype, _ = self._get_lazy_proxy()

        assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)
        assert not lazy_proxy._is_materialized

        assert lazy_proxy == "category"
        assert isinstance(lazy_proxy, pd.CategoricalDtype)
        assert isinstance(lazy_proxy, pandas.CategoricalDtype)
        assert str(lazy_proxy) == "category"
        assert str(lazy_proxy) == str(actual_dtype)
        assert not lazy_proxy.ordered
        assert not lazy_proxy._is_materialized

        # Further, there are all checks that materialize categories
        assert lazy_proxy == actual_dtype
        assert actual_dtype == lazy_proxy
        assert repr(lazy_proxy) == repr(actual_dtype)
        assert lazy_proxy.categories.equals(actual_dtype.categories)
        assert lazy_proxy._is_materialized

    def test_proxy_as_dtype(self):
        """Verify that proxy can be used as an actual dtype."""
        lazy_proxy, actual_dtype, _ = self._get_lazy_proxy()

        assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)
        assert not lazy_proxy._is_materialized

        modin_df2, pandas_df2 = create_test_dfs({"c": [2, 2, 3, 4, 5, 6]})
        eval_general(
            (modin_df2, lazy_proxy),
            (pandas_df2, actual_dtype),
            lambda args: args[0].astype({"c": args[1]}),
        )

    def test_proxy_with_pandas_constructor(self):
        """Verify that users still can use pandas' constructor using `type(cat)(...)` notation."""
        lazy_proxy, _, _ = self._get_lazy_proxy()
        assert isinstance(lazy_proxy, LazyProxyCategoricalDtype)

        new_cat_values = pandas.Index([3, 4, 5])
        new_category_dtype = type(lazy_proxy)(categories=new_cat_values, ordered=True)
        assert not lazy_proxy._is_materialized
        assert new_category_dtype._is_materialized
        assert new_category_dtype.categories.equals(new_cat_values)
        assert new_category_dtype.ordered


def test_infer_objects_single_partition():
    data = {"a": ["s", 2, 3]}
    modin_df = pd.DataFrame(data).iloc[1:]
    pandas_df = pandas.DataFrame(data).iloc[1:]
    modin_result = modin_df.infer_objects()
    pandas_result = pandas_df.infer_objects()

    df_equals(modin_result, pandas_result)
    assert modin_result.dtypes.equals(pandas_result.dtypes)


@pytest.mark.parametrize(
    "infer_objects", bool_arg_values, ids=arg_keys("infer_objects", bool_arg_keys)
)
@pytest.mark.parametrize(
    "convert_string", bool_arg_values, ids=arg_keys("convert_string", bool_arg_keys)
)
@pytest.mark.parametrize(
    "convert_integer", bool_arg_values, ids=arg_keys("convert_integer", bool_arg_keys)
)
@pytest.mark.parametrize(
    "convert_boolean", bool_arg_values, ids=arg_keys("convert_boolean", bool_arg_keys)
)
@pytest.mark.parametrize(
    "convert_floating", bool_arg_values, ids=arg_keys("convert_floating", bool_arg_keys)
)
@pytest.mark.exclude_in_sanity
def test_convert_dtypes_single_partition(
    infer_objects, convert_string, convert_integer, convert_boolean, convert_floating
):
    # Sanity check, copied from pandas documentation:
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.convert_dtypes.html
    data = {
        "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
        "b": pd.Series(["x", "y", "z"], dtype=np.dtype("O")),
        "c": pd.Series([True, False, np.nan], dtype=np.dtype("O")),
        "d": pd.Series(["h", "i", np.nan], dtype=np.dtype("O")),
        "e": pd.Series([10, np.nan, 20], dtype=np.dtype("float")),
        "f": pd.Series([np.nan, 100.5, 200], dtype=np.dtype("float")),
    }
    kwargs = {
        "infer_objects": infer_objects,
        "convert_string": convert_string,
        "convert_integer": convert_integer,
        "convert_boolean": convert_boolean,
        "convert_floating": convert_floating,
    }
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_result = modin_df.convert_dtypes(**kwargs)
    pandas_result = pandas_df.convert_dtypes(**kwargs)
    assert modin_result.dtypes.equals(pandas_result.dtypes)


@pytest.mark.parametrize("dtype_backend", ["numpy_nullable", "pyarrow"])
def test_convert_dtypes_dtype_backend(dtype_backend):
    data = {
        "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
        "b": pd.Series(["x", "y", "z"], dtype=np.dtype("O")),
        "c": pd.Series([True, False, np.nan], dtype=np.dtype("O")),
        "d": pd.Series(["h", "i", np.nan], dtype=np.dtype("O")),
        "e": pd.Series([10, np.nan, 20], dtype=np.dtype("float")),
        "f": pd.Series([np.nan, 100.5, 200], dtype=np.dtype("float")),
    }

    def comparator(df1, df2):
        df_equals(df1, df2)
        df_equals(df1.dtypes, df2.dtypes)

    eval_general(
        *create_test_dfs(data),
        lambda df: df.convert_dtypes(dtype_backend=dtype_backend),
        comparator=comparator,
    )


@pytest.mark.skipif(
    current_execution_is_native(),
    reason="NativeQueryCompiler does not contain partitions.",
)
def test_convert_dtypes_multiple_row_partitions():
    # Column 0 should have string dtype
    modin_part1 = pd.DataFrame(["a"]).convert_dtypes()
    # Column 0 should have an int dtype
    modin_part2 = pd.DataFrame([1]).convert_dtypes()
    modin_df = pd.concat([modin_part1, modin_part2])
    if StorageFormat.get() == "Pandas":
        assert modin_df._query_compiler._modin_frame._partitions.shape == (2, 1)
    pandas_df = pandas.DataFrame(["a", 1], index=[0, 0])
    # The initial dataframes should be the same
    df_equals(modin_df, pandas_df)
    # TODO(https://github.com/modin-project/modin/pull/3805): delete
    # this assert once df_equals checks dtypes
    assert modin_df.dtypes.equals(pandas_df.dtypes)
    modin_result = modin_df.convert_dtypes()
    pandas_result = pandas_df.convert_dtypes()
    df_equals(modin_result, pandas_result)
    assert modin_result.dtypes.equals(pandas_result.dtypes)


def test_convert_dtypes_5653():
    modin_part1 = pd.DataFrame({"col1": ["a", "b", "c", "d"]})
    modin_part2 = pd.DataFrame({"col1": [None, None, None, None]})
    modin_df = pd.concat([modin_part1, modin_part2])
    if StorageFormat.get() == "Pandas":
        assert modin_df._query_compiler._modin_frame._partitions.shape == (2, 1)
    modin_df = modin_df.convert_dtypes()
    assert len(modin_df.dtypes) == 1
    assert modin_df.dtypes.iloc[0] == "string"


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("bound_type", ["list", "series"], ids=["list", "series"])
@pytest.mark.exclude_in_sanity
def test_clip(request, data, axis, bound_type):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        ind_len = (
            len(modin_df.index)
            if not pandas.DataFrame()._get_axis_number(axis)
            else len(modin_df.columns)
        )
        # set bounds
        lower, upper = np.sort(random_state.randint(RAND_LOW, RAND_HIGH, 2))

        # test only upper scalar bound
        modin_result = modin_df.clip(None, upper, axis=axis)
        pandas_result = pandas_df.clip(None, upper, axis=axis)
        df_equals(modin_result, pandas_result)

        # test lower and upper scalar bound
        modin_result = modin_df.clip(lower, upper, axis=axis)
        pandas_result = pandas_df.clip(lower, upper, axis=axis)
        df_equals(modin_result, pandas_result)

        lower = random_state.randint(RAND_LOW, RAND_HIGH, ind_len)
        upper = random_state.randint(RAND_LOW, RAND_HIGH, ind_len)

        if bound_type == "series":
            modin_lower = pd.Series(lower)
            pandas_lower = pandas.Series(lower)
            modin_upper = pd.Series(upper)
            pandas_upper = pandas.Series(upper)
        else:
            modin_lower = pandas_lower = lower
            modin_upper = pandas_upper = upper

        # test lower and upper list bound on each column
        modin_result = modin_df.clip(modin_lower, modin_upper, axis=axis)
        pandas_result = pandas_df.clip(pandas_lower, pandas_upper, axis=axis)
        df_equals(modin_result, pandas_result)

        # test only upper list bound on each column
        modin_result = modin_df.clip(np.nan, modin_upper, axis=axis)
        pandas_result = pandas_df.clip(np.nan, pandas_upper, axis=axis)
        df_equals(modin_result, pandas_result)

        with pytest.raises(ValueError):
            modin_df.clip(lower=[1, 2, 3], axis=None)


def test_clip_4485():
    modin_result = pd.DataFrame([1]).clip([3])
    pandas_result = pandas.DataFrame([1]).clip([3])
    df_equals(modin_result, pandas_result)


def test_drop():
    frame_data = {"A": [1, 2, 3, 4], "B": [0, 1, 2, 3]}
    simple = pandas.DataFrame(frame_data)
    modin_simple = pd.DataFrame(frame_data)
    df_equals(modin_simple.drop("A", axis=1), simple[["B"]])
    df_equals(modin_simple.drop(["A", "B"], axis="columns"), simple[[]])
    df_equals(modin_simple.drop([0, 1, 3], axis=0), simple.loc[[2], :])
    df_equals(modin_simple.drop([0, 3], axis="index"), simple.loc[[1, 2], :])

    pytest.raises(KeyError, modin_simple.drop, 5)
    pytest.raises(KeyError, modin_simple.drop, "C", axis=1)
    pytest.raises(KeyError, modin_simple.drop, [1, 5])
    pytest.raises(KeyError, modin_simple.drop, ["A", "C"], axis=1)

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
    with warns_that_defaulting_to_pandas_if(
        not df_or_series_using_native_execution(df)
    ):
        df.drop(index="length", level=1)


def test_drop_api_equivalence():
    # equivalence of the labels/axis and index/columns API's
    frame_data = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]

    modin_df = pd.DataFrame(frame_data, index=["a", "b", "c"], columns=["d", "e", "f"])

    modin_df1 = modin_df.drop("a")
    modin_df2 = modin_df.drop(index="a")
    df_equals(modin_df1, modin_df2)

    modin_df1 = modin_df.drop("d", axis=1)
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
def test_drop_transpose(data):
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


def test_droplevel():
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
@pytest.mark.parametrize("ignore_index", [True, False], ids=["True", "False"])
@pytest.mark.exclude_in_sanity
def test_drop_duplicates(data, keep, subset, ignore_index):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_df.drop_duplicates(
            keep=keep, inplace=False, subset=subset, ignore_index=ignore_index
        )
    except Exception as err:
        with pytest.raises(type(err)):
            modin_df.drop_duplicates(
                keep=keep, inplace=False, subset=subset, ignore_index=ignore_index
            )
    else:
        sort_if_range_partitioning(
            pandas_df.drop_duplicates(
                keep=keep, inplace=False, subset=subset, ignore_index=ignore_index
            ),
            modin_df.drop_duplicates(
                keep=keep, inplace=False, subset=subset, ignore_index=ignore_index
            ),
        )

    try:
        pandas_df.drop_duplicates(
            keep=keep, inplace=True, subset=subset, ignore_index=ignore_index
        )
    except Exception as err:
        with pytest.raises(type(err)):
            modin_df.drop_duplicates(
                keep=keep, inplace=True, subset=subset, ignore_index=ignore_index
            )
    else:
        modin_df.drop_duplicates(
            keep=keep, inplace=True, subset=subset, ignore_index=ignore_index
        )
        sort_if_range_partitioning(modin_df, pandas_df)


def test_drop_duplicates_with_missing_index_values():
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
    modin_df = pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])
    modin_result = modin_df.sort_values(["id", "time"]).drop_duplicates(["id"])
    pandas_result = pandas_df.sort_values(["id", "time"]).drop_duplicates(["id"])
    sort_if_range_partitioning(modin_result, pandas_result)


def test_drop_duplicates_after_sort():
    data = [
        {"value": 1, "time": 2},
        {"value": 1, "time": 1},
        {"value": 2, "time": 1},
        {"value": 2, "time": 2},
    ]
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = modin_df.sort_values(["value", "time"]).drop_duplicates(["value"])
    pandas_result = pandas_df.sort_values(["value", "time"]).drop_duplicates(["value"])
    sort_if_range_partitioning(modin_result, pandas_result)


def test_drop_duplicates_with_repeated_index_values():
    # This tests for issue #4467: https://github.com/modin-project/modin/issues/4467
    data = [[0], [1], [0]]
    index = [0, 0, 0]
    modin_df, pandas_df = create_test_dfs(data, index=index)
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.drop_duplicates(),
        comparator=sort_if_range_partitioning,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("how", ["any", "all"], ids=["any", "all"])
def test_dropna(data, axis, how):
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
def test_dropna_inplace(data):
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
def test_dropna_multiple_axes(data):
    modin_df = pd.DataFrame(data)

    with pytest.raises(TypeError):
        modin_df.dropna(how="all", axis=[0, 1])
    with pytest.raises(TypeError):
        modin_df.dropna(how="all", axis=(0, 1))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dropna_subset(request, data):
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
@pytest.mark.parametrize("axis,subset", [(0, list("EF")), (1, [4, 5])])
def test_dropna_subset_error(data, axis, subset):
    eval_general(
        *create_test_dfs(data),
        lambda df: df.dropna(axis=axis, subset=subset),
        expected_exception=KeyError(["E", "F"]),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("astype", ["category", "int32", "float"])
def test_insert_dtypes(data, astype, request):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    # categories with NaN works incorrect for now
    if astype == "category" and pandas_df.iloc[:, 0].isnull().any():
        return

    expected_exception = None
    if "int32-float_nan_data" in request.node.callspec.id:
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/7026")
    eval_insert(
        modin_df,
        pandas_df,
        col="TypeSaver",
        value=lambda df: df.iloc[:, 0].astype(astype),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("loc", [-3, 0, 3])
def test_insert_loc(data, loc):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    expected_exception = None
    if loc == -3:
        expected_exception = ValueError("unbounded slice")
    eval_insert(
        modin_df,
        pandas_df,
        loc=loc,
        value=lambda df: df.iloc[:, 0],
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_insert(data):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    eval_insert(
        modin_df, pandas_df, col="Duplicate", value=lambda df: df[df.columns[0]]
    )
    eval_insert(modin_df, pandas_df, col="Scalar", value=100)
    eval_insert(
        pd.DataFrame(columns=list("ab")),
        pandas.DataFrame(columns=list("ab")),
        col="Series insert",
        value=lambda df: df[df.columns[0]],
    )
    eval_insert(
        modin_df,
        pandas_df,
        col="DataFrame insert",
        value=lambda df: df[[df.columns[0]]],
    )
    eval_insert(
        modin_df,
        pandas_df,
        col="Different indices",
        value=lambda df: df[[df.columns[0]]].set_index(df.index[::-1]),
    )
    eval_insert(
        modin_df,
        pandas_df,
        col="2d list insert",
        value=lambda df: [[1, 2]] * len(df),
    )

    # Bad inserts
    eval_insert(
        modin_df,
        pandas_df,
        col="Bad Column",
        value=lambda df: df,
        expected_exception=ValueError(
            f"Expected a one-dimensional object, got a DataFrame with {len(pandas_df.columns)} columns instead."
        ),
    )
    eval_insert(
        modin_df,
        pandas_df,
        col="Too Short",
        value=lambda df: list(df[df.columns[0]])[:-1],
        expected_exception=ValueError(
            f"Length of values ({len(pandas_df)-1}) does not match length of index ({len(pandas_df)})"
        ),
    )
    eval_insert(
        modin_df,
        pandas_df,
        col=lambda df: df.columns[0],
        value=lambda df: df[df.columns[0]],
        expected_exception=ValueError("cannot insert 2d list insert, already exists"),
    )
    eval_insert(
        modin_df,
        pandas_df,
        loc=lambda df: len(df.columns) + 100,
        col="Bad Loc",
        value=100,
        expected_exception=IndexError(
            f"index {len(pandas_df.columns) + 100} is out of bounds for axis 0 with size {len(pandas_df.columns)}"
        ),
    )


def test_insert_4407():
    data = {"col1": [1, 2, 3], "col2": [2, 3, 4]}
    modin_df, pandas_df = create_test_dfs(data)

    def comparator(df1, df2):
        assert_series_equal(df1.dtypes, df2.dtypes, check_index=False)
        return df_equals(df1, df2)

    for idx, value in enumerate(
        (pandas_df.to_numpy(), np.array([[1]] * 3), np.array([[1, 2, 3], [4, 5, 6]]))
    ):
        expected_exception = None
        if idx == 0:
            expected_exception = ValueError(
                "Expected a 1D array, got an array with shape (3, 2)"
            )
        elif idx == 2:
            # FIXME: https://github.com/modin-project/modin/issues/7080
            expected_exception = False
        eval_insert(
            modin_df,
            pandas_df,
            loc=0,
            col=f"test_col{idx}",
            value=value,
            comparator=lambda df1, df2: comparator(df1, df2),
            expected_exception=expected_exception,
        )


def test_insert_modin_array():
    from modin.numpy import array

    data = {"col1": [1, 2, 3], "col2": [2, 3, 4]}
    modin_df1, modin_df2 = pd.DataFrame(data), pd.DataFrame(data)
    np_value = np.array([7, 7, 7])
    md_np_value = array(np_value)

    modin_df1.insert(1, "new_col", np_value)
    modin_df2.insert(1, "new_col", md_np_value)
    df_equals(modin_df1, modin_df2)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ndim(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.ndim == pandas_df.ndim


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_notna(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.notna(), pandas_df.notna())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_notnull(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.notnull(), pandas_df.notnull())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_round(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.round(), pandas_df.round())
    df_equals(modin_df.round(1), pandas_df.round(1))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_set_axis(data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    x = pandas.DataFrame()._get_axis_number(axis)
    index = modin_df.columns if x else modin_df.index
    labels = ["{0}_{1}".format(index[i], i) for i in range(modin_df.shape[x])]

    eval_general(
        modin_df, pandas_df, lambda df: df.set_axis(labels, axis=axis, copy=True)
    )

    modin_df_copy = modin_df.copy()
    modin_df = modin_df.set_axis(labels, axis=axis, copy=False)

    # Check that the copy and original are different
    try:
        df_equals(modin_df, modin_df_copy)
    except AssertionError:
        assert True
    else:
        assert False

    pandas_df = pandas_df.set_axis(labels, axis=axis)
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("drop", bool_arg_values, ids=arg_keys("drop", bool_arg_keys))
@pytest.mark.parametrize(
    "append", bool_arg_values, ids=arg_keys("append", bool_arg_keys)
)
def test_set_index(request, data, drop, append):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        modin_result = modin_df.set_index(key, drop=drop, append=append, inplace=False)
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
def test_shape(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.shape == pandas_df.shape


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_size(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.size == pandas_df.size


def test_squeeze():
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
def test_transpose(data):
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
        ({"C": [1, 2, 3], "B": [400, 500, 600]}, {"B": [4, 5, 6], "A": [7, 8, 9]}),
        (
            {"A": ["a", "b", "c"], "B": ["x", "y", "z"]},
            {"B": ["d", "e", "f", "g", "h", "i"]},
        ),
        ({"A": [1, 2, 3], "B": [400, 500, 600]}, {"B": [4, np.nan, 6]}),
    ],
)
@pytest.mark.parametrize("errors", ["raise", "ignore"])
def test_update(data, other_data, errors):
    modin_df, pandas_df = create_test_dfs(data)
    other_modin_df, other_pandas_df = create_test_dfs(other_data)
    expected_exception = None
    if errors == "raise":
        expected_exception = ValueError("Data overlaps.")
    eval_general(
        modin_df,
        pandas_df,
        lambda df: (
            df.update(other_modin_df, errors=errors)
            if isinstance(df, pd.DataFrame)
            else df.update(other_pandas_df, errors=errors)
        ),
        __inplace__=True,
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___neg__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.__neg__()
    except Exception as err:
        with pytest.raises(type(err)):
            modin_df.__neg__()
    else:
        modin_result = modin_df.__neg__()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___invert__(data, request):
    expected_exception = None
    if "float_nan_data" in request.node.callspec.id:
        # FIXME: https://github.com/modin-project/modin/issues/7081
        expected_exception = False
    eval_general(
        *create_test_dfs(data), lambda df: ~df, expected_exception=expected_exception
    )


def test___invert___bool():
    data = [False]
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_result = ~modin_df
    pandas_result = ~pandas_df
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___delitem__(request, data):
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
def test___nonzero__(data):
    modin_df = pd.DataFrame(data)

    with pytest.raises(ValueError):
        # Always raises ValueError
        modin_df.__nonzero__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___abs__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = abs(pandas_df)
    except Exception as err:
        with pytest.raises(type(err)):
            abs(modin_df)
    else:
        modin_result = abs(modin_df)
        df_equals(modin_result, pandas_result)


def test___round__():
    data = test_data_values[0]
    eval_general(pd.DataFrame(data), pandas.DataFrame(data), lambda df: df.__round__())


@pytest.mark.parametrize(
    "get_index",
    [
        pytest.param(lambda idx: None, id="None_idx"),
        pytest.param(lambda idx: ["a", "b", "c"], id="No_intersection_idx"),
        pytest.param(lambda idx: idx, id="Equal_idx"),
        pytest.param(lambda idx: idx[::-1], id="Reversed_idx"),
    ],
)
@pytest.mark.parametrize(
    "get_columns",
    [
        pytest.param(lambda idx: None, id="None_idx"),
        pytest.param(lambda idx: ["a", "b", "c"], id="No_intersection_idx"),
        pytest.param(lambda idx: idx, id="Equal_idx"),
        pytest.param(lambda idx: idx[::-1], id="Reversed_idx"),
    ],
)
@pytest.mark.parametrize("dtype", [None, "str"])
@pytest.mark.exclude_in_sanity
def test_constructor_from_modin_series(get_index, get_columns, dtype):
    modin_df, pandas_df = create_test_dfs(test_data_values[0])

    modin_data = {f"new_col{i}": modin_df.iloc[:, i] for i in range(modin_df.shape[1])}
    pandas_data = {
        f"new_col{i}": pandas_df.iloc[:, i] for i in range(pandas_df.shape[1])
    }

    index = get_index(modin_df.index)
    columns = get_columns(list(modin_data.keys()))

    new_modin = pd.DataFrame(modin_data, index=index, columns=columns, dtype=dtype)
    new_pandas = pandas.DataFrame(
        pandas_data, index=index, columns=columns, dtype=dtype
    )
    df_equals(new_modin, new_pandas)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_constructor(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)
    df_equals(pandas_df, modin_df)

    pandas_df = pandas.DataFrame({k: pandas.Series(v) for k, v in data.items()})
    modin_df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})
    df_equals(pandas_df, modin_df)


def test_pyarrow_constructor():
    pa = pytest.importorskip("pyarrow")

    data = [[Decimal("3.19"), None], [None, Decimal("-1.23")]]
    df_equals(*create_test_dfs(data, dtype=pd.ArrowDtype(pa.decimal128(3, scale=2))))


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
def test_constructor_dtypes(data):
    modin_df, pandas_df = create_test_dfs(data)
    df_equals(modin_df, pandas_df)


def test_constructor_columns_and_index():
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


def test_constructor_from_index():
    data = pd.Index([1, 2, 3], name="pricing_date")
    modin_df, pandas_df = create_test_dfs(data)
    df_equals(modin_df, pandas_df)


def test_insert_datelike_string_issue_7371():
    # When a new value is inserted into a frame, we call pandas.api.types.pandas_dtype(value) to
    # extract the dtype of an object like a pandas Series or numpy array. When a scalar value is passed,
    # this usually raises a TypeError, so we construct a local pandas Series from the object and
    # extract the dtype from there.
    # When the passed value is a date-like string, pandas will instead raise a ValueError because
    # it tries to parse it as a numpy structured dtype. After fixing GH#7371, we now catch
    # ValueError in addition to TypeError to handle this case.
    modin_df = pd.DataFrame({"a": [0]})
    modin_df["c"] = "2020-01-01"
    pandas_df = pandas.DataFrame({"a": [0]})
    pandas_df["c"] = "2020-01-01"
    df_equals(modin_df, pandas_df)
