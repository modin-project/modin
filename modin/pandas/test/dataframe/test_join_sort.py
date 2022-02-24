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
import matplotlib
import modin.pandas as pd
from modin.utils import to_pandas

from modin.pandas.test.utils import (
    random_state,
    df_equals,
    arg_keys,
    test_data_values,
    test_data_keys,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    test_data,
    generate_multiindex,
    eval_general,
    rotate_decimal_digits_or_symbols,
    extra_test_parameters,
    default_to_pandas_ignore_string,
)
from modin.config import NPartitions
from modin.test.test_utils import warns_that_defaulting_to_pandas

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_combine(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    modin_df.combine(modin_df + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2)
    pandas_df.combine(
        pandas_df + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2
    )


@pytest.mark.parametrize(
    "test_data, test_data2",
    [
        (
            np.random.uniform(0, 100, size=(2**6, 2**6)),
            np.random.uniform(0, 100, size=(2**7, 2**6)),
        ),
        (
            np.random.uniform(0, 100, size=(2**7, 2**6)),
            np.random.uniform(0, 100, size=(2**6, 2**6)),
        ),
        (
            np.random.uniform(0, 100, size=(2**6, 2**6)),
            np.random.uniform(0, 100, size=(2**6, 2**7)),
        ),
        (
            np.random.uniform(0, 100, size=(2**6, 2**7)),
            np.random.uniform(0, 100, size=(2**6, 2**6)),
        ),
    ],
)
def test_join(test_data, test_data2):
    modin_df = pd.DataFrame(
        test_data,
        columns=["col{}".format(i) for i in range(test_data.shape[1])],
        index=pd.Index([i for i in range(1, test_data.shape[0] + 1)], name="key"),
    )
    pandas_df = pandas.DataFrame(
        test_data,
        columns=["col{}".format(i) for i in range(test_data.shape[1])],
        index=pandas.Index([i for i in range(1, test_data.shape[0] + 1)], name="key"),
    )
    modin_df2 = pd.DataFrame(
        test_data2,
        columns=["col{}".format(i) for i in range(test_data2.shape[1])],
        index=pd.Index([i for i in range(1, test_data2.shape[0] + 1)], name="key"),
    )
    pandas_df2 = pandas.DataFrame(
        test_data2,
        columns=["col{}".format(i) for i in range(test_data2.shape[1])],
        index=pandas.Index([i for i in range(1, test_data2.shape[0] + 1)], name="key"),
    )

    hows = ["inner", "left", "right", "outer"]
    ons = ["col33", "col34"]
    sorts = [False, True]
    for i in range(4):
        for j in range(2):
            modin_result = modin_df.join(
                modin_df2,
                how=hows[i],
                on=ons[j],
                sort=sorts[j],
                lsuffix="_caller",
                rsuffix="_other",
            )
            pandas_result = pandas_df.join(
                pandas_df2,
                how=hows[i],
                on=ons[j],
                sort=sorts[j],
                lsuffix="_caller",
                rsuffix="_other",
            )
            df_equals(modin_result, pandas_result)

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


@pytest.mark.parametrize(
    "test_data, test_data2",
    [
        (
            np.random.uniform(0, 100, size=(2**6, 2**6)),
            np.random.uniform(0, 100, size=(2**7, 2**6)),
        ),
        (
            np.random.uniform(0, 100, size=(2**7, 2**6)),
            np.random.uniform(0, 100, size=(2**6, 2**6)),
        ),
        (
            np.random.uniform(0, 100, size=(2**6, 2**6)),
            np.random.uniform(0, 100, size=(2**6, 2**7)),
        ),
        (
            np.random.uniform(0, 100, size=(2**6, 2**7)),
            np.random.uniform(0, 100, size=(2**6, 2**6)),
        ),
    ],
)
def test_merge(test_data, test_data2):
    modin_df = pd.DataFrame(
        test_data,
        columns=["col{}".format(i) for i in range(test_data.shape[1])],
        index=pd.Index([i for i in range(1, test_data.shape[0] + 1)], name="key"),
    )
    pandas_df = pandas.DataFrame(
        test_data,
        columns=["col{}".format(i) for i in range(test_data.shape[1])],
        index=pandas.Index([i for i in range(1, test_data.shape[0] + 1)], name="key"),
    )
    modin_df2 = pd.DataFrame(
        test_data2,
        columns=["col{}".format(i) for i in range(test_data2.shape[1])],
        index=pd.Index([i for i in range(1, test_data2.shape[0] + 1)], name="key"),
    )
    pandas_df2 = pandas.DataFrame(
        test_data2,
        columns=["col{}".format(i) for i in range(test_data2.shape[1])],
        index=pandas.Index([i for i in range(1, test_data2.shape[0] + 1)], name="key"),
    )

    hows = ["left", "inner"]
    ons = ["col33", ["col33", "col34"]]
    sorts = [False, True]
    for i in range(2):
        for j in range(2):
            modin_result = modin_df.merge(
                modin_df2, how=hows[i], on=ons[j], sort=sorts[j]
            )
            pandas_result = pandas_df.merge(
                pandas_df2, how=hows[i], on=ons[j], sort=sorts[j]
            )
            df_equals(modin_result, pandas_result)

            modin_result = modin_df.merge(
                modin_df2,
                how=hows[i],
                left_on="key",
                right_on="key",
                sort=sorts[j],
            )
            pandas_result = pandas_df.merge(
                pandas_df2,
                how=hows[i],
                left_on="key",
                right_on="key",
                sort=sorts[j],
            )
            df_equals(modin_result, pandas_result)

    # Test for issue #1771
    modin_df = pd.DataFrame({"name": np.arange(40)})
    modin_df2 = pd.DataFrame({"name": [39], "position": [0]})
    pandas_df = pandas.DataFrame({"name": np.arange(40)})
    pandas_df2 = pandas.DataFrame({"name": [39], "position": [0]})
    modin_result = modin_df.merge(modin_df2, on="name", how="inner")
    pandas_result = pandas_df.merge(pandas_df2, on="name", how="inner")
    df_equals(modin_result, pandas_result)

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

    # Cannot merge a Series without a name
    ps = pandas.Series(frame_data2.get("col1"))
    ms = pd.Series(frame_data2.get("col1"))
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.merge(ms if isinstance(df, pd.DataFrame) else ps),
    )

    # merge a Series with a name
    ps = pandas.Series(frame_data2.get("col1"), name="col1")
    ms = pd.Series(frame_data2.get("col1"), name="col1")
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.merge(ms if isinstance(df, pd.DataFrame) else ps),
    )

    with pytest.raises(TypeError):
        modin_df.merge("Non-valid type")


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys)
)
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
def test_sort_index(axis, ascending, na_position):
    data = test_data["float_nan_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    # Change index value so sorting will actually make a difference
    if axis == 0:
        length = len(modin_df.index)
        for df in [modin_df, pandas_df]:
            df.index = [(i - length / 2) % length for i in range(length)]

    # Add NaNs to sorted index
    for df in [modin_df, pandas_df]:
        sort_index = df.axes[axis]
        df.set_axis(
            [np.nan if i % 2 == 0 else sort_index[i] for i in range(len(sort_index))],
            axis=axis,
            inplace=True,
        )

    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.sort_index(
            axis=axis, ascending=ascending, na_position=na_position
        ),
    )


@pytest.mark.parametrize("axis", ["rows", "columns"])
def test_sort_index_inplace(axis):
    data = test_data["int_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    for df in [modin_df, pandas_df]:
        df.sort_index(axis=axis, inplace=True)
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize(
    "sort_remaining", bool_arg_values, ids=arg_keys("sort_remaining", bool_arg_keys)
)
def test_sort_multiindex(sort_remaining):
    data = test_data["int_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    for index in ["index", "columns"]:
        new_index = generate_multiindex(len(getattr(modin_df, index)))
        for df in [modin_df, pandas_df]:
            setattr(df, index, new_index)

    for kwargs in [{"level": 0}, {"axis": 0}, {"axis": 1}]:
        with warns_that_defaulting_to_pandas():
            df_equals(
                modin_df.sort_index(sort_remaining=sort_remaining, **kwargs),
                pandas_df.sort_index(sort_remaining=sort_remaining, **kwargs),
            )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "by",
    [
        pytest.param(
            "first",
            marks=pytest.mark.skipif(not extra_test_parameters, reason="extra"),
        ),
        pytest.param(
            "first,last",
            marks=pytest.mark.skipif(not extra_test_parameters, reason="extra"),
        ),
        "first,last,middle",
        pytest.param(
            "multiindex_level0",
            marks=pytest.mark.xfail_executions(
                ["PandasOnPython", "PandasOnRay", "PandasOnDask"],
                reason="multiindex levels do not work",
            ),
        ),
        pytest.param(
            "multiindex_level1,multiindex_level0",
            marks=pytest.mark.xfail_executions(
                ["PandasOnPython", "PandasOnRay", "PandasOnDask"],
                reason="multiindex levels do not work",
            ),
        ),
        pytest.param(
            "multiindex_level0,last,first,multiindex_level1",
            marks=pytest.mark.xfail_executions(
                ["PandasOnPython", "PandasOnRay", "PandasOnDask"],
                reason="multiindex levels do not work",
            ),
        ),
    ],
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "ascending",
    bool_arg_values + ["list_first_True", "list_first_False"],
    ids=arg_keys("ascending", bool_arg_keys + ["list_first_True", "list_first_False"]),
)
@pytest.mark.parametrize(
    "inplace", bool_arg_values, ids=arg_keys("inplace", bool_arg_keys)
)
@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            "mergesort",
            marks=pytest.mark.skipif(not extra_test_parameters, reason="extra"),
        ),
        "quicksort",
        pytest.param(
            "heapsort",
            marks=pytest.mark.skipif(not extra_test_parameters, reason="extra"),
        ),
    ],
)
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
@pytest.mark.parametrize(
    "ignore_index",
    bool_arg_values,
    ids=arg_keys("ignore_index", bool_arg_keys),
)
@pytest.mark.parametrize("key", [None, rotate_decimal_digits_or_symbols])
def test_sort_values(
    data, by, axis, ascending, inplace, kind, na_position, ignore_index, key
):
    if ascending is None:
        pytest.skip("None is not a valid value for ascending.")
    if (axis == 1 or axis == "columns") and ignore_index:
        pytest.skip("Pandas bug #39426 which is fixed in Pandas 1.3")

    if ascending is None and key is not None:
        pytest.skip("Pandas bug #41318")

    if "multiindex" in by:
        index = generate_multiindex(len(data[list(data.keys())[0]]), nlevels=2)
        columns = generate_multiindex(len(data.keys()), nlevels=2)
        data = {columns[ind]: data[key] for ind, key in enumerate(data)}
    else:
        index = None
        columns = None

    modin_df = pd.DataFrame(data, index=index, columns=columns)
    pandas_df = pandas.DataFrame(data, index=index, columns=columns)

    index = modin_df.index if axis == 1 or axis == "columns" else modin_df.columns

    # Parse "by" spec
    by_list = []
    for b in by.split(","):
        if b == "first":
            by_list.append(index[0])
        elif b == "last":
            by_list.append(index[-1])
        elif b == "middle":
            by_list.append(index[len(index) // 2])
        elif b.startswith("multiindex_level"):
            by_list.append(index.names[int(b[len("multiindex_level") :])])
        else:
            raise Exception('Unknown "by" specifier:' + b)

    # Create "ascending" list
    if ascending in ["list_first_True", "list_first_False"]:
        start = 0 if ascending == "list_first_False" else 1
        ascending = [i & 1 > 0 for i in range(start, len(by_list) + start)]

    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.sort_values(
            by_list,
            axis=axis,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
            ignore_index=ignore_index,
            key=key,
        ),
        __inplace__=inplace,
    )


def test_sort_values_with_duplicates():
    modin_df = pd.DataFrame({"col": [2, 1, 1]}, index=[1, 1, 0])
    pandas_df = pandas.DataFrame({"col": [2, 1, 1]}, index=[1, 1, 0])

    key = modin_df.columns[0]
    modin_result = modin_df.sort_values(key, inplace=False)
    pandas_result = pandas_df.sort_values(key, inplace=False)
    df_equals(modin_result, pandas_result)

    modin_df.sort_values(key, inplace=True)
    pandas_df.sort_values(key, inplace=True)
    df_equals(modin_df, pandas_df)


def test_sort_values_with_string_index():
    modin_df = pd.DataFrame({"col": [25, 17, 1]}, index=["ccc", "bbb", "aaa"])
    pandas_df = pandas.DataFrame({"col": [25, 17, 1]}, index=["ccc", "bbb", "aaa"])

    key = modin_df.columns[0]
    modin_result = modin_df.sort_values(key, inplace=False)
    pandas_result = pandas_df.sort_values(key, inplace=False)
    df_equals(modin_result, pandas_result)

    modin_df.sort_values(key, inplace=True)
    pandas_df.sort_values(key, inplace=True)
    df_equals(modin_df, pandas_df)


def test_where():
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


@pytest.mark.parametrize("align_axis", ["index", "columns"])
@pytest.mark.parametrize("keep_shape", [False, True])
@pytest.mark.parametrize("keep_equal", [False, True])
def test_compare(align_axis, keep_shape, keep_equal):
    kwargs = {
        "align_axis": align_axis,
        "keep_shape": keep_shape,
        "keep_equal": keep_equal,
    }
    frame_data1 = random_state.randn(100, 10)
    frame_data2 = random_state.randn(100, 10)
    pandas_df = pandas.DataFrame(frame_data1, columns=list("abcdefghij"))
    pandas_df2 = pandas.DataFrame(frame_data2, columns=list("abcdefghij"))
    modin_df = pd.DataFrame(frame_data1, columns=list("abcdefghij"))
    modin_df2 = pd.DataFrame(frame_data2, columns=list("abcdefghij"))

    modin_result = modin_df.compare(modin_df2, **kwargs)
    pandas_result = pandas_df.compare(pandas_df2, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)

    modin_result = modin_df2.compare(modin_df, **kwargs)
    pandas_result = pandas_df2.compare(pandas_df, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)

    series_data1 = ["a", "b", "c", "d", "e"]
    series_data2 = ["a", "a", "c", "b", "e"]
    pandas_series1 = pandas.Series(series_data1)
    pandas_series2 = pandas.Series(series_data2)
    modin_series1 = pd.Series(series_data1)
    modin_series2 = pd.Series(series_data2)

    modin_result = modin_series1.compare(modin_series2, **kwargs)
    pandas_result = pandas_series1.compare(pandas_series2, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)

    modin_result = modin_series2.compare(modin_series1, **kwargs)
    pandas_result = pandas_series2.compare(pandas_series1, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)
