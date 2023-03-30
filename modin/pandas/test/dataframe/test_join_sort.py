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
    create_test_dfs,
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
from modin.config import NPartitions, Engine, StorageFormat
from modin.test.test_utils import warns_that_defaulting_to_pandas

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)

# Initialize env for storage format detection in @pytest.mark.*
pd.DataFrame()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_combine(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    modin_df.combine(modin_df + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2)
    pandas_df.combine(
        pandas_df + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2
    )


@pytest.mark.xfail(
    StorageFormat.get() == "Hdk", reason="https://github.com/intel-ai/hdk/issues/264"
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


@pytest.mark.parametrize("has_index_cache", [True, False])
def test_merge_on_index(has_index_cache):
    modin_df1, pandas_df1 = create_test_dfs(
        {
            "idx_key1": [1, 2, 3, 4],
            "idx_key2": [2, 3, 4, 5],
            "idx_key3": [3, 4, 5, 6],
            "data_col1": [10, 2, 3, 4],
            "col_key1": [3, 4, 5, 6],
            "col_key2": [3, 4, 5, 6],
        }
    )

    modin_df1 = modin_df1.set_index(["idx_key1", "idx_key2"])
    pandas_df1 = pandas_df1.set_index(["idx_key1", "idx_key2"])

    modin_df2, pandas_df2 = create_test_dfs(
        {
            "idx_key1": [4, 3, 2, 1],
            "idx_key2": [5, 4, 3, 2],
            "idx_key3": [6, 5, 4, 3],
            "data_col2": [10, 2, 3, 4],
            "col_key1": [6, 5, 4, 3],
            "col_key2": [6, 5, 4, 3],
        }
    )

    modin_df2 = modin_df2.set_index(["idx_key2", "idx_key3"])
    pandas_df2 = pandas_df2.set_index(["idx_key2", "idx_key3"])

    def setup_cache():
        if has_index_cache:
            modin_df1.index  # triggering index materialization
            modin_df2.index
            assert modin_df1._query_compiler._modin_frame._index_cache is not None
            assert modin_df2._query_compiler._modin_frame._index_cache is not None
        else:
            # Propagate deferred indices to partitions
            # The change in index is not automatically handled by Modin. See #3941.
            modin_df1.index = modin_df1.index
            modin_df1._to_pandas()
            modin_df1._query_compiler._modin_frame._index_cache = None
            modin_df2.index = modin_df2.index
            modin_df2._to_pandas()
            modin_df2._query_compiler._modin_frame._index_cache = None

    for on in (
        ["col_key1", "idx_key1"],
        ["col_key1", "idx_key2"],
        ["col_key1", "idx_key3"],
        ["idx_key1"],
        ["idx_key2"],
        ["idx_key3"],
    ):
        setup_cache()
        eval_general(
            (modin_df1, modin_df2),
            (pandas_df1, pandas_df2),
            lambda dfs: dfs[0].merge(dfs[1], on=on),
        )

    for left_on, right_on in (
        (["idx_key1"], ["col_key1"]),
        (["col_key1"], ["idx_key3"]),
        (["idx_key1"], ["idx_key3"]),
        (["idx_key2"], ["idx_key2"]),
        (["col_key1", "idx_key2"], ["col_key2", "idx_key2"]),
    ):
        setup_cache()
        eval_general(
            (modin_df1, modin_df2),
            (pandas_df1, pandas_df2),
            lambda dfs: dfs[0].merge(dfs[1], left_on=left_on, right_on=right_on),
        )


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

    dfs = [modin_df, pandas_df]
    # Add NaNs to sorted index
    for idx in range(len(dfs)):
        sort_index = dfs[idx].axes[axis]
        dfs[idx] = dfs[idx].set_axis(
            [np.nan if i % 2 == 0 else sort_index[i] for i in range(len(sort_index))],
            axis=axis,
            copy=False,
        )
    modin_df, pandas_df = dfs

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

    # If index is preserved and `key` function is ``None``,
    # it could be sorted along rows differently from pandas.
    # The order of NA rows, sorted by HDK, is different (but still valid)
    # from pandas. To make the index identical to pandas, we add the
    # index names to 'by'.
    by_index_names = None
    if (
        StorageFormat.get() == "Hdk"
        and not ignore_index
        and key is None
        and (axis == 0 or axis == "rows")
    ):
        by_index_names = []
    if "multiindex" in by:
        index = generate_multiindex(len(data[list(data.keys())[0]]), nlevels=2)
        columns = generate_multiindex(len(data.keys()), nlevels=2)
        data = {columns[ind]: data[key] for ind, key in enumerate(data)}
        if by_index_names is not None:
            by_index_names.extend(index.names)
    elif by_index_names is not None:
        index = pd.RangeIndex(0, len(next(iter(data.values()))), name="test_idx")
        columns = None
        by_index_names.append(index.name)
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

    if by_index_names is not None:
        by_list.extend(by_index_names)

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


def test_sort_values_descending_with_only_two_bins():
    # test case from https://github.com/modin-project/modin/issues/5781
    part1 = pd.DataFrame({"a": [1, 2, 3, 4]})
    part2 = pd.DataFrame({"a": [5, 6, 7, 8]})

    modin_df = pd.concat([part1, part2])
    pandas_df = modin_df._to_pandas()

    if StorageFormat.get() == "Pandas":
        assert modin_df._query_compiler._modin_frame._partitions.shape == (2, 1)

    eval_general(
        modin_df, pandas_df, lambda df: df.sort_values(by="a", ascending=False)
    )


@pytest.mark.parametrize("ascending", [True, False])
def test_sort_values_with_one_partition(ascending):
    # Test case from https://github.com/modin-project/modin/issues/5859
    modin_df, pandas_df = create_test_dfs(
        np.array([["hello", "goodbye"], ["hello", "Hello"]])
    )

    if StorageFormat.get() == "Pandas":
        assert modin_df._query_compiler._modin_frame._partitions.shape == (1, 1)

    eval_general(
        modin_df, pandas_df, lambda df: df.sort_values(by=1, ascending=ascending)
    )


def test_sort_overpartitioned_df():
    # First we test when the final df will have only 1 row and column partition.
    data = [[4, 5, 6], [1, 2, 3]]
    modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
    pandas_df = pandas.DataFrame(data)

    eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

    # Next we test when the final df will only have 1 row, but starts with multiple column
    # partitions.
    data = [list(range(100)), list(range(100, 200))]
    modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
    pandas_df = pandas.DataFrame(data)

    eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

    # Next we test when the final df will have multiple row partitions.
    data = np.random.choice(650, 650, replace=False).reshape((65, 10))
    modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
    pandas_df = pandas.DataFrame(data)

    eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

    old_nptns = NPartitions.get()
    NPartitions.put(24)
    try:
        # Next we test when there's only one row per partition.
        data = np.random.choice(650, 650, replace=False).reshape((65, 10))
        modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(
            drop=True
        )
        pandas_df = pandas.DataFrame(data)

        eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

        # And again, when there's more than one column partition.
        data = np.random.choice(6500, 6500, replace=False).reshape((65, 100))
        modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(
            drop=True
        )
        pandas_df = pandas.DataFrame(data)

        eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

        # Additionally, we should test when we have a number of partitions
        # that doesn't divide cleanly into our desired number of partitions.
        # In this case, we start with 17 partitions, and want 2.
        NPartitions.put(21)
        data = np.random.choice(6500, 6500, replace=False).reshape((65, 100))
        modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(
            drop=True
        )
        pandas_df = pandas.DataFrame(data)

        eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

    finally:
        NPartitions.put(old_nptns)


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


@pytest.mark.skipif(
    Engine.get() not in ["Ray", "Dask", "Unidist"],
    reason="We only need to test this case where sort does not default to pandas.",
)
@pytest.mark.parametrize("ascending", [True, False], ids=["True", "False"])
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
def test_sort_values_with_only_one_non_na_row_in_partition(ascending, na_position):
    pandas_df = pandas.DataFrame(
        np.random.rand(1000, 100), columns=[f"col {i}" for i in range(100)]
    )
    # Need to ensure that one of the partitions has all NA values except for one row
    pandas_df.iloc[340:] = np.NaN
    pandas_df.iloc[-1] = -4.0
    modin_df = pd.DataFrame(pandas_df)
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.sort_values(
            "col 3", ascending=ascending, na_position=na_position
        ),
    )


@pytest.mark.skipif(
    Engine.get() not in ["Ray", "Dask", "Unidist"],
    reason="We only need to test this case where sort does not default to pandas.",
)
def test_sort_values_with_sort_key_on_partition_boundary():
    modin_df = pd.DataFrame(
        np.random.rand(1000, 100), columns=[f"col {i}" for i in range(100)]
    )
    sort_key = modin_df.columns[modin_df._query_compiler._modin_frame.column_widths[0]]
    eval_general(modin_df, modin_df._to_pandas(), lambda df: df.sort_values(sort_key))


def test_where():
    columns = list("abcdefghij")

    frame_data = random_state.randn(100, 10)
    modin_df, pandas_df = create_test_dfs(frame_data, columns=columns)
    pandas_cond_df = pandas_df % 5 < 2
    modin_cond_df = modin_df % 5 < 2

    pandas_result = pandas_df.where(pandas_cond_df, -pandas_df)
    modin_result = modin_df.where(modin_cond_df, -modin_df)
    assert all((to_pandas(modin_result) == pandas_result).all())

    # Test that we choose the right values to replace when `other` == `True`
    # everywhere.
    other_data = np.full(shape=pandas_df.shape, fill_value=True)
    modin_other, pandas_other = create_test_dfs(other_data, columns=columns)
    pandas_result = pandas_df.where(pandas_cond_df, pandas_other)
    modin_result = modin_df.where(modin_cond_df, modin_other)
    df_equals(modin_result, pandas_result)

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


def test_where_different_axis_order():
    # Test `where` when `cond`, `df`, and `other` each have columns and index
    # in different orders.
    data = test_data["float_nan_data"]
    pandas_df = pandas.DataFrame(data)
    pandas_cond_df = pandas_df % 5 < 2
    pandas_cond_df = pandas_cond_df.reindex(
        columns=pandas_df.columns[::-1], index=pandas_df.index[::-1]
    )
    pandas_other_df = -pandas_df
    pandas_other_df = pandas_other_df.reindex(
        columns=pandas_df.columns[-1:].append(pandas_df.columns[:-1]),
        index=pandas_df.index[-1:].append(pandas_df.index[:-1]),
    )

    modin_df = pd.DataFrame(pandas_df)
    modin_cond_df = pd.DataFrame(pandas_cond_df)
    modin_other_df = pd.DataFrame(pandas_other_df)

    pandas_result = pandas_df.where(pandas_cond_df, pandas_other_df)
    modin_result = modin_df.where(modin_cond_df, modin_other_df)
    df_equals(modin_result, pandas_result)


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
