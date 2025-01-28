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

import functools
import sys
import unittest.mock as mock

import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import (
    CpuCount,
    Engine,
    MinColumnPartitionSize,
    MinRowPartitionSize,
    NPartitions,
    RangePartitioning,
    context,
)
from modin.core.dataframe.algebra import Fold
from modin.core.dataframe.algebra.default2pandas import DataFrameDefault
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.core.dataframe.pandas.dataframe.utils import ColumnInfo, ShuffleSortFunctions
from modin.core.dataframe.pandas.metadata import (
    DtypesDescriptor,
    LazyProxyCategoricalDtype,
    ModinDtypes,
)
from modin.core.execution.utils import remote_function
from modin.core.storage_formats import PandasQueryCompiler
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.distributed.dataframe.pandas import from_partitions
from modin.tests.pandas.utils import (
    create_test_dfs,
    df_equals,
    eval_general,
    test_data_values,
)
from modin.utils import try_cast_to_pandas

NPartitions.put(4)

if Engine.get() == "Ray":
    import ray

    from modin.core.execution.ray.common import RayWrapper
    from modin.core.execution.ray.common.deferred_execution import MetaList
    from modin.core.execution.ray.implementations.pandas_on_ray.partitioning import (
        PandasOnRayDataframeColumnPartition,
        PandasOnRayDataframePartition,
        PandasOnRayDataframeRowPartition,
    )

    block_partition_class = PandasOnRayDataframePartition
    virtual_column_partition_class = PandasOnRayDataframeColumnPartition
    virtual_row_partition_class = PandasOnRayDataframeRowPartition
    put = RayWrapper.put
    deploy = RayWrapper.deploy
    materialize = RayWrapper.materialize
elif Engine.get() == "Dask":
    from modin.core.execution.dask.common import DaskWrapper
    from modin.core.execution.dask.implementations.pandas_on_dask.partitioning import (
        PandasOnDaskDataframeColumnPartition,
        PandasOnDaskDataframePartition,
        PandasOnDaskDataframeRowPartition,
    )

    # initialize modin dataframe to initialize dask
    pd.DataFrame()

    def put(x):
        return DaskWrapper.put(x, hash=False)

    block_partition_class = PandasOnDaskDataframePartition
    virtual_column_partition_class = PandasOnDaskDataframeColumnPartition
    virtual_row_partition_class = PandasOnDaskDataframeRowPartition
    deploy = DaskWrapper.deploy
    materialize = DaskWrapper.materialize
elif Engine.get() == "Unidist":
    from modin.core.execution.unidist.common import UnidistWrapper
    from modin.core.execution.unidist.implementations.pandas_on_unidist.partitioning import (
        PandasOnUnidistDataframeColumnPartition,
        PandasOnUnidistDataframePartition,
        PandasOnUnidistDataframeRowPartition,
    )

    block_partition_class = PandasOnUnidistDataframePartition
    virtual_column_partition_class = PandasOnUnidistDataframeColumnPartition
    virtual_row_partition_class = PandasOnUnidistDataframeRowPartition
    put = UnidistWrapper.put
elif Engine.get() == "Python":
    from modin.core.execution.python.common import PythonWrapper
    from modin.core.execution.python.implementations.pandas_on_python.partitioning import (
        PandasOnPythonDataframeColumnPartition,
        PandasOnPythonDataframePartition,
        PandasOnPythonDataframeRowPartition,
    )

    def put(x):
        return PythonWrapper.put(x, hash=False)

    def deploy(func, args=tuple()):
        return func(*args)

    def materialize(arg):
        return arg

    block_partition_class = PandasOnPythonDataframePartition
    virtual_column_partition_class = PandasOnPythonDataframeColumnPartition
    virtual_row_partition_class = PandasOnPythonDataframeRowPartition
else:
    raise NotImplementedError(
        f"These test suites are not implemented for the '{Engine.get()}' engine"
    )


def construct_modin_df_by_scheme(pandas_df, partitioning_scheme):
    """
    Build ``modin.pandas.DataFrame`` from ``pandas.DataFrame`` according the `partitioning_scheme`.

    Parameters
    ----------
    pandas_df : pandas.DataFrame
    partitioning_scheme : dict[{"row_lengths", "column_widths"}] -> list of ints

    Returns
    -------
    modin.pandas.DataFrame
    """
    index = pandas_df.index
    columns = pandas_df.columns
    row_lengths = partitioning_scheme["row_lengths"]
    column_widths = partitioning_scheme["column_widths"]
    new_length = sum(row_lengths)
    new_width = sum(column_widths)
    new_index = index if len(index) == new_length else index[:new_length]
    new_columns = columns if len(columns) == new_width else columns[:new_width]

    row_partitions = split_result_of_axis_func_pandas(
        axis=0,
        num_splits=len(row_lengths),
        result=pandas_df,
        min_block_size=MinRowPartitionSize.get(),
        length_list=row_lengths,
    )
    partitions = [
        split_result_of_axis_func_pandas(
            axis=1,
            num_splits=len(column_widths),
            result=row_part,
            min_block_size=MinColumnPartitionSize.get(),
            length_list=column_widths,
        )
        for row_part in row_partitions
    ]

    md_df = from_partitions(
        [[put(part) for part in row_parts] for row_parts in partitions],
        axis=None,
        index=new_index,
        columns=new_columns,
        row_lengths=row_lengths,
        column_widths=column_widths,
    )
    return md_df


def validate_partitions_cache(df, axis=None):
    """
    Assert that the ``PandasDataframe`` shape caches correspond to the actual partition's shapes.

    Parameters
    ----------
    df : PandasDataframe
    axis : int, optional
        An axis to verify the cache for. If not specified, verify cache for both of the axes.
    """
    axis = [0, 1] if axis is None else [axis]

    axis_lengths = [df._row_lengths_cache, df._column_widths_cache]

    for ax in axis:
        assert axis_lengths[ax] is not None
        assert df._partitions.shape[ax] == len(axis_lengths[ax])

    for i in range(df._partitions.shape[0]):
        for j in range(df._partitions.shape[1]):
            if 0 in axis:
                assert df._partitions[i, j].length() == axis_lengths[0][i]
            if 1 in axis:
                assert df._partitions[i, j].width() == axis_lengths[1][j]


def assert_has_no_cache(df, axis=0):
    """
    Assert that the passed dataframe has no labels and no lengths cache along the specified axis.

    Parameters
    ----------
    df : modin.pandas.DataFrame
    axis : int, default: 0
    """
    mf = df._query_compiler._modin_frame
    if axis == 0:
        assert not mf.has_materialized_index and mf._row_lengths_cache is None
    else:
        assert not mf.has_materialized_columns and mf._column_widths_cache is None


def remove_axis_cache(df, axis=0, remove_lengths=True):
    """
    Remove index/columns cache for the passed dataframe.

    Parameters
    ----------
    df : modin.pandas.DataFrame
    axis : int, default: 0
        0 - remove index cache, 1 - remove columns cache.
    remove_lengths : bool, default: True
        Whether to remove row lengths/column widths cache.
    """
    mf = df._query_compiler._modin_frame
    if axis == 0:
        mf.set_index_cache(None)
        if remove_lengths:
            mf._row_lengths_cache = None
    else:
        mf.set_columns_cache(None)
        if remove_lengths:
            mf._column_widths_cache = None


def test_aligning_blocks():
    # Test problem when modin frames have the same number of rows, but different
    # blocks (partition.list_of_blocks). See #2322 for details
    accm = pd.DataFrame(["-22\n"] * 162)
    accm = accm.iloc[2:, :]
    accm.reset_index(drop=True, inplace=True)
    accm["T"] = pd.Series(["24.67\n"] * 145)

    # see #2322 for details
    try_cast_to_pandas(accm)  # force materialization


def test_aligning_blocks_with_duplicated_index():
    # Same problem as in `test_aligning_blocks` but with duplicated values in index.
    data11 = [0, 1]
    data12 = [2, 3]

    data21 = [0]
    data22 = [1, 2, 3]

    df1 = pd.concat((pd.DataFrame(data11), pd.DataFrame(data12)))
    df2 = pd.concat((pd.DataFrame(data21), pd.DataFrame(data22)))

    try_cast_to_pandas(df1 - df2)  # force materialization


def test_aligning_partitions():
    data = [0, 1, 2, 3, 4, 5]
    modin_df1, _ = create_test_dfs({"a": data, "b": data})
    modin_df = modin_df1.loc[:2]

    modin_df2 = pd.concat((modin_df, modin_df))

    modin_df2["c"] = modin_df1["b"]
    try_cast_to_pandas(modin_df2)  # force materialization


@pytest.mark.parametrize("row_labels", [None, [("a", "")], ["a"]])
@pytest.mark.parametrize("col_labels", [None, ["a1"], [("c1", "z")]])
def test_take_2d_labels_or_positional(row_labels, col_labels):
    kwargs = {
        "index": [["a", "b", "c", "d"], ["", "", "x", "y"]],
        "columns": [["a1", "b1", "c1", "d1"], ["", "", "z", "x"]],
    }
    md_df, pd_df = create_test_dfs(np.random.rand(4, 4), **kwargs)

    _row_labels = slice(None) if row_labels is None else row_labels
    _col_labels = slice(None) if col_labels is None else col_labels
    pd_df = pd_df.loc[_row_labels, _col_labels]
    modin_frame = md_df._query_compiler._modin_frame
    new_modin_frame = modin_frame.take_2d_labels_or_positional(
        row_labels=row_labels, col_labels=col_labels
    )
    md_df._query_compiler._modin_frame = new_modin_frame

    df_equals(md_df, pd_df)


@pytest.mark.parametrize("has_partitions_shape_cache", [True, False])
@pytest.mark.parametrize("has_frame_shape_cache", [True, False])
def test_apply_func_to_both_axis(has_partitions_shape_cache, has_frame_shape_cache):
    """
    Test ``modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe.apply_select_indices`` functionality of broadcasting non-distributed items.
    """
    data = test_data_values[0]

    md_df, pd_df = create_test_dfs(data)
    values = pd_df.values + 1

    pd_df.iloc[:, :] = values

    modin_frame = md_df._query_compiler._modin_frame

    if has_frame_shape_cache:
        # Explicitly compute rows & columns shapes to store this info in frame's cache
        modin_frame.row_lengths
        modin_frame.column_widths
    else:
        # Explicitly reset frame's cache
        modin_frame._row_lengths_cache = None
        modin_frame._column_widths_cache = None

    for row in modin_frame._partitions:
        for part in row:
            if has_partitions_shape_cache:
                # Explicitly compute partition shape to store this info in its cache
                part.length()
                part.width()
            else:
                # Explicitly reset partition's shape cache
                part._length_cache = None
                part._width_cache = None

    def func_to_apply(partition, row_internal_indices, col_internal_indices, item):
        partition.iloc[row_internal_indices, col_internal_indices] = item
        return partition

    new_modin_frame = modin_frame.apply_select_indices(
        axis=None,
        func=func_to_apply,
        # Passing none-slices does not trigger shapes recomputation and so the cache is untouched.
        row_labels=slice(None),
        col_labels=slice(None),
        keep_remaining=True,
        new_index=pd_df.index,
        new_columns=pd_df.columns,
        item_to_distribute=values,
    )
    md_df._query_compiler._modin_frame = new_modin_frame

    df_equals(md_df, pd_df)


@pytest.mark.parametrize(
    "test_type",
    [
        "many_small_dfs",
        "concatted_df_with_small_dfs",
        "large_df_plus_small_dfs",
    ],
)
@pytest.mark.parametrize(
    "set_num_partitions",
    [1, 4],
    indirect=True,
)
def test_rebalance_partitions(test_type, set_num_partitions):
    num_partitions = NPartitions.get()
    if test_type == "many_small_dfs":
        small_dfs = [
            pd.DataFrame(
                [[i + j for j in range(0, 1000)]],
                columns=[f"col{j}" for j in range(0, 1000)],
                index=pd.Index([i]),
            )
            for i in range(1, 100001, 1000)
        ]
        large_df = pd.concat(small_dfs)
        col_length = 100
    elif test_type == "concatted_df_with_small_dfs":
        small_dfs = [
            pd.DataFrame(
                [[i + j for j in range(0, 1000)]],
                columns=[f"col{j}" for j in range(0, 1000)],
                index=pd.Index([i]),
            )
            for i in range(1, 100001, 1000)
        ]
        large_df = pd.concat([pd.concat(small_dfs)] + small_dfs[:3])
        col_length = 103
    else:
        large_df = pd.DataFrame(
            [[i + j for j in range(1, 1000)] for i in range(0, 100000, 1000)],
            columns=[f"col{j}" for j in range(1, 1000)],
            index=pd.Index(list(range(0, 100000, 1000))),
        )
        small_dfs = [
            pd.DataFrame(
                [[i + j for j in range(0, 1000)]],
                columns=[f"col{j}" for j in range(0, 1000)],
                index=pd.Index([i]),
            )
            for i in range(1, 4001, 1000)
        ]
        large_df = pd.concat([large_df] + small_dfs[:3])
        col_length = 103
    large_modin_frame = large_df._query_compiler._modin_frame
    assert large_modin_frame._partitions.shape == (
        num_partitions,
        num_partitions,
    ), "Partitions were not rebalanced after concat."
    assert all(
        isinstance(ptn, large_modin_frame._partition_mgr_cls._column_partitions_class)
        for ptn in large_modin_frame._partitions.flatten()
    )
    # The following check tests that we can correctly form full-axis virtual partitions
    # over the orthogonal axis from non-full-axis virtual partitions.

    def col_apply_func(col):
        assert len(col) == col_length, "Partial axis partition detected."
        return col + 1

    large_apply_result = large_df.apply(col_apply_func)
    large_apply_result_frame = large_apply_result._query_compiler._modin_frame
    assert large_apply_result_frame._partitions.shape == (
        num_partitions,
        num_partitions,
    ), "Partitions list shape is incorrect."
    assert all(
        isinstance(ptn, large_apply_result_frame._partition_mgr_cls._partition_class)
        for ptn in large_apply_result_frame._partitions.flatten()
    ), "Partitions are not block partitioned after column-wise apply."
    large_df = pd.DataFrame(
        query_compiler=large_df._query_compiler.__constructor__(large_modin_frame)
    )
    # The following check tests that we can correctly form full-axis virtual partitions
    # over the same axis from non-full-axis virtual partitions.

    def row_apply_func(row):
        assert len(row) == 1000, "Partial axis partition detected."
        return row + 1

    large_apply_result = large_df.apply(row_apply_func, axis=1)
    large_apply_result_frame = large_apply_result._query_compiler._modin_frame
    assert large_apply_result_frame._partitions.shape == (
        num_partitions,
        num_partitions,
    ), "Partitions list shape is incorrect."
    assert all(
        isinstance(ptn, large_apply_result_frame._partition_mgr_cls._partition_class)
        for ptn in large_apply_result_frame._partitions.flatten()
    ), "Partitions are not block partitioned after row-wise apply."

    large_apply_result = large_df.applymap(lambda x: x)
    large_apply_result_frame = large_apply_result._query_compiler._modin_frame
    assert large_apply_result_frame._partitions.shape == (
        num_partitions,
        num_partitions,
    ), "Partitions list shape is incorrect."
    assert all(
        isinstance(ptn, large_apply_result_frame._partition_mgr_cls._partition_class)
        for ptn in large_apply_result_frame._partitions.flatten()
    ), "Partitions are not block partitioned after element-wise apply."


@pytest.mark.parametrize(
    "axis,virtual_partition_class",
    ((0, virtual_column_partition_class), (1, virtual_row_partition_class)),
    ids=["partitions_spanning_all_columns", "partitions_spanning_all_rows"],
)
class TestDrainVirtualPartitionCallQueue:
    """Test draining virtual partition call queues.

    Test creating a virtual partition made of block partitions and/or one or
    more layers of virtual partitions, draining the top-level partition's
    call queue, and getting the result.

    In all these test cases, the full_axis argument doesn't matter for
    correctness because it only affects `apply`, which is not used here.
    Still, virtual partition users are not supposed to create full-axis
    virtual partitions out of other full-axis virtual partitions, so
    set full_axis to False everywhere.
    """

    def test_from_virtual_partitions_with_call_queues(
        self,
        axis,
        virtual_partition_class,
    ):
        # reverse the dataframe along the virtual partition axis.
        def reverse(df):
            return df.iloc[::-1, :] if axis == 0 else df.iloc[:, ::-1]

        level_zero_blocks_first = [
            block_partition_class(put(pandas.DataFrame([0]))),
            block_partition_class(put(pandas.DataFrame([1]))),
        ]
        level_one_virtual_first = virtual_partition_class(
            level_zero_blocks_first, full_axis=False
        )
        level_one_virtual_first = level_one_virtual_first.add_to_apply_calls(reverse)
        level_zero_blocks_second = [
            block_partition_class(put(pandas.DataFrame([2]))),
            block_partition_class(put(pandas.DataFrame([3]))),
        ]
        level_one_virtual_second = virtual_partition_class(
            level_zero_blocks_second, full_axis=False
        )
        level_one_virtual_second = level_one_virtual_second.add_to_apply_calls(reverse)
        level_two_virtual = virtual_partition_class(
            [level_one_virtual_first, level_one_virtual_second], full_axis=False
        )
        level_two_virtual.drain_call_queue()
        if axis == 0:
            expected_df = pandas.DataFrame([1, 0, 3, 2], index=[0, 0, 0, 0])
        else:
            expected_df = pandas.DataFrame([[1, 0, 3, 2]], columns=[0, 0, 0, 0])
        df_equals(
            level_two_virtual.to_pandas(),
            expected_df,
        )

    def test_from_block_and_virtual_partition_with_call_queues(
        self, axis, virtual_partition_class
    ):
        # make a function that reverses the dataframe along the virtual
        # partition axis.
        # for testing axis == 0, start with two 2-rows-by-1-column blocks. for
        # axis == 1, start with two 1-rows-by-2-column blocks.
        def reverse(df):
            return df.iloc[::-1, :] if axis == 0 else df.iloc[:, ::-1]

        block_data = [[0, 1], [2, 3]] if axis == 0 else [[[0, 1]], [[2, 3]]]
        level_zero_blocks = [
            block_partition_class(put(pandas.DataFrame(block_data[0]))),
            block_partition_class(put(pandas.DataFrame(block_data[1]))),
        ]
        level_zero_blocks[0] = level_zero_blocks[0].add_to_apply_calls(reverse)
        level_one_virtual = virtual_partition_class(
            level_zero_blocks[1], full_axis=False
        )
        level_one_virtual = level_one_virtual.add_to_apply_calls(reverse)
        level_two_virtual = virtual_partition_class(
            [level_zero_blocks[0], level_one_virtual], full_axis=False
        )
        level_two_virtual.drain_call_queue()
        if axis == 0:
            expected_df = pandas.DataFrame([1, 0, 3, 2], index=[1, 0, 1, 0])
        else:
            expected_df = pandas.DataFrame([[1, 0, 3, 2]], columns=[1, 0, 1, 0])
        df_equals(level_two_virtual.to_pandas(), expected_df)

    def test_virtual_partition_call_queues_at_three_levels(
        self, axis, virtual_partition_class
    ):
        block = block_partition_class(put(pandas.DataFrame([1])))
        level_one_virtual = virtual_partition_class([block], full_axis=False)
        level_one_virtual = level_one_virtual.add_to_apply_calls(
            lambda df: pandas.concat([df, pandas.DataFrame([2])])
        )
        level_two_virtual = virtual_partition_class(
            [level_one_virtual], full_axis=False
        )
        level_two_virtual = level_two_virtual.add_to_apply_calls(
            lambda df: pandas.concat([df, pandas.DataFrame([3])])
        )
        level_three_virtual = virtual_partition_class(
            [level_two_virtual], full_axis=False
        )
        level_three_virtual = level_three_virtual.add_to_apply_calls(
            lambda df: pandas.concat([df, pandas.DataFrame([4])])
        )
        level_three_virtual.drain_call_queue()
        df_equals(
            level_three_virtual.to_pandas(),
            pd.DataFrame([1, 2, 3, 4], index=[0, 0, 0, 0]),
        )


@pytest.mark.parametrize(
    "virtual_partition_class",
    (virtual_column_partition_class, virtual_row_partition_class),
    ids=["partitions_spanning_all_columns", "partitions_spanning_all_rows"],
)
def test_virtual_partition_apply_not_returning_pandas_dataframe(
    virtual_partition_class,
):
    # see https://github.com/modin-project/modin/issues/4811

    partition = virtual_partition_class(
        block_partition_class(put(pandas.DataFrame())), full_axis=False
    )

    apply_result = partition.apply(lambda df: 1).get()
    assert apply_result == 1


@pytest.mark.skipif(
    Engine.get() != "Ray",
    reason="Only ray.wait() does not take duplicate object refs.",
)
def test_virtual_partition_dup_object_ref():
    # See https://github.com/modin-project/modin/issues/5045
    frame_c = pd.DataFrame(np.zeros((100, 20), dtype=np.float32, order="C"))
    frame_c = [frame_c] * 20
    df = pd.concat(frame_c)
    partition = df._query_compiler._modin_frame._partitions.flatten()[0]
    obj_refs = partition.list_of_blocks
    assert len(obj_refs) != len(
        set(obj_refs)
    ), "Test setup did not contain duplicate objects"
    # The below call to wait() should not crash
    partition.wait()


__test_reorder_labels_cache_axis_positions = [
    pytest.param(lambda index: None, id="no_reordering"),
    pytest.param(lambda index: np.arange(len(index) - 1, -1, -1), id="reordering_only"),
    pytest.param(
        lambda index: [0, 1, 2, len(index) - 3, len(index) - 2, len(index) - 1],
        id="projection_only",
    ),
    pytest.param(
        lambda index: np.repeat(np.arange(len(index)), repeats=3), id="size_grow"
    ),
]


@pytest.mark.parametrize("row_positions", __test_reorder_labels_cache_axis_positions)
@pytest.mark.parametrize("col_positions", __test_reorder_labels_cache_axis_positions)
@pytest.mark.parametrize(
    "partitioning_scheme",
    [
        pytest.param(
            lambda df: {
                "row_lengths": [df.shape[0]],
                "column_widths": [df.shape[1]],
            },
            id="single_partition",
        ),
        pytest.param(
            lambda df: {
                "row_lengths": [32, max(0, df.shape[0] - 32)],
                "column_widths": [32, max(0, df.shape[1] - 32)],
            },
            id="two_unbalanced_partitions",
        ),
        pytest.param(
            lambda df: {
                "row_lengths": [df.shape[0] // NPartitions.get()] * NPartitions.get(),
                "column_widths": [df.shape[1] // NPartitions.get()] * NPartitions.get(),
            },
            id="perfect_partitioning",
        ),
        pytest.param(
            lambda df: {
                "row_lengths": [2**i for i in range(NPartitions.get())],
                "column_widths": [2**i for i in range(NPartitions.get())],
            },
            id="unbalanced_partitioning_equals_npartition",
        ),
        pytest.param(
            lambda df: {
                "row_lengths": [2] * (df.shape[0] // 2),
                "column_widths": [2] * (df.shape[1] // 2),
            },
            id="unbalanced_partitioning",
        ),
    ],
)
def test_reorder_labels_cache(
    row_positions,
    col_positions,
    partitioning_scheme,
):
    pandas_df = pandas.DataFrame(test_data_values[0])

    md_df = construct_modin_df_by_scheme(pandas_df, partitioning_scheme(pandas_df))
    md_df = md_df._query_compiler._modin_frame

    result = md_df._reorder_labels(
        row_positions(md_df.index), col_positions(md_df.columns)
    )
    validate_partitions_cache(result)


def test_reorder_labels_dtypes():
    pandas_df = pandas.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [1.0, 2.4, 3.4, 4.5],
            "c": ["a", "b", "c", "d"],
            "d": pd.to_datetime([1, 2, 3, 4], unit="D"),
        }
    )

    md_df = construct_modin_df_by_scheme(
        pandas_df,
        partitioning_scheme={
            "row_lengths": [len(pandas_df)],
            "column_widths": [
                len(pandas_df) // 2,
                len(pandas_df) // 2 + len(pandas_df) % 2,
            ],
        },
    )
    md_df = md_df._query_compiler._modin_frame

    result = md_df._reorder_labels(
        row_positions=None, col_positions=np.arange(len(md_df.columns) - 1, -1, -1)
    )
    df_equals(result.dtypes, result.to_pandas().dtypes)


@pytest.mark.parametrize(
    "left_partitioning, right_partitioning, ref_with_cache_available, ref_with_no_cache",
    # Note: this test takes into consideration that `MinRowPartitionSize == 32`,
    # `MinColumnPartitionSize == 32` and `NPartitions == 4`
    [
        (
            [2],
            [2],
            1,  # the num_splits is computed like (2 + 2 = 4 / chunk_size = 1 split)
            2,  # the num_splits is just splits sum (1 + 1 == 2)
        ),
        (
            [24],
            [54],
            3,  # the num_splits is computed like (24 + 54 = 78 / chunk_size = 3 splits)
            2,  # the num_splits is just splits sum (1 + 1 == 2)
        ),
        (
            [2],
            [299],
            4,  # the num_splits is bounded by NPartitions (2 + 299 = 301 / chunk_size = 10 splits -> bound by 4)
            2,  # the num_splits is just splits sum (1 + 1 == 2)
        ),
        (
            [32, 32],
            [128],
            4,  # the num_splits is bounded by NPartitions (32 + 32 + 128 = 192 / chunk_size = 6 splits -> bound by 4)
            3,  # the num_splits is just splits sum (2 + 1 == 3)
        ),
        (
            [128] * 7,
            [128] * 6,
            4,  # the num_splits is bounded by NPartitions (128 * 7 + 128 * 6 = 1664 / chunk_size = 52 splits -> bound by 4)
            4,  # the num_splits is just splits sum bound by NPartitions (7 + 6 = 13 splits -> 4 splits)
        ),
    ],
)
@pytest.mark.parametrize(
    "modify_config",
    [{NPartitions: 4, MinRowPartitionSize: 32, MinColumnPartitionSize: 32}],
    indirect=True,
)
def test_merge_partitioning(
    left_partitioning,
    right_partitioning,
    ref_with_cache_available,
    ref_with_no_cache,
    modify_config,
):
    from modin.core.storage_formats.pandas.utils import merge_partitioning

    left_df = pandas.DataFrame(
        [np.arange(sum(left_partitioning)) for _ in range(sum(left_partitioning))]
    )
    right_df = pandas.DataFrame(
        [np.arange(sum(right_partitioning)) for _ in range(sum(right_partitioning))]
    )

    left = construct_modin_df_by_scheme(
        left_df, {"row_lengths": left_partitioning, "column_widths": left_partitioning}
    )._query_compiler._modin_frame
    right = construct_modin_df_by_scheme(
        right_df,
        {"row_lengths": right_partitioning, "column_widths": right_partitioning},
    )._query_compiler._modin_frame

    assert left.row_lengths == left.column_widths == left_partitioning
    assert right.row_lengths == right.column_widths == right_partitioning

    res = merge_partitioning(left, right, axis=0)
    assert res == ref_with_cache_available

    res = merge_partitioning(left, right, axis=1)
    assert res == ref_with_cache_available

    (
        left._row_lengths_cache,
        left._column_widths_cache,
        right._row_lengths_cache,
        right._column_widths_cache,
    ) = [None] * 4

    res = merge_partitioning(left, right, axis=0)
    assert res == ref_with_no_cache
    # Verifying that no computations are being triggered
    assert all(
        cache is None
        for cache in (
            left._row_lengths_cache,
            left._column_widths_cache,
            right._row_lengths_cache,
            right._column_widths_cache,
        )
    )

    res = merge_partitioning(left, right, axis=1)
    assert res == ref_with_no_cache
    # Verifying that no computations are being triggered
    assert all(
        cache is None
        for cache in (
            left._row_lengths_cache,
            left._column_widths_cache,
            right._row_lengths_cache,
            right._column_widths_cache,
        )
    )


def test_merge_with_bad_partitioning():
    # https://github.com/modin-project/modin/pull/7229

    left_partitioning = [256]
    right_partitioning = [32, 32, 32, 32]

    left_df = pandas.DataFrame(
        [np.arange(sum(left_partitioning)) for _ in range(sum(left_partitioning))]
    )
    right_df = pandas.DataFrame(
        [np.arange(sum(right_partitioning)) for _ in range(sum(right_partitioning))]
    )

    left = construct_modin_df_by_scheme(
        left_df, {"row_lengths": left_partitioning, "column_widths": left_partitioning}
    )
    right = construct_modin_df_by_scheme(
        right_df,
        {"row_lengths": right_partitioning, "column_widths": right_partitioning},
    )

    left_frame = left._query_compiler._modin_frame
    right_frame = right._query_compiler._modin_frame
    assert left_frame.row_lengths == left_frame.column_widths == left_partitioning
    assert right_frame.row_lengths == right_frame.column_widths == right_partitioning

    # just a dummy value
    return_value = pd.DataFrame([1, 2, 3, 4])._query_compiler
    with mock.patch.object(
        left._query_compiler, "repartition", return_value=return_value
    ) as repartition:
        _ = left.merge(right)
        repartition.assert_called_once_with(axis=0)


def test_groupby_with_empty_partition():
    # see #5461 for details
    md_df = construct_modin_df_by_scheme(
        pandas_df=pandas.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 5, 6]}),
        partitioning_scheme={"row_lengths": [2, 2], "column_widths": [2]},
    )
    md_res = md_df.query("a > 1", engine="python")
    grp_obj = md_res.groupby("a")
    # check index error due to partitioning mismatching
    grp_obj.count()

    md_df = construct_modin_df_by_scheme(
        pandas_df=pandas.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 5, 6]}),
        partitioning_scheme={"row_lengths": [2, 2], "column_widths": [2]},
    )
    md_res = md_df.query("a > 1", engine="python")
    grp_obj = md_res.groupby(md_res["a"])
    grp_obj.count()


@pytest.mark.parametrize("set_num_partitions", [2], indirect=True)
def test_repartitioning(set_num_partitions):
    """
    This test verifies that 'keep_partitioning=False' doesn't actually preserve partitioning.

    For more details see: https://github.com/modin-project/modin/issues/5621
    """
    assert NPartitions.get() == 2

    pandas_df = pandas.DataFrame(
        {"a": [1, 1, 2, 2], "b": [3, 4, 5, 6], "c": [1, 2, 3, 4], "d": [4, 5, 6, 7]}
    )

    modin_df = construct_modin_df_by_scheme(
        pandas_df=pandas.DataFrame(
            {"a": [1, 1, 2, 2], "b": [3, 4, 5, 6], "c": [1, 2, 3, 4], "d": [4, 5, 6, 7]}
        ),
        partitioning_scheme={"row_lengths": [4], "column_widths": [2, 2]},
    )

    modin_frame = modin_df._query_compiler._modin_frame

    assert modin_frame._partitions.shape == (1, 2)
    assert modin_frame.column_widths == [2, 2]

    res = modin_frame.apply_full_axis(
        axis=1,
        func=lambda df: df,
        keep_partitioning=False,
        new_index=[0, 1, 2, 3],
        new_columns=["a", "b", "c", "d"],
    )

    assert res._partitions.shape == (1, 1)
    assert res.column_widths == [4]
    df_equals(res._partitions[0, 0].to_pandas(), pandas_df)
    df_equals(res.to_pandas(), pandas_df)


@pytest.mark.parametrize("col_name", ["numeric_col", "non_numeric_col"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("num_pivots", [3, 2, 1])
@pytest.mark.parametrize("all_pivots_are_unique", [True, False])
def test_split_partitions_kernel(
    col_name, ascending, num_pivots, all_pivots_are_unique
):
    """
    This test verifies proper work of the `split_partitions_using_pivots_for_sort` function
    used in partitions reshuffling.

    The function being tested splits the passed dataframe into parts according
    to the 'pivots' indicating boundary values for the parts.

    Parameters
    ----------
    col_name : {"numeric_col", "non_numeric_col"}
        The tested function takes a key column name to which the pivot values belong.
        The function may behave differently depending on the type of that column.
    ascending : {True, False}
        The split parts are returned either in ascending or descending order.
        This parameter helps us to test both of the cases.
    num_pivots : {3, 2, 1}
        The function's behavior may depend on the number of boundary values being passed.
    all_pivots_are_unique : {True, False}
        Duplicate pivot values cause empty partitions to be produced. This parameter helps
        to verify that the function still behaves correctly in such cases.
    """
    random_state = np.random.RandomState(42)

    df = pandas.DataFrame(
        {
            "numeric_col": range(9),
            "non_numeric_col": list("abcdefghi"),
        }
    )
    min_val, max_val = df[col_name].iloc[0], df[col_name].iloc[-1]

    # Selecting random boundary values for the key column
    pivots = random_state.choice(df[col_name], num_pivots, replace=False)
    if not all_pivots_are_unique:
        # Making the 'pivots' contain only duplicate values
        pivots = np.repeat(pivots[0], num_pivots)
    # The tested function assumes that we pass pivots in the ascending order
    pivots = np.sort(pivots)

    # Randomly reordering rows in the dataframe
    df = df.reindex(random_state.permutation(df.index))
    bins = ShuffleSortFunctions.split_partitions_using_pivots_for_sort(
        df,
        [
            ColumnInfo(
                name=col_name,
                is_numeric=pandas.api.types.is_numeric_dtype(df.dtypes[col_name]),
                pivots=pivots,
            )
        ],
        ascending=ascending,
    )

    # Building reference bounds to make the result verification simpler
    bounds = np.concatenate([[min_val], pivots, [max_val]])
    if not ascending:
        # If the order is descending we want bounds to be in the descending order as well:
        # Ex: bounds = [0, 2, 5, 10] for ascending and [10, 5, 2, 0] for descending.
        bounds = bounds[::-1]

    for idx, part in enumerate(bins):
        if ascending:
            # Check that each part is in the range of 'bound[i] <= part <= bound[i + 1]'
            # Example, if the `pivots` were [2, 5] and the min/max values for the colum are min=0, max=10
            # Then each part satisfies: 0 <= part[0] <= 2; 2 <= part[1] <= 5; 5 <= part[2] <= 10
            assert (
                (bounds[idx] <= part[col_name]) & (part[col_name] <= bounds[idx + 1])
            ).all()
        else:
            # Check that each part is in the range of 'bound[i + 1] <= part <= bound[i]'
            # Example, if the `pivots` were [2, 5] and the min/max values for the colum are min=0, max=10
            # Then each part satisfies: 5 <= part[0] <= 10; 2 <= part[1] <= 5; 0 <= part[2] <= 2
            assert (
                (bounds[idx + 1] <= part[col_name]) & (part[col_name] <= bounds[idx])
            ).all()


@pytest.mark.parametrize("col_name", ["numeric_col", "non_numeric_col"])
@pytest.mark.parametrize("ascending", [True, False])
def test_split_partitions_with_empty_pivots(col_name, ascending):
    """
    This test verifies that the splitting function performs correctly when an empty pivots list is passed.
    The expected behavior is to return a single split consisting of the exact copy of the input dataframe.
    """
    df = pandas.DataFrame(
        {
            "numeric_col": range(9),
            "non_numeric_col": list("abcdefghi"),
        }
    )

    result = ShuffleSortFunctions.split_partitions_using_pivots_for_sort(
        df,
        [
            ColumnInfo(
                name=col_name,
                is_numeric=pandas.api.types.is_numeric_dtype(df.dtypes[col_name]),
                pivots=[],
            )
        ],
        ascending=ascending,
    )
    # We're expecting to recieve a single split here
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0].equals(df)


@pytest.mark.parametrize("ascending", [True, False])
def test_shuffle_partitions_with_empty_pivots(ascending):
    """
    This test verifies that the `PartitionMgr.shuffle_partitions` method can handle empty pivots list.
    """
    modin_frame = pd.DataFrame(
        np.array([["hello", "goodbye"], ["hello", "Hello"]])
    )._query_compiler._modin_frame

    assert modin_frame._partitions.shape == (1, 1)

    column_name = modin_frame.columns[1]

    shuffle_functions = ShuffleSortFunctions(
        # These are the parameters we pass in the `.sort_by()` implementation
        modin_frame,
        columns=column_name,
        ascending=ascending,
        ideal_num_new_partitions=1,
    )

    new_partitions = modin_frame._partition_mgr_cls.shuffle_partitions(
        modin_frame._partitions,
        index=0,
        shuffle_functions=shuffle_functions,
        final_shuffle_func=lambda df: df.sort_values(column_name),
    )
    ref = modin_frame.to_pandas().sort_values(column_name)
    res = new_partitions[0, 0].get()

    assert new_partitions.shape == (1, 1)
    assert ref.equals(res)


@pytest.mark.parametrize("ascending", [True, False])
def test_split_partition_preserve_names(ascending):
    """
    This test verifies that the dataframes being split by ``split_partitions_using_pivots_for_sort``
    preserve their index/column names.
    """
    df = pandas.DataFrame(
        {
            "numeric_col": range(9),
            "non_numeric_col": list("abcdefghi"),
        }
    )
    index_name = "custom_name"
    df.index.name = index_name
    df.columns.name = index_name

    # Pivots that contain empty bins
    pivots = [2, 2, 5, 7]
    splits = ShuffleSortFunctions.split_partitions_using_pivots_for_sort(
        df,
        [ColumnInfo(name="numeric_col", is_numeric=True, pivots=pivots)],
        ascending=ascending,
    )

    for part in splits:
        assert part.index.name == index_name
        assert part.columns.name == index_name


@pytest.mark.parametrize("has_cols_metadata", [True, False])
@pytest.mark.parametrize("has_dtypes_metadata", [True, False])
def test_merge_preserves_metadata(has_cols_metadata, has_dtypes_metadata):
    df1 = pd.DataFrame({"a": [1, 1, 2, 2], "b": list("abcd")})
    df2 = pd.DataFrame({"a": [4, 2, 1, 3], "b": list("bcaf"), "c": [3, 2, 1, 0]})

    modin_frame = df1._query_compiler._modin_frame

    if has_cols_metadata:
        # Verify that there were initially materialized metadata
        assert modin_frame.has_materialized_columns
    else:
        modin_frame._columns_cache = None

    if has_dtypes_metadata:
        # Verify that there were initially materialized metadata
        assert modin_frame.has_materialized_dtypes
    else:
        modin_frame.set_dtypes_cache(None)

    res = df1.merge(df2, on="b")._query_compiler._modin_frame

    if has_cols_metadata:
        assert res.has_materialized_columns
        if has_dtypes_metadata:
            assert res.has_materialized_dtypes
        else:
            # Verify that no materialization was triggered
            assert not res.has_materialized_dtypes
            assert not modin_frame.has_materialized_dtypes
    else:
        # Verify that no materialization was triggered
        assert not res.has_materialized_columns
        assert not res.has_materialized_dtypes
        assert not modin_frame.has_materialized_columns
        if not has_dtypes_metadata:
            assert not modin_frame.has_materialized_dtypes


def test_binary_op_preserve_dtypes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})

    def setup_cache(df, has_cache=True):
        if has_cache:
            _ = df.dtypes
            assert df._query_compiler.frame_has_materialized_dtypes
        else:
            df._query_compiler.set_frame_dtypes_cache(None)
            assert not df._query_compiler.frame_has_materialized_dtypes
        return df

    def assert_cache(df, has_cache=True):
        assert not (has_cache ^ df._query_compiler.frame_has_materialized_dtypes)

    # Check when `other` is a non-distributed object
    assert_cache(setup_cache(df) + 2.0)
    assert_cache(setup_cache(df) + {"a": 2.0, "b": 4})
    assert_cache(setup_cache(df) + [2.0, 4])
    assert_cache(setup_cache(df) + np.array([2.0, 4]))

    # Check when `other` is a dataframe
    other = pd.DataFrame({"b": [3, 4, 5], "c": [4.0, 5.0, 6.0]})
    assert_cache(setup_cache(df) + setup_cache(other, has_cache=True))
    assert_cache(setup_cache(df) + setup_cache(other, has_cache=False), has_cache=False)

    # Check when `other` is a series
    other = pd.Series({"b": 3.0, "c": 4.0})
    assert_cache(setup_cache(df) + setup_cache(other, has_cache=True))
    assert_cache(setup_cache(df) + setup_cache(other, has_cache=False), has_cache=False)


@pytest.mark.parametrize("axis", [0, 1])
def test_concat_dont_materialize_opposite_axis(axis):
    data = {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}
    df1, df2 = pd.DataFrame(data), pd.DataFrame(data)

    def assert_no_cache(df, axis):
        if axis:
            assert not df._query_compiler.frame_has_materialized_columns
        else:
            assert not df._query_compiler.frame_has_materialized_index

    def remove_cache(df, axis):
        if axis:
            df._query_compiler.set_frame_columns_cache(None)
        else:
            df._query_compiler.set_frame_index_cache(None)
        assert_no_cache(df, axis)
        return df

    df1, df2 = remove_cache(df1, axis), remove_cache(df2, axis)

    df_concated = pd.concat((df1, df2), axis=axis)
    assert_no_cache(df1, axis)
    assert_no_cache(df2, axis)
    assert_no_cache(df_concated, axis)


def test_setitem_bool_preserve_dtypes():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 5, 6]})
    indexer = pd.Series([True, False, True, False])

    assert df._query_compiler.frame_has_materialized_dtypes

    # slice(None) as a col_loc
    df.loc[indexer] = 2.0
    assert df._query_compiler.frame_has_materialized_dtypes

    # list as a col_loc
    df.loc[indexer, ["a", "b"]] = 2.0
    assert df._query_compiler.frame_has_materialized_dtypes

    # scalar as a col_loc
    df.loc[indexer, "a"] = 2.0
    assert df._query_compiler.frame_has_materialized_dtypes


def test_setitem_unhashable_preserve_dtypes():
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert df._query_compiler.frame_has_materialized_dtypes

    df2 = pd.DataFrame([[9, 9], [5, 5]])
    assert df2._query_compiler.frame_has_materialized_dtypes

    df[[1, 2]] = df2
    assert df._query_compiler.frame_has_materialized_dtypes


@pytest.mark.parametrize("modify_config", [{RangePartitioning: True}], indirect=True)
def test_groupby_size_shuffling(modify_config):
    # verifies that 'groupby.size()' works with reshuffling implementation
    # https://github.com/modin-project/modin/issues/6367
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 5, 6]})
    modin_frame = df._query_compiler._modin_frame

    with mock.patch.object(
        modin_frame,
        "_apply_func_to_range_partitioning",
        wraps=modin_frame._apply_func_to_range_partitioning,
    ) as shuffling_method:
        try_cast_to_pandas(df.groupby("a").size())

    shuffling_method.assert_called()


@pytest.mark.parametrize(
    "kwargs",
    [dict(axis=0, labels=[]), dict(axis=1, labels=["a"]), dict(axis=1, labels=[])],
)
def test_reindex_preserve_dtypes(kwargs):
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [3, 4, 5, 6]})

    reindexed_df = df.reindex(**kwargs)
    assert reindexed_df._query_compiler.frame_has_materialized_dtypes


class TestModinIndexIds:
    @staticmethod
    def _patch_get_index(df, axis=0):
        """Patch the ``.index``/``.columns`` attribute of the passed dataframe."""
        if axis == 0:
            return mock.patch.object(
                type(df),
                "index",
                new_callable=mock.PropertyMock,
                wraps=functools.partial(type(df).index.__get__, df),
            )
        else:
            return mock.patch.object(
                type(df),
                "columns",
                new_callable=mock.PropertyMock,
                wraps=functools.partial(type(df).columns.__get__, df),
            )

    def test_setitem_without_copartition(self):
        """Test that setitem for identical indices works without materializing the axis."""
        # simple insertion
        df = pd.DataFrame({f"col{i}": np.arange(256) for i in range(64)})
        remove_axis_cache(df)

        col = df["col0"]
        assert_has_no_cache(col)
        assert_has_no_cache(df)

        # insert the column back and check that no index computation were triggered
        with self._patch_get_index(df) as get_index_patch:
            df["col0"] = col
            # check that no cache computation was triggered
            assert_has_no_cache(df)
            assert_has_no_cache(col)
        get_index_patch.assert_not_called()

        # insertion with few map operations
        df = pd.DataFrame({f"col{i}": np.arange(256) for i in range(64)})
        remove_axis_cache(df)

        col = df["col0"]
        # perform some operations that doesn't modify index labels and partitioning
        col = col * 2 + 10
        assert_has_no_cache(col)
        assert_has_no_cache(df)

        # insert the modified column back and check that no index computation were triggered
        with self._patch_get_index(df) as get_index_patch:
            df["col0"] = col
            # check that no cache computation was triggered
            assert_has_no_cache(df)
            assert_has_no_cache(col)
        get_index_patch.assert_not_called()

    @pytest.mark.parametrize("axis", [0, 1])
    def test_concat_without_copartition(self, axis):
        """Test that concatenation for frames with identical indices works without materializing the axis."""
        df1 = pd.DataFrame({f"col{i}": np.arange(256) for i in range(64)})
        remove_axis_cache(df1, axis)

        # perform some operations that doesn't modify index labels and partitioning
        df2 = df1.abs().applymap(lambda df: df * 2)

        with self._patch_get_index(df1, axis) as get_index_patch:
            res = pd.concat([df1, df2], axis=axis ^ 1)
            # check that no cache computation was triggered
            assert_has_no_cache(df1, axis)
            assert_has_no_cache(df2, axis)
            assert_has_no_cache(res, axis)
        get_index_patch.assert_not_called()

    def test_index_updates_ref(self):
        """Test that copying the default ModinIndex to a new frame updates frame reference with the new one."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        remove_axis_cache(df1)

        modin_frame1 = df1._query_compiler._modin_frame
        # verify that index cache is 'default' and so holds a reference to the `modin_frame`
        assert modin_frame1._index_cache._is_default_callable

        ref_count_before = sys.getrefcount(modin_frame1)

        df2 = df1 + 1
        modin_frame2 = df2._query_compiler._modin_frame
        # verify that new index cache is also the 'default' one
        assert modin_frame2._index_cache._is_default_callable
        # verify that there's no new references being created to the old frame
        assert sys.getrefcount(modin_frame1) == ref_count_before

    def test_index_updates_axis(self):
        """Verify that the ModinIndex `axis` attribute is updated when copied to a new frame but for an opposit axis."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        remove_axis_cache(df1)

        # now index becomes columns and vice-versa, this means that the 'default callable'
        # of the ModinIndex now has to update its axis
        df2 = df1.T

        idx1 = df1._query_compiler._modin_frame._index_cache
        idx2 = df2._query_compiler._modin_frame._index_cache

        cols1 = df1._query_compiler._modin_frame._columns_cache
        cols2 = df2._query_compiler._modin_frame._columns_cache

        # check that we can compare df.index == df.T.columns & df.columns == df.T.index
        # without triggering any axis materialization
        assert (
            idx1._index_id == cols2._index_id and idx1._lengths_id == cols2._lengths_id
        )
        assert (
            cols1._index_id == idx2._index_id and cols1._lengths_id == idx2._lengths_id
        )

        # check that when the materialization is triggered for the transposed frame it produces proper labels
        assert df2.index.equals(pandas.Index(["a", "b"]))
        assert df2.columns.equals(pandas.Index([0, 1, 2]))

    def test_filter_empties_resets_lengths(self):
        """Verify that filtering out empty partitions affects ``ModinIndex._lengths_id`` field."""
        # case1: partitioning is modified by '._filter_empties()', meaning that '._lengths_id' should be changed
        md_df = construct_modin_df_by_scheme(
            pandas.DataFrame({"a": [1, 1, 2, 2]}),
            {"row_lengths": [2, 2], "column_widths": [1]},
        )
        mf = md_df.query("a < 2")._query_compiler._modin_frame
        mf.index  # trigger index materialization

        old_cache = mf._index_cache
        assert mf._partitions.shape == (2, 1)

        mf._filter_empties()
        new_cache = mf._index_cache

        assert new_cache._index_id == old_cache._index_id
        assert new_cache._lengths_id != old_cache._lengths_id
        assert new_cache._lengths_cache != old_cache._lengths_cache

        # case2: partitioning is NOT modified by '._filter_empties()', meaning that '._lengths_id' should stay the same
        md_df = construct_modin_df_by_scheme(
            pandas.DataFrame({"a": [1, 1, 2, 2]}),
            {"row_lengths": [2, 2], "column_widths": [1]},
        )
        mf = md_df._query_compiler._modin_frame

        old_cache = mf._index_cache
        assert mf._partitions.shape == (2, 1)

        mf._filter_empties()
        new_cache = mf._index_cache

        assert new_cache._index_id == old_cache._index_id
        assert new_cache._lengths_id == old_cache._lengths_id
        assert new_cache._lengths_cache == old_cache._lengths_cache

    def test_binops_without_repartitioning(self):
        """Test that binary operations for identical indices works without materializing the axis."""
        df = pd.DataFrame({f"col{i}": np.arange(256) for i in range(64)})
        remove_axis_cache(df)

        col1 = df["col1"]
        assert_has_no_cache(col1)
        assert_has_no_cache(df)

        col2 = df["col2"]
        assert_has_no_cache(col2)
        assert_has_no_cache(df)

        # perform a binary op and insert the result back then check that no index computation were triggered
        with self._patch_get_index(df) as get_index_df:
            df["result"] = col1 + col2
            # check that no cache computation was triggered
            assert_has_no_cache(df)
            assert_has_no_cache(col1)
            assert_has_no_cache(col2)
        get_index_df.assert_not_called()


def test_skip_set_columns():
    """
    Verifies that the mechanism of skipping the actual ``._set_columns()`` call in case
    the new columns are identical to the previous ones works properly.

    In this test, we rely on the ``modin_frame._deferred_column`` attribute.
    The new indices propagation is done lazily, and the ``deferred_column`` attribute
    indicates whether there's a new indices propagation pending.
    """
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 4, 5]})
    df.columns = ["col1", "col10"]
    # Verifies that the new columns were successfully set in case they're actually new
    assert df._query_compiler._modin_frame._deferred_column
    assert np.all(df.columns.values == ["col1", "col10"])

    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 4, 5]})
    df.columns = ["col1", "col2"]
    # Verifies that the new columns weren't set if they're equal to the previous ones
    assert not df._query_compiler._modin_frame._deferred_column

    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 4, 5]})
    df.columns = pandas.Index(["col1", "col2"], name="new name")
    # Verifies that the new columns were successfully set in case they's new metadata
    assert df.columns.name == "new name"

    df = pd.DataFrame(
        {("a", "col1"): [1, 2, 3], ("a", "col2"): [3, 4, 5], ("b", "col1"): [6, 7, 8]}
    )
    df.columns = df.columns.copy()
    # Verifies that the new columns weren't set if they're equal to the previous ones
    assert not df._query_compiler._modin_frame._deferred_column

    df = pd.DataFrame(
        {("a", "col1"): [1, 2, 3], ("a", "col2"): [3, 4, 5], ("b", "col1"): [6, 7, 8]}
    )
    new_cols = df.columns[::-1]
    df.columns = new_cols
    # Verifies that the new columns were successfully set in case they're actually new
    assert df._query_compiler._modin_frame._deferred_column
    assert df.columns.equals(new_cols)

    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 4, 5]})
    remove_axis_cache(df, axis=1)
    df.columns = ["col1", "col2"]
    # Verifies that the computation of the old columns wasn't triggered for the sake
    # of equality comparison, in this case the new columns should be set unconditionally,
    # meaning that the '_deferred_column' has to be True
    assert df._query_compiler._modin_frame._deferred_column


def test_query_dispatching():
    """
    Test whether the logic of determining whether the passed query
    can be performed row-wise works correctly in ``PandasQueryCompiler.rowwise_query()``.

    The tested method raises a ``NotImpementedError`` if the query cannot be performed row-wise
    and raises nothing if it can.
    """
    qc = pd.DataFrame(
        {"a": [1], "b": [2], "c": [3], "d": [4], "e": [5]}
    )._query_compiler

    local_var = 10  # noqa: F841 (unused variable)

    # these queries should be performed row-wise (so no exception)
    qc.rowwise_query("a < 1")
    qc.rowwise_query("a < b")
    qc.rowwise_query("a < (b + @local_var) * c > 10")

    # these queries cannot be performed row-wise (so they must raise an exception)
    with pytest.raises(NotImplementedError):
        qc.rowwise_query("a < b[0]")
    with pytest.raises(NotImplementedError):
        qc.rowwise_query("a < b.min()")
    with pytest.raises(NotImplementedError):
        qc.rowwise_query("a < (b + @local_var + (b - e.min())) * c > 10")
    with pytest.raises(NotImplementedError):
        qc.rowwise_query("a < b.size")


def test_sort_values_cache():
    """
    Test that the column widths cache after ``.sort_values()`` is valid:
    https://github.com/modin-project/modin/issues/6607
    """
    # 1 row partition and 2 column partitions, in this case '.sort_values()' will use
    # row-wise implementation and so the column widths WILL NOT be changed
    modin_df = construct_modin_df_by_scheme(
        pandas.DataFrame({f"col{i}": range(100) for i in range(64)}),
        partitioning_scheme={"row_lengths": [100], "column_widths": [32, 32]},
    )
    mf_initial = modin_df._query_compiler._modin_frame

    mf_res = modin_df.sort_values("col0")._query_compiler._modin_frame
    # check that row-wise implementation was indeed used (col widths were not changed)
    assert mf_res._column_widths_cache == [32, 32]
    # check that the cache and actual col widths match
    validate_partitions_cache(mf_res, axis=1)
    # check that the initial frame's cache wasn't changed
    assert mf_initial._column_widths_cache == [32, 32]
    validate_partitions_cache(mf_initial, axis=1)

    # 2 row partition and 2 column partitions, in this case '.sort_values()' will use
    # range-partitioning implementation and so the column widths WILL be changed
    modin_df = construct_modin_df_by_scheme(
        pandas.DataFrame({f"col{i}": range(100) for i in range(64)}),
        partitioning_scheme={"row_lengths": [50, 50], "column_widths": [32, 32]},
    )
    mf_initial = modin_df._query_compiler._modin_frame

    mf_res = modin_df.sort_values("col0")._query_compiler._modin_frame
    # check that range-partitioning implementation was indeed used (col widths were changed)
    assert mf_res._column_widths_cache == [64]
    # check that the cache and actual col widths match
    validate_partitions_cache(mf_res, axis=1)
    # check that the initial frame's cache wasn't changed
    assert mf_initial._column_widths_cache == [32, 32]
    validate_partitions_cache(mf_initial, axis=1)


def test_apply_full_axis_preserve_widths():
    md_df = construct_modin_df_by_scheme(
        pandas.DataFrame(
            {"a": [1, 2, 3, 4], "b": [3, 4, 5, 6], "c": [6, 7, 8, 9], "d": [0, 1, 2, 3]}
        ),
        {"row_lengths": [2, 2], "column_widths": [2, 2]},
    )._query_compiler._modin_frame

    assert md_df._row_lengths_cache == [2, 2]
    assert md_df._column_widths_cache == [2, 2]

    def func(df):
        if df.iloc[0, 0] == 1:
            return pandas.DataFrame(
                {"a": [1, 2, 3], "b": [3, 4, 5], "c": [6, 7, 8], "d": [0, 1, 2]}
            )
        else:
            return pandas.DataFrame({"a": [4], "b": [6], "c": [9], "d": [3]})

    res = md_df.apply_full_axis(
        func=func,
        axis=1,
        new_index=[0, 1, 2, 3],
        new_columns=["a", "b", "c", "d"],
        keep_partitioning=True,
    )
    col_widths_cache = res._column_widths_cache
    actual_column_widths = [part.width() for part in res._partitions[0]]

    assert col_widths_cache == actual_column_widths
    assert res._row_lengths_cache is None


def test_apply_full_axis_preserve_lengths():
    md_df = construct_modin_df_by_scheme(
        pandas.DataFrame(
            {"a": [1, 2, 3, 4], "b": [3, 4, 5, 6], "c": [6, 7, 8, 9], "d": [0, 1, 2, 3]}
        ),
        {"row_lengths": [2, 2], "column_widths": [2, 2]},
    )._query_compiler._modin_frame

    assert md_df._row_lengths_cache == [2, 2]
    assert md_df._column_widths_cache == [2, 2]

    def func(df):
        if df.iloc[0, 0] == 1:
            return pandas.DataFrame({"a": [3, 2, 3, 4], "b": [3, 4, 5, 6]})
        else:
            return pandas.DataFrame({"c": [9, 5, 6, 7]})

    res = md_df.apply_full_axis(
        func=func,
        axis=0,
        new_index=[0, 1, 2, 3],
        new_columns=["a", "b", "c"],
        keep_partitioning=True,
    )

    row_lengths_cache = res._row_lengths_cache
    actual_row_lengths = [part.length() for part in res._partitions[:, 0]]

    assert row_lengths_cache == actual_row_lengths
    assert res._column_widths_cache is None


class DummyFuture:
    """
    A dummy object emulating future's behaviour, this class is used in ``test_call_queue_serialization``.

    It stores a random numeric value representing its data and `was_materialized` state.
    Initially this object is considered to be serialized, the state can be changed by calling
    the ``.materialize()`` method.
    """

    def __init__(self):
        self._value = np.random.randint(0, 1_000_000)
        self._was_materialized = False

    def materialize(self):
        self._was_materialized = True
        return self

    def __eq__(self, other):
        if isinstance(other, type(self)) and self._value == other._value:
            return True
        return False


class TestModinDtypes:
    """Test ``ModinDtypes`` and ``DtypesDescriptor`` classes."""

    schema = pandas.Series(
        {
            "a": np.dtype("int64"),
            "b": np.dtype(float),
            "c": np.dtype(bool),
            "d": np.dtype(bool),
            "e": np.dtype("object"),
        }
    )

    def get_columns_order(self, cols):
        """Return a value to be passed as ``DtypesDescriptor(columns_order=...)`` parameter."""
        return {i: col for i, col in enumerate(cols)}

    class DummyDf:
        def __init__(self, schema):
            self._schema = pandas.Series(schema)
            # record calls to verify that we haven't materialized more than needed
            self.history = []

        def _compute_dtypes(self, subset=None):
            self.history.append(("_compute_dtypes", subset))
            return self._schema if subset is None else self._schema[subset]

        @property
        def columns(self):
            self.history.append(("columns",))
            return self._schema.index

        @property
        def has_materialized_columns(self):
            # False, to make descriptor avoid materialization at all cost
            return False

    def test_get_dtypes_set_modin_dtypes(self):
        """Test that ``ModinDtypes.get_dtypes_set()`` correctly propagates this request to the underlying value."""
        res = ModinDtypes(lambda: self.schema).get_dtypes_set()
        exp = set(self.schema.values)
        assert res == exp

        res = ModinDtypes(self.schema).get_dtypes_set()
        exp = set(self.schema.values)
        assert res == exp

        res = ModinDtypes(
            DtypesDescriptor(
                self.schema[["a", "b", "e"]], remaining_dtype=np.dtype(bool)
            )
        ).get_dtypes_set()
        exp = set(self.schema.values)
        assert res == exp

    def test_get_dtypes_set_desc(self):
        """
        Test that ``DtypesDescriptor.get_dtypes_set()`` returns valid values and doesn't
        trigger unnecessary computations.
        """
        df = self.DummyDf(self.schema)
        desc = DtypesDescriptor(
            self.schema[["a", "b"]], know_all_names=False, parent_df=df
        )
        res = desc.get_dtypes_set()
        exp = self.schema.values
        assert res == set(exp)
        # since 'know_all_names=False', we first have to retrieve columns
        # in order to determine missing dtypes and then call '._compute_dtypes()'
        # only on a subset
        assert len(df.history) == 2 and df.history == [
            ("columns",),
            ("_compute_dtypes", ["c", "d", "e"]),
        ]

        df = self.DummyDf(self.schema)
        desc = DtypesDescriptor(
            self.schema[["a", "b"]],
            cols_with_unknown_dtypes=["c", "d", "e"],
            parent_df=df,
        )
        res = desc.get_dtypes_set()
        exp = self.schema.values
        assert res == set(exp)
        # here we already know names for cols with unknown dtypes, so only
        # calling '._compute_dtypes()' on a subset
        assert len(df.history) == 1 and df.history[0] == (
            "_compute_dtypes",
            ["c", "d", "e"],
        )

        df = self.DummyDf(self.schema[["a", "b", "c", "d"]])
        desc = DtypesDescriptor(
            self.schema[["a", "b"]], remaining_dtype=np.dtype(bool), parent_df=df
        )
        res = desc.get_dtypes_set()
        exp = self.schema[["a", "b", "c", "d"]].values
        assert res == set(exp)
        # we don't need to access 'parent_df' in order to get dtypes set, as we
        # can infer it from 'known_dtypes' and 'remaining_dtype'
        assert len(df.history) == 0

        df = self.DummyDf(self.schema)
        desc = DtypesDescriptor(know_all_names=False, parent_df=df)
        res = desc.get_dtypes_set()
        exp = self.schema.values
        assert res == set(exp)
        # compute dtypes for all columns
        assert len(df.history) == 1 and df.history[0] == ("_compute_dtypes", None)

        df = self.DummyDf(self.schema)
        desc = DtypesDescriptor(
            cols_with_unknown_dtypes=self.schema.index.tolist(), parent_df=df
        )
        res = desc.get_dtypes_set()
        exp = self.schema.values
        assert res == set(exp)
        # compute dtypes for all columns
        assert len(df.history) == 1 and df.history[0] == (
            "_compute_dtypes",
            self.schema.index.tolist(),
        )

        df = self.DummyDf(self.schema)
        desc = DtypesDescriptor(
            cols_with_unknown_dtypes=["a", "b", "e"],
            remaining_dtype=np.dtype(bool),
            parent_df=df,
        )
        res = desc.get_dtypes_set()
        exp = self.schema.values
        assert res == set(exp)
        # here we already know names for cols with unknown dtypes, so only
        # calling '._compute_dtypes()' on a subset
        assert len(df.history) == 1 and df.history[0] == (
            "_compute_dtypes",
            ["a", "b", "e"],
        )

    def test_lazy_get_modin_dtypes(self):
        """Test that ``ModinDtypes.lazy_get()`` correctly propagates this request to the underlying value."""
        res = ModinDtypes(self.schema).lazy_get(["b", "c", "a"])
        exp = self.schema[["b", "c", "a"]]
        assert res._value.equals(exp)

        res = ModinDtypes(lambda: self.schema).lazy_get(["b", "c", "a"])
        exp = self.schema[["b", "c", "a"]]
        assert callable(res._value)
        assert res._value().equals(exp)

        res = ModinDtypes(
            DtypesDescriptor(
                self.schema[["a", "b"]], cols_with_unknown_dtypes=["c", "d", "e"]
            )
        ).lazy_get(["b", "c", "a"])
        exp = DtypesDescriptor(
            self.schema[["a", "b"]],
            cols_with_unknown_dtypes=["c"],
            columns_order={0: "b", 1: "c", 2: "a"},
        )
        assert res._value.equals(exp)

    def test_lazy_get_desc(self):
        """
        Test that ``DtypesDescriptor.lazy_get()`` work properly.

        In this test we never specify `parent_df` for a descriptor, verifying that
        ``.lazy_get()`` never triggers any computations.
        """
        desc = DtypesDescriptor(self.schema[["a", "b"]])
        subset = ["a", "c", "e"]
        res = desc.lazy_get(subset)
        exp = DtypesDescriptor(
            self.schema[subset[:1]],
            cols_with_unknown_dtypes=subset[1:],
            columns_order=self.get_columns_order(subset),
        )
        assert res.equals(exp)

        desc = DtypesDescriptor(self.schema[["a", "b"]], remaining_dtype=np.dtype(bool))
        subset = ["a", "c", "d"]
        res = desc.lazy_get(subset)
        exp = DtypesDescriptor(
            # dtypes for 'c' and 'b' were infered from 'remaining_dtype' parameter
            self.schema[subset],
            columns_order=self.get_columns_order(subset),
            _schema_is_known=True,
        )
        assert res.equals(exp)

        desc = DtypesDescriptor()
        subset = ["a", "c", "d"]
        res = desc.lazy_get(subset)
        exp = DtypesDescriptor(
            cols_with_unknown_dtypes=subset,
            columns_order=self.get_columns_order(subset),
        )
        assert res.equals(exp)

        desc = DtypesDescriptor(remaining_dtype=np.dtype(bool))
        subset = ["c", "d"]
        res = desc.lazy_get(subset)
        exp = DtypesDescriptor(
            # dtypes for 'c' and 'd' were infered from 'remaining_dtype' parameter
            self.schema[subset],
            columns_order=self.get_columns_order(subset),
            _schema_is_known=True,
        )
        assert res.equals(exp)

    def test_concat_axis_0(self):
        res = DtypesDescriptor.concat(
            [
                DtypesDescriptor(self.schema[["a", "b"]]),
                DtypesDescriptor(self.schema[["c", "d"]]),
            ]
        )
        # simply concat known schemas
        exp = DtypesDescriptor(self.schema[["a", "b", "c", "d"]])
        assert res.equals(exp)

        res = DtypesDescriptor.concat(
            [
                DtypesDescriptor(self.schema[["a", "b"]]),
                DtypesDescriptor(remaining_dtype=np.dtype(bool)),
            ]
        )
        # none of the descriptors had missing column names, so we can preserve 'remaining_dtype'
        exp = DtypesDescriptor(self.schema[["a", "b"]], remaining_dtype=np.dtype(bool))
        assert res.equals(exp)

        res = DtypesDescriptor.concat(
            [
                DtypesDescriptor(self.schema[["a", "b"]], know_all_names=False),
                DtypesDescriptor(remaining_dtype=np.dtype(bool)),
            ]
        )
        # can't preserve 'remaining_dtype' since first descriptor has unknown column names
        exp = DtypesDescriptor(self.schema[["a", "b"]], know_all_names=False)
        assert res.equals(exp)

        res = DtypesDescriptor.concat(
            [
                DtypesDescriptor(self.schema[["a", "b"]]),
                DtypesDescriptor(
                    cols_with_unknown_dtypes=["d", "e"], know_all_names=False
                ),
                DtypesDescriptor(remaining_dtype=np.dtype(bool)),
            ]
        )
        # can't preserve 'remaining_dtype' since second descriptor has unknown column names
        exp = DtypesDescriptor(
            self.schema[["a", "b"]],
            cols_with_unknown_dtypes=["d", "e"],
            know_all_names=False,
        )
        assert res.equals(exp)

        res = DtypesDescriptor.concat(
            [
                DtypesDescriptor(
                    self.schema[["a", "b"]],
                ),
                DtypesDescriptor(
                    cols_with_unknown_dtypes=["d", "e"],
                ),
                DtypesDescriptor(remaining_dtype=np.dtype(bool)),
            ]
        )
        # none of the descriptors had missing column names, so we can preserve 'remaining_dtype'
        exp = DtypesDescriptor(
            self.schema[["a", "b"]],
            cols_with_unknown_dtypes=["d", "e"],
            remaining_dtype=np.dtype(bool),
        )
        assert res.equals(exp)

        res = DtypesDescriptor.concat(
            [
                DtypesDescriptor(
                    self.schema[["a", "b"]], remaining_dtype=np.dtype(bool)
                ),
                DtypesDescriptor(
                    cols_with_unknown_dtypes=["d", "e"], remaining_dtype=np.dtype(float)
                ),
                DtypesDescriptor(remaining_dtype=np.dtype(bool)),
            ]
        )
        # remaining dtypes don't match, so we drop them and set 'know_all_names=False'
        exp = DtypesDescriptor(
            self.schema[["a", "b"]],
            cols_with_unknown_dtypes=["d", "e"],
            know_all_names=False,
        )
        assert res.equals(exp)

    @pytest.mark.parametrize(
        "initial_dtypes, result_cols_with_known_dtypes, result_cols_with_unknown_dtypes",
        [
            [
                # initial dtypes (cols_with_known_dtypes, cols_with_unknown_dtypes, remaining_dtype):
                # dtypes for all columns are known
                [
                    (["a", "b", "c", "d"], [], None),
                    (["a", "b", "e", "d"], [], None),
                    (["a", "b"], [], None),
                ],
                # result_cols_with_known_dtypes:
                # all dtypes were known in the beginning, expecting the same
                # for the result
                ["a", "b", "c", "d", "e"],
                # result_cols_with_unknown_dtypes
                [],
            ],
            [
                # initial dtypes (cols_with_known_dtypes, cols_with_unknown_dtypes, remaining_dtype)
                [
                    (["a", "b"], ["c", "d"], None),
                    (["a", "b", "d"], ["e"], None),
                    (["a", "b"], [], None),
                ],
                # result_cols_with_known_dtypes:
                # across all dataframes, dtypes were only known for 'a' and 'b' columns
                ["a", "b"],
                # result_cols_with_unknown_dtypes
                ["c", "d", "e"],
            ],
            [
                # initial dtypes (cols_with_known_dtypes, cols_with_unknown_dtypes, remaining_dtype):
                # the 'e' column in the second frame is missing here, emulating 'know_all_names=False' case
                [
                    (["a", "b"], ["c", "d"], None),
                    (["a", "b", "d"], [], None),
                    (["a", "b"], [], None),
                ],
                # result_cols_with_known_dtypes
                ["a", "b"],
                # result_cols_with_unknown_dtypes:
                # the missing 'e' column will be deducted from the resulted frame after '.concat()'
                ["c", "d", "e"],
            ],
            [
                # initial dtypes (cols_with_known_dtypes, cols_with_unknown_dtypes, remaining_dtype)
                # the 'c' column in the first frame is described using 'remaining_dtype'
                [
                    (["a", "b", "d"], [], np.dtype(bool)),
                    (["a", "b", "e", "d"], [], None),
                    (["a", "b"], [], None),
                ],
                # result_cols_with_known_dtypes:
                # remaining dtypes are not supported by 'concat(axis=0)', so dtype for the 'c'
                # column is missing here
                ["a", "b", "e", "d"],
                # result_cols_with_unknown_dtypes:
                ["c"],
            ],
        ],
    )
    def test_concat_axis_1(
        self,
        initial_dtypes,
        result_cols_with_known_dtypes,
        result_cols_with_unknown_dtypes,
    ):
        """
        Test that ``DtypesDescriptor.concat(axis=1)`` works as expected.

        Parameters
        ----------
        initial_dtypes : list of tuples: (cols_with_known_dtypes, cols_with_unknown_dtypes, remaining_dtype)
            Describe how to build ``DtypesDescriptor`` for each of the three dataframes.
        result_cols_with_known_dtypes : list of labels
            Column names for which dtypes has to be determined after ``.concat()``.
        result_cols_with_unknown_dtypes : list of labels
            Column names for which dtypes has to be unknown after ``.concat()``.
        """
        md_df1, pd_df1 = create_test_dfs(
            {
                "a": [1, 2, 3],
                "b": [3.5, 4.5, 5.5],
                "c": [True, False, True],
                "d": ["a", "b", "c"],
            }
        )
        md_df2, pd_df2 = create_test_dfs(
            {
                "a": [1.5, 2.5, 3.5],
                "b": [3.5, 4.5, 5.5],
                "e": [True, False, True],
                "d": ["a", "b", "c"],
            }
        )
        md_df3, pd_df3 = create_test_dfs({"a": [1, 2, 3], "b": [3.5, 4.5, 5.5]})

        for md_df, (known_cols, unknown_cols, remaining_dtype) in zip(
            [md_df1, md_df2, md_df3], initial_dtypes
        ):
            known_dtypes = {col: md_df.dtypes[col] for col in known_cols}
            know_all_names = (
                len(known_cols) + len(unknown_cols) == len(md_df.columns)
                or remaining_dtype is not None
            )
            # setting columns cache to 'None', in order to prevent completing 'dtypes' with the materialized columns
            md_df._query_compiler.set_frame_columns_cache(None)
            md_df._query_compiler.set_frame_dtypes_cache(
                ModinDtypes(
                    DtypesDescriptor(
                        known_dtypes,
                        unknown_cols,
                        remaining_dtype,
                        know_all_names=know_all_names,
                    )
                )
            )
        md_dtypes = pd.concat(
            [md_df1, md_df2, md_df3]
        )._query_compiler._modin_frame._dtypes
        pd_dtypes = pandas.concat([pd_df1, pd_df2, pd_df3]).dtypes
        if len(result_cols_with_known_dtypes) == len(pd_dtypes):
            md_dtypes = (
                md_dtypes if isinstance(md_dtypes, pandas.Series) else md_dtypes._value
            )
            assert isinstance(md_dtypes, pandas.Series)
            assert md_dtypes.equals(pd_dtypes)
        else:
            assert set(md_dtypes._value._known_dtypes.keys()) == set(
                result_cols_with_known_dtypes
            )
            # reindexing to ensure proper order
            md_known_dtypes = pandas.Series(md_dtypes._value._known_dtypes).reindex(
                result_cols_with_known_dtypes
            )
            assert md_known_dtypes.equals(pd_dtypes[result_cols_with_known_dtypes])
            assert set(md_dtypes._value._cols_with_unknown_dtypes) == set(
                result_cols_with_unknown_dtypes
            )

    def test_ModinDtypes_duplicated_concat(self):
        # test that 'ModinDtypes' is able to perform dtypes concatenation on duplicated labels
        # if all of them are Serieses
        res = ModinDtypes.concat([pandas.Series([np.dtype("int64")], index=["a"])] * 2)
        assert isinstance(res._value, pandas.Series)
        assert res._value.equals(
            pandas.Series([np.dtype("int64"), np.dtype("int64")], index=["a", "a"])
        )

        # test that 'ModinDtypes.concat' with duplicated labels raises when not all dtypes are materialized
        with pytest.raises(NotImplementedError):
            res = ModinDtypes.concat(
                [
                    pandas.Series([np.dtype("int64")], index=["a"]),
                    DtypesDescriptor(cols_with_unknown_dtypes=["a"]),
                ]
            )

    def test_update_parent(self):
        """
        Test that updating parents in ``DtypesDescriptor`` also propagates to stored lazy categoricals.
        """
        # 'df1' will have a materialized 'pandas.Series' as dtypes cache
        df1 = pd.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]}).astype({"a": "category"})
        assert isinstance(df1.dtypes["a"], LazyProxyCategoricalDtype)

        # 'df2' will have a 'DtypesDescriptor' with unknown dtypes for a column 'c'
        df2 = pd.DataFrame({"c": [2, 3, 4]})
        df2._query_compiler.set_frame_dtypes_cache(None)
        dtypes_cache = df2._query_compiler._modin_frame._dtypes
        assert isinstance(
            dtypes_cache._value, DtypesDescriptor
        ) and dtypes_cache._value._cols_with_unknown_dtypes == ["c"]

        # concatenating 'df1' and 'df2' to get a 'DtypesDescriptor' storing lazy categories
        # in its 'known_dtypes' field
        res = pd.concat([df1, df2], axis=1)
        old_parent = df1._query_compiler._modin_frame
        new_parent = res._query_compiler._modin_frame
        dtypes_cache = new_parent._dtypes._value

        # verifying that the reference for lazy categories to a new parent was updated
        assert dtypes_cache._parent_df is new_parent
        assert dtypes_cache._known_dtypes["a"]._parent is new_parent
        assert old_parent._dtypes["a"]._parent is old_parent

    @pytest.mark.parametrize(
        "initial_dtypes, result_dtypes",
        [
            [
                DtypesDescriptor(
                    {"a": np.dtype("int64"), "b": np.dtype(float), "c": np.dtype(float)}
                ),
                DtypesDescriptor(
                    cols_with_unknown_dtypes=["col1", "col2", "col3"],
                    columns_order={0: "col1", 1: "col2", 2: "col3"},
                ),
            ],
            [
                DtypesDescriptor(
                    {
                        "a": np.dtype("int64"),
                        "b": np.dtype(float),
                        "c": np.dtype(float),
                    },
                    columns_order={0: "a", 1: "b", 2: "c"},
                ),
                DtypesDescriptor(
                    {
                        "col1": np.dtype("int64"),
                        "col2": np.dtype(float),
                        "col3": np.dtype(float),
                    },
                    columns_order={0: "col1", 1: "col2", 2: "col3"},
                ),
            ],
            [
                DtypesDescriptor(
                    {"a": np.dtype("int64"), "b": np.dtype(float)},
                    cols_with_unknown_dtypes=["c"],
                    columns_order={0: "a", 1: "b", 2: "c"},
                ),
                DtypesDescriptor(
                    {"col1": np.dtype("int64"), "col2": np.dtype(float)},
                    cols_with_unknown_dtypes=["col3"],
                    columns_order={0: "col1", 1: "col2", 2: "col3"},
                ),
            ],
            [
                DtypesDescriptor(
                    {"a": np.dtype("int64")},
                    cols_with_unknown_dtypes=["c"],
                    know_all_names=False,
                ),
                DtypesDescriptor(
                    cols_with_unknown_dtypes=["col1", "col2", "col3"],
                    columns_order={0: "col1", 1: "col2", 2: "col3"},
                ),
            ],
            [
                DtypesDescriptor(
                    {"a": np.dtype("int64")}, remaining_dtype=np.dtype(float)
                ),
                DtypesDescriptor(
                    cols_with_unknown_dtypes=["col1", "col2", "col3"],
                    columns_order={0: "col1", 1: "col2", 2: "col3"},
                ),
            ],
            [
                lambda: pandas.Series(
                    [np.dtype("int64"), np.dtype(float), np.dtype(float)],
                    index=["a", "b", "c"],
                ),
                lambda: pandas.Series(
                    [np.dtype("int64"), np.dtype(float), np.dtype(float)],
                    index=["col1", "col2", "col3"],
                ),
            ],
            [
                pandas.Series(
                    [np.dtype("int64"), np.dtype(float), np.dtype(float)],
                    index=["a", "b", "c"],
                ),
                pandas.Series(
                    [np.dtype("int64"), np.dtype(float), np.dtype(float)],
                    index=["col1", "col2", "col3"],
                ),
            ],
        ],
    )
    def test_set_index_dataframe(self, initial_dtypes, result_dtypes):
        """Test that changing labels for a dataframe also updates labels of dtypes."""
        df = pd.DataFrame(
            {"a": [1, 2, 3], "b": [3.0, 4.0, 5.0], "c": [3.2, 4.5, 5.4]}
        )._query_compiler._modin_frame
        df.set_columns_cache(None)
        if isinstance(initial_dtypes, DtypesDescriptor):
            initial_dtypes = ModinDtypes(initial_dtypes)

        df.set_dtypes_cache(initial_dtypes)
        df.columns = ["col1", "col2", "col3"]

        if result_dtypes is not None:
            if callable(result_dtypes):
                assert callable(df._dtypes._value)
                assert df._dtypes._value().equals(result_dtypes())
            else:
                assert df._dtypes._value.equals(result_dtypes)
        assert df.dtypes.index.equals(pandas.Index(["col1", "col2", "col3"]))

    def test_set_index_with_dupl_labels(self):
        """Verify that setting duplicated columns doesn't propagate any errors to a user."""
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [3.5, 4.4, 5.5, 6.6]})
        # making sure that dtypes are represented by an unmaterialized dtypes-descriptor
        df._query_compiler.set_frame_dtypes_cache(None)

        df.columns = ["a", "a"]
        assert df.dtypes.equals(
            pandas.Series([np.dtype(int), np.dtype("float64")], index=["a", "a"])
        )

    def test_reset_index_mi_columns(self):
        # reproducer from: https://github.com/modin-project/modin/issues/6904
        md_df, pd_df = create_test_dfs({"a": [1, 1, 2, 2], "b": [3, 3, 4, 4]})
        eval_general(
            md_df,
            pd_df,
            lambda df: df.groupby("a").agg({"b": ["min", "std"]}).reset_index().dtypes,
        )

    def test_concat_mi(self):
        """
        Verify that concatenating dfs with non-MultiIndex and MultiIndex columns results into valid indices for lazy dtypes.
        """
        md_df1, pd_df1 = create_test_dfs({"a": [1, 1, 2, 2], "b": [3, 3, 4, 4]})
        md_df2, pd_df2 = create_test_dfs(
            {("l1", "v1"): [1, 1, 2, 2], ("l1", "v2"): [3, 3, 4, 4]}
        )

        # Drop actual dtypes in order to use partially-known dtypes
        md_df1._query_compiler.set_frame_dtypes_cache(None)
        md_df2._query_compiler.set_frame_dtypes_cache(None)

        md_res = pd.concat([md_df1, md_df2], axis=1)
        pd_res = pandas.concat([pd_df1, pd_df2], axis=1)
        df_equals(md_res.dtypes, pd_res.dtypes)


class TestZeroComputationDtypes:
    """
    Test cases that shouldn't trigger dtypes computation during their execution.
    """

    @pytest.mark.parametrize("self_dtype", ["materialized", "partial", "unknown"])
    @pytest.mark.parametrize(
        "value, value_dtype",
        [
            [3.5, np.dtype(float)],
            [[3.5, 2.4], np.dtype(float)],
            [np.array([3.5, 2.4]), np.dtype(float)],
            [pd.Series([3.5, 2.4]), np.dtype(float)],
        ],
    )
    def test_preserve_dtypes_setitem(self, self_dtype, value, value_dtype):
        """
        Test that ``df[single_existing_column] = value`` preserves dtypes cache.
        """
        with mock.patch.object(PandasDataframe, "_compute_dtypes") as patch:
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [3, 4]})
            if self_dtype == "materialized":
                assert df._query_compiler.frame_has_materialized_dtypes
            elif self_dtype == "partial":
                df._query_compiler.set_frame_dtypes_cache(
                    ModinDtypes(
                        DtypesDescriptor(
                            {"a": np.dtype("int64")},
                            cols_with_unknown_dtypes=["b", "c"],
                        )
                    )
                )
            elif self_dtype == "unknown":
                df._query_compiler.set_frame_dtypes_cache(None)
            else:
                raise NotImplementedError(self_dtype)

            df["b"] = value

            if self_dtype == "materialized":
                result_dtype = pandas.Series(
                    [np.dtype("int64"), value_dtype, np.dtype("int64")],
                    index=["a", "b", "c"],
                )
                assert df._query_compiler.frame_has_materialized_dtypes
                assert df.dtypes.equals(result_dtype)
            elif self_dtype == "partial":
                result_dtype = DtypesDescriptor(
                    {"a": np.dtype("int64"), "b": value_dtype},
                    cols_with_unknown_dtypes=["c"],
                    columns_order={0: "a", 1: "b", 2: "c"},
                )
                df._query_compiler._modin_frame._dtypes._value.equals(result_dtype)
            elif self_dtype == "unknown":
                result_dtype = DtypesDescriptor(
                    {"b": value_dtype},
                    cols_with_unknown_dtypes=["a", "b"],
                    columns_order={0: "a", 1: "b", 2: "c"},
                )
                df._query_compiler._modin_frame._dtypes._value.equals(result_dtype)
            else:
                raise NotImplementedError(self_dtype)

        patch.assert_not_called()

    @pytest.mark.parametrize("self_dtype", ["materialized", "partial", "unknown"])
    @pytest.mark.parametrize(
        "value, value_dtype",
        [
            [3.5, np.dtype(float)],
            [[3.5, 2.4], np.dtype(float)],
            [np.array([3.5, 2.4]), np.dtype(float)],
            [pd.Series([3.5, 2.4]), np.dtype(float)],
        ],
    )
    def test_preserve_dtypes_insert(self, self_dtype, value, value_dtype):
        with mock.patch.object(PandasDataframe, "_compute_dtypes") as patch:
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            if self_dtype == "materialized":
                assert df._query_compiler.frame_has_materialized_dtypes
            elif self_dtype == "partial":
                df._query_compiler.set_frame_dtypes_cache(
                    ModinDtypes(
                        DtypesDescriptor(
                            {"a": np.dtype("int64")}, cols_with_unknown_dtypes=["b"]
                        )
                    )
                )
            elif self_dtype == "unknown":
                df._query_compiler.set_frame_dtypes_cache(None)
            else:
                raise NotImplementedError(self_dtype)

            df.insert(loc=0, column="c", value=value)

            if self_dtype == "materialized":
                result_dtype = pandas.Series(
                    [value_dtype, np.dtype("int64"), np.dtype("int64")],
                    index=["c", "a", "b"],
                )
                assert df._query_compiler.frame_has_materialized_dtypes
                assert df.dtypes.equals(result_dtype)
            elif self_dtype == "partial":
                result_dtype = DtypesDescriptor(
                    {"a": np.dtype("int64"), "c": value_dtype},
                    cols_with_unknown_dtypes=["b"],
                    columns_order={0: "c", 1: "a", 2: "b"},
                )
                df._query_compiler._modin_frame._dtypes._value.equals(result_dtype)
            elif self_dtype == "unknown":
                result_dtype = DtypesDescriptor(
                    {"c": value_dtype},
                    cols_with_unknown_dtypes=["a", "b"],
                    columns_order={0: "c", 1: "a", 2: "b"},
                )
                df._query_compiler._modin_frame._dtypes._value.equals(result_dtype)
            else:
                raise NotImplementedError(self_dtype)

        patch.assert_not_called()

    def test_get_dummies_case(self):
        with mock.patch.object(PandasDataframe, "_compute_dtypes") as patch:
            df = pd.DataFrame(
                {"items": [1, 2, 3, 4], "b": [3, 3, 4, 4], "c": [1, 0, 0, 1]}
            )
            res = pd.get_dummies(df, columns=["b", "c"])
            cols = [col for col in res.columns if col != "items"]
            res[cols] = res[cols] / res[cols].mean()

            assert res._query_compiler.frame_has_materialized_dtypes

        patch.assert_not_called()

    @pytest.mark.parametrize("has_materialized_index", [True, False])
    @pytest.mark.parametrize("drop", [True, False])
    def test_preserve_dtypes_reset_index(self, drop, has_materialized_index):
        with mock.patch.object(PandasDataframe, "_compute_dtypes") as patch:
            # case 1: 'df' has complete dtype by default
            df = pd.DataFrame({"a": [1, 2, 3]})
            if has_materialized_index:
                assert df._query_compiler.frame_has_materialized_index
            else:
                df._query_compiler.set_frame_index_cache(None)
                assert not df._query_compiler.frame_has_materialized_index
            assert df._query_compiler.frame_has_materialized_dtypes

            res = df.reset_index(drop=drop)
            if drop:
                # we droped the index, so columns and dtypes shouldn't change
                assert res._query_compiler.frame_has_materialized_dtypes
                assert res.dtypes.equals(df.dtypes)
            else:
                if has_materialized_index:
                    # we should have inserted index dtype into the descriptor,
                    # and since both of them are materialized, the result should be
                    # materialized too
                    assert res._query_compiler.frame_has_materialized_dtypes
                    assert res.dtypes.equals(
                        pandas.Series(
                            [np.dtype("int64"), np.dtype("int64")], index=["index", "a"]
                        )
                    )
                else:
                    # we now know that there are cols with unknown name and dtype in our dataframe,
                    # so the resulting dtypes should contain information only about original column
                    expected_dtypes = DtypesDescriptor(
                        {"a": np.dtype("int64")},
                        know_all_names=False,
                    )
                    assert res._query_compiler._modin_frame._dtypes._value.equals(
                        expected_dtypes
                    )

            # case 2: 'df' has partial dtype by default
            df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
            df._query_compiler.set_frame_dtypes_cache(
                ModinDtypes(
                    DtypesDescriptor(
                        {"a": np.dtype("int64")}, cols_with_unknown_dtypes=["b"]
                    )
                )
            )
            if has_materialized_index:
                assert df._query_compiler.frame_has_materialized_index
            else:
                df._query_compiler.set_frame_index_cache(None)
                assert not df._query_compiler.frame_has_materialized_index

            res = df.reset_index(drop=drop)
            if drop:
                # we droped the index, so columns and dtypes shouldn't change
                assert res._query_compiler._modin_frame._dtypes._value.equals(
                    df._query_compiler._modin_frame._dtypes._value
                )
            else:
                if has_materialized_index:
                    # we should have inserted index dtype into the descriptor,
                    # the resulted dtype should have information about 'index' and 'a' columns,
                    # and miss dtype info for 'b' column
                    expected_dtypes = DtypesDescriptor(
                        {"index": np.dtype("int64"), "a": np.dtype("int64")},
                        cols_with_unknown_dtypes=["b"],
                        columns_order={0: "index", 1: "a", 2: "b"},
                    )
                    assert res._query_compiler._modin_frame._dtypes._value.equals(
                        expected_dtypes
                    )
                else:
                    # we miss info about the 'index' column since it wasn't materialized at
                    # the time of 'reset_index()' and we're still missing dtype info for 'b' column
                    expected_dtypes = DtypesDescriptor(
                        {"a": np.dtype("int64")},
                        cols_with_unknown_dtypes=["b"],
                        know_all_names=False,
                    )
                    assert res._query_compiler._modin_frame._dtypes._value.equals(
                        expected_dtypes
                    )

        patch.assert_not_called()

    def test_groupby_index_dtype(self):
        with mock.patch.object(PandasDataframe, "_compute_dtypes") as patch:
            # case 1: MapReduce impl, Series as an output of groupby
            df = pd.DataFrame({"a": [1, 2, 2], "b": [3, 4, 5]})
            res = df.groupby("a").size().reset_index(name="new_name")
            res_dtypes = res._query_compiler._modin_frame._dtypes._value
            assert "a" in res_dtypes._known_dtypes
            assert res_dtypes._known_dtypes["a"] == np.dtype("int64")

            # case 2: ExperimentalImpl impl, Series as an output of groupby
            RangePartitioning.put(True)
            try:
                df = pd.DataFrame({"a": [1, 2, 2], "b": [3, 4, 5]})
                res = df.groupby("a").size().reset_index(name="new_name")
                res_dtypes = res._query_compiler._modin_frame._dtypes._value
                assert "a" in res_dtypes._known_dtypes
                assert res_dtypes._known_dtypes["a"] == np.dtype("int64")
            finally:
                RangePartitioning.put(False)

            # case 3: MapReduce impl, DataFrame as an output of groupby
            df = pd.DataFrame({"a": [1, 2, 2], "b": [3, 4, 5]})
            res = df.groupby("a").sum().reset_index()
            res_dtypes = res._query_compiler._modin_frame._dtypes._value
            assert "a" in res_dtypes._known_dtypes
            assert res_dtypes._known_dtypes["a"] == np.dtype("int64")

            # case 4: ExperimentalImpl impl, DataFrame as an output of groupby
            RangePartitioning.put(True)
            try:
                df = pd.DataFrame({"a": [1, 2, 2], "b": [3, 4, 5]})
                res = df.groupby("a").sum().reset_index()
                res_dtypes = res._query_compiler._modin_frame._dtypes._value
                assert "a" in res_dtypes._known_dtypes
                assert res_dtypes._known_dtypes["a"] == np.dtype("int64")
            finally:
                RangePartitioning.put(False)

            # case 5: FullAxis impl, DataFrame as an output of groupby
            df = pd.DataFrame({"a": [1, 2, 2], "b": [3, 4, 5]})
            res = df.groupby("a").quantile().reset_index()
            res_dtypes = res._query_compiler._modin_frame._dtypes._value
            assert "a" in res_dtypes._known_dtypes
            assert res_dtypes._known_dtypes["a"] == np.dtype("int64")

        patch.assert_not_called()


@pytest.mark.skipif(Engine.get() != "Ray", reason="Ray specific")
@pytest.mark.parametrize("mode", [None, "Auto", "On", "Off"])
def test_ray_lazy_exec_mode(mode):
    import ray

    from modin.config import LazyExecution
    from modin.core.execution.ray.common.deferred_execution import DeferredExecution
    from modin.core.execution.ray.common.utils import ObjectIDType
    from modin.core.execution.ray.implementations.pandas_on_ray.partitioning import (
        PandasOnRayDataframePartition,
    )

    orig_mode = LazyExecution.get()
    try:
        if mode is None:
            mode = LazyExecution.get()
        else:
            LazyExecution.put(mode)
            assert mode == LazyExecution.get()

        df = pandas.DataFrame({"A": [1, 2, 3]})
        part = PandasOnRayDataframePartition(ray.put(df))

        def func(df):
            return len(df)

        ray_func = ray.put(func)

        if mode == "Auto":
            assert isinstance(part.apply(ray_func)._data_ref, ObjectIDType)
            assert isinstance(
                part.add_to_apply_calls(ray_func)._data_ref, DeferredExecution
            )
        elif mode == "On":
            assert isinstance(part.apply(ray_func)._data_ref, DeferredExecution)
            assert isinstance(
                part.add_to_apply_calls(ray_func)._data_ref, DeferredExecution
            )
        elif mode == "Off":
            assert isinstance(part.apply(ray_func)._data_ref, ObjectIDType)
            assert isinstance(part.add_to_apply_calls(ray_func)._data_ref, ObjectIDType)
        else:
            pytest.fail(f"Invalid value: {mode}")
    finally:
        LazyExecution.put(orig_mode)


@pytest.mark.skipif(Engine.get() != "Ray", reason="Ray specific")
def test_materialization_hook_serialization():
    @ray.remote(num_returns=1)
    def f1():
        return [1, 2, 3]

    @ray.remote(num_returns=1)
    def f2(i):
        return i

    hook = MetaList(f1.remote())[2]
    assert ray.get(f2.remote(hook)) == 3


def test_remote_function():
    def get_func():
        @remote_function
        def remote_func(arg):
            return arg

        return remote_func

    def get_capturing_func(arg):
        @remote_function
        def remote_func():
            return arg

        return remote_func

    if Engine.get() in ("Ray", "Unidist"):
        from modin.core.execution.utils import _remote_function_cache

        cache_len = len(_remote_function_cache)
        assert get_func() is get_func()
        assert get_func() in _remote_function_cache.values()
        assert get_capturing_func(1) not in _remote_function_cache.values()
        assert len(_remote_function_cache) == cache_len + 1

    assert materialize(deploy(get_func(), [123])) == 123
    assert get_capturing_func(1) is not get_capturing_func(2)
    assert (
        materialize(deploy(get_capturing_func(1)))
        + materialize(deploy(get_capturing_func(2)))
        == 3
    )


@pytest.mark.parametrize(
    "partitioning_scheme,expected_map_approach",
    [
        pytest.param(
            lambda df: {
                "row_lengths": [df.shape[0] // CpuCount.get()] * CpuCount.get(),
                "column_widths": [df.shape[1]],
            },
            "map_partitions",
            id="one_column_partition",
        ),
        pytest.param(
            lambda df: {
                "row_lengths": [df.shape[0] // (CpuCount.get() * 2)]
                * (CpuCount.get() * 2),
                "column_widths": [df.shape[1]],
            },
            "map_partitions_joined_by_column",
            id="very_long_column_partition",
        ),
        pytest.param(
            lambda df: {
                "row_lengths": [df.shape[0] // CpuCount.get()] * CpuCount.get(),
                "column_widths": [df.shape[1] // CpuCount.get()] * CpuCount.get(),
            },
            "map_axis_partitions",
            id="perfect_partitioning",
        ),
    ],
)
def test_dynamic_partitioning(partitioning_scheme, expected_map_approach):
    data_size = MinRowPartitionSize.get() * CpuCount.get()
    data = {f"col{i}": np.ones(data_size) for i in range(data_size)}
    df = pandas.DataFrame(data)

    modin_df = construct_modin_df_by_scheme(df, partitioning_scheme(df))
    partitions = modin_df._query_compiler._modin_frame._partitions
    partition_mgr_cls = modin_df._query_compiler._modin_frame._partition_mgr_cls

    with mock.patch.object(
        partition_mgr_cls,
        expected_map_approach,
        wraps=getattr(partition_mgr_cls, expected_map_approach),
    ) as expected_method:
        with context(DynamicPartitioning=True):
            partition_mgr_cls.map_partitions(partitions, lambda x: x * 2)
            expected_method.assert_called()


@pytest.mark.parametrize("npartitions", [7, CpuCount.get() * 2])
def test_map_partitions_joined_by_column(npartitions):
    with context(NPartitions=npartitions):
        ncols = MinColumnPartitionSize.get()
        nrows = MinRowPartitionSize.get() * CpuCount.get() * 2
        data = {f"col{i}": np.ones(nrows) for i in range(ncols)}
        df = pd.DataFrame(data)
        partitions = df._query_compiler._modin_frame._partitions
        partition_mgr_cls = df._query_compiler._modin_frame._partition_mgr_cls

        def map_func(df, first_arg, extra_arg=0):
            return df.map(lambda x: (x * first_arg) + extra_arg)

        column_splits = 2
        map_func_args = (2,)
        map_func_kwargs = {"extra_arg": 1}

        # this approach doesn't work if column_splits == 0
        with pytest.raises(ValueError):
            partition_mgr_cls.map_partitions_joined_by_column(
                partitions, 0, map_func, map_func_args, map_func_kwargs
            )

        result_partitions = partition_mgr_cls.map_partitions_joined_by_column(
            partitions,
            column_splits,
            map_func,
            map_func_args,
            map_func_kwargs,
        )
        assert (
            result_partitions.shape == partitions.shape
        ), "The result has a different split than the original."
        for i in range(result_partitions.shape[0]):
            assert np.all(
                result_partitions[i][0].to_numpy() == 3
            ), "Invalid map function result."


def test_fold_operator():
    new_index = list(range(500, 1000))
    new_columns = ["b"]

    initial_df = pandas.DataFrame({"a": range(0, 1000)})
    modin_df = pd.DataFrame(initial_df)
    expected_df = pandas.DataFrame(
        list(range(0, 1000, 2)), index=new_index, columns=new_columns
    )

    def filter_func(df):
        result = df[df.index % 2 == 0]
        result.index = new_index
        result.columns = new_columns
        return result

    PandasQueryCompiler.filter_func = Fold.register(filter_func)

    def filter_modin_dataframe1(df):
        return df.__constructor__(
            query_compiler=df._query_compiler.filter_func(
                fold_axis=0,
                new_index=new_index,
                new_columns=new_columns,
            )
        )

    pd.DataFrame.filter_dataframe1 = filter_modin_dataframe1

    filtered_df = modin_df.filter_dataframe1()

    df_equals(filtered_df, expected_df)

    def filter_modin_dataframe2(df):
        return df.__constructor__(
            query_compiler=df._query_compiler.filter_func(fold_axis=0)
        )

    pd.DataFrame.filter_dataframe2 = filter_modin_dataframe2

    filtered_df = modin_df.filter_dataframe2()

    df_equals(filtered_df, expected_df)


def test_default_property_warning_name():
    # Test that when a property defaults to pandas, the raised warning mentions the full name of
    # the pandas property rather than a hex address

    @property
    def _test_default_property(df):
        return "suspicious sentinel value"

    @property
    def qc_test_default_property(qc):
        return DataFrameDefault.register(_test_default_property)(qc)

    PandasQueryCompiler.qc_test_default_property = qc_test_default_property

    @property
    def dataframe_test_default_property(df):
        return df._query_compiler.qc_test_default_property

    pd.DataFrame.dataframe_test_default_property = dataframe_test_default_property

    with pytest.warns(
        UserWarning,
        match="<function DataFrame.<property fget:_test_default_property>> is not currently supported",
    ):
        pd.DataFrame([[1]]).dataframe_test_default_property
