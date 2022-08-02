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
from modin.pandas.test.utils import (
    create_test_dfs,
    test_data_values,
    df_equals,
)
from modin.config import NPartitions, Engine

import pandas
import pytest

NPartitions.put(4)

if Engine.get() == "Ray":
    import ray
    from modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition import (
        PandasOnRayDataframePartition,
    )
    from modin.core.execution.ray.implementations.pandas_on_ray.partitioning.virtual_partition import (
        PandasOnRayDataframeColumnPartition,
        PandasOnRayDataframeRowPartition,
    )

    block_partition_class = PandasOnRayDataframePartition
    virtual_column_partition_class = PandasOnRayDataframeColumnPartition
    virtual_row_partition_class = PandasOnRayDataframeRowPartition
    put = ray.put
elif Engine.get() == "Dask":
    from modin.core.execution.dask.implementations.pandas_on_dask.partitioning.virtual_partition import (
        PandasOnDaskDataframeColumnPartition,
        PandasOnDaskDataframeRowPartition,
    )
    from modin.core.execution.dask.implementations.pandas_on_dask.partitioning.partition import (
        PandasOnDaskDataframePartition,
    )
    from modin.core.execution.dask.common.engine_wrapper import DaskWrapper

    # initialize modin dataframe to initialize dask
    pd.DataFrame()

    def put(x):
        return DaskWrapper.put(x, hash=False)

    block_partition_class = PandasOnDaskDataframePartition
    virtual_column_partition_class = PandasOnDaskDataframeColumnPartition
    virtual_row_partition_class = PandasOnDaskDataframeRowPartition


def test_aligning_blocks():
    # Test problem when modin frames have the same number of rows, but different
    # blocks (partition.list_of_blocks). See #2322 for details
    accm = pd.DataFrame(["-22\n"] * 162)
    accm = accm.iloc[2:, :]
    accm.reset_index(drop=True, inplace=True)
    accm["T"] = pd.Series(["24.67\n"] * 145)

    # see #2322 for details
    repr(accm)


def test_aligning_blocks_with_duplicated_index():
    # Same problem as in `test_aligning_blocks` but with duplicated values in index.
    data11 = [0, 1]
    data12 = [2, 3]

    data21 = [0]
    data22 = [1, 2, 3]

    df1 = pd.DataFrame(data11).append(pd.DataFrame(data12))
    df2 = pd.DataFrame(data21).append(pd.DataFrame(data22))

    repr(df1 - df2)


def test_aligning_partitions():
    data = [0, 1, 2, 3, 4, 5]
    modin_df1, _ = create_test_dfs({"a": data, "b": data})
    modin_df = modin_df1.loc[:2]

    modin_df2 = modin_df.append(modin_df)

    modin_df2["c"] = modin_df1["b"]
    repr(modin_df2)


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
        modin_frame._row_lengths
        modin_frame._column_widths
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


@pytest.mark.skipif(
    Engine.get() not in ("Dask", "Ray"),
    reason="Rebalancing partitions is only supported for Dask and Ray engines",
)
@pytest.mark.parametrize(
    "test_type",
    [
        "many_small_dfs",
        "concatted_df_with_small_dfs",
        "large_df_plus_small_dfs",
    ],
)
def test_rebalance_partitions(test_type):
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
        NPartitions.get(),
        NPartitions.get(),
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

    large_df = large_df.apply(col_apply_func)
    new_large_modin_frame = large_df._query_compiler._modin_frame
    assert new_large_modin_frame._partitions.shape == (
        NPartitions.get(),
        NPartitions.get(),
    ), "Partitions list shape is incorrect."
    assert all(
        isinstance(ptn, new_large_modin_frame._partition_mgr_cls._partition_class)
        for ptn in new_large_modin_frame._partitions.flatten()
    ), "Partitions are not block partitioned after apply."
    large_df = pd.DataFrame(
        query_compiler=large_df._query_compiler.__constructor__(large_modin_frame)
    )
    # The following check tests that we can correctly form full-axis virtual partitions
    # over the same axis from non-full-axis virtual partitions.

    def row_apply_func(row):
        assert len(row) == 1000, "Partial axis partition detected."
        return row + 1

    large_df = large_df.apply(row_apply_func, axis=1)
    new_large_modin_frame = large_df._query_compiler._modin_frame
    assert new_large_modin_frame._partitions.shape == (
        4,
        4,
    ), "Partitions list shape is incorrect."
    assert all(
        isinstance(ptn, new_large_modin_frame._partition_mgr_cls._partition_class)
        for ptn in new_large_modin_frame._partitions.flatten()
    ), "Partitions are not block partitioned after apply."


@pytest.mark.skipif(
    Engine.get() not in ("Dask", "Ray"),
    reason="Only Dask and Ray engines have virtual partitions.",
)
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
