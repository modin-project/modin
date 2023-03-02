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
from modin.config import NPartitions, Engine, MinPartitionSize
from modin.distributed.dataframe.pandas import from_partitions
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas

import numpy as np
import pandas
import pytest

NPartitions.put(4)

if Engine.get() == "Ray":
    from modin.core.execution.ray.implementations.pandas_on_ray.partitioning import (
        PandasOnRayDataframePartition,
    )
    from modin.core.execution.ray.implementations.pandas_on_ray.partitioning import (
        PandasOnRayDataframeColumnPartition,
        PandasOnRayDataframeRowPartition,
    )
    from modin.core.execution.ray.common import RayWrapper

    block_partition_class = PandasOnRayDataframePartition
    virtual_column_partition_class = PandasOnRayDataframeColumnPartition
    virtual_row_partition_class = PandasOnRayDataframeRowPartition
    put = RayWrapper.put
elif Engine.get() == "Dask":
    from modin.core.execution.dask.implementations.pandas_on_dask.partitioning import (
        PandasOnDaskDataframeColumnPartition,
        PandasOnDaskDataframeRowPartition,
    )
    from modin.core.execution.dask.implementations.pandas_on_dask.partitioning import (
        PandasOnDaskDataframePartition,
    )
    from modin.core.execution.dask.common import DaskWrapper

    # initialize modin dataframe to initialize dask
    pd.DataFrame()

    def put(x):
        return DaskWrapper.put(x, hash=False)

    block_partition_class = PandasOnDaskDataframePartition
    virtual_column_partition_class = PandasOnDaskDataframeColumnPartition
    virtual_row_partition_class = PandasOnDaskDataframeRowPartition


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
    row_partitions = split_result_of_axis_func_pandas(
        axis=0,
        num_splits=len(partitioning_scheme["row_lengths"]),
        result=pandas_df,
        length_list=partitioning_scheme["row_lengths"],
    )
    partitions = [
        split_result_of_axis_func_pandas(
            axis=1,
            num_splits=len(partitioning_scheme["column_widths"]),
            result=row_part,
            length_list=partitioning_scheme["column_widths"],
        )
        for row_part in row_partitions
    ]

    md_df = from_partitions(
        [[put(part) for part in row_parts] for row_parts in partitions], axis=None
    )
    return md_df


@pytest.fixture
def modify_config(request):
    values = request.param
    old_values = {}

    for key, value in values.items():
        old_values[key] = key.get()
        key.put(value)

    yield  # waiting for the test to be completed
    # restoring old parameters
    for key, value in old_values.items():
        key.put(value)


def validate_partitions_cache(df):
    """Assert that the ``PandasDataframe`` shape caches correspond to the actual partition's shapes."""
    row_lengths = df._row_lengths_cache
    column_widths = df._column_widths_cache

    assert row_lengths is not None
    assert column_widths is not None
    assert df._partitions.shape[0] == len(row_lengths)
    assert df._partitions.shape[1] == len(column_widths)

    for i in range(df._partitions.shape[0]):
        for j in range(df._partitions.shape[1]):
            assert df._partitions[i, j].length() == row_lengths[i]
            assert df._partitions[i, j].width() == column_widths[j]


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


@pytest.mark.skipif(
    Engine.get() not in ("Dask", "Ray"),
    reason="Only Dask and Ray engines have virtual partitions.",
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
    # Note: this test takes into consideration that `MinPartitionSize == 32` and `NPartitions == 4`
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
    "modify_config", [{NPartitions: 4, MinPartitionSize: 32}], indirect=True
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
