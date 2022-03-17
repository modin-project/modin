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

import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.distributed.dataframe.pandas import unwrap_partitions, from_partitions
from modin.config import Engine, NPartitions
from modin.pandas.test.utils import df_equals
from modin.pandas.indexing import compute_sliced_len
from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

PartitionClass = (
    FactoryDispatcher.get_factory().io_cls.frame_cls._partition_mgr_cls._partition_class
)

if Engine.get() == "Ray":
    import ray

    put_func = ray.put
    get_func = ray.get
    FutureType = ray.ObjectRef
elif Engine.get() == "Dask":
    from modin.core.execution.dask.common.engine_wrapper import DaskWrapper
    from distributed import Future

    put_func = lambda x: DaskWrapper.put(x)  # noqa: E731
    get_func = lambda x: DaskWrapper.materialize(x)  # noqa: E731
    FutureType = Future
elif Engine.get() == "Python":
    put_func = lambda x: x  # noqa: E731
    get_func = lambda x: x  # noqa: E731
    FutureType = object
else:
    raise NotImplementedError(
        f"'{Engine.get()}' engine is not supported by these test suites"
    )

NPartitions.put(4)
# HACK: implicit engine initialization (Modin issue #2989)
pd.DataFrame([])


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_unwrap_partitions(axis):
    data = np.random.randint(0, 100, size=(2**16, 2**8))
    df = pd.DataFrame(data)

    if axis is None:
        expected_partitions = df._query_compiler._modin_frame._partitions
        actual_partitions = np.array(unwrap_partitions(df, axis=axis))
        assert (
            expected_partitions.shape[0] == actual_partitions.shape[0]
            and expected_partitions.shape[1] == expected_partitions.shape[1]
        )
        for row_idx in range(expected_partitions.shape[0]):
            for col_idx in range(expected_partitions.shape[1]):
                if Engine.get() == "Ray":
                    assert (
                        expected_partitions[row_idx][col_idx].oid
                        == actual_partitions[row_idx][col_idx]
                    )
                if Engine.get() == "Dask":
                    assert (
                        expected_partitions[row_idx][col_idx].future
                        == actual_partitions[row_idx][col_idx]
                    )
    else:
        expected_axis_partitions = (
            df._query_compiler._modin_frame._partition_mgr_cls.axis_partition(
                df._query_compiler._modin_frame._partitions, axis ^ 1
            )
        )
        expected_axis_partitions = [
            axis_partition.force_materialization().unwrap(squeeze=True)
            for axis_partition in expected_axis_partitions
        ]
        actual_axis_partitions = unwrap_partitions(df, axis=axis)
        assert len(expected_axis_partitions) == len(actual_axis_partitions)
        for item_idx in range(len(expected_axis_partitions)):
            if Engine.get() in ["Ray", "Dask"]:
                df_equals(
                    get_func(expected_axis_partitions[item_idx]),
                    get_func(actual_axis_partitions[item_idx]),
                )


@pytest.mark.parametrize("column_widths", [None, "column_widths"])
@pytest.mark.parametrize("row_lengths", [None, "row_lengths"])
@pytest.mark.parametrize("columns", [None, "columns"])
@pytest.mark.parametrize("index", [None, "index"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_from_partitions(axis, index, columns, row_lengths, column_widths):
    num_rows = 2**16
    num_cols = 2**8
    data = np.random.randint(0, 100, size=(num_rows, num_cols))
    df1, df2 = pandas.DataFrame(data), pandas.DataFrame(data)
    expected_df = pandas.concat([df1, df2], axis=1 if axis is None else axis)

    index = expected_df.index if index == "index" else None
    columns = expected_df.columns if columns == "columns" else None
    row_lengths = (
        None
        if row_lengths is None
        else [num_rows, num_rows]
        if axis == 0
        else [num_rows]
    )
    column_widths = (
        None
        if column_widths is None
        else [num_cols]
        if axis == 0
        else [num_cols, num_cols]
    )

    if Engine.get() == "Ray":
        if axis is None:
            futures = [[put_func(df1), put_func(df2)]]
        else:
            futures = [put_func(df1), put_func(df2)]
    if Engine.get() == "Dask":
        if axis is None:
            futures = [put_func([df1, df2], hash=False)]
        else:
            futures = put_func([df1, df2], hash=False)
    actual_df = from_partitions(
        futures,
        axis,
        index=index,
        columns=columns,
        row_lengths=row_lengths,
        column_widths=column_widths,
    )
    df_equals(expected_df, actual_df)


@pytest.mark.parametrize("columns", ["original_col", "new_col"])
@pytest.mark.parametrize("index", ["original_idx", "new_idx"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_from_partitions_mismatched_labels(axis, index, columns):
    num_rows = 2**16
    num_cols = 2**8
    expected_df = pd.DataFrame(np.random.randint(0, 100, size=(num_rows, num_cols)))
    partitions = unwrap_partitions(expected_df, axis=axis)

    index = (
        expected_df.index
        if index == "original_idx"
        else [f"row{i}" for i in expected_df.index]
    )
    columns = (
        expected_df.columns
        if columns == "original_col"
        else [f"col{i}" for i in expected_df.columns]
    )

    expected_df.index = index
    expected_df.columns = columns
    actual_df = from_partitions(partitions, axis=axis, index=index, columns=columns)
    df_equals(expected_df, actual_df)


@pytest.mark.parametrize("row_labels", [[0, 2], slice(None)])
@pytest.mark.parametrize("col_labels", [[0, 2], slice(None)])
@pytest.mark.parametrize("is_length_future", [False, True])
@pytest.mark.parametrize("is_width_future", [False, True])
def test_mask_preserve_cache(row_labels, col_labels, is_length_future, is_width_future):
    def deserialize(obj):
        if isinstance(obj, FutureType):
            return get_func(obj)
        return obj

    def compute_length(indices, length):
        if not isinstance(indices, slice):
            return len(indices)
        return compute_sliced_len(indices, length)

    df = pandas.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8], "c": [9, 10, 11, 12]})
    obj_id = put_func(df)

    partition_shape = [
        put_func(len(df)) if is_length_future else len(df),
        put_func(len(df.columns)) if is_width_future else len(df.columns),
    ]

    source_partition = PartitionClass(obj_id, *partition_shape)
    masked_partition = source_partition.mask(
        row_labels=row_labels, col_labels=col_labels
    )

    expected_length = compute_length(row_labels, len(df))
    expected_width = compute_length(col_labels, len(df.columns))

    # Check that the cache is preserved
    assert expected_length == deserialize(masked_partition._length_cache)
    assert expected_width == deserialize(masked_partition._width_cache)
    # Check that the cache is interpreted properly
    assert expected_length == masked_partition.length()
    assert expected_width == masked_partition.width()
    # Recompute shape explicitly to check that the cached data was correct
    expected_length, expected_width = [
        masked_partition._length_cache,
        masked_partition._width_cache,
    ]
    masked_partition._length_cache = None
    masked_partition._width_cache = None
    assert expected_length == masked_partition.length()
    assert expected_width == masked_partition.width()
