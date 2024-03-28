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
from modin.config import Engine, NPartitions
from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
from modin.distributed.dataframe.pandas import from_partitions, unwrap_partitions
from modin.pandas.indexing import compute_sliced_len
from modin.tests.pandas.utils import df_equals, test_data

PartitionClass = (
    FactoryDispatcher.get_factory().io_cls.frame_cls._partition_mgr_cls._partition_class
)

if Engine.get() == "Ray":
    from modin.core.execution.ray.common import RayWrapper
    from modin.core.execution.ray.common.utils import ObjectIDType

    put_func = RayWrapper.put
    get_func = RayWrapper.materialize
    is_future = lambda obj: isinstance(obj, ObjectIDType)  # noqa: E731
elif Engine.get() == "Dask":
    from distributed import Future

    from modin.core.execution.dask.common import DaskWrapper

    # Looks like there is a key collision;
    # https://github.com/dask/distributed/issues/3703#issuecomment-619446739
    # recommends to use `hash=False`. Perhaps this should be the default value of `put`.
    put_func = lambda obj: DaskWrapper.put(obj, hash=False)  # noqa: E731
    get_func = DaskWrapper.materialize
    is_future = lambda obj: isinstance(obj, Future)  # noqa: E731
elif Engine.get() == "Unidist":
    from unidist import is_object_ref

    from modin.core.execution.unidist.common import UnidistWrapper

    put_func = UnidistWrapper.put
    get_func = UnidistWrapper.materialize
    is_future = is_object_ref
elif Engine.get() == "Python":
    put_func = lambda x: x  # noqa: E731
    get_func = lambda x: x  # noqa: E731
    is_future = lambda obj: isinstance(obj, object)  # noqa: E731
else:
    raise NotImplementedError(
        f"'{Engine.get()}' engine is not supported by these test suites"
    )

NPartitions.put(4)
# HACK: implicit engine initialization (Modin issue #2989)
pd.DataFrame([])


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("reverse_index", [True, False])
@pytest.mark.parametrize("reverse_columns", [True, False])
def test_unwrap_partitions(axis, reverse_index, reverse_columns):
    data = test_data["int_data"]

    def get_df(lib, data):
        df = lib.DataFrame(data)
        if reverse_index:
            df.index = df.index[::-1]
        if reverse_columns:
            df.columns = df.columns[::-1]
        return df

    df = get_df(pd, data)
    # `df` should not have propagated the index and column updates to its
    # partitions yet. The partitions of `expected_df` should have the updated
    # metadata because we construct `expected_df` directly from the updated
    # pandas dataframe.
    expected_df = pd.DataFrame(get_df(pandas, data))
    expected_partitions = expected_df._query_compiler._modin_frame._partitions
    if axis is None:
        actual_partitions = np.array(unwrap_partitions(df, axis=axis))
        assert expected_partitions.shape == actual_partitions.shape
        for row_idx in range(expected_partitions.shape[0]):
            for col_idx in range(expected_partitions.shape[1]):
                df_equals(
                    get_func(expected_partitions[row_idx][col_idx].list_of_blocks[0]),
                    get_func(actual_partitions[row_idx][col_idx]),
                )
    else:
        expected_axis_partitions = (
            expected_df._query_compiler._modin_frame._partition_mgr_cls.axis_partition(
                expected_partitions, axis ^ 1
            )
        )
        expected_axis_partitions = [
            axis_partition.force_materialization().unwrap(squeeze=True)
            for axis_partition in expected_axis_partitions
        ]
        actual_axis_partitions = unwrap_partitions(df, axis=axis)
        assert len(expected_axis_partitions) == len(actual_axis_partitions)
        for item_idx in range(len(expected_axis_partitions)):
            if Engine.get() in ["Ray", "Dask", "Unidist"]:
                df_equals(
                    get_func(expected_axis_partitions[item_idx]),
                    get_func(actual_axis_partitions[item_idx]),
                )


def test_unwrap_virtual_partitions():
    # see #5164 for details
    data = test_data["int_data"]
    df = pd.DataFrame(data)
    virtual_partitioned_df = pd.concat([df] * 10)
    actual_partitions = np.array(unwrap_partitions(virtual_partitioned_df, axis=None))
    expected_df = pd.concat([pd.DataFrame(data)] * 10)
    expected_partitions = expected_df._query_compiler._modin_frame._partitions
    assert expected_partitions.shape == actual_partitions.shape

    for row_idx in range(expected_partitions.shape[0]):
        for col_idx in range(expected_partitions.shape[1]):
            df_equals(
                get_func(
                    expected_partitions[row_idx][col_idx]
                    .force_materialization()
                    .list_of_blocks[0]
                ),
                get_func(actual_partitions[row_idx][col_idx]),
            )


@pytest.mark.parametrize("column_widths", [None, "column_widths"])
@pytest.mark.parametrize("row_lengths", [None, "row_lengths"])
@pytest.mark.parametrize("columns", [None, "columns"])
@pytest.mark.parametrize("index", [None, "index"])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_from_partitions(axis, index, columns, row_lengths, column_widths):
    data = test_data["int_data"]
    df1, df2 = pandas.DataFrame(data), pandas.DataFrame(data)
    num_rows, num_cols = df1.shape
    expected_df = pandas.concat([df1, df2], axis=1 if axis is None else axis)

    index = expected_df.index if index == "index" else None
    columns = expected_df.columns if columns == "columns" else None
    row_lengths = (
        None
        if row_lengths is None
        else [num_rows, num_rows] if axis == 0 else [num_rows]
    )
    column_widths = (
        None
        if column_widths is None
        else [num_cols] if axis == 0 else [num_cols, num_cols]
    )
    futures = []
    if axis is None:
        futures = [[put_func(df1), put_func(df2)]]
    else:
        futures = [put_func(df1), put_func(df2)]
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
    expected_df = pd.DataFrame(test_data["int_data"])
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
        if is_future(obj):
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
