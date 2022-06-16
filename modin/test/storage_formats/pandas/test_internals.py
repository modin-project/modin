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
from modin.pandas.test.utils import create_test_dfs, test_data_values, df_equals
from modin.config import NPartitions, Engine

import pytest

NPartitions.put(4)


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


small_dfs = [
    pd.DataFrame(
        [[i + j for j in range(0, 1000)]],
        columns=[f"col{j}" for j in range(1, 1001)],
        index=pd.Index([i - 1]),
    )
    for i in range(1, 10001, 100)
]


@pytest.mark.skipif(
    Engine.get() not in ("Dask", "Ray"),
    reason="Rebalancing partitions is only supported for Dask and Ray engines",
)
@pytest.mark.parametrize(
    "large_df",
    [pd.concat(small_dfs), pd.concat([pd.concat(small_dfs)] + small_dfs[:3])],
)
def test_rebalance_partitions(large_df):
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
        assert len(col) == 100, "Partial axis partition detected."
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
