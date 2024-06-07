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


"""Collection of utility functions for the PandasDataFrame."""

import pandas
from pandas.api.types import union_categoricals

from modin.error_message import ErrorMessage


def concatenate(dfs, copy=True):
    """
    Concatenate pandas DataFrames with saving 'category' dtype.

    All dataframes' columns must be equal to each other.

    Parameters
    ----------
    dfs : list
        List of pandas DataFrames to concatenate.
    copy : bool, default: True
        Make explicit copy when creating dataframe.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame.
    """
    for df in dfs:
        assert df.columns.equals(dfs[0].columns)
    for i in dfs[0].columns.get_indexer_for(dfs[0].select_dtypes("category").columns):
        columns = [df.iloc[:, i] for df in dfs]
        all_categorical_parts_are_empty = None
        has_non_categorical_parts = False
        for col in columns:
            if isinstance(col.dtype, pandas.CategoricalDtype):
                if all_categorical_parts_are_empty is None:
                    all_categorical_parts_are_empty = len(col) == 0
                    continue
                all_categorical_parts_are_empty &= len(col) == 0
            else:
                has_non_categorical_parts = True
        # 'union_categoricals' raises an error if some of the passed values don't have categorical dtype,
        # if it happens, we only want to continue when all parts with categorical dtypes are actually empty.
        # This can happen if there were an aggregation that discards categorical dtypes and that aggregation
        # doesn't properly do so for empty partitions
        if has_non_categorical_parts and all_categorical_parts_are_empty:
            continue
        union = union_categoricals(columns)
        for df in dfs:
            df.isetitem(
                i, pandas.Categorical(df.iloc[:, i], categories=union.categories)
            )
    # `ValueError: buffer source array is read-only` if copy==False
    if len(dfs) == 1 and copy:
        # concat doesn't make a copy if len(dfs) == 1,
        # so do it explicitly
        return dfs[0].copy()
    return pandas.concat(dfs, copy=copy)


def create_pandas_df_from_partitions(
    partition_data,
    partition_shape,
    called_from_remote=False,
    new_index=None,
    new_columns=None,
):
    """
    Convert partition data of multiple dataframes to a single dataframe.

    Parameters
    ----------
    partition_data : list
        List of pandas DataFrames or list of Object references holding pandas DataFrames.
    partition_shape : int or tuple
        Shape of the partitions NumPy array.
    called_from_remote : bool, default: False
        Flag used to check if explicit copy should be done in concat.
    new_index : pandas.Index, optional
        Index for propagation into internal partitions.
        Optimization allowing to do this in one remote kernel.
    new_columns : pandas.Index, optional
        Columns for propagation into internal partitions.
        Optimization allowing to do this in one remote kernel.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame.
    """
    if all(
        isinstance(obj, (pandas.DataFrame, pandas.Series)) for obj in partition_data
    ):
        height, width, *_ = tuple(partition_shape) + (0,)
        # restore 2d array
        objs = iter(partition_data)
        partition_data = [[next(objs) for _ in range(width)] for __ in range(height)]
    else:
        # Partitions do not always contain pandas objects.
        # This implementation comes from the fact that calling `partition.get`
        # function is not always equivalent to `partition.to_pandas`.
        partition_data = [[obj.to_pandas() for obj in part] for part in partition_data]
    if all(isinstance(part, pandas.Series) for row in partition_data for part in row):
        axis = 0
    elif all(
        isinstance(part, pandas.DataFrame) for row in partition_data for part in row
    ):
        axis = 1
    else:
        ErrorMessage.catch_bugs_and_request_email(True)

    def is_part_empty(part):
        return part.empty and (
            not isinstance(part, pandas.DataFrame) or (len(part.columns) == 0)
        )

    df_rows = [
        pandas.concat([part for part in row], axis=axis, copy=False)
        for row in partition_data
        if not all(is_part_empty(part) for part in row)
    ]

    # to reduce peak memory consumption
    del partition_data

    if len(df_rows) == 0:
        res = pandas.DataFrame()
    else:
        res = concatenate(df_rows, copy=not called_from_remote)

    if new_index is not None:
        res.index = new_index
    if new_columns is not None:
        res.columns = new_columns

    return res
