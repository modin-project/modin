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

"""Collection of algebra utility functions, mostly for internal use."""

import numpy as np
import pandas
from typing import Callable, Union

from modin.config import NPartitions


def build_sort_functions(
    modin_frame: "PandasDataframe",
    columns: list,
    method: str,
    ascending: Union[list, bool],
    **kwargs: dict
) -> "dict[str, Callable]":
    """
    Return a dictionary containing the functions necessary to perform a sort.

    Parameters
    ----------
    modin_frame : PandasDataframe
        A reference to the frame calling these sort functions.
    columns : list[str]
        The list of column names to sort by.
    method : str
        The method to use for picking quantiles.
    ascending : list[bool] or bool
        The ascending flag (or a list of ascending flags for each column).
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    dict :
        A dictionary containing the functions to pick quantiles, pick overall quantiles, and split
        partitions for sorting.
    """

    def sample_fn(partition, A=100, k=0.05, q=0.1):
        return get_partition_quantiles_for_sort(
            partition, columns, A=A, k=k, q=q, method=method
        )

    def pivot_fn(samples):
        return pick_pivots_from_quantiles_for_sort(
            modin_frame, samples, columns, method
        )

    def split_fn(partition, pivots):
        return split_partitions_using_pivots_for_sort(
            modin_frame, partition, columns, pivots, ascending, **kwargs
        )

    return {
        "sample_function": sample_fn,
        "pivot_function": pivot_fn,
        "split_function": split_fn,
    }


def get_partition_quantiles_for_sort(
    df: pandas.DataFrame,
    columns: list,
    A: int = 100,
    k: float = 0.05,
    q: float = 0.1,
    method: str = "linear",
) -> "tuple[pandas.DataFrame, np.ndarray]":
    """
    Pick quantiles over the given partition.

    This function is applied to each row-axis partition in parallel to select quantiles
    from each. It samples using the following algorithm:
        * If there are <= A (100) rows in the dataframe, we pick quantiles over the entire dataframe.
        * If there are A (100) < # of rows <= (A * (1 - k))/(q - k) (1900), we pick quantiles over the first A (100) rows, plus a sample
        of k (5%) of the remaining rows.
        * If there are > (A * (1 - k))/(q - k) (1900) rows, we pick quantiles over a sample of q (10%) of all of the rows.

    These numbers are a heuristic. They were picked such that the size of the sample
    in scenario 2 scales well. In other words, we picked A, k, and q such that:
        A + (len(df) - A)*k = q*len(df)
    Where q is the proportion we sample in scenario 3, k is the proportion we sample
    of remaining rows in scenario 2, and A is the threshold below which we just return
    the entire dataframe, instead of sampling (scenario 1).
    q = 0.1 and k = 0.05 were picked such that this formula holds for A = 100.

    Parameters
    ----------
    df : pandas.Dataframe
        The dataframe to pick quantiles from.
    columns : list[str]
        The columns to pick quantiles from. Only the first column in the list will be used.
    A : int, default: 100
        A heuristic that defines "small" dataframes.
    k : float, default: 0.05
        A heuristic used for sampling.
    q : float, default: 0.1
        A hueristic used for sampling.
    method : str, default: linear
        The method to use when picking quantiles.

    Returns
    -------
    pandas.Dataframe, np.ndarray:
        The partition that was provided as input, as well as the quantiles for the partition.

    Notes
    -----
    quantiles are only computed over the first column of the sort.
    """
    quantiles = [i / (NPartitions.get() * 2) for i in range(NPartitions.get() * 2)]
    # Heuristic for a "small" df we will compute quantiles over entirety of.
    if len(df) <= A:
        return df, np.quantile(df[columns[0]], quantiles, method=method)
    # Heuristic for a "medium" df where we will include first 100 (A) rows, and sample
    # of remaining rows when computing quantiles.
    if len(df) <= (A * (1 - k)) / (q - k):
        return df, np.quantile(
            np.concatenate(
                (
                    df[columns[0]][:100].values,
                    df[columns[0]][100:].sample(frac=k),
                )
            ),
            quantiles,
            method=method,
        )
    # Heuristic for a "large" df where we will sample 10% (q) of all rows to compute quantiles
    # over.
    return df, np.quantile(df[columns[0]].sample(frac=q), quantiles, method=method)


def pick_pivots_from_quantiles_for_sort(
    df: "PandasDataframe", samples: np.ndarray, columns: list, method: str = "linear"
) -> np.ndarray:
    """
    Determine quantiles from the given samples.

    This function takes as input the quantiles calculated over all partitions from
    `sample_func` defined above, and determines a final NPartitions.get() * 2 quantiles
    to use to roughly sort the entire dataframe. It does so by collating all the samples
    and computing NPartitions.get() * 2 quantiles for the overall set.

    Parameters
    ----------
    df : PandasDataframe
        The Modin Dataframe calling this function.
    samples : np.ndarray
        The samples computed by ``get_partition_quantiles_for_sort``.
    columns : list[str]
        The columns to sort by.
    method : str, default: linear
        The method to use when picking quantiles.

    Returns
    -------
    np.ndarray
        A list of overall quantiles.
    """
    # We don't call `np.unique` on the samples, since if a quantile shows up in multiple
    # partition's samples, this is probably an indicator of skew in the dataset, and we
    # want our final partitions to take this into account.
    all_pivots = np.array(samples).flatten()
    quantiles = [i / (NPartitions.get() * 2) for i in range(NPartitions.get() * 2)]
    overall_quantiles = np.quantile(all_pivots, quantiles, method=method)
    if df.dtypes[columns[0]] != object:
        overall_quantiles[0] = np.NINF
        overall_quantiles[-1] = np.inf
    return overall_quantiles


def split_partitions_using_pivots_for_sort(
    modin_frame: "PandasDataframe",
    df: pandas.DataFrame,
    columns: list,
    pivots: np.ndarray,
    ascending: Union[list, bool],
    **kwargs: dict
) -> "tuple[pandas.DataFrame]":
    """
    Split the given dataframe into the partitions specified by `pivots`.

    This function takes as input a row-axis partition, as well as the quantiles determined
    by the `pivot_func` defined above. It then splits the input dataframe into NPartitions.get() * 2
    dataframes, with the elements in the i-th split belonging to the i-th partition, as determined
    by the quantiles we're using.

    Parameters
    ----------
    modin_frame : PandasDataframe
        The Modin Dataframe calling this function.
    df : pandas.Dataframe
        The partition to split.
    columns : list[str]
        The columns to sort by.
    pivots : np.ndarray
        The quantiles to use to split the data.
    ascending : list[bool] or bool
        The ascending flag (or a list of ascending flags for each column).
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    list[pandas.DataFrame]
        A list of the splits from this partition.
    """
    if isinstance(ascending, list):
        ascending = ascending[0]
    if not ascending and modin_frame.dtypes[columns[0]] != object:
        pivots = pivots[::-1]
    na_rows = df[df[columns[0]].isna()]
    if modin_frame.dtypes[columns[0]] != object:
        groupby_col = np.digitize(df[columns[0]].squeeze(), pivots) - 1
    else:
        groupby_col = (
            np.searchsorted(pivots, df[columns[0]].squeeze(), side="right") - 1
        )
        if not ascending:
            groupby_col = (len(pivots) - 1) - groupby_col
    grouped = df.groupby(groupby_col)
    groups = [
        grouped.get_group(i)
        if i in grouped.keys
        else pandas.DataFrame(columns=df.columns)
        for i in range(len(pivots))
    ]
    index_to_insert_na_vals = -1 if kwargs.get("na_position", "last") == "last" else 0
    groups[index_to_insert_na_vals] = pandas.concat(
        [groups[index_to_insert_na_vals], na_rows]
    )
    return tuple(groups)
