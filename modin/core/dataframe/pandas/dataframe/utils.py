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
from typing import Callable, Union, TYPE_CHECKING, Optional

from modin.config import NPartitions

if TYPE_CHECKING:
    from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe


def build_sort_functions(
    modin_frame: "PandasDataframe",
    columns: list,
    method: str,
    ascending: Union[list, bool],
    **kwargs: dict,
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

    def terasort_sample_fn(partition):
        return pick_samples_for_quantiles(
            partition, columns, len(modin_frame._partitions), len(modin_frame.index)
        )

    # def original_sample_fn(partition, A=100, k=0.05, q=0.1):
    # key = kwargs.get("key", None)
    #     return get_partition_quantiles_for_sort(
    #         partition, columns, A=A, k=k, q=q, method=method, key
    #     )

    sample_fn = terasort_sample_fn

    def pivot_fn(samples):
        key = kwargs.get("key", None)
        return pick_pivots_from_quantiles_for_sort(
            modin_frame, samples, columns, method, key
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


def _find_quantiles(
    df: Union[pandas.DataFrame, pandas.Series], quantiles: list, method: str
) -> np.ndarray:
    """
    Find quantiles of a given dataframe using the specified method.

    We use this method to provide backwards compatibility with NumPy versions < 1.23 (e.g. when
    the user is using Modin in compat mode). This is basically a wrapper around `np.quantile` that
    ensures we provide the correct `method` argument - i.e. if we are dealing with objects (which
    may or may not support algebra), we do not want to use a method to find quantiles that will
    involve algebra operations (e.g. mean) between the objects, since that may fail.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        The data to pick quantiles from.
    quantiles : list[float]
        The quantiles to compute.
    method : str
        The method to use. `linear` if dealing with numeric types, otherwise `inverted_cdf`.

    Returns
    -------
    np.ndarray
        A NumPy array with the quantiles of the data.
    """
    if method == "linear":
        # This is the default method for finding quantiles, so it does not need to be specified,
        # which keeps backwards compatibility with older versions of NumPy that do not have a
        # `method` keyword argument in np.quantile.
        return np.quantile(df, quantiles)
    else:
        try:
            return np.quantile(df, quantiles, method=method)
        except Exception:
            # In this case, we're dealing with an array of objects, but the current version of
            # NumPy does not have a `method` kwarg. We need to use the older kwarg, `interpolation`
            # instead.
            return np.quantile(df, quantiles, interpolation="lower")


def pick_samples_for_quantiles(
    df: pandas.DataFrame,
    columns: list,
    num_partitions: int,
    length: int,
) -> np.ndarray:
    """
    Pick samples over the given partition.

    This function picks samples from the given partition using the TeraSort algorithm - each
    value is sampled with probability 1 / m * ln(n * t) where m = total_length / num_partitions,
    t = num_partitions, and n = total_length.

    Parameters
    ----------
    df : pandas.Dataframe
        The dataframe to pick samples from.
    columns : list[str]
        The columns to pick quantiles from. Only the first column in the list will be used.
    num_partitions : int
        The number of partitions.
    length : int
        The total length.

    Returns
    -------
    np.ndarray:
        The samples for the partition.

    Notes
    -----
    samples are only computed over the first column of the sort.
    """
    m = length / num_partitions
    probability = (1 / m) * np.log(num_partitions * length)
    return df[columns[0]].sample(frac=probability).to_numpy()


def get_partition_quantiles_for_sort(
    df: pandas.DataFrame,
    columns: list,
    A: int = 100,
    k: float = 0.05,
    q: float = 0.1,
    method: str = "linear",
    key: Optional[Callable] = None,
) -> np.ndarray:
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
    key : Callable, default: None
        The sort key to use.

    Returns
    -------
    np.ndarray:
        The quantiles for the partition.

    Notes
    -----
    quantiles are only computed over the first column of the sort.
    """
    quantiles = [i / (NPartitions.get()) for i in range(1, NPartitions.get())]
    # Heuristic for a "small" df we will compute quantiles over entirety of.
    if len(df) <= A:
        col_to_find_quantiles = df[columns[0]]
    # Heuristic for a "medium" df where we will include first 100 (A) rows, and sample
    # of remaining rows when computing quantiles.
    elif len(df) <= (A * (1 - k)) / (q - k):
        col_to_find_quantiles = np.concatenate(
            (
                df[columns[0]].iloc[:100].values,
                df[columns[0]].iloc[100:].sample(frac=k),
            )
        )
    # Heuristic for a "large" df where we will sample 10% (q) of all rows to compute quantiles
    # over.
    else:
        col_to_find_quantiles = df[columns[0]].sample(frac=q)
    if key is not None:
        col_to_find_quantiles = key(col_to_find_quantiles)
    return _find_quantiles(col_to_find_quantiles, quantiles, method)


def pick_pivots_from_quantiles_for_sort(
    df: "PandasDataframe",
    samples: np.ndarray,
    columns: list,
    method: str = "linear",
    key: Optional[Callable] = None,
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
    key : Callable, default: None
        The key to use on the samples when picking pivots.

    Returns
    -------
    np.ndarray
        A list of overall quantiles.
    """
    # We don't call `np.unique` on the samples, since if a quantile shows up in multiple
    # partition's samples, this is probably an indicator of skew in the dataset, and we
    # want our final partitions to take this into account.
    if isinstance(samples[0], np.ndarray):
        all_pivots = np.concatenate(samples).flatten()
    else:
        all_pivots = np.array(samples).flatten()
    if key is not None:
        all_pivots = key(all_pivots)
    # We don't want to pick very many quantiles if we have a very small dataframe.
    num_quantiles = len(df._partitions)
    quantiles = [i / num_quantiles for i in range(1, num_quantiles)]
    overall_quantiles = _find_quantiles(all_pivots, quantiles, method)
    return overall_quantiles


def split_partitions_using_pivots_for_sort(
    modin_frame: "PandasDataframe",
    df: pandas.DataFrame,
    columns: list,
    pivots: np.ndarray,
    ascending: Union[list, bool],
    **kwargs: dict,
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
    key = kwargs.pop("key", None)
    na_rows = df[df[columns[0]].isna()]
    non_na_rows = df[~df[columns[0]].isna()]
    cols_to_digitize = non_na_rows[columns[0]]
    if key is not None:
        cols_to_digitize = key(cols_to_digitize)
    if modin_frame.dtypes[columns[0]] != object:
        groupby_col = np.digitize(cols_to_digitize.squeeze(), pivots)
        if not ascending and len(np.unique(pivots)) == 1 and len(pivots) != 1:
            groupby_col = len(pivots) - groupby_col
    else:
        groupby_col = np.searchsorted(pivots, cols_to_digitize.squeeze(), side="right")
        if not ascending:
            groupby_col = len(pivots) - groupby_col
    grouped = non_na_rows.groupby(groupby_col)
    groups = [
        grouped.get_group(i)
        if i in grouped.keys
        else pandas.DataFrame(columns=df.columns).astype(df.dtypes)
        for i in range(len(pivots) + 1)
    ]
    index_to_insert_na_vals = -1 if kwargs.get("na_position", "last") == "last" else 0
    groups[index_to_insert_na_vals] = pandas.concat(
        [groups[index_to_insert_na_vals], na_rows]
    ).astype(df.dtypes)
    return tuple(groups)
