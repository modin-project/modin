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

"""Collection of algebra utility functions, used to shuffle data across partitions."""

import numpy as np
import pandas
from typing import Callable, Union, Optional, TYPE_CHECKING
from collections import namedtuple

if TYPE_CHECKING:
    from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe


ShuffleFunctions = namedtuple(
    "ShuffleFunctions", ["sample_function", "pivot_function", "split_function"]
)


def build_sort_functions(
    modin_frame: "PandasDataframe",
    column: str,
    method: str,
    ascending: Union[list, bool],
    ideal_num_new_partitions: int,
    **kwargs: dict,
) -> ShuffleFunctions:
    """
    Return a named tuple containing the functions necessary to perform a sort.

    Parameters
    ----------
    modin_frame : PandasDataframe
        The frame calling these sort functions.
    column : str
        The major column name to sort by.
    method : str
        The method to use for picking quantiles.
    ascending : bool
        The ascending flag.
    ideal_num_new_partitions : int
        The ideal number of new partitions.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    ShuffleFunctions :
        A named tuple containing the functions to pick quantiles, choose pivot points, and split
        partitions for sorting.
    """

    def sample_fn(partition):
        return pick_samples_for_quantiles(
            partition[column], ideal_num_new_partitions, len(modin_frame.index)
        )

    def pivot_fn(samples):
        key = kwargs.get("key", None)
        return pick_pivots_from_samples_for_sort(
            samples, ideal_num_new_partitions, method, key
        )

    def split_fn(partition, pivots):
        return split_partitions_using_pivots_for_sort(
            modin_frame, partition, column, pivots, ascending, **kwargs
        )

    return ShuffleFunctions(
        sample_function=sample_fn, pivot_function=pivot_fn, split_function=split_fn
    )


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
        The masked dataframe to pick samples from.
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
    This sampling algorithm is inspired by TeraSort. You can find more information about TeraSort
    and the sampling algorithm at https://www.cse.cuhk.edu.hk/~taoyf/paper/sigmod13-mr.pdf.
    """
    m = length / num_partitions
    probability = (1 / m) * np.log(num_partitions * length)
    return df.sample(frac=probability).to_numpy()


def pick_pivots_from_samples_for_sort(
    samples: "list[np.ndarray]",
    ideal_num_new_partitions: int,
    method: str = "linear",
    key: Optional[Callable] = None,
) -> np.ndarray:
    """
    Determine quantiles from the given samples.

    This function takes as input the quantiles calculated over all partitions from
    `sample_func` defined above, and determines a final NPartitions.get() quantiles
    to use to roughly sort the entire dataframe. It does so by collating all the samples
    and computing NPartitions.get() quantiles for the overall set.

    Parameters
    ----------
    samples : list[np.ndarray]
        The samples computed by ``get_partition_quantiles_for_sort``.
    ideal_num_new_partitions : int
        The ideal number of new partitions.
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
    # We need to use numpy to concatenate the samples since the sample from each partition is
    # a NumPy array, and we want one flattened array of samples.
    all_pivots = np.concatenate(samples).flatten()
    if key is not None:
        all_pivots = key(all_pivots)
    # We don't want to pick very many quantiles if we have a very small dataframe.
    num_quantiles = ideal_num_new_partitions
    quantiles = [i / num_quantiles for i in range(1, num_quantiles)]
    # If we only desire 1 partition, we need to ensure that we're not trying to find quantiles
    # from an empty list of pivots. We use `np.nan` as a dummy value.
    if len(quantiles) > 0:
        return _find_quantiles(all_pivots, quantiles, method)
    return np.array([np.nan])


def split_partitions_using_pivots_for_sort(
    modin_frame: "PandasDataframe",
    df: pandas.DataFrame,
    column: str,
    pivots: np.ndarray,
    ascending: bool,
    **kwargs: dict,
) -> "tuple[pandas.DataFrame, ...]":
    """
    Split the given dataframe into the partitions specified by `pivots`.

    This function takes as input a row-axis partition, as well as the quantiles determined
    by the `pivot_func` defined above. It then splits the input dataframe into NPartitions.get()
    dataframes, with the elements in the i-th split belonging to the i-th partition, as determined
    by the quantiles we're using.

    Parameters
    ----------
    modin_frame : PandasDataframe
        The Modin Dataframe calling this function.
    df : pandas.Dataframe
        The partition to split.
    column : str
        The major column to sort by.
    pivots : np.ndarray
        The quantiles to use to split the data.
    ascending : bool
        The ascending flag.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    tuple[pandas.DataFrame]
        A tuple of the splits from this partition.
    """
    # If `ascending=False` and we are dealing with a numeric dtype, we can pass in a reversed list
    # of pivots, and `np.digitize` will work correctly. For object dtypes, we use `np.searchsorted`
    # which breaks when we reverse the pivots.
    if not ascending and pandas.api.types.is_numeric_dtype(modin_frame.dtypes[column]):
        # `key` is already applied to `pivots` in the `pick_pivots_from_samples_for_sort` function.
        pivots = pivots[::-1]
    key = kwargs.pop("key", None)
    na_index = df[column].isna()
    na_rows = df[na_index]
    non_na_rows = df[~na_index]
    cols_to_digitize = non_na_rows[column]
    if key is not None:
        cols_to_digitize = key(cols_to_digitize)
    if pandas.api.types.is_numeric_dtype(modin_frame.dtypes[column]):
        groupby_col = np.digitize(cols_to_digitize.squeeze(), pivots)
        # `np.digitize` returns results based off of the sort order of the pivots it is passed.
        # When we only have one unique value in our pivots, `np.digitize` assumes that the pivots
        # are sorted in ascending order, and gives us results based off of that assumption - so if
        # we actually want to sort in descending order, we need to swap the new indices.
        if not ascending and len(np.unique(pivots)) == 1 and len(pivots) != 1:
            groupby_col = len(pivots) - groupby_col
    else:
        groupby_col = np.searchsorted(pivots, cols_to_digitize.squeeze(), side="right")
        # Since np.searchsorted requires the pivots to be in ascending order, if we want to sort
        # in descending order, we need to swap the new indices.
        if not ascending:
            groupby_col = len(pivots) - groupby_col
    if len(non_na_rows) == 1:
        groups = [
            pandas.DataFrame(columns=df.columns).astype(df.dtypes)
            if i != groupby_col
            else non_na_rows
            for i in range(len(pivots) + 1)
        ]
    else:
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
