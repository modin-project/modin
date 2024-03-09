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

import abc
from collections import namedtuple
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype

from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings

if TYPE_CHECKING:
    from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe

ColumnInfo = namedtuple("ColumnInfo", ["name", "pivots", "is_numeric"])


class ShuffleFunctions:
    """
    Defines an interface to perform the sampling, quantiles picking, and the splitting stages for the range-partitioning building.

    Parameters
    ----------
    modin_frame : PandasDataframe
        The frame to build the range-partitioning for.
    columns : str or list of strings
        The column/columns to use as a key.
    ascending : bool
        Whether the ranges should be in ascending or descending order.
    ideal_num_new_partitions : int
        The ideal number of new partitions.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self, modin_frame, columns, ascending, ideal_num_new_partitions, **kwargs
    ):
        pass

    @abc.abstractmethod
    def sample_fn(self, partition: pandas.DataFrame) -> pandas.DataFrame:
        """
        Pick samples over the given partition.

        Parameters
        ----------
        partition : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame:
            The samples for the partition.
        """
        pass

    @abc.abstractmethod
    def pivot_fn(self, samples: "list[pandas.DataFrame]") -> int:
        """
        Determine quantiles from the given samples and save it for the future ``.split_fn()`` calls.

        Parameters
        ----------
        samples : list of pandas.DataFrames

        Returns
        -------
        int
            The number of bins the ``.split_fn()`` will return.
        """
        pass

    @abc.abstractmethod
    def split_fn(self, partition: pandas.DataFrame) -> "tuple[pandas.DataFrame, ...]":
        """
        Split the given dataframe into the range-partitions defined by the preceding call of the ``.pivot_fn()``.

        Parameters
        ----------
        partition : pandas.DataFrame

        Returns
        -------
        tuple of pandas.DataFrames

        Notes
        -----
        In order to call this method you must call the ``.pivot_fn()`` first.
        """
        pass


@_inherit_docstrings(ShuffleFunctions)
class ShuffleSortFunctions(ShuffleFunctions):
    """
    Perform the sampling, quantiles picking, and the splitting stages for the range-partitioning building.

    Parameters
    ----------
    modin_frame : PandasDataframe
        The frame to build the range-partitioning for.
    columns : str or list of strings
        The column/columns to use as a key.
    ascending : bool
        Whether the ranges should be in ascending or descending order.
    ideal_num_new_partitions : int
        The ideal number of new partitions.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        modin_frame: "PandasDataframe",
        columns: Union[str, list],
        ascending: Union[list, bool],
        ideal_num_new_partitions: int,
        **kwargs: dict,
    ):
        self.frame_len = len(modin_frame)
        self.ideal_num_new_partitions = ideal_num_new_partitions
        self.columns = columns if is_list_like(columns) else [columns]
        self.ascending = ascending
        self.kwargs = kwargs.copy()
        self.columns_info = None

    def sample_fn(self, partition: pandas.DataFrame) -> pandas.DataFrame:
        return self.pick_samples_for_quantiles(
            partition[self.columns], self.ideal_num_new_partitions, self.frame_len
        )

    def pivot_fn(self, samples: "list[pandas.DataFrame]") -> int:
        key = self.kwargs.get("key", None)
        samples = pandas.concat(samples, axis=0, copy=False)

        columns_info: "list[ColumnInfo]" = []
        number_of_groups = 1
        cols = []
        for col in samples.columns:
            num_pivots = int(self.ideal_num_new_partitions / number_of_groups)
            if num_pivots < 2 and len(columns_info):
                break
            column_val = samples[col].to_numpy()
            cols.append(col)
            is_numeric = is_numeric_dtype(column_val.dtype)

            # When we are not sorting numbers, we need our quantiles to not do arithmetic on the values
            method = "linear" if is_numeric else "inverted_cdf"
            pivots = self.pick_pivots_from_samples_for_sort(
                column_val, num_pivots, method, key
            )
            columns_info.append(ColumnInfo(col, pivots, is_numeric))
            number_of_groups *= len(pivots) + 1
        self.columns_info = columns_info
        return number_of_groups

    def split_fn(
        self,
        partition: pandas.DataFrame,
    ) -> "tuple[pandas.DataFrame, ...]":
        ErrorMessage.catch_bugs_and_request_email(
            failure_condition=self.columns_info is None,
            extra_log="The 'split_fn' doesn't have proper metadata, the probable reason is that it was called before 'pivot_fn'",
        )
        return self.split_partitions_using_pivots_for_sort(
            partition,
            self.columns_info,
            self.ascending,
            **self.kwargs,
        )

    @staticmethod
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
            return np.unique(np.quantile(df, quantiles))
        else:
            try:
                return np.unique(np.quantile(df, quantiles, method=method))
            except Exception:
                # In this case, we're dealing with an array of objects, but the current version of
                # NumPy does not have a `method` kwarg. We need to use the older kwarg, `interpolation`
                # instead.
                return np.unique(np.quantile(df, quantiles, interpolation="lower"))

    @staticmethod
    def pick_samples_for_quantiles(
        df: pandas.DataFrame,
        num_partitions: int,
        length: int,
    ) -> pandas.DataFrame:
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
        pandas.DataFrame:
            The samples for the partition.

        Notes
        -----
        This sampling algorithm is inspired by TeraSort. You can find more information about TeraSort
        and the sampling algorithm at https://www.cse.cuhk.edu.hk/~taoyf/paper/sigmod13-mr.pdf.
        """
        m = length / num_partitions
        probability = (1 / m) * np.log(num_partitions * length)
        return df.sample(frac=probability)

    @classmethod
    def pick_pivots_from_samples_for_sort(
        cls,
        samples: np.ndarray,
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
        samples : np.ndarray
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
        if key is not None:
            samples = key(samples)
        # We don't want to pick very many quantiles if we have a very small dataframe.
        num_quantiles = ideal_num_new_partitions
        quantiles = [i / num_quantiles for i in range(1, num_quantiles)]
        # If we only desire 1 partition, we need to ensure that we're not trying to find quantiles
        # from an empty list of pivots.
        if len(quantiles) > 0:
            return cls._find_quantiles(samples, quantiles, method)
        return np.array([])

    @staticmethod
    def split_partitions_using_pivots_for_sort(
        df: pandas.DataFrame,
        columns_info: "list[ColumnInfo]",
        ascending: bool,
        **kwargs: dict,
    ) -> "tuple[pandas.DataFrame, ...]":
        """
        Split the given dataframe into the partitions specified by `pivots` in `columns_info`.

        This function takes as input a row-axis partition, as well as the quantiles determined
        by the `pivot_func` defined above. It then splits the input dataframe into NPartitions.get()
        dataframes, with the elements in the i-th split belonging to the i-th partition, as determined
        by the quantiles we're using.

        Parameters
        ----------
        df : pandas.Dataframe
            The partition to split.
        columns_info : list of ColumnInfo
            Information regarding keys and pivots for range partitioning.
        ascending : bool
            The ascending flag.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[pandas.DataFrame]
            A tuple of the splits from this partition.
        """
        if len(columns_info) == 0:
            # We can return the dataframe with zero changes if there were no pivots passed
            return (df,)

        na_index = (
            df[[col_info.name for col_info in columns_info]].isna().squeeze(axis=1)
        )
        if na_index.ndim == 2:
            na_index = na_index.any(axis=1)
        na_rows = df[na_index]
        non_na_rows = df[~na_index]

        def get_group(grp, key, df):
            """Get a group with the `key` from the `grp`, if it doesn't exist return an empty slice of `df`."""
            try:
                return grp.get_group(key)
            except KeyError:
                return pandas.DataFrame(index=df.index[:0], columns=df.columns).astype(
                    df.dtypes
                )

        groupby_codes = []
        group_keys = []
        for col_info in columns_info:
            pivots = col_info.pivots
            if len(pivots) == 0:
                continue
            # If `ascending=False` and we are dealing with a numeric dtype, we can pass in a reversed list
            # of pivots, and `np.digitize` will work correctly. For object dtypes, we use `np.searchsorted`
            # which breaks when we reverse the pivots.
            if not ascending and col_info.is_numeric:
                # `key` is already applied to `pivots` in the `pick_pivots_from_samples_for_sort` function.
                pivots = pivots[::-1]
            group_keys.append(range(len(pivots) + 1))
            key = kwargs.pop("key", None)
            cols_to_digitize = non_na_rows[col_info.name]
            if key is not None:
                cols_to_digitize = key(cols_to_digitize)

            if col_info.is_numeric:
                groupby_col = np.digitize(cols_to_digitize.squeeze(), pivots)
                # `np.digitize` returns results based off of the sort order of the pivots it is passed.
                # When we only have one unique value in our pivots, `np.digitize` assumes that the pivots
                # are sorted in ascending order, and gives us results based off of that assumption - so if
                # we actually want to sort in descending order, we need to swap the new indices.
                if not ascending and len(np.unique(pivots)) == 1:
                    groupby_col = len(pivots) - groupby_col
            else:
                groupby_col = np.searchsorted(
                    pivots, cols_to_digitize.squeeze(), side="right"
                )
                # Since np.searchsorted requires the pivots to be in ascending order, if we want to sort
                # in descending order, we need to swap the new indices.
                if not ascending:
                    groupby_col = len(pivots) - groupby_col
            groupby_codes.append(groupby_col)

        if len(group_keys) == 0:
            # We can return the dataframe with zero changes if there were no pivots passed
            return (df,)
        elif len(group_keys) == 1:
            group_keys = group_keys[0]
        else:
            group_keys = pandas.MultiIndex.from_product(group_keys)

        if len(non_na_rows) == 1:
            groups = [
                # taking an empty slice for an index's metadata
                (
                    pandas.DataFrame(index=df.index[:0], columns=df.columns).astype(
                        df.dtypes
                    )
                    if key != groupby_codes[0]
                    else non_na_rows
                )
                for key in group_keys
            ]
        else:
            grouped = non_na_rows.groupby(groupby_codes)
            groups = [get_group(grouped, key, df) for key in group_keys]
        index_to_insert_na_vals = (
            -1 if kwargs.get("na_position", "last") == "last" else 0
        )
        groups[index_to_insert_na_vals] = pandas.concat(
            [groups[index_to_insert_na_vals], na_rows]
        ).astype(df.dtypes)
        return tuple(groups)


def lazy_metadata_decorator(apply_axis=None, axis_arg=-1, transpose=False):
    """
    Lazily propagate metadata for the ``PandasDataframe``.

    This decorator first adds the minimum required reindexing operations
    to each partition's queue of functions to be lazily applied for
    each PandasDataframe in the arguments by applying the function
    run_f_on_minimally_updated_metadata. The decorator also sets the
    flags for deferred metadata synchronization on the function result
    if necessary.

    Parameters
    ----------
    apply_axis : str, default: None
        The axes on which to apply the reindexing operations to the `self._partitions` lazily.
        Case None: No lazy metadata propagation.
        Case "both": Add reindexing operations on both axes to partition queue.
        Case "opposite": Add reindexing operations complementary to given axis.
        Case "rows": Add reindexing operations on row axis to partition queue.
    axis_arg : int, default: -1
        The index or column axis.
    transpose : bool, default: False
        Boolean for if a transpose operation is being used.

    Returns
    -------
    Wrapped Function.
    """

    def decorator(f):
        from functools import wraps

        @wraps(f)
        def run_f_on_minimally_updated_metadata(self, *args, **kwargs):
            from .dataframe import PandasDataframe

            for obj in (
                [self]
                + [o for o in args if isinstance(o, PandasDataframe)]
                + [v for v in kwargs.values() if isinstance(v, PandasDataframe)]
                + [
                    d
                    for o in args
                    if isinstance(o, list)
                    for d in o
                    if isinstance(d, PandasDataframe)
                ]
                + [
                    d
                    for _, o in kwargs.items()
                    if isinstance(o, list)
                    for d in o
                    if isinstance(d, PandasDataframe)
                ]
            ):
                if apply_axis == "both":
                    if obj._deferred_index and obj._deferred_column:
                        obj._propagate_index_objs(axis=None)
                    elif obj._deferred_index:
                        obj._propagate_index_objs(axis=0)
                    elif obj._deferred_column:
                        obj._propagate_index_objs(axis=1)
                elif apply_axis == "opposite":
                    if "axis" not in kwargs:
                        axis = args[axis_arg]
                    else:
                        axis = kwargs["axis"]
                    if axis == 0 and obj._deferred_column:
                        obj._propagate_index_objs(axis=1)
                    elif axis == 1 and obj._deferred_index:
                        obj._propagate_index_objs(axis=0)
                elif apply_axis == "rows":
                    obj._propagate_index_objs(axis=0)
            result = f(self, *args, **kwargs)
            if apply_axis is None and not transpose:
                result._deferred_index = self._deferred_index
                result._deferred_column = self._deferred_column
            elif apply_axis is None and transpose:
                result._deferred_index = self._deferred_column
                result._deferred_column = self._deferred_index
            elif apply_axis == "opposite":
                if axis == 0:
                    result._deferred_index = self._deferred_index
                else:
                    result._deferred_column = self._deferred_column
            elif apply_axis == "rows":
                result._deferred_column = self._deferred_column
            return result

        return run_f_on_minimally_updated_metadata

    return decorator


def add_missing_categories_to_groupby(
    dfs,
    by,
    operator,
    initial_columns,
    combined_cols,
    is_udf_agg,
    kwargs,
    initial_dtypes=None,
):
    """
    Generate values for missing categorical values to be inserted into groupby result.

    This function is used to emulate behavior of ``groupby(observed=False)`` parameter,
    it takes groupby result that was computed using ``groupby(observed=True)``
    and computes results for categorical values that are not presented in `dfs`.

    Parameters
    ----------
    dfs : list of pandas.DataFrames
        Row partitions containing groupby results.
    by : list of hashable
        Column labels that were used to perform groupby.
    operator : callable
        Aggregation function that was used during groupby.
    initial_columns : pandas.Index
        Column labels of the original dataframe.
    combined_cols : pandas.Index
        Column labels of the groupby result.
    is_udf_agg : bool
        Whether ``operator`` is a UDF.
    kwargs : dict
        Parameters that were passed to ``groupby(by, **kwargs)``.
    initial_dtypes : pandas.Series, optional
        Dtypes of the original dataframe. If not specified, assume it's ``int64``.

    Returns
    -------
    masks : dict[int, pandas.DataFrame]
        Mapping between partition idx and a dataframe with results for missing categorical values
        to insert to this partition.
    new_combined_cols : pandas.Index
        New column labels of the groupby result. If ``is_udf_agg is True``, then ``operator``
        may change the resulted columns.
    """
    kwargs["observed"] = False
    new_combined_cols = combined_cols

    ### At first we need to compute missing categorical values
    indices = [df.index for df in dfs]
    # total_index contains all categorical values that resided in the result,
    # missing values are computed differently depending on whether we're grouping
    # on multiple groupers or not
    total_index = indices[0].append(indices[1:])
    if isinstance(total_index, pandas.MultiIndex):
        if all(
            not isinstance(level, pandas.CategoricalIndex)
            for level in total_index.levels
        ):
            return {}, new_combined_cols
        missing_cats_dtype = {
            name: (
                level.dtype
                if isinstance(level.dtype, pandas.CategoricalDtype)
                # it's a bit confusing but we have to convert the remaining 'by' columns to categoricals
                # in order to compute a proper fill value later in the code
                else pandas.CategoricalDtype(level)
            )
            for level, name in zip(total_index.levels, total_index.names)
        }
        # if we're grouping on multiple groupers, then the missing categorical values is a
        # carthesian product of (actual_missing_categorical_values X all_values_of_another_groupers)
        complete_index = pandas.MultiIndex.from_product(
            [
                value.categories.astype(total_level.dtype)
                for total_level, value in zip(
                    total_index.levels, missing_cats_dtype.values()
                )
            ],
            names=by,
        )
        missing_index = complete_index[~complete_index.isin(total_index)]
    else:
        if not isinstance(total_index, pandas.CategoricalIndex):
            return {}, new_combined_cols
        # if we're grouping on a single grouper then we simply compute the difference
        # between categorical values in the result and the values defined in categorical dtype
        missing_index = total_index.categories.difference(total_index.values)
        missing_cats_dtype = {by[0]: pandas.CategoricalDtype(missing_index)}
    missing_index.names = by

    if len(missing_index) == 0:
        return {}, new_combined_cols

    ### At this stage we want to get a fill_value for missing categorical values
    if is_udf_agg and isinstance(total_index, pandas.MultiIndex):
        # if grouping on multiple columns and aggregating with an UDF, then the
        # fill value is always `np.NaN`
        missing_values = pandas.DataFrame({0: [np.NaN]})
    else:
        # In case of a udf aggregation we're forced to run the operator against each
        # missing category, as in theory it can return different results for each
        # empty group. In other cases it's enough to run the operator against a single
        # missing categorical and then broadcast the fill value to each missing value
        if not is_udf_agg:
            missing_cats_dtype = {
                key: pandas.CategoricalDtype(value.categories[:1])
                for key, value in missing_cats_dtype.items()
            }

        empty_df = pandas.DataFrame(columns=initial_columns)
        # HACK: default 'object' dtype doesn't fit our needs, as most of the aggregations
        # fail on a non-numeric columns, ideally, we need dtypes of the original dataframe,
        # however, 'int64' also works fine here if the original schema is not available
        empty_df = empty_df.astype(
            "int64" if initial_dtypes is None else initial_dtypes
        )
        empty_df = empty_df.astype(missing_cats_dtype)
        missing_values = operator(empty_df.groupby(by, **kwargs))

    if is_udf_agg and not isinstance(total_index, pandas.MultiIndex):
        missing_values = missing_values.drop(columns=by, errors="ignore")
        new_combined_cols = pandas.concat(
            [
                pandas.DataFrame(columns=combined_cols),
                missing_values.iloc[:0],
            ],
            axis=0,
            join="outer",
        ).columns
    else:
        # HACK: If the aggregation has failed, the result would be empty. Assuming the
        # fill value to be `np.NaN` here (this may not always be correct!!!)
        fill_value = np.NaN if len(missing_values) == 0 else missing_values.iloc[0, 0]
        missing_values = pandas.DataFrame(
            fill_value, index=missing_index, columns=combined_cols
        )

    # restoring original categorical dtypes for the indices (MultiIndex already have proper dtypes)
    if not isinstance(missing_values.index, pandas.MultiIndex):
        missing_values.index = missing_values.index.astype(total_index.dtype)

    ### Then we decide to which missing categorical values should go to which partition
    if not kwargs["sort"]:
        # If the result is allowed to be unsorted, simply insert all the missing
        # categories to the last partition
        mask = {len(indices) - 1: missing_values}
        return mask, new_combined_cols

    # If the result has to be sorted, we have to assign missing categoricals to proper partitions.
    # For that purpose we define bins with corner values of each partition and then using either
    # np.digitize or np.searchsorted find correct bins for each missing categorical value.
    # Example: part0-> [0, 1, 2]; part1-> [3, 4, 10, 12]; part2-> [15, 17, 20, 100]
    #          bins -> [2, 12] # took last values of each partition excluding the last partition
    #                            (every value that's matching 'x > part[-2][-1]' should go to the
    #                             last partition, meaning that including the last value of the last
    #                             partitions doesn't make sense)
    #          missing_cats ->                    [-2, 5, 6, 14, 21, 120]
    #          np.digitize(missing_cats, bins) -> [ 0, 1, 1,  2,  2,  2]
    #                                               ^-- mapping between values and partition idx to insert
    bins = []
    old_bins_to_new = {}
    offset = 0
    # building bins by taking last values of each partition excluding the last partition
    for idx in indices[:-1]:
        if len(idx) == 0:
            # if a partition is empty, we can't use its values to define a bin, thus we simply
            # skip it and remember the number of skipped partitions as an 'offset'
            offset += 1
            continue
        # remember the number of skipped partitions before this bin, in order to restore original
        # indexing at the end
        old_bins_to_new[len(bins)] = offset
        # for MultiIndices we always use the very first level for bins as using multiple levels
        # doesn't affect the result
        bins.append(idx[-1][0] if isinstance(idx, pandas.MultiIndex) else idx[-1])
    old_bins_to_new[len(bins)] = offset

    if len(bins) == 0:
        # insert values to the first non-empty partition
        return {old_bins_to_new.get(0, 0): missing_values}, new_combined_cols

    # we used the very first level of MultiIndex to build bins, meaning that we also have
    # to use values of the first index's level for 'digitize'
    lvl_zero = (
        missing_values.index.levels[0]
        if isinstance(missing_values.index, pandas.MultiIndex)
        else missing_values.index
    )
    if pandas.api.types.is_any_real_numeric_dtype(lvl_zero):
        part_idx = np.digitize(lvl_zero, bins, right=True)
    else:
        part_idx = np.searchsorted(bins, lvl_zero)

    ### In the end we build a dictionary mapping partition index to a dataframe with missing categoricals
    ### to be inserted into this partition
    masks = {}
    if isinstance(total_index, pandas.MultiIndex):
        for idx, values in pandas.RangeIndex(len(lvl_zero)).groupby(part_idx).items():
            masks[idx] = missing_values[
                pandas.Index(missing_values.index.codes[0]).isin(values)
            ]
    else:
        frame_idx = missing_values.index.to_frame()
        for idx, values in lvl_zero.groupby(part_idx).items():
            masks[idx] = missing_values[frame_idx.iloc[:, 0].isin(values)]

    # Restore the original indexing by adding the amount of skipped missing partitions
    masks = {key + old_bins_to_new[key]: value for key, value in masks.items()}
    return masks, new_combined_cols
