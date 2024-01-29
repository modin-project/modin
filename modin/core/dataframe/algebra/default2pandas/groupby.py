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

"""Module houses default GroupBy functions builder class."""

import warnings
from typing import Any

import pandas
from pandas.core.dtypes.common import is_list_like

# Defines a set of string names of functions that are executed in a transform-way in groupby
from pandas.core.groupby.base import transformation_kernels

from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable

from .default import DefaultMethod


# FIXME: there is no sence of keeping `GroupBy` and `GroupByDefault` logic in a different
# classes. They should be combined.
class GroupBy:
    """Builder for GroupBy aggregation functions."""

    agg_aliases = [
        "agg",
        "dict_agg",
        pandas.core.groupby.DataFrameGroupBy.agg,
        pandas.core.groupby.DataFrameGroupBy.aggregate,
    ]

    @staticmethod
    def is_transformation_kernel(agg_func: Any) -> bool:
        """
        Check whether a passed aggregation function is a transformation.

        Transformation means that the result of the function will be broadcasted
        to the frame's original shape.

        Parameters
        ----------
        agg_func : Any

        Returns
        -------
        bool
        """
        return hashable(agg_func) and agg_func in transformation_kernels.union(
            # these methods are also producing transpose-like result in a sense we understand it
            # (they're non-aggregative functions), however are missing in the pandas dictionary
            {"nth", "head", "tail"}
        )

    @classmethod
    def _call_groupby(cls, df, *args, **kwargs):  # noqa: PR01
        """Call .groupby() on passed `df`."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return df.groupby(*args, **kwargs)

    @classmethod
    def validate_by(cls, by):
        """
        Build valid `by` parameter for `pandas.DataFrame.groupby`.

        Cast all DataFrames in `by` parameter to Series or list of Series in case
        of multi-column frame.

        Parameters
        ----------
        by : DateFrame, Series, index label or list of such
            Object which indicates groups for GroupBy.

        Returns
        -------
        Series, index label or list of such
            By parameter with all DataFrames casted to Series.
        """

        def try_cast_series(df):
            """Cast one-column frame to Series."""
            if isinstance(df, pandas.DataFrame):
                df = df.squeeze(axis=1)
            if not isinstance(df, pandas.Series):
                return df
            if df.name == MODIN_UNNAMED_SERIES_LABEL:
                df.name = None
            return df

        if isinstance(by, pandas.DataFrame):
            by = [try_cast_series(column) for _, column in by.items()]
        elif isinstance(by, pandas.Series):
            by = [try_cast_series(by)]
        elif isinstance(by, list):
            by = [try_cast_series(o) for o in by]
        return by

    @classmethod
    def inplace_applyier_builder(cls, key, func=None):
        """
        Bind actual aggregation function to the GroupBy aggregation method.

        Parameters
        ----------
        key : callable
            Function that takes GroupBy object and evaluates passed aggregation function.
        func : callable or str, optional
            Function that takes DataFrame and aggregate its data. Will be applied
            to each group at the grouped frame.

        Returns
        -------
        callable,
            Function that executes aggregation under GroupBy object.
        """
        inplace_args = [] if func is None else [func]

        def inplace_applyier(grp, *func_args, **func_kwargs):
            return key(grp, *inplace_args, *func_args, **func_kwargs)

        return inplace_applyier

    @classmethod
    def get_func(cls, key, **kwargs):
        """
        Extract aggregation function from groupby arguments.

        Parameters
        ----------
        key : callable or str
            Default aggregation function. If aggregation function is not specified
            via groupby arguments, then `key` function is used.
        **kwargs : dict
            GroupBy arguments that may contain aggregation function.

        Returns
        -------
        callable
            Aggregation function.

        Notes
        -----
        There are two ways of how groupby aggregation can be invoked:
            1. Explicitly with query compiler method: `qc.groupby_sum()`.
            2. By passing aggregation function as an argument: `qc.groupby_agg("sum")`.
        Both are going to produce the same result, however in the first case actual aggregation
        function can be extracted from the method name, while for the second only from the method arguments.
        """
        if "agg_func" in kwargs:
            return cls.inplace_applyier_builder(key, kwargs["agg_func"])
        elif "func_dict" in kwargs:
            return cls.inplace_applyier_builder(key, kwargs["func_dict"])
        else:
            return cls.inplace_applyier_builder(key)

    @classmethod
    def build_aggregate_method(cls, key):
        """
        Build function for `QueryCompiler.groupby_agg` that can be executed as default-to-pandas.

        Parameters
        ----------
        key : callable or str
            Default aggregation function. If aggregation function is not specified
            via groupby arguments, then `key` function is used.

        Returns
        -------
        callable
            Function that executes groupby aggregation.
        """

        def fn(
            df,
            by,
            axis,
            groupby_kwargs,
            agg_args,
            agg_kwargs,
            drop=False,
            **kwargs,
        ):
            """Group DataFrame and apply aggregation function to each group."""
            by = cls.validate_by(by)

            grp = cls._call_groupby(df, by, axis=axis, **groupby_kwargs)
            agg_func = cls.get_func(key, **kwargs)
            result = agg_func(grp, *agg_args, **agg_kwargs)

            return result

        return fn

    @classmethod
    def build_groupby_reduce_method(cls, agg_func):
        """
        Build function for `QueryCompiler.groupby_*` that can be executed as default-to-pandas.

        Parameters
        ----------
        agg_func : callable or str
            Default aggregation function. If aggregation function is not specified
            via groupby arguments, then `agg_func` function is used.

        Returns
        -------
        callable
            Function that executes groupby aggregation.
        """

        def fn(
            df, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False, **kwargs
        ):
            """Group DataFrame and apply aggregation function to each group."""
            if not isinstance(by, (pandas.Series, pandas.DataFrame)):
                by = cls.validate_by(by)
                grp = cls._call_groupby(df, by, axis=axis, **groupby_kwargs)
                grp_agg_func = cls.get_func(agg_func, **kwargs)
                return grp_agg_func(
                    grp,
                    *agg_args,
                    **agg_kwargs,
                )

            if isinstance(by, pandas.DataFrame):
                by = by.squeeze(axis=1)
            if (
                drop
                and isinstance(by, pandas.Series)
                and by.name in df
                and df[by.name].equals(by)
            ):
                by = [by.name]
            if isinstance(by, pandas.DataFrame):
                df = pandas.concat([df] + [by[[o for o in by if o not in df]]], axis=1)
                by = list(by.columns)

            groupby_kwargs = groupby_kwargs.copy()
            as_index = groupby_kwargs.pop("as_index", True)
            groupby_kwargs["as_index"] = True

            grp = cls._call_groupby(df, by, axis=axis, **groupby_kwargs)
            func = cls.get_func(agg_func, **kwargs)
            result = func(grp, *agg_args, **agg_kwargs)
            method = kwargs.get("method")

            if isinstance(result, pandas.Series):
                result = result.to_frame(
                    MODIN_UNNAMED_SERIES_LABEL if result.name is None else result.name
                )

            if not as_index:
                if isinstance(by, pandas.Series):
                    # 1. If `drop` is True then 'by' Series represents a column from the
                    #    source frame and so the 'by' is internal.
                    # 2. If method is 'size' then any 'by' is considered to be internal.
                    #    This is a hacky legacy from the ``groupby_size`` implementation:
                    #    https://github.com/modin-project/modin/issues/3739
                    internal_by = (by.name,) if drop or method == "size" else tuple()
                else:
                    internal_by = by

                cls.handle_as_index_for_dataframe(
                    result,
                    internal_by,
                    by_cols_dtypes=(
                        df.index.dtypes.values
                        if isinstance(df.index, pandas.MultiIndex)
                        else (df.index.dtype,)
                    ),
                    by_length=len(by),
                    drop=drop,
                    method=method,
                    inplace=True,
                )

            if result.index.name == MODIN_UNNAMED_SERIES_LABEL:
                result.index.name = None

            return result

        return fn

    @classmethod
    def is_aggregate(cls, key):  # noqa: PR01
        """Check whether `key` is an alias for pandas.GroupBy.aggregation method."""
        return key in cls.agg_aliases

    @classmethod
    def build_groupby(cls, func):
        """
        Build function that groups DataFrame and applies aggregation function to the every group.

        Parameters
        ----------
        func : callable or str
            Default aggregation function. If aggregation function is not specified
            via groupby arguments, then `func` function is used.

        Returns
        -------
        callable
            Function that takes pandas DataFrame and does GroupBy aggregation.
        """
        if cls.is_aggregate(func):
            return cls.build_aggregate_method(func)
        return cls.build_groupby_reduce_method(func)

    @classmethod
    def handle_as_index_for_dataframe(
        cls,
        result,
        internal_by_cols,
        by_cols_dtypes=None,
        by_length=None,
        selection=None,
        partition_idx=0,
        drop=True,
        method=None,
        inplace=False,
    ):
        """
        Handle `as_index=False` parameter for the passed GroupBy aggregation result.

        Parameters
        ----------
        result : DataFrame
            Frame containing GroupBy aggregation result computed with `as_index=True`
            parameter (group names are located at the frame's index).
        internal_by_cols : list-like
            Internal 'by' columns.
        by_cols_dtypes : list-like, optional
            Data types of the internal 'by' columns. Required to do special casing
            in case of categorical 'by'. If not specified, assume that there is no
            categorical data in 'by'.
        by_length : int, optional
            Amount of keys to group on (including frame columns and external objects like list, Series, etc.)
            If not specified, consider `by_length` to be equal ``len(internal_by_cols)``.
        selection : label or list of labels, optional
            Set of columns that were explicitly selected for aggregation (for example
            via dict-aggregation). If not specified assuming that aggregation was
            applied to all of the available columns.
        partition_idx : int, default: 0
            Positional index of the current partition.
        drop : bool, default: True
            Indicates whether or not any of the `by` data came from the same frame.
        method : str, optional
            Name of the groupby function. This is a hint to be able to do special casing.
            Note: this parameter is a legacy from the ``groupby_size`` implementation,
            it's a hacky one and probably will be removed in the future: https://github.com/modin-project/modin/issues/3739.
        inplace : bool, default: False
            Modify the DataFrame in place (do not create a new object).

        Returns
        -------
        DataFrame
            GroupBy aggregation result with the considered `as_index=False` parameter.
        """
        if not inplace:
            result = result.copy()

        reset_index, drop, lvls_to_drop, cols_to_drop = cls.handle_as_index(
            result_cols=result.columns,
            result_index_names=result.index.names,
            internal_by_cols=internal_by_cols,
            by_cols_dtypes=by_cols_dtypes,
            by_length=by_length,
            selection=selection,
            partition_idx=partition_idx,
            drop=drop,
            method=method,
        )

        if len(lvls_to_drop) > 0:
            result.index = result.index.droplevel(lvls_to_drop)
        if len(cols_to_drop) > 0:
            result.drop(columns=cols_to_drop, inplace=True)
        if reset_index:
            result.reset_index(drop=drop, inplace=True)
        return result

    @staticmethod
    def handle_as_index(
        result_cols,
        result_index_names,
        internal_by_cols,
        by_cols_dtypes=None,
        by_length=None,
        selection=None,
        partition_idx=0,
        drop=True,
        method=None,
    ):
        """
        Compute hints to process ``as_index=False`` parameter for the GroupBy result.

        This function resolves naming conflicts of the index levels to insert and the column labels
        for the GroupBy result. The logic of this function assumes that the initial GroupBy result
        was computed as ``as_index=True``.

        Parameters
        ----------
        result_cols : pandas.Index
            Columns of the GroupBy result.
        result_index_names : list-like
            Index names of the GroupBy result.
        internal_by_cols : list-like
            Internal 'by' columns.
        by_cols_dtypes : list-like, optional
            Data types of the internal 'by' columns. Required to do special casing
            in case of categorical 'by'. If not specified, assume that there is no
            categorical data in 'by'.
        by_length : int, optional
            Amount of keys to group on (including frame columns and external objects like list, Series, etc.)
            If not specified, consider `by_length` to be equal ``len(internal_by_cols)``.
        selection : label or list of labels, optional
            Set of columns that were explicitly selected for aggregation (for example
            via dict-aggregation). If not specified assuming that aggregation was
            applied to all of the available columns.
        partition_idx : int, default: 0
            Positional index of the current partition.
        drop : bool, default: True
            Indicates whether or not any of the `by` data came from the same frame.
        method : str, optional
            Name of the groupby function. This is a hint to be able to do special casing.
            Note: this parameter is a legacy from the ``groupby_size`` implementation,
            it's a hacky one and probably will be removed in the future: https://github.com/modin-project/modin/issues/3739.

        Returns
        -------
        reset_index : bool
            Indicates whether to reset index to the default one (0, 1, 2 ... n) at this partition.
        drop_index : bool
            If `reset_index` is True, indicates whether to drop all index levels (True) or insert them into the
            resulting columns (False).
        lvls_to_drop : list of ints
            Contains numeric indices of the levels of the result index to drop as intersected.
        cols_to_drop : list of labels
            Contains labels of the columns to drop from the result as intersected.

        Examples
        --------
        >>> groupby_result = compute_groupby_without_processing_as_index_parameter()
        >>> if not as_index:
        >>>     reset_index, drop, lvls_to_drop, cols_to_drop = handle_as_index(**extract_required_params(groupby_result))
        >>>     if len(lvls_to_drop) > 0:
        >>>         groupby_result.index = groupby_result.index.droplevel(lvls_to_drop)
        >>>     if len(cols_to_drop) > 0:
        >>>         groupby_result = groupby_result.drop(columns=cols_to_drop)
        >>>     if reset_index:
        >>>         groupby_result_with_processed_as_index_parameter = groupby_result.reset_index(drop=drop)
        >>> else:
        >>>     groupby_result_with_processed_as_index_parameter = groupby_result
        """
        if by_length is None:
            by_length = len(internal_by_cols)

        reset_index = method != "transform" and (by_length > 0 or selection is not None)

        # If the method is "size" then the result contains only one unique named column
        # and we don't have to worry about any naming conflicts, so inserting all of
        # the "by" into the result (just a fast-path)
        if method == "size":
            return reset_index, False, [], []

        # Pandas logic of resolving naming conflicts is the following:
        #   1. If any categorical is in 'by' and 'by' is multi-column, then the categorical
        #      index is prioritized: drop intersected columns and insert all of the 'by' index
        #      levels to the frame as columns.
        #   2. Otherwise, aggregation result is prioritized: drop intersected index levels and
        #      insert the filtered ones to the frame as columns.
        if by_cols_dtypes is not None:
            keep_index_levels = (
                by_length > 1
                and selection is None
                and any(isinstance(x, pandas.CategoricalDtype) for x in by_cols_dtypes)
            )
        else:
            keep_index_levels = False

        # 1. We insert 'by'-columns to the result at the beginning of the frame and so only to the
        #    first partition, if partition_idx != 0 we just drop the index. If there are no columns
        #    that are required to drop (keep_index_levels is True) then we can exit here.
        # 2. We don't insert 'by'-columns to the result if 'by'-data came from a different
        #    frame (drop is False), there's only one exception for this rule: if the `method` is "size",
        #    so if (drop is False) and method is not "size" we just drop the index and so can exit here.
        if (not keep_index_levels and partition_idx != 0) or (
            not drop and method != "size"
        ):
            return reset_index, True, [], []

        if not isinstance(internal_by_cols, pandas.Index):
            if not is_list_like(internal_by_cols):
                internal_by_cols = [internal_by_cols]
            internal_by_cols = pandas.Index(internal_by_cols)

        internal_by_cols = (
            internal_by_cols[
                ~internal_by_cols.str.startswith(MODIN_UNNAMED_SERIES_LABEL, na=False)
            ]
            if hasattr(internal_by_cols, "str")
            else internal_by_cols
        )

        if selection is not None and not isinstance(selection, pandas.Index):
            selection = pandas.Index(selection)

        lvls_to_drop = []
        cols_to_drop = []

        if not keep_index_levels:
            # We want to insert only these internal-by-cols that are not presented
            # in the result in order to not create naming conflicts
            if selection is None:
                cols_to_insert = frozenset(internal_by_cols) - frozenset(result_cols)
            else:
                cols_to_insert = frozenset(
                    # We have to use explicit 'not in' check and not just difference
                    # of sets because of specific '__contains__' operator in case of
                    # scalar 'col' and MultiIndex 'selection'.
                    col
                    for col in internal_by_cols
                    if col not in selection
                )
        else:
            cols_to_insert = internal_by_cols
            # We want to drop such internal-by-cols that are presented
            # in the result in order to not create naming conflicts
            cols_to_drop = frozenset(internal_by_cols) & frozenset(result_cols)

        if partition_idx == 0:
            lvls_to_drop = [
                i
                for i, name in enumerate(result_index_names)
                if name not in cols_to_insert
            ]
        else:
            lvls_to_drop = result_index_names

        drop = False
        if len(lvls_to_drop) == len(result_index_names):
            drop = True
            lvls_to_drop = []

        return reset_index, drop, lvls_to_drop, cols_to_drop


class SeriesGroupBy(GroupBy):
    """Builder for GroupBy aggregation functions for Series."""

    @classmethod
    def _call_groupby(cls, df, *args, **kwargs):  # noqa: PR01
        """Call .groupby() on passed `df` squeezed to Series."""
        # We can end up here by two means - either by "true" call
        # like Series().groupby() or by df.groupby()[item].

        if len(df.columns) == 1:
            # Series().groupby() case
            return df.squeeze(axis=1).groupby(*args, **kwargs)
        # In second case surrounding logic will supplement grouping columns,
        # so we need to drop them after grouping is over; our originally
        # selected column is always the first, so use it
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return df.groupby(*args, **kwargs)[df.columns[0]]


class GroupByDefault(DefaultMethod):
    """Builder for default-to-pandas GroupBy aggregation functions."""

    _groupby_cls = GroupBy

    OBJECT_TYPE = "GroupBy"

    @classmethod
    def register(cls, func, **kwargs):
        """
        Build default-to-pandas GroupBy aggregation function.

        Parameters
        ----------
        func : callable or str
            Default aggregation function. If aggregation function is not specified
            via groupby arguments, then `func` function is used.
        **kwargs : kwargs
            Additional arguments that will be passed to function builder.

        Returns
        -------
        callable
            Functiom that takes query compiler and defaults to pandas to do GroupBy
            aggregation.
        """
        return super().register(
            cls._groupby_cls.build_groupby(func), fn_name=func.__name__, **kwargs
        )

    # This specifies a `pandas.DataFrameGroupBy` method to pass the `agg_func` to,
    # it's based on `how` to apply it. Going by pandas documentation:
    #   1. `.aggregate(func)` applies func row/column wise.
    #   2. `.apply(func)` applies func to a DataFrames, holding a whole group (group-wise).
    #   3. `.transform(func)` is the same as `.apply()` but also broadcast the `func`
    #      result to the group's original shape.
    #   4. 'direct' mode means that the passed `func` has to be applied directly
    #      to the `pandas.DataFrameGroupBy` object.
    _aggregation_methods_dict = {
        "axis_wise": pandas.core.groupby.DataFrameGroupBy.aggregate,
        "group_wise": pandas.core.groupby.DataFrameGroupBy.apply,
        "transform": pandas.core.groupby.DataFrameGroupBy.transform,
        "direct": lambda grp, func, *args, **kwargs: func(grp, *args, **kwargs),
    }

    @classmethod
    def get_aggregation_method(cls, how):
        """
        Return `pandas.DataFrameGroupBy` method that implements the passed `how` UDF applying strategy.

        Parameters
        ----------
        how : {"axis_wise", "group_wise", "transform"}
            `how` parameter of the ``BaseQueryCompiler.groupby_agg``.

        Returns
        -------
        callable(pandas.DataFrameGroupBy, callable, *args, **kwargs) -> [pandas.DataFrame | pandas.Series]

        Notes
        -----
        Visit ``BaseQueryCompiler.groupby_agg`` doc-string for more information about `how` parameter.
        """
        return cls._aggregation_methods_dict[how]


class SeriesGroupByDefault(GroupByDefault):
    """Builder for default-to-pandas GroupBy aggregation functions for Series."""

    _groupby_cls = SeriesGroupBy

    _aggregation_methods_dict = {
        "axis_wise": pandas.core.groupby.SeriesGroupBy.aggregate,
        "group_wise": pandas.core.groupby.SeriesGroupBy.apply,
        "transform": pandas.core.groupby.SeriesGroupBy.transform,
        "direct": lambda grp, func, *args, **kwargs: func(grp, *args, **kwargs),
    }
