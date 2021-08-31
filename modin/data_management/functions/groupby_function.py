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

"""Module houses GroupBy functions builder class."""

import pandas
from pandas.core.dtypes.common import is_list_like

from .mapreducefunction import MapReduceFunction
from modin.utils import try_cast_to_pandas, hashable


class GroupbyReduceFunction(MapReduceFunction):
    """Builder class for GroupBy aggregation functions."""

    @classmethod
    def call(cls, map_func, reduce_func=None, **call_kwds):
        """
        Build template GroupBy aggregation function.

        Resulted function is applied in parallel via MapReduce algorithm.

        Parameters
        ----------
        map_func : str, dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            If `str` this parameter will be treated as a function name to register,
            so `map_func` and `reduce_func` will be grabbed from `groupby_reduce_functions`.
            If dict or callable then this will be treated as a function to apply to the `GroupByObject`
            at the map phase.
        reduce_func : str, dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame, optional
            Function to apply to the `GroupByObject` at the reduce phase. If not specified
            will be set the same as 'map_func'.
        **call_kwds : kwargs
            Kwargs that will be passed to the returned function.

        Returns
        -------
        callable
            Function that takes query compiler and executes GroupBy aggregation
            with MapReduce algorithm.
        """
        if isinstance(map_func, str):

            def build_fn(name):
                return lambda df, *args, **kwargs: getattr(df, name)(*args, **kwargs)

            map_func, reduce_func = map(build_fn, groupby_reduce_functions[map_func])
        if reduce_func is None:
            reduce_func = map_func
        assert not (
            isinstance(map_func, dict) ^ isinstance(reduce_func, dict)
        ) and not (
            callable(map_func) ^ callable(reduce_func)
        ), "Map and reduce functions must be either both dict or both callable."

        return lambda *args, **kwargs: cls.caller(
            *args, map_func=map_func, reduce_func=reduce_func, **kwargs, **call_kwds
        )

    @classmethod
    # FIXME:
    #   1. Remove `drop` parameter since it isn't used.
    #   2. `map_func` is not supposed to be `None`
    #   3. Case when `map_args` or `groupby_args` is `None` (its default value) is unhandled.
    def map(
        cls,
        df,
        other=None,
        axis=0,
        by=None,
        groupby_args=None,
        map_func=None,
        map_args=None,
        drop=False,
        selection=None,
    ):
        """
        Execute Map phase of GroupbyReduce.

        Groups DataFrame and applies map function. Groups will be
        preserved in the results index for the following reduce phase.

        Parameters
        ----------
        df : pandas.DataFrame
            Serialized frame to group.
        other : pandas.DataFrame, optional
            Serialized frame, whose columns are used to determine the groups.
            If not specified, `by` parameter is used.
        axis : {0, 1}, default: 0
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        by : level index name or list of such labels, optional
            Index levels, that is used to determine groups.
            If not specified, `other` parameter is used.
        groupby_args : dict, optional
            Dictionary which carries arguments for `pandas.DataFrame.groupby`.
        map_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame, default: None
            Function to apply to the `GroupByObject`.
        map_args : dict, optional
            Arguments which will be passed to `map_func`.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.
        selection : label or list of labels, optional
            Set of columns to apply aggregation on, by default aggregation is applied
            to all of the available columns.

        Returns
        -------
        pandas.DataFrame
            GroupBy aggregation result for one particular partition.
        """
        # Set `as_index` to True to track the metadata of the grouping object
        # It is used to make sure that between phases we are constructing the
        # right index and placing columns in the correct order.
        groupby_args["as_index"] = True
        groupby_args["observed"] = True
        partition_selection = (
            selection
            if selection is None
            else (
                # If `selection` is not a list then the only possible partition to which
                # kernel function could be applied is the one containing the single
                # selected column, so we don't need to compute the potentially expensive intersection
                df.columns.intersection(selection)
                if is_list_like(selection)
                else selection
            )
        )

        if other is not None:
            # Other is a broadcasted partition that represents 'by' columns
            # Concatenate it with 'df' to group on its columns names
            if not drop:
                other = other.squeeze(axis=axis ^ 1)
            if isinstance(other, pandas.DataFrame):
                df = pandas.concat(
                    [df, other[other.columns.difference(df.columns)]],
                    axis=1,
                )
                other = list(other.columns)
            by_part = other
        else:
            by_part = by

        apply_func = cls.try_filter_dict(map_func, df)
        grp = df.groupby(by=by_part, axis=axis, **groupby_args)
        if partition_selection is not None:
            grp = grp[partition_selection]
        result = apply_func(grp, **map_args)
        # Result could not always be a frame, so wrapping it into DataFrame
        return pandas.DataFrame(result)

    @classmethod
    # FIXME:
    #   1. spread `**kwargs` into an actual function arguments.
    #   2. `reduce_func` is not supposed to be `None`
    #   3. Case when `reduce_args` or `groupby_args` is `None` (its default value)
    #      is unhandled.
    def reduce(
        cls,
        df,
        partition_idx=0,
        axis=0,
        groupby_args=None,
        reduce_func=None,
        reduce_args=None,
        drop=False,
        selection=None,
        partition_selection=None,
        method=None,
    ):
        """
        Execute Reduce phase of GroupbyReduce.

        Combines groups from the Map phase and applies reduce function.

        Parameters
        ----------
        df : pandas.DataFrame
            Serialized frame which contain groups to combine.
        partition_idx : int, default: 0
            Internal index of column partition to which this function is applied.
        axis : {0, 1}, default: 0
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_args : dict, optional
            Dictionary which carries arguments for `pandas.DataFrame.groupby`.
        reduce_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame, default: None
            Function to apply to the `GroupByObject`.
        reduce_args : dict, optional
            Arguments which will be passed to `reduce_func`.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.
        selection : label or list of labels, optional
            Set of columns to apply aggregation on, by default aggregation is applied
            to all of the available columns.
        partition_selection : list of labels, optional
            Set of columns at this particular partition to which aggregation was applied
            at the Map phase. If not specified assuming that aggregation at this partition
            was applied to all of the columns listed in the `selection` parameter.
            **Note:** in the natural execution flow, this parameter is computed automatically
            at the partition manager and is passed to this function if `selection` is specified.
        method : str, optional
            Name of the groupby function. This is a hint to be able to do special casing.

        Returns
        -------
        pandas.DataFrame
            GroupBy aggregation result.
        """
        # Wrapping names into an Index should be unnecessary, however
        # there is a bug in pandas with intersection that forces us to do so:
        # https://github.com/pandas-dev/pandas/issues/39699
        by_part = pandas.Index(df.index.names)
        to_drop = df.columns.intersection(by_part)
        if selection is not None:
            to_drop = to_drop.difference(
                selection if is_list_like(selection) else (selection,)
            )
        if drop and len(to_drop) > 0:
            df.drop(columns=to_drop, errors="ignore", inplace=True)

        groupby_args = groupby_args.copy()
        as_index = groupby_args["as_index"]

        # Set `as_index` to True to track the metadata of the grouping object
        groupby_args["as_index"] = True

        # since now index levels contain out 'by', in the reduce phace
        # we want to group on these levels
        groupby_args["level"] = list(range(len(df.index.names)))

        apply_func = cls.try_filter_dict(reduce_func, df)
        grp = df.groupby(axis=axis, **groupby_args)
        if partition_selection is not None and not isinstance(reduce_func, dict):
            # Some of the selected columns could've been dropped in the Map phase as non-suitable
            # for the current aggregation, so re-selecting columns for this particular partition.
            current_partition_selection = df.columns.intersection(partition_selection)
            if not is_list_like(selection) and len(current_partition_selection) > 0:
                assert (
                    len(current_partition_selection) == 1
                ), f"Single-dimensional object has non-single dimensional selection: {current_partition_selection}."
                current_partition_selection = current_partition_selection[0]
            grp = grp[current_partition_selection]
        result = apply_func(grp, **reduce_args)

        if isinstance(result, pandas.Series):
            result = result.to_frame()

        if not as_index:
            drop, lvls_to_drop = cls.handle_as_index(
                result.columns,
                result.index.names,
                by_part,
                selection=selection,
                partition_selection=partition_selection,
                partition_idx=partition_idx,
                drop=drop,
                method=method,
            )
            if len(lvls_to_drop) > 0:
                result.index = result.index.droplevel(lvls_to_drop)
            result = result.reset_index(drop=drop)

        # Result could not always be a frame, so wrapping it into DataFrame
        return pandas.DataFrame(result)

    @classmethod
    def caller(
        cls,
        query_compiler,
        by,
        axis,
        groupby_args,
        map_args,
        map_func,
        reduce_func,
        reduce_args,
        numeric_only=True,
        drop=False,
        selection=None,
        method=None,
        default_to_pandas_func=None,
    ):
        """
        Execute GroupBy aggregation with MapReduce approach.

        Parameters
        ----------
        query_compiler : BaseQueryCompiler
            Frame to group.
        by : BaseQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        axis : {0, 1}, default: 0
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_args : dict
            Dictionary which carries arguments for pandas.DataFrame.groupby.
        map_args : dict
            Arguments which will be passed to `map_func`.
        map_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Map phase.
        reduce_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Reduce phase.
        reduce_args : dict
            Arguments which will be passed to `reduce_func`.
        numeric_only : bool, default: True
            Whether or not to drop non-numeric columns before executing GroupBy.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.
        selection : list of labels, optional
            Set of columns to apply aggregation on, by default aggregation is applied
            to all of the available columns.
        method : str, optional
            Name of the GroupBy aggregation function. This is a hint to be able to do special casing.
        default_to_pandas_func : callable(pandas.DataFrameGroupBy) -> pandas.DataFrame, optional
            The pandas aggregation function equivalent to the `map_func + reduce_func`.
            Used in case of defaulting to pandas. If not specified `map_func` is used.

        Returns
        -------
        The same type as `query_compiler`
            QueryCompiler which carries the result of GroupBy aggregation.
        """
        if groupby_args.get("level", None) is None and (
            not (isinstance(by, (type(query_compiler))) or hashable(by))
            or isinstance(by, pandas.Grouper)
        ):
            by = try_cast_to_pandas(by, squeeze=True)
            if default_to_pandas_func is None:
                default_to_pandas_func = (
                    (lambda grp: grp.agg(map_func))
                    if isinstance(map_func, dict)
                    else map_func
                )

            def groupby_reduce(df):
                grp = df.groupby(by=by, axis=axis, **groupby_args)
                if selection is not None:
                    grp = grp[selection]
                return default_to_pandas_func(grp, **map_args)

            return query_compiler.default_to_pandas(groupby_reduce)
        assert axis == 0, "Can only groupby reduce with axis=0"

        # If columns are explicitly selected for aggregation we mustn't drop
        # them as non-suitable for numeric aggregation
        if numeric_only and selection is None:
            qc = query_compiler.getitem_column_array(
                query_compiler._modin_frame.numeric_columns(include_bool=True)
            )
        else:
            qc = query_compiler

        map_fn, reduce_fn = cls.build_map_reduce_functions(
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_func=map_func,
            map_args=map_args,
            reduce_func=reduce_func,
            reduce_args=reduce_args,
            drop=drop,
            selection=selection,
            method=method,
        )

        # If `by` is a ModinFrame, then its partitions will be broadcasted to every
        # `self` partition in a way determined by engine (modin_frame.groupby_reduce)
        # Otherwise `by` was already bound to the Map function in `build_map_reduce_functions`.
        broadcastable_by = getattr(by, "_modin_frame", None)
        apply_indices = None
        if isinstance(map_func, dict):
            apply_indices = tuple(map_func.keys())
        if selection is not None and apply_indices is None:
            apply_indices = selection if is_list_like(selection) else (selection,)
        new_modin_frame = qc._modin_frame.groupby_reduce(
            axis, broadcastable_by, map_fn, reduce_fn, selection=apply_indices
        )

        result = query_compiler.__constructor__(new_modin_frame)

        if result.index.name == "__reduced__":
            result.index.name = None
        return result

    @staticmethod
    def try_filter_dict(agg_func, df):
        """
        Build aggregation function to apply to each group at this particular partition.

        If it's dictionary aggregation — filters aggregation dictionary for keys which
        this particular partition contains, otherwise do nothing with passed function.

        Parameters
        ----------
        agg_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Aggregation function.
        df : pandas.DataFrame
            Serialized partition which contains available columns.

        Returns
        -------
        Callable
            Aggregation function that can be safely applied to this particular partition.
        """
        if not isinstance(agg_func, dict):
            return agg_func
        partition_dict = {k: v for k, v in agg_func.items() if k in df.columns}
        return lambda grp: grp.agg(partition_dict)

    @classmethod
    def build_map_reduce_functions(
        cls,
        by,
        axis,
        groupby_args,
        map_func,
        map_args,
        reduce_func,
        reduce_args,
        drop,
        selection=None,
        method=None,
    ):
        """
        Bind appropriate arguments to map and reduce functions.

        Parameters
        ----------
        by : BaseQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        axis : {0, 1}, default: 0
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_args : dict
            Dictionary which carries arguments for pandas.DataFrame.groupby.
        map_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Map phase.
        map_args : dict
            Arguments which will be passed to `map_func`.
        reduce_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Reduce phase.
        reduce_args : dict
            Arguments which will be passed to `reduce_func`.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.
        selection : list of labels, optional
            Set of columns to apply aggregation on, by default aggregation is applied
            to all of the available columns.
        method : str, optional
            Name of the GroupBy aggregation function. This is a hint to be able to do special casing.

        Returns
        -------
        Tuple of callable
            Tuple of map and reduce functions with bound arguments.
        """
        # if by is a query compiler, then it will be broadcasted explicit via
        # groupby_reduce method of the modin frame and so we don't want secondary
        # implicit broadcastion via passing it as an function argument.
        if hasattr(by, "_modin_frame"):
            by = None

        if isinstance(map_func, dict):
            assert (
                selection is None
            ), "Can't handle 'selection' and dictionary aggregation at the same time."

        def _map(df, other=None, **kwargs):
            def wrapper(df, other=None):
                return cls.map(
                    df,
                    other,
                    axis=axis,
                    by=by,
                    groupby_args=groupby_args.copy(),
                    map_func=map_func,
                    map_args=map_args,
                    drop=drop,
                    selection=selection,
                    **kwargs,
                )

            try:
                result = wrapper(df, other)
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            except ValueError:
                result = wrapper(df.copy(), other if other is None else other.copy())
            return result

        def _reduce(df, **call_kwargs):
            def wrapper(df):
                return cls.reduce(
                    df,
                    axis=axis,
                    groupby_args=groupby_args,
                    reduce_func=reduce_func,
                    reduce_args=reduce_args,
                    drop=drop,
                    method=method,
                    selection=selection,
                    **call_kwargs,
                )

            try:
                result = wrapper(df)
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            except ValueError:
                result = wrapper(df.copy())
            return result

        return _map, _reduce

    @staticmethod
    def handle_as_index(
        result_cols,
        result_index_names,
        internal_by_cols,
        selection=None,
        partition_selection=None,
        partition_idx=0,
        drop=True,
        method=None,
    ):
        """
        Help to handle ``as_index=False`` parameter for the GroupBy result.

        This function resolve naming conflicts of the index levels to insert and the column labels
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
        selection : label or list of labels, optional
            Set of columns to which aggregation was applied. If not specified
            assuming that aggregation was applied to all of the available columns.
        partition_selection : label or list of labels, optional
            Set of columns at this particular partition to which aggregation was applied.
            If not specified assuming that aggregation at this partition was applied
            to all of the columns listed in the `selection` parameter.
        partition_idx : int, default: 0
            Positional index of the current partition.
        drop : bool, default: True
            Indicates whether or not all of the `by` data came from the same frame.
        method : str, optional
            Name of the groupby function. This is a hint to be able to do special casing.
            Note: this parameter is a legacy from the ``groupby_size`` implementation,
            it's a hacky one and probably will be removed in the future.

        Returns
        -------
        drop : bool
            Indicates whether to drop all index levels (True) or insert them into the
            resulting columns (False).
        lvls_to_drop : list of ints
            Contains numeric indices of the levels of the result index to drop as intersected.

        Examples
        --------
        >>> groupby_result = compute_groupby_without_processing_as_index_parameter()
        >>> if not as_index:
        >>>     drop, lvls_to_drop = handle_as_index(**extract_required_params(groupby_result))
        >>>     if len(lvls_to_drop) > 0:
        >>>         groupby_result.index = groupby_result.index.droplevel(lvls_to_drop)
        >>>     groupby_result_with_processed_as_index_parameter = groupby_result.reset_index(drop=drop)
        >>> else:
        >>>     groupby_result_with_processed_as_index_parameter = groupby_result
        """
        # 1. We insert by-columns to the result at the beginning of the frame and so
        #    only to the first partition, if partition_idx != 0 we just dropping the index
        # 2. We don't insert by-columns to the result if by-data came from a different
        #    frame (drop is False), there's only one exception for this rule when method is "size",
        #    so if (drop is False) and method is not "size" we just drop the index.
        if partition_idx != 0 or (not drop and method != "size"):
            return True, []

        # If the method is "size" then the result contains only one unique named column
        # and we don't have to worry about any naming conflicts, so inserting all of
        # the "by" into the result (just a fast-path)
        if method == "size":
            return False, []

        if selection is not None:
            selection = selection if is_list_like(selection) else (selection,)
            if partition_selection is None:
                partition_selection = selection
            partition_selection = (
                partition_selection
                if is_list_like(partition_selection)
                else (partition_selection,)
            )
            if len(result_cols) != len(partition_selection):
                cols_failed_to_select = pandas.Index(partition_selection).difference(
                    result_cols
                )
                # If some of the selected 'by' columns were dropped as non-suitable for
                # the current aggregation it's allowed to insert them back. Dropping
                # them from the selection to be able to do so:
                selection = pandas.Index(selection).difference(cols_failed_to_select)

        if not isinstance(internal_by_cols, pandas.Index):
            if not is_list_like(internal_by_cols):
                internal_by_cols = [internal_by_cols]
            internal_by_cols = pandas.Index(internal_by_cols)

        internal_by_cols = (
            internal_by_cols[~internal_by_cols.str.match(r"__reduced__.*", na=False)]
            if hasattr(internal_by_cols, "str")
            else internal_by_cols
        )

        # We want to insert such internal-by-cols that are not presented
        # in the result in order to not create naming conflicts
        cols_to_insert = (
            internal_by_cols.copy()
            if selection is None and internal_by_cols.nlevels != result_cols.nlevels
            else internal_by_cols.difference(
                result_cols if selection is None else selection
            )
        )

        lvls_to_drop = [
            i for i, name in enumerate(result_index_names) if name not in cols_to_insert
        ]

        drop = False
        if len(lvls_to_drop) == len(result_index_names):
            drop = True
            lvls_to_drop = []

        return drop, lvls_to_drop


# This dict is a map for function names and their equivalents in MapReduce
groupby_reduce_functions = {
    "all": ("all", "all"),
    "any": ("any", "any"),
    "count": ("count", "sum"),
    "max": ("max", "max"),
    "min": ("min", "min"),
    "prod": ("prod", "prod"),
    "size": ("size", "sum"),
    "sum": ("sum", "sum"),
}
