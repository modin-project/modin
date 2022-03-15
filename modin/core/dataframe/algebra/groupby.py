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

"""Module houses builder class for GroupByReduce operator."""

from collections.abc import Container
import pandas

from .tree_reduce import TreeReduce
from .default2pandas.groupby import GroupBy
from modin.utils import try_cast_to_pandas, hashable
from modin.error_message import ErrorMessage


class GroupByReduce(TreeReduce):
    """Builder class for GroupBy aggregation functions."""

    @classmethod
    def call(cls, map_func, reduce_func=None, **call_kwds):
        """
        Build template GroupBy aggregation function.

        Resulted function is applied in parallel via TreeReduce algorithm.

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
            with TreeReduce algorithm.
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
    def map(
        cls,
        df,
        map_func,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        other=None,
        by=None,
    ):
        """
        Execute Map phase of GroupByReduce.

        Groups DataFrame and applies map function. Groups will be
        preserved in the results index for the following reduce phase.

        Parameters
        ----------
        df : pandas.DataFrame
            Serialized frame to group.
        map_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject`.
        axis : {0, 1}
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_kwargs : dict
            Dictionary which carries arguments for `pandas.DataFrame.groupby`.
        agg_args : list-like
            Positional arguments to pass to the aggregation functions.
        agg_kwargs : dict
            Keyword arguments to pass to the aggregation functions.
        other : pandas.DataFrame, optional
            Serialized frame, whose columns are used to determine the groups.
            If not specified, `by` parameter is used.
        by : level index name or list of such labels, optional
            Index levels, that is used to determine groups.
            If not specified, `other` parameter is used.

        Returns
        -------
        pandas.DataFrame
            GroupBy aggregation result for one particular partition.
        """
        # Set `as_index` to True to track the metadata of the grouping object
        # It is used to make sure that between phases we are constructing the
        # right index and placing columns in the correct order.
        groupby_kwargs["as_index"] = True
        groupby_kwargs["observed"] = True
        # We have to filter func-dict BEFORE inserting broadcasted 'by' columns
        # to avoid multiple aggregation results for 'by' cols in case they're
        # present in the func-dict:
        apply_func = cls.try_filter_dict(map_func, df)
        if other is not None:
            # Other is a broadcasted partition that represents 'by' columns
            # Concatenate it with 'df' to group on its columns names
            other = other.squeeze(axis=axis ^ 1)
            if isinstance(other, pandas.DataFrame):
                df = pandas.concat(
                    [df] + [other[[o for o in other if o not in df]]],
                    axis=1,
                )
                other = list(other.columns)
            by_part = other
        else:
            by_part = by

        result = apply_func(
            df.groupby(by=by_part, axis=axis, **groupby_kwargs), *agg_args, **agg_kwargs
        )
        # Result could not always be a frame, so wrapping it into DataFrame
        return pandas.DataFrame(result)

    @classmethod
    def reduce(
        cls,
        df,
        reduce_func,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        partition_idx=0,
        drop=False,
        method=None,
    ):
        """
        Execute Reduce phase of GroupByReduce.

        Combines groups from the Map phase and applies reduce function.

        Parameters
        ----------
        df : pandas.DataFrame
            Serialized frame which contain groups to combine.
        reduce_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject`.
        axis : {0, 1}
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_kwargs : dict
            Dictionary which carries arguments for `pandas.DataFrame.groupby`.
        agg_args : list-like
            Positional arguments to pass to the aggregation functions.
        agg_kwargs : dict
            Keyword arguments to pass to the aggregation functions.
        partition_idx : int, default: 0
            Internal index of column partition to which this function is applied.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.
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
        if drop:
            to_drop = df.columns.intersection(by_part)
            if isinstance(reduce_func, dict):
                to_drop = to_drop.difference(reduce_func.keys())
            if len(to_drop) > 0:
                df.drop(columns=by_part, errors="ignore", inplace=True)

        groupby_kwargs = groupby_kwargs.copy()
        as_index = groupby_kwargs["as_index"]

        # Set `as_index` to True to track the metadata of the grouping object
        groupby_kwargs["as_index"] = True

        # since now index levels contain out 'by', in the reduce phace
        # we want to group on these levels
        groupby_kwargs["level"] = list(range(len(df.index.names)))

        apply_func = cls.try_filter_dict(reduce_func, df)
        result = apply_func(
            df.groupby(axis=axis, **groupby_kwargs), *agg_args, **agg_kwargs
        )

        if not as_index:
            GroupBy.handle_as_index_for_dataframe(
                result,
                by_part,
                by_cols_dtypes=(
                    df.index.dtypes.values
                    if isinstance(df.index, pandas.MultiIndex)
                    else (df.index.dtype,)
                ),
                by_length=len(by_part),
                selection=reduce_func.keys() if isinstance(reduce_func, dict) else None,
                partition_idx=partition_idx,
                drop=drop,
                method=method,
                inplace=True,
            )
        # Result could not always be a frame, so wrapping it into DataFrame
        return pandas.DataFrame(result)

    @classmethod
    def caller(
        cls,
        query_compiler,
        by,
        map_func,
        reduce_func,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
        method=None,
        default_to_pandas_func=None,
    ):
        """
        Execute GroupBy aggregation with TreeReduce approach.

        Parameters
        ----------
        query_compiler : BaseQueryCompiler
            Frame to group.
        by : BaseQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        map_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Map phase.
        reduce_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Reduce phase.
        axis : {0, 1}
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_kwargs : dict
            Dictionary which carries arguments for pandas.DataFrame.groupby.
        agg_args : list-like
            Positional arguments to pass to the aggregation functions.
        agg_kwargs : dict
            Keyword arguments to pass to the aggregation functions.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.
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
        if (
            axis != 0
            or groupby_kwargs.get("level", None) is None
            and (
                not (isinstance(by, (type(query_compiler))) or hashable(by))
                or isinstance(by, pandas.Grouper)
            )
        ):
            by = try_cast_to_pandas(by, squeeze=True)
            # Since 'by' may be a 2D query compiler holding columns to group by,
            # to_pandas will also produce a pandas DataFrame containing them.
            # So splitting 2D 'by' into a list of 1D Series using 'GroupBy.validate_by':
            by = GroupBy.validate_by(by)
            if default_to_pandas_func is None:
                default_to_pandas_func = (
                    (lambda grp: grp.agg(map_func))
                    if isinstance(map_func, dict)
                    else map_func
                )
            return query_compiler.default_to_pandas(
                lambda df: default_to_pandas_func(
                    df.groupby(by=by, axis=axis, **groupby_kwargs),
                    *agg_args,
                    **agg_kwargs,
                )
            )

        # The bug only occurs in the case of Categorical 'by', so we might want to check whether any of
        # the 'by' dtypes is Categorical before going into this branch, however triggering 'dtypes'
        # computation if they're not computed may take time, so we don't do it
        if not groupby_kwargs.get("sort", True) and isinstance(
            by, type(query_compiler)
        ):
            ErrorMessage.missmatch_with_pandas(
                operation="df.groupby(categorical_by, sort=False)",
                message=(
                    "the groupby keys will be sorted anyway, although the 'sort=False' was passed. "
                    + "See the following issue for more details: "
                    + "https://github.com/modin-project/modin/issues/3571"
                ),
            )
            groupby_kwargs = groupby_kwargs.copy()
            groupby_kwargs["sort"] = True

        map_fn, reduce_fn = cls.build_map_reduce_functions(
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            map_func=map_func,
            reduce_func=reduce_func,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
            method=method,
        )

        # If `by` is a ModinFrame, then its partitions will be broadcasted to every
        # `self` partition in a way determined by engine (modin_frame.groupby_reduce)
        # Otherwise `by` was already bound to the Map function in `build_map_reduce_functions`.
        broadcastable_by = getattr(by, "_modin_frame", None)
        apply_indices = list(map_func.keys()) if isinstance(map_func, dict) else None
        new_modin_frame = query_compiler._modin_frame.groupby_reduce(
            axis, broadcastable_by, map_fn, reduce_fn, apply_indices=apply_indices
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
        groupby_kwargs,
        map_func,
        reduce_func,
        agg_args,
        agg_kwargs,
        drop=False,
        method=None,
    ):
        """
        Bind appropriate arguments to map and reduce functions.

        Parameters
        ----------
        by : BaseQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        axis : {0, 1}
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_kwargs : dict
            Dictionary which carries arguments for pandas.DataFrame.groupby.
        map_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Map phase.
        reduce_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Reduce phase.
        agg_args : list-like
            Positional arguments to pass to the aggregation functions.
        agg_kwargs : dict
            Keyword arguments to pass to the aggregation functions.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.
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

        def _map(df, other=None, **kwargs):
            def wrapper(df, other=None):
                return cls.map(
                    df,
                    other=other,
                    axis=axis,
                    by=by,
                    groupby_kwargs=groupby_kwargs.copy(),
                    map_func=map_func,
                    agg_args=agg_args,
                    agg_kwargs=agg_kwargs,
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
                    groupby_kwargs=groupby_kwargs,
                    reduce_func=reduce_func,
                    agg_args=agg_args,
                    agg_kwargs=agg_kwargs,
                    drop=drop,
                    method=method,
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


# This dict is a map for function names and their equivalents in TreeReduce
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


def _is_reduce_function_with_depth(fn, depth: int = 0):
    """
    Check whether all functions defined by `fn` are groupby reductions.

    If true, all functions defined by `fn` can be implemented with TreeReduce.
    This is a recursive helper function for is_reduce_function.

    Parameters
    ----------
    fn : Any
        Function to test.
    depth : int, default: 0
        How many nested containers we are within for this check.
            - if it's 0, then we're outside of any container, and `fn` could be
              either function name or container of function names/renamers.
            - if it's 1, then we're inside container of function
              names/renamers.`fn` must be either function name or renamer.
              renamer is some container which length == 2, where the first
              element is the new column name and the second is the function
              name.

    Returns
    -------
    bool
        Whether all functions defined by `fn` are reductions.
    """
    if not isinstance(fn, str) and isinstance(fn, Container):
        assert depth == 0 or (
            depth > 0 and len(fn) == 2
        ), f"Got the renamer with incorrect length, expected 2 got {len(fn)}."
        return (
            all(_is_reduce_function_with_depth(f, depth + 1) for f in fn)
            if depth == 0
            else _is_reduce_function_with_depth(fn[1], depth + 1)
        )
    return isinstance(fn, str) and fn in groupby_reduce_functions


def is_reduce_function(fn):
    """
    Check whether all functions defined by `fn` are groupby reductions.

    If true, all functions defined by `fn` can be implemented with TreeReduce.

    Parameters
    ----------
    fn : Any
        Function to test.

    Returns
    -------
    bool
        Whether all functions defined by `fn` are reductions.
    """
    return _is_reduce_function_with_depth(fn, depth=0)
