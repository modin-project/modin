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

from typing import Callable, Optional, Union
import pandas

from .mapreducefunction import MapReduceFunction
from modin.utils import try_cast_to_pandas, hashable


class GroupbyReduceFunction(MapReduceFunction):
    @classmethod
    def register(
        cls,
        map_func: Union[str, dict, Callable],
        reduce_func: Optional[Union[dict, Callable]] = None,
        *reg_args,
        **reg_kwargs,
    ):
        """
        Build template groupby function, which  MapReduce

        Parameters
        ----------
        map_func: str, callable or dict,
            If 'str' this parameter will be treated as a function name to register,
            so 'map_func' and 'reduce_func' will be grabbed from 'GROUPBY_REDUCE_FUNCTIONS'.
            If dict or callable then this will be treated as a function to apply to each group
            at the map phase.
        reduce_func: callable or dict (optional),
            Function to apply to each group at the reduce phase. If not specified
            will be set the same as 'map_func'.
        *reg_args: args,
            Args that will be passed to the returned function.
        **reg_kwargs: kwargs,
            Kwargs that will be passed to the returned function.

        Returns
        -------
        Callable,
            Function that executes GroupBy aggregation with MapReduce algorithm.

        Notes
        -----
        Method obtains QueryCompiler
          0  1
        a
        b
        -------
        a
        a
        """
        if isinstance(map_func, str):

            def build_fn(name):
                return lambda df, *args, **kwargs: getattr(df, name)(*args, **kwargs)

            map_func, reduce_func = map(build_fn, GROUPBY_REDUCE_FUNCTIONS[map_func])
        if reduce_func is None:
            reduce_func = map_func
        assert not (
            isinstance(map_func, dict) ^ isinstance(reduce_func, dict)
        ) and not (
            callable(map_func) ^ callable(reduce_func)
        ), "Map and reduce functions must be either both dict or both callable."

        return lambda *args, **kwargs: cls.groupby_reduce_function(
            *args,
            *reg_args,
            map_func=map_func,
            reduce_func=reduce_func,
            **kwargs,
            **reg_kwargs,
        )

    @classmethod
    def map(
        cls,
        df,
        other=None,
        axis=0,
        by=None,
        groupby_args=None,
        map_func=None,
        map_args=None,
    ):
        """
        Executes Map phase of GroupbyReduce:
        Groups DataFrame and applies map function to each group. Groups will be
        preserved in the results index for the following reduce phase.

        Parameters
        ----------
        df: pandas.DataFrame,
            Serialized frame to group.
        other: pandas.DataFrame (optional),
            Serialized frame, whose columns used to determine the groups.
            If not specified, `by` parameter is used.
        axis: int, (default 0)
            Axis to group and execute function along.
        by: level index name or list of such labels
            Index levels, that is used to determine groups.
        groupby_args: dict,
            Dictionary which carries arguments for pandas.DataFrame.groupby
        map_func: callable,
            Function to apply to each group.
        map_args: dict,
            Arguments which will be passed to `map_func`.

        Returns
        -------
        pandas.DataFrame
            GroupBy result for one particular partition.
        """
        # Set `as_index` to True to track the metadata of the grouping object
        # It is used to make sure that between phases we are constructing the
        # right index and placing columns in the correct order.
        groupby_args["as_index"] = True
        groupby_args["observed"] = True
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

        apply_func = cls.try_filter_dict(map_func, df)
        result = apply_func(
            df.groupby(by=by_part, axis=axis, **groupby_args), **map_args
        )
        # Result could not always be a frame, so wrapping it into DataFrame
        return pandas.DataFrame(result)

    @classmethod
    def reduce(
        cls,
        df,
        partition_idx=0,
        axis=0,
        groupby_args=None,
        reduce_func=None,
        reduce_args=None,
        drop=False,
        method=None,
    ):
        """
        Executes Reduce phase of GroupbyReduce:
        Combines groups from the Map phase and applies reduce function to each group.

        Parameters
        ----------
        df: pandas.DataFrame,
            Serialized frame which contain groups to combine.
        partition_idx: int,
            Internal index of column partition to which this function is applied.
        axis: int, (default 0)
            Axis to group and execute function along.
        groupby_args: dict,
            Dictionary which carries arguments for pandas.DataFrame.groupby
        reduce_func: callable,
            Function to apply to each group.
        reduce_args: dict,
            Arguments which will be passed to `reduce_func`.
        drop: bool, (default False)
            Indicates whether or not by-data came from the same frame?
        method: str, (optional)
            Name of the groupby function. This is a hint to do special casing.

        Returns
        -------
        pandas.DataFrame
            GroupBy result.
        """
        # Wrapping names into an Index should be unnecessary, however
        # there is a bug in pandas with intersection that forces us to do so:
        # https://github.com/pandas-dev/pandas/issues/39699
        by_part = pandas.Index(df.index.names)
        if drop and len(df.columns.intersection(by_part)) > 0:
            df.drop(columns=by_part, errors="ignore", inplace=True)

        groupby_args = groupby_args.copy()
        as_index = groupby_args["as_index"]

        # Set `as_index` to True to track the metadata of the grouping object
        groupby_args["as_index"] = True

        # since now index levels contain out 'by', in the reduce phace
        # we want to group on these levels
        groupby_args["level"] = list(range(len(df.index.names)))

        apply_func = cls.try_filter_dict(reduce_func, df)
        result = apply_func(df.groupby(axis=axis, **groupby_args), **reduce_args)

        if not as_index:
            insert_levels = partition_idx == 0 and (drop or method == "size")
            result.reset_index(drop=not insert_levels, inplace=True)
        # Result could not always be a frame, so wrapping it into DataFrame
        return pandas.DataFrame(result)

    @classmethod
    def groupby_reduce_function(
        cls,
        query_compiler,
        by,
        axis,
        groupby_args,
        map_args,
        map_func,
        numeric_only=True,
        **kwargs,
    ):
        """
        Builds Map and Reduce function that processes GroupBy and executes them.

        query_compiler: QueryCompiler,
            frame to group
        by: QueryCompiler, column or index label, Grouper or list of such
            determine groups
        axis: int,
            axis to group and apply function
        groupby_args: dict,
            Dictionary which carries arguments for pandas.DataFrame.groupby
        map_args: dict,
            Arguments which will be passed to `map_func`.
        map_func: callable,
            Function to apply to each group at the Map phase.
        numeric_only: bool (default True),
            Whether or not to drop non-numeric columns before executing GroupBy.
        **kwargs: kwargs,
            Additional parameters to build Map and Reduce functions.

        Returns
        -------
        QueryCompiler
            QueryCompiler which carries the result of GroupBy aggregation.
        """
        # Currently we're defaulting to pandas if
        if groupby_args.get("level", None) is None and (
            not (isinstance(by, (type(query_compiler))) or hashable(by))
            or isinstance(by, pandas.Grouper)
        ):
            by = try_cast_to_pandas(by, squeeze=True)
            default_func = kwargs.get(
                "default_to_pandas_func",
                (lambda grp: grp.agg(map_func))
                if isinstance(map_func, dict)
                else map_func,
            )
            return query_compiler.default_to_pandas(
                lambda df: default_func(
                    df.groupby(by=by, axis=axis, **groupby_args), **map_args
                )
            )
        assert axis == 0, "Can only groupby reduce with axis=0"

        if numeric_only:
            qc = query_compiler.getitem_column_array(
                query_compiler._modin_frame._numeric_columns(True)
            )
        else:
            qc = query_compiler

        map_fn, reduce_fn = cls.build_map_reduce_functions(
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_func=map_func,
            map_args=map_args,
            **kwargs,
        )

        # If `by` is a ModinFrame, then its partitions will be broadcasted to every
        # `self` partition in a way, determined by engine (modin_frame.groupby_reduce)
        # Otherwise `by` was already binded to the Map function in `build_map_reduce_functions`.
        broadcastable_by = getattr(by, "_modin_frame", None)
        apply_indices = list(map_func.keys()) if isinstance(map_func, dict) else None
        new_modin_frame = qc._modin_frame.groupby_reduce(
            axis, broadcastable_by, map_fn, reduce_fn, apply_indices=apply_indices
        )

        result = query_compiler.__constructor__(new_modin_frame)
        if result.index.name == "__reduced__":
            result.index.name = None
        return result

    @staticmethod
    def try_filter_dict(agg_func, df):
        """
        If it's dictionary aggregation filters aggregation dictionary for keys which
        this particular partition contains, otherwise do nothing with passed function.

        Parameters
        ----------
        agg_func: callable or dict,
            GroupBy function.
        df: pandas.DataFrame,
            Serialized partition which contains available columns.

        Returns
        -------
        Callable
            Function that can be safely applied to each group.
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
        method=None,
    ):
        """
        Binds appropriate arguments to map and reduce functions.

        Returns
        -------
        Tuple of callable
            Tuple of map and reduce function with binded arguments.
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
                    other,
                    axis=axis,
                    by=by,
                    groupby_args=groupby_args.copy(),
                    map_func=map_func,
                    map_args=map_args,
                    **kwargs,
                )

            try:
                result = wrapper(df, other)
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            except ValueError:
                result = wrapper(df.copy(), other if other is None else other.copy())
            return result

        def _reduce(df, **kwargs):
            def wrapper(df):
                return cls.reduce(
                    df,
                    axis=axis,
                    groupby_args=groupby_args,
                    reduce_func=reduce_func,
                    reduce_args=reduce_args,
                    drop=drop,
                    method=method,
                    **kwargs,
                )

            try:
                result = wrapper(df)
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            except ValueError:
                result = wrapper(df.copy())
            return result

        return _map, _reduce


# This dict is a map for function names and their equivalents in MapReduce
GROUPBY_REDUCE_FUNCTIONS = {
    "all": ("all", "all"),
    "any": ("any", "any"),
    "count": ("count", "sum"),
    "max": ("max", "max"),
    "min": ("min", "min"),
    "prod": ("prod", "prod"),
    "size": ("size", "sum"),
    "sum": ("sum", "sum"),
}
