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

import pandas

from .mapreducefunction import MapReduceFunction
from modin.utils import try_cast_to_pandas, hashable


class GroupbyReduceFunction(MapReduceFunction):
    @classmethod
    def call(cls, map_func, reduce_func=None, **call_kwds):
        """
        Build GroupbyReduce function.

        Parameters
        ----------
        map_func: str, callable or dict,
            If 'str' this parameter will be treated as a function name to register,
            so 'map_func' and 'reduce_func' will be grabbed from 'groupby_reduce_functions'.
            If dict or callable then this will be treated as a function to apply to each group
            at the map phase.
        reduce_func: callable or dict (optional),
            A function to apply to each group at the reduce phase. If not specified
            will be set the same as 'map_func'.
        **call_kwds: kwargs,
            Kwargs that will be passed to the returned function.

        Returns
        -------
        Callable,
            Function that executes GroupBy aggregation with MapReduce algorithm.
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
        other=None,
        axis=0,
        by=None,
        groupby_args=None,
        map_func=None,
        map_args=None,
        drop=False,
    ):
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
        **kwargs,
    ):
        # Wrapping names into an Index should be unnecessary, however
        # there is a bug in pandas with intersection that forces us to do so:
        # https://github.com/pandas-dev/pandas/issues/39699
        by_part = pandas.Index(df.index.names)
        if drop and len(df.columns.intersection(by_part)) > 0:
            df.drop(columns=by_part, errors="ignore", inplace=True)

        groupby_args = groupby_args.copy()
        method = kwargs.get("method", None)
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
    def caller(
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
        if groupby_args.get("level", None) is None and (
            not (isinstance(by, (type(query_compiler))) or hashable(by))
            or isinstance(by, pandas.Grouper)
        ):
            by = try_cast_to_pandas(by, squeeze=True)
            default_func = (
                (lambda grp: grp.agg(map_func))
                if isinstance(map_func, dict)
                else map_func
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
        **kwargs,
    ):
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
                    drop=drop,
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
                    **kwargs,
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
