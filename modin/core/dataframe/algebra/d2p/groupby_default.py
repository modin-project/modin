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

from .default import DefaultMethod

import pandas


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
            if df.name == "__reduced__":
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

        def inplace_applyier(grp, **func_kwargs):
            return key(grp, *inplace_args, **func_kwargs)

        return inplace_applyier

    @classmethod
    # FIXME: `grp` parameter is redundant and should be removed
    def get_func(cls, grp, key, **kwargs):
        """
        Extract aggregation function from groupby arguments.

        Parameters
        ----------
        grp : pandas.DataFrameGroupBy
            GroupBy object to apply aggregation on.
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
        There is two ways of how groupby aggregation can be invoked:
            1. Explicitly with query compiler method: `qc.groupby_sum()`.
            2. By passing aggregation function as an argument: `qc.groupby_agg("sum")`.
        Both are going to produce the same result, however in the first case actual aggregation
        function can be extracted from the method name, while for the second only from the method arguments.
        """
        if "agg_func" in kwargs:
            return kwargs["agg_func"]
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
            groupby_args,
            agg_args,
            axis=0,
            is_multi_by=None,
            drop=False,
            **kwargs
        ):
            """Group DataFrame and apply aggregation function to each group."""
            by = cls.validate_by(by)

            grp = df.groupby(by, axis=axis, **groupby_args)
            agg_func = cls.get_func(grp, key, **kwargs)
            result = (
                grp.agg(agg_func, **agg_args)
                if isinstance(agg_func, dict)
                else agg_func(grp, **agg_args)
            )

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
            df,
            by,
            axis,
            groupby_args,
            map_args,
            numeric_only=True,
            drop=False,
            **kwargs
        ):
            """Group DataFrame and apply aggregation function to each group."""
            if not isinstance(by, (pandas.Series, pandas.DataFrame)):
                by = cls.validate_by(by)
                return agg_func(
                    df.groupby(by=by, axis=axis, **groupby_args), **map_args
                )

            if numeric_only:
                df = df.select_dtypes(include="number")

            by = by.squeeze(axis=1)
            if (
                drop
                and isinstance(by, pandas.Series)
                and by.name in df
                and df[by.name].equals(by)
            ):
                by = by.name
            if isinstance(by, pandas.DataFrame):
                df = pandas.concat([df] + [by[[o for o in by if o not in df]]], axis=1)
                by = list(by.columns)

            groupby_args = groupby_args.copy()
            as_index = groupby_args.pop("as_index", True)
            groupby_args["as_index"] = True

            grp = df.groupby(by, axis=axis, **groupby_args)
            result = agg_func(grp, **map_args)

            if isinstance(result, pandas.Series):
                result = result.to_frame()

            if not as_index:
                if (
                    len(result.index.names) == 1 and result.index.names[0] is None
                ) or all([name in result.columns for name in result.index.names]):
                    drop = False
                elif kwargs.get("method") == "size":
                    drop = True
                result = result.reset_index(drop=not drop)

            if result.index.name == "__reduced__":
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


class GroupByDefault(DefaultMethod):
    """Builder for default-to-pandas GroupBy aggregation functions."""

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
        return cls.call(GroupBy.build_groupby(func), fn_name=func.__name__, **kwargs)
