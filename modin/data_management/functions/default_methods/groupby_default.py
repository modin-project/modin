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

from .default import DefaultMethod

from modin.utils import try_cast_to_pandas
import re
import pandas


class GroupBy:
    @classmethod
    def validate_by(cls, by):
        def try_cast_series(df):
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
        inplace_args = [] if func is None else [func]

        if isinstance(key, str):
            key = getattr(pandas.core.groupby.DataFrameGroupBy, key)

        def inplace_applyier(grp, **func_kwargs):
            return key(grp, *inplace_args, **func_kwargs)

        return inplace_applyier

    @classmethod
    def get_by_names(cls, by):
        return [o.name if isinstance(o, pandas.Series) else o for o in by]

    @classmethod
    def materialize_by(cls, df, by):
        return [df[o] for o in by if o in df]

    @classmethod
    def get_func(cls, grp, key, **kwargs):
        if "agg_func" in kwargs:
            return kwargs["agg_func"]
        elif "func_dict" in kwargs:
            return cls.inplace_applyier_builder(key, kwargs["func_dict"])
        else:
            return cls.inplace_applyier_builder(key)

    @classmethod
    def build_aggregate_method(cls, key):
        def fn(df, by, groupby_args, agg_args, axis=0, drop=False, **kwargs):
            by = cls.validate_by(by)
            groupby_args = groupby_args.copy()
            as_index = groupby_args.pop("as_index", True)
            groupby_args["as_index"] = True

            grp = df.groupby(by, axis=axis, **groupby_args)
            agg_func = cls.get_func(grp, key, **kwargs)
            result = agg_func(grp, **agg_args)

            if as_index:
                return result
            else:
                if result.index.name is None or result.index.name in result.columns:
                    drop = False
                return result.reset_index(drop=not drop)

        return fn

    @classmethod
    def build_groupby_reduce_method(cls, key):
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
            if not isinstance(by, (pandas.Series, pandas.DataFrame)):
                grp = df.groupby(by=by, axis=axis, **groupby_args)
                if callable(key):
                    agg_func = key
                else:
                    agg_func = cls.get_func(grp, key, **kwargs)
                return agg_func(grp, **map_args)
            # breakpoint()
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
            if callable(key):
                agg_func = key
            else:
                agg_func = cls.get_func(grp, key, **kwargs)
            result = agg_func(grp, **map_args)

            if not as_index:
                if (
                    len(result.index.names) == 1 and result.index.names[0] is None
                ) or all([name in result.columns for name in result.index.names]):
                    drop = False
                result = result.reset_index(drop=not drop)

            if result.index.name == "__reduced__":
                result.index.name = None

            return result

        return fn

    def __getattr__(self, key):
        if key == "aggregate":
            return self.build_aggregate_method(key)
        else:
            return self.build_groupby_reduce_method(key)


class GroupByDefault(DefaultMethod):
    methods_translator = {
        "agg": "aggregate",
        "dict_agg": "aggregate",
        pandas.core.groupby.DataFrameGroupBy.agg: "aggregate",
        pandas.core.groupby.DataFrameGroupBy.aggregate: "aggregate",
    }

    @classmethod
    def is_aggregate(cls, key):
        return key in cls.methods_translator

    @classmethod
    def register(cls, func, **kwargs):
        if callable(func) and not cls.is_aggregate(func):
            func = GroupBy.build_groupby_reduce_method(func)
        elif isinstance(func, str):
            func = re.findall(r"groupby_(.*)", func)[0]
        func = cls.methods_translator.get(func, func)
        return cls.call(func, obj_type=GroupBy(), **kwargs)

    @classmethod
    def build_wrapper(cls, fn, fn_name=None):
        wrapper = cls.build_default_to_pandas(fn)

        def args_cast(*args, **kwargs):
            if len(args) > 0:
                self = args[0]
                args = args[1:]
            else:
                self = kwargs.pop("query_compiler")

            args = try_cast_to_pandas(args)
            kwargs = try_cast_to_pandas(kwargs)
            return wrapper(self, *args, **kwargs)

        if fn_name is None:
            fn_name = fn.__name__
        if not isinstance(fn_name, str):
            fn_name = fn_name.__name__

        # setting proper function name that will be printed in default to pandas warning
        args_cast.__name__ = fn_name
        return args_cast
