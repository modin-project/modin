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

from modin.backends.pandas.query_compiler import PandasQueryCompiler

import cudf
import numpy as np
import pandas
import cupy as cp

from modin.error_message import ErrorMessage
from modin.data_management.functions import (
    MapFunction,
    MapReduceFunction,
    ReductionFunction,
    BinaryFunction,
)
from pandas.core.dtypes.common import (
    is_list_like,
)

# from modin.pandas.utils import try_cast_to_pandas


def _str_map(func_name):
    """
    Create a function that call method of property `str` of the series.
    Parameters
    ----------
    func_name
        The method of `str`, which will be applied.
    Returns
    -------
        A callable function to be applied in the partitions
    Notes
    -----
    This applies callable methods of `Series.str`.
    """

    def str_op_builder(df, *args, **kwargs):
        str_s = df.iloc[:, 0].str  # TODO(apolakof): Find out how to change to squeeze.
        return getattr(cudf.core.column.string.StringMethods, func_name)(
            str_s, *args, **kwargs
        )

    return str_op_builder


def _dt_prop_map(property_name):
    """
    Create a function that call property of property `dt` of the series.

    Parameters
    ----------
    property_name
        The property of `dt`, which will be applied.

    Returns
    -------
        A callable function to be applied in the partitions

    Notes
    -----
    This applies non-callable properties of `Series.dt`.
    """

    def dt_op_builder(df, *args, **kwargs):
        prop_val = getattr(df.iloc[:, 0].dt, property_name)
        if isinstance(prop_val, cudf.Series):
            return prop_val.to_frame()
        elif isinstance(prop_val, cudf.DataFrame):
            return prop_val
        else:
            return cudf.DataFrame([prop_val])

    return dt_op_builder


def _dt_func_map(func_name):
    """
    Create a function that call method of property `dt` of the series.

    Parameters
    ----------
    func_name
        The method of `dt`, which will be applied.

    Returns
    -------
        A callable function to be applied in the partitions

    Notes
    -----
    This applies callable methods of `Series.dt`.
    """

    def dt_op_builder(df, *args, **kwargs):
        dt_s = df.iloc[:, 0].dt
        return cudf.DataFrame(getattr(cudf.Series.dt, func_name)(dt_s, *args, **kwargs))

    return dt_op_builder


# Use cupy streams for parallelism because cupy does not have support for DataFrame.apply.
# https://github.com/rapidsai/cudf/issues/925
def _build_apply_func(func, axis=0, reduce_func=None, *args, **kwargs):
    pass
    """
    if not reduce_func:
        reduce_func = lambda l: cudf.DataFrame(l)
    if axis == 0:

        def apply_wrapper(df):
            ncols = len(df.columns)
            res = []
            map_streams = []
            stop_events = []
            for i in range(ncols):
                map_streams.append(cp.cuda.stream.Stream())
            cupy_arr = df.values
            for i in range(ncols):
                stream = map_streams[i]
                with stream:
                    cupy_chunk = func(df.iloc[:, i], *args, **kwargs)
                    res.append(cupy_chunk)
                stop_event = stream.record()
                stop_events.append(stop_event)
            return reduce_func(res)

    else:

        def apply_wrapper(df):
            nrows = len(df.index)
            N = 1000
            res = []
            map_streams = []
            stop_events = []
            for i in range(N):
                map_streams.append(cp.cuda.stream.Stream())
            chunk_size = nrows // N
            for i in range(N):
                stream = map_streams[i]
                with stream:
                    cudf_series = df.iloc[i * chunk_size : (i + 1) * chunk_size]
                    res.append(func(cudf_series, *args, **kwargs))
                stop_event = stream.record()
                stop_events.append(stop_event)
            return reduce_func(res)

    return apply_wrapper
    """


class cuDFQueryCompiler(PandasQueryCompiler):
    def drop(self, index=None, columns=None):
        if index is not None:
            # The unique here is to avoid duplicating rows with the same name
            index = np.sort(
                self.index.get_indexer_for(self.index[~self.index.isin(index)].unique())
            )
        if columns is not None:
            # The unique here is to avoid duplicating columns with the same name
            columns = np.sort(
                self.columns.get_indexer_for(
                    self.columns[~self.columns.isin(columns)].unique()
                )
            )
        new_modin_frame = self._modin_frame.mask(
            row_numeric_idx=index, col_numeric_idx=columns
        )
        return self.__constructor__(new_modin_frame)

    def default_to_pandas(self, pandas_op, *args, **kwargs):
        ErrorMessage.default_to_pandas(str(pandas_op))
        args = (a.to_pandas() if isinstance(a, type(self)) else a for a in args)
        kwargs = {
            k: v.to_pandas if isinstance(v, type(self)) else v
            for k, v in kwargs.items()
        }

        result = pandas_op(self.to_pandas(), *args, **kwargs)
        if isinstance(result, pandas.Series):
            result = result.to_frame()
        print(f"result={result}, frame={self.to_pandas()}")  # DEBUG
        if isinstance(result, pandas.DataFrame):
            return self.from_pandas(result, type(self._modin_frame))
        else:
            return result

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.

    # def merge(self, right, **kwargs):
    #     return JoinFunction.register(cudf.DataFrame.merge)(self, right=right, **kwargs)

    # TODO(lepl3): Hacky solution meanwhile we decide whether or not to implement
    # from scratch pivot or cudf release pivot_table.
    def pivot_table(
        self,
        index,
        values,
        columns,
        aggfunc,
        fill_value,
        margins,
        dropna,
        margins_name,
        observed,
    ):
        by = [index] + columns
        by = self.getitem_column_array(by)
        groupby_args = {
            "level": None,
            "sort": True,
            "as_index": True,
            "group_keys": True,
            "squeeze": False,
        }
        map_args = {}
        reduce_args = {}
        if isinstance(aggfunc, list):
            raise NotImplementedError("Multiple aggfunc are not yet suppoted")

        try:
            groupby_func = getattr(self, f"groupby_{aggfunc}")
        except AttributeError:
            raise NotImplementedError(
                f"{aggfunc}: Not implemented for MODIN-GPU pivot_table"
            )

        temp_qc = groupby_func(
            by=by,
            axis=0,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=False,
            drop=True,
        )

        # TODO(lepl3): Avoid bringing data to CPU.
        if isinstance(values, str):
            values = [values]
        pandas_tmp = temp_qc.getitem_column_array(values).to_pandas()
        return self.from_pandas(
            pandas_tmp.unstack([*range(1, len(columns) + 1)]), type(self._modin_frame)
        )

    # THIS WILL COME WITH CUDF 0.16.0
    # def unstack(self, level=-1, fill_value=None)
    #     return MapFunction.register(cudf.DataFrame.unstack, dtypes='copy')
    # End

    invert = MapFunction.register(cudf.DataFrame.__invert__)
    eq = BinaryFunction.register(cudf.DataFrame.__eq__)
    floordiv = BinaryFunction.register(cudf.DataFrame.floordiv)
    ge = BinaryFunction.register(cudf.DataFrame.__ge__)
    gt = BinaryFunction.register(cudf.DataFrame.__gt__)
    le = BinaryFunction.register(cudf.DataFrame.__le__)
    lt = BinaryFunction.register(cudf.DataFrame.__lt__)
    add = BinaryFunction.register(cudf.DataFrame.add)
    mod = BinaryFunction.register(cudf.DataFrame.mod)
    mul = BinaryFunction.register(cudf.DataFrame.mul)
    ne = BinaryFunction.register(cudf.DataFrame.__ne__)
    pow = BinaryFunction.register(cudf.DataFrame.rpow)
    rfloordiv = BinaryFunction.register(cudf.DataFrame.rfloordiv)
    rmod = BinaryFunction.register(cudf.DataFrame.rmod)
    rpow = BinaryFunction.register(cudf.DataFrame.rpow)
    rsub = BinaryFunction.register(cudf.DataFrame.rsub)
    rtruediv = BinaryFunction.register(cudf.DataFrame.rtruediv)
    sub = BinaryFunction.register(cudf.DataFrame.sub)
    truediv = BinaryFunction.register(cudf.DataFrame.truediv)
    # TODO(lepl3): Investigate how cudf handles bool operator in differnt axis
    __and__ = BinaryFunction.register(cudf.DataFrame.__and__)
    __or__ = BinaryFunction.register(cudf.DataFrame.__or__)
    __rand__ = BinaryFunction.register(cudf.DataFrame.__rand__)
    __ror__ = BinaryFunction.register(cudf.DataFrame.__ror__)
    __rxor__ = BinaryFunction.register(cudf.DataFrame.__rxor__)
    __xor__ = BinaryFunction.register(cudf.DataFrame.__xor__)

    # TODO(kvu35): Figure out why PandasQueryCompiler requires two passes
    def where(self, cond, other, **kwargs):
        # CuDF does not support these arguments
        kwargs.pop("axis")
        kwargs.pop("level")
        if other is np.nan:
            other = None
        # This will be a Series of scalars to be applied based on the condition
        # dataframe.
        if isinstance(cond, type(self)) and len(self.columns) == 1:

            def where_builder_series(df, cond):
                # Convert both to series since where will only work with two series anyways
                df = df[df.columns[0]]
                cond = cond[cond.columns[0]]
                return df.where(cond, other, **kwargs).to_frame()

            new_modin_frame = self._modin_frame._binary_op(
                where_builder_series, cond._modin_frame, join_type="left"
            )
            return self.__constructor__(new_modin_frame)

    # Reindex/reset_index (may shuffle data)
    def reindex(self, axis, labels, **kwargs):
        """Fits a new index for this Manager.

        Args:
            axis: The axis index object to target the reindex on.
            labels: New labels to conform 'axis' on to.

        Returns:
            A new QueryCompiler with updated data and new index.
        """
        new_index = self.index if axis else labels
        new_columns = labels if axis else self.columns
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis,
            lambda df: df.reindex(labels=labels, axis=axis, **kwargs),
            new_index=new_index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def mean(self, **kwargs):
        # Note possible to do this all in one map reduce by creating a large dataframe with multiple
        # columns in it and concatinating
        def map_func(x):
            x = x.iloc[:, 0]
            return cudf.DataFrame({"N": [x.count()], "sum": [x.sum()]})

        def reduce_func(x):
            N = x["N"].sum()
            sum_of_x = x["sum"].sum()
            return cudf.Series([sum_of_x / N], name="mean")

        new_modin_frame = self._modin_frame._map_reduce(
            0,
            map_func,
            reduce_func,
        )
        return self.__constructor__(new_modin_frame)

    def to_datetime(self, **kwargs):
        unit = kwargs["unit"]
        return MapFunction.register(
            lambda df: cudf.to_datetime(
                df if len(df.columns) > 1 else df.iloc[:, 0], **kwargs
            ),
            # FIXME (kvu35): for some reason dtypes is just returning the default
            # dtype even though the backend dataframes are casted to the write dtype
            dtypes=np.dtype(f"datetime64[{unit}]"),
        )(self)

    ## Str Methods
    str_startswith = MapFunction.register(_str_map("startswith"), dtypes=np.bool)

    ## Dt Methods
    dt_date = MapFunction.register(_dt_prop_map("date"))
    dt_time = MapFunction.register(_dt_prop_map("time"))
    dt_timetz = MapFunction.register(_dt_prop_map("timetz"))
    dt_year = MapFunction.register(_dt_prop_map("year"))
    dt_month = MapFunction.register(_dt_prop_map("month"))
    dt_day = MapFunction.register(_dt_prop_map("day"))
    dt_hour = MapFunction.register(_dt_prop_map("hour"))
    dt_minute = MapFunction.register(_dt_prop_map("minute"))
    dt_second = MapFunction.register(_dt_prop_map("second"))
    dt_microsecond = MapFunction.register(_dt_prop_map("microsecond"))
    dt_nanosecond = MapFunction.register(_dt_prop_map("nanosecond"))
    dt_week = MapFunction.register(_dt_prop_map("week"))
    dt_weekofyear = MapFunction.register(_dt_prop_map("weekofyear"))
    dt_dayofweek = MapFunction.register(_dt_prop_map("dayofweek"))
    dt_weekday = MapFunction.register(_dt_prop_map("weekday"))
    dt_dayofyear = MapFunction.register(_dt_prop_map("dayofyear"))
    dt_quarter = MapFunction.register(_dt_prop_map("quarter"))
    dt_is_month_start = MapFunction.register(_dt_prop_map("is_month_start"))
    dt_is_month_end = MapFunction.register(_dt_prop_map("is_month_end"))
    dt_is_quarter_start = MapFunction.register(_dt_prop_map("is_quarter_start"))
    dt_is_quarter_end = MapFunction.register(_dt_prop_map("is_quarter_end"))
    dt_is_year_start = MapFunction.register(_dt_prop_map("is_year_start"))
    dt_is_year_end = MapFunction.register(_dt_prop_map("is_year_end"))
    dt_is_leap_year = MapFunction.register(_dt_prop_map("is_leap_year"))
    dt_daysinmonth = MapFunction.register(_dt_prop_map("daysinmonth"))
    dt_days_in_month = MapFunction.register(_dt_prop_map("days_in_month"))
    dt_tz = MapReduceFunction.register(
        _dt_prop_map("tz"), lambda df: cudf.DataFrame(df.iloc[0]), axis=0
    )
    dt_freq = MapReduceFunction.register(
        _dt_prop_map("freq"), lambda df: cudf.DataFrame(df.iloc[0]), axis=0
    )
    dt_to_period = MapFunction.register(_dt_func_map("to_period"))
    dt_to_pydatetime = MapFunction.register(_dt_func_map("to_pydatetime"))
    dt_tz_localize = MapFunction.register(_dt_func_map("tz_localize"))
    dt_tz_convert = MapFunction.register(_dt_func_map("tz_convert"))
    dt_normalize = MapFunction.register(_dt_func_map("normalize"))
    dt_strftime = MapFunction.register(_dt_func_map("strftime"))
    dt_round = MapFunction.register(_dt_func_map("round"))
    dt_floor = MapFunction.register(_dt_func_map("floor"))
    dt_ceil = MapFunction.register(_dt_func_map("ceil"))
    dt_month_name = MapFunction.register(_dt_func_map("month_name"))
    dt_day_name = MapFunction.register(_dt_func_map("day_name"))
    dt_to_pytimedelta = MapFunction.register(_dt_func_map("to_pytimedelta"))

    # cudf timedelta does not support total_seconds so here we convert to a pandas timedelta
    def dt_total_seconds(self):
        df = self.to_pandas().squeeze()
        return self.from_pandas(df.dt.total_seconds().to_frame(), self._modin_frame)

    dt_seconds = MapFunction.register(_dt_prop_map("seconds"))
    dt_days = MapFunction.register(_dt_prop_map("days"))
    dt_microseconds = MapFunction.register(_dt_prop_map("microseconds"))
    dt_nanoseconds = MapFunction.register(_dt_prop_map("nanoseconds"))
    dt_components = MapFunction.register(
        _dt_prop_map("components"), validate_columns=True
    )
    dt_qyear = MapFunction.register(_dt_prop_map("qyear"))
    dt_start_time = MapFunction.register(_dt_prop_map("start_time"))
    dt_end_time = MapFunction.register(_dt_prop_map("end_time"))
    dt_to_timestamp = MapFunction.register(_dt_func_map("to_timestamp"))

    def isin(self, values):
        if isinstance(values, pandas.Index):
            values = values.to_list()
        return MapFunction.register(lambda x: x.isin(values), dtypes=np.bool)(self)

    def unique(self, **kwargs):
        def unique_wrapper(df, **kwargs):
            if isinstance(df, cudf.DataFrame):
                df = df.iloc[:, 0]
            # cudf doesn't currently support cuda string arrays, so we must deserialize and
            # serialize the data using regular python lists
            if df.dtype == object:
                return cudf.DataFrame(list(df.unique().to_array()), dtype=object)
            else:
                return cudf.DataFrame(df.unique())

        new_modin_frame = self._modin_frame._map_reduce(0, unique_wrapper)
        return self.__constructor__(new_modin_frame)

    def nunique(self, **kwargs):
        return ReductionFunction.register(lambda x: cudf.Series([len(x)]))(
            self.unique()
        )

    def std(self, **kwargs):
        # Note possible to do this all in one map reduce by creating a large dataframe with multiple
        # columns in it and concatinating
        def map_func(x):
            x = x.iloc[:, 0]
            return cudf.DataFrame(
                {"N": [x.count()], "sum_of_X2": [(x ** 2).sum()], "sum": [x.sum()]}
            )

        def reduce_func(x):
            N = x["N"].sum()
            sum_of_X2 = x["sum_of_X2"].sum()
            X2_of_sum = x["sum"].sum() ** 2
            V = (sum_of_X2 - X2_of_sum / N) / (N - 1)
            return cudf.Series([np.sqrt(V)], name="std")

        new_modin_frame = self._modin_frame._map_reduce(
            0,
            map_func,
            reduce_func,
        )
        return self.__constructor__(new_modin_frame)

    # TODO: Only supports series right now. No support for args
    # Find the variance using the given formula V = (sum(X^2) - sum(X)^2/N)/(N-1)
    # Correct for degrees of freedom
    def var(self, **kwargs):
        # Note possible to do this all in one map reduce by creating a large dataframe with multiple
        # columns in it and concatinating
        def map_func(x):
            x = x.iloc[:, 0]
            return cudf.DataFrame(
                {"N": [x.count()], "sum_of_X2": [(x ** 2).sum()], "sum": [x.sum()]}
            )

        def reduce_func(x):
            N = x["N"].sum()
            sum_of_X2 = x["sum_of_X2"].sum()
            X2_of_sum = x["sum"].sum() ** 2
            V = (sum_of_X2 - X2_of_sum / N) / (N - 1)
            return cudf.Series([V], name="var")

        new_modin_frame = self._modin_frame._map_reduce(
            0,
            map_func,
            reduce_func,
        )
        return self.__constructor__(new_modin_frame)

    def fillna(self, **kwargs):
        kwargs.pop("downcast")
        axis = kwargs.get("axis", 0)
        value = kwargs.get("value")
        method = kwargs.get("method", None)
        limit = kwargs.get("limit", None)
        full_axis = method is not None or limit is not None
        if isinstance(value, dict):
            kwargs.pop("value")

            def fillna(df):
                func_dict = {c: value[c] for c in value if c in df.columns}
                return df.fillna(value=func_dict, **kwargs)

        else:

            def fillna(df):
                return df.fillna(**kwargs)

        if full_axis:
            new_modin_frame = self._modin_frame._fold(axis, fillna)
        else:
            new_modin_frame = self._modin_frame._map(fillna)
        return self.__constructor__(new_modin_frame)

    # END Abstract map across rows/columns

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    def quantile_for_list_of_values(self, **kwargs):
        pass

    # Indexing
    def view(self, index=None, columns=None):
        return self.__constructor__(
            self._modin_frame.mask(row_numeric_idx=index, col_numeric_idx=columns)
        )

    def sort_rows_by_column_values(self, by, ascending=True, **kwargs):
        kwargs.pop("inplace", None)
        kwargs.pop("kind", None)
        kwargs["ascending"] = ascending
        kwargs["by"] = by
        new_modin_frame = self._modin_frame._apply_full_axis(
            0,
            lambda df: df.sort_values(**kwargs),
            dtypes="copy",
            new_columns=self.columns,
        )
        return self.__constructor__(new_modin_frame)

    def sort_columns_by_row_values(self, by, ascending=True, **kwargs):
        kwargs.pop("inplace", None)
        kwargs.pop("kind", None)
        kwargs["ascending"] = ascending
        kwargs["by"] = by
        new_modin_frame = self._modin_frame._apply_full_axis(
            1,
            lambda df: df.sort_values(**kwargs).to_frame(),
            dtypes="copy",
            new_columns=self.columns,
        )
        return self.__constructor__(new_modin_frame)

    def reset_index(self, **kwargs):
        new_modin_frame = self._modin_frame._map(
            lambda x: x.reset_index(**kwargs),
            dtypes=None,
            validate_columns=True,
        )
        new_modin_frame.index = pandas.RangeIndex(len(self.index))
        return self.__constructor__(new_modin_frame)

    def apply(self, func, axis, *args, **kwargs):
        # if any of args contain modin object, we should
        # convert it to pandas
        # args = try_cast_to_pandas(args)
        # kwargs = try_cast_to_pandas(kwargs)
        # we have to build the apply function directly because cudf does not have support for
        # apply
        func = _build_apply_func(func, axis, *args, **kwargs)
        if isinstance(func, str):
            return self._apply_text_func_elementwise(func, axis, *args, **kwargs)
        elif callable(func):
            return self._callable_func(func, axis, *args, **kwargs)
        elif isinstance(func, dict):
            return self._dict_func(func, axis, *args, **kwargs)
        elif is_list_like(func):
            return self._list_like_func(func, axis, *args, **kwargs)
        else:
            pass

    def _apply_text_func_elementwise(self, func, axis, *args, **kwargs):
        assert isinstance(func, str)
        new_modin_frame = self._modin_frame._apply_full_axis(axis, func)
        return self.__constructor__(new_modin_frame)

    def _callable_func(self, func, axis, *args, **kwargs):
        # TODO (kvu35): Validate index
        new_modin_frame = self._modin_frame._apply_full_axis(axis, func)
        return self.__constructor__(new_modin_frame)

    def _dict_func(self, func, axis, *args, **kwargs):
        kwargs["func"] = func
        if "axis" not in kwargs:
            kwargs["axis"] = axis

        def dict_apply_builder(df, func_dict={}):
            # Sometimes `apply` can return a `Series`, but we require that internally
            # all objects are `DataFrame`s.
            return pandas.DataFrame(df.apply(func_dict, *args, **kwargs))

        func = {k: v if callable(v) else v for k, v in func.items()}
        return self.__constructor__(
            self._modin_frame._apply_full_axis_select_indices(
                axis, dict_apply_builder, func, keep_remaining=False
            )
        )

    def _list_like_func(self, func, axis, *args, **kwargs):
        kwargs["func"] = func
        # When the function is list-like, the function names become the index/columns
        new_index = (
            [f if isinstance(f, str) else f.__name__ for f in func]
            if axis == 0
            else self.index
        )
        new_columns = (
            [f if isinstance(f, str) else f.__name__ for f in func]
            if axis == 1
            else self.columns
        )
        func = [f if callable(f) else f for f in func]
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis,
            lambda df: pandas.DataFrame(df.apply(func, axis, *args, **kwargs)),
            new_index=new_index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def applymap(self, func, out_dtype=None):
        # TODO (kvu35): Add other unsupported dtypes into the if statement
        # CuDF does not have support for apply operations with columns that are string or
        # categorical, so we have to default to pandas.
        if self.dtypes.isin([np.dtype("O")]).any():
            return self.default_to_pandas(lambda df: df.applymap(func))

        def applymap_wrapper(df):
            for col in df.columns:
                df[col] = df[col].applymap(func, out_dtype)
            return df

        return MapFunction.register(applymap_wrapper)(self)

    # Perform the duplicated api.
    def duplicated(self):
        fused_frame = self._modin_frame._apply_full_axis(1, lambda x: x)
        uniques = self._modin_frame._apply_full_axis(1, lambda x: x.drop_duplicates())

        def make_boolean_indexor(df, upstream_df):
            if not upstream_df:
                n_upstream_vals = 0
                all_vals = df.reset_index(drop=True)
            else:
                n_upstream_vals = len(upstream_df.index)
                all_vals = cudf.concat([upstream_df, df], ignore_index=True)
            original_len = len(df.index)
            # Get a frame that contains the unique values of the current partition and every
            # partition before. The index objects contains the indicies of these values plus an
            # offset equal to n_upstream_vals.
            partition_uniques = all_vals.drop_duplicates()
            indicies_of_uniques = partition_uniques.index.values - n_upstream_vals
            # Curate away the first n_upstream_vals because these values do not belong in the current
            # partition. They are distributed amongst the upstream.
            indicies_of_uniques = indicies_of_uniques[indicies_of_uniques >= 0]
            indexor = cp.zeros(original_len, dtype=bool)
            indexor[indicies_of_uniques] = True
            return cudf.Series(indexor).to_frame()

        bool_indexor = fused_frame.trickle_down(make_boolean_indexor, uniques)
        return self.__constructor__(bool_indexor)

    def hash_values(self):
        new_modin_frame = self._modin_frame._apply_full_axis(
            1,
            lambda x: cudf.Series(x.hash_columns()),
            dtypes="copy",
        )
        return self.__constructor__(new_modin_frame)

    dropna = MapFunction.register(cudf.DataFrame.dropna, validate_index=True)
    replace = MapFunction.register(cudf.DataFrame.replace)

    def value_counts(self, **kwargs):
        """
        Return a QueryCompiler of Series containing counts of unique values.

        Returns
        -------
        cuDFQueryCompiler
        """
        if kwargs.get("bins", None) is not None:
            new_modin_frame = self._modin_frame._apply_full_axis(
                0, lambda df: df.iloc[:, 0].value_counts(**kwargs)
            )
            return self.__constructor__(new_modin_frame)

        def map_func(df, *args, **kwargs):
            return df.iloc[:, 0].value_counts(**kwargs).to_frame()

        def reduce_func(df, *args, **kwargs):
            normalize = kwargs.get("normalize", False)
            sort = kwargs.get("sort", True)
            ascending = kwargs.get("ascending", False)
            dropna = kwargs.get("dropna", True)

            # For pandas compliance we have to count the occurence of all categories even if they
            # don't appear in the array. In this case, they will just have a count of 0.
            if isinstance(df.index, cudf.core.index.CategoricalIndex):
                categories = df.index.categories.dropna().astype("category")
                zeros = cudf.DataFrame(cp.zeros(len(categories)), index=categories)
                df = cudf.concat([df, zeros])
            try:
                result = df.groupby(by=df.index, sort=False, dropna=dropna).sum()
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            except (ValueError):
                result = df.copy().groupby(by=df.index, sort=False, dropna=dropna).sum()

            result = result.iloc[:, 0]
            if normalize:
                result = result / result.sum()

            result = result.sort_values(ascending=ascending) if sort else result
            # TODO (kvu35): Figure out why PandasOnRayQueryCompiler sort the dataframe here
            return result

        return MapReduceFunction.register(
            map_func, reduce_func, axis=0, preserve_index=False
        )(self, **kwargs)

    def _resample_func(
        self, resample_args, func_name, new_columns=None, df_op=None, *args, **kwargs
    ):
        op = getattr(pandas.core.resample.Resampler, func_name)
        return self.default_to_pandas(
            lambda x, *args, **kwargs: op(x.resample(*resample_args), *args, **kwargs)
        )

    # No choice but to default to pandas here because CuDF only supports diff on numeric types. The
    # network security notebook requires a diff on a datetime series. If this line causes too much
    # latency, maybe we can see there is a workaround using cupy instead?
    def diff(self, *args, **kwargs):
        return self.default_to_pandas(pandas.DataFrame.diff, *args, **kwargs)

    # def __del__(self):
    #     self.free()
