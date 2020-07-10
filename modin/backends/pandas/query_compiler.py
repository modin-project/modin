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

import numpy as np
import pandas
from pandas.core.dtypes.common import (
    is_list_like,
    is_numeric_dtype,
    is_datetime_or_timedelta_dtype,
)
from pandas.core.base import DataError

from modin.backends.base.query_compiler import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.data_management.functions import (
    FoldFunction,
    MapFunction,
    MapReduceFunction,
    ReductionFunction,
    BinaryFunction,
    GroupbyReduceFunction,
)


def _get_axis(axis):
    if axis == 0:
        return lambda self: self._modin_frame.index
    else:
        return lambda self: self._modin_frame.columns


def _set_axis(axis):
    if axis == 0:

        def set_axis(self, idx):
            self._modin_frame.index = idx

    else:

        def set_axis(self, cols):
            self._modin_frame.columns = cols

    return set_axis


def _str_map(func_name):
    def str_op_builder(df, *args, **kwargs):
        str_s = df.squeeze(axis=1).str
        return getattr(pandas.Series.str, func_name)(str_s, *args, **kwargs).to_frame()

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
        prop_val = getattr(df.squeeze(axis=1).dt, property_name)
        if isinstance(prop_val, pandas.Series):
            return prop_val.to_frame()
        elif isinstance(prop_val, pandas.DataFrame):
            return prop_val
        else:
            return pandas.DataFrame([prop_val])

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
        dt_s = df.squeeze(axis=1).dt
        return pandas.DataFrame(
            getattr(pandas.Series.dt, func_name)(dt_s, *args, **kwargs)
        )

    return dt_op_builder


def copy_df_for_func(func):
    """
    Create a function that copies the dataframe, likely because `func` is inplace.

    Parameters
    ----------
    func : callable
        The function, usually updates a dataframe inplace.

    Returns
    -------
    callable
        A callable function to be applied in the partitions
    """

    def caller(df, *args, **kwargs):
        df = df.copy()
        func(df, *args, **kwargs)
        return df

    return caller


class PandasQueryCompiler(BaseQueryCompiler):
    """This class implements the logic necessary for operating on partitions
        with a Pandas backend. This logic is specific to Pandas."""

    def __init__(self, modin_frame):
        self._modin_frame = modin_frame

    def default_to_pandas(self, pandas_op, *args, **kwargs):
        """Default to pandas behavior.

        Parameters
        ----------
        pandas_op : callable
            The operation to apply, must be compatible pandas DataFrame call
        args
            The arguments for the `pandas_op`
        kwargs
            The keyword arguments for the `pandas_op`

        Returns
        -------
        PandasQueryCompiler
            The result of the `pandas_op`, converted back to PandasQueryCompiler

        Note
        ----
        This operation takes a distributed object and converts it directly to pandas.
        """
        ErrorMessage.default_to_pandas(str(pandas_op))
        args = (a.to_pandas() if isinstance(a, type(self)) else a for a in args)
        kwargs = {
            k: v.to_pandas if isinstance(v, type(self)) else v
            for k, v in kwargs.items()
        }

        result = pandas_op(self.to_pandas(), *args, **kwargs)
        if isinstance(result, pandas.Series):
            result = result.to_frame()
        if isinstance(result, pandas.DataFrame):
            return self.from_pandas(result, type(self._modin_frame))
        else:
            return result

    def to_pandas(self):
        return self._modin_frame.to_pandas()

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    @property
    def dtypes(self):
        return self._modin_frame.dtypes

    # END Index, columns, and dtypes objects

    # Metadata modification methods
    def add_prefix(self, prefix, axis=1):
        return self.__constructor__(self._modin_frame.add_prefix(prefix, axis))

    def add_suffix(self, suffix, axis=1):
        return self.__constructor__(self._modin_frame.add_suffix(suffix, axis))

    # END Metadata modification methods

    # Copy
    # For copy, we don't want a situation where we modify the metadata of the
    # copies if we end up modifying something here. We copy all of the metadata
    # to prevent that.
    def copy(self):
        return self.__constructor__(self._modin_frame.copy())

    # END Copy

    # Append/Concat/Join (Not Merge)
    # The append/concat/join operations should ideally never trigger remote
    # compute. These operations should only ever be manipulations of the
    # metadata of the resulting object. It should just be a simple matter of
    # appending the other object's blocks and adding np.nan columns for the new
    # columns, if needed. If new columns are added, some compute may be
    # required, though it can be delayed.
    #
    # Currently this computation is not delayed, and it may make a copy of the
    # DataFrame in memory. This can be problematic and should be fixed in the
    # future. TODO (devin-petersohn): Delay reindexing

    def concat(self, axis, other, **kwargs):
        """Concatenates two objects together.

        Args:
            axis: The axis index object to join (0 for columns, 1 for index).
            other: The other_index to concat with.

        Returns:
            Concatenated objects.
        """
        if not isinstance(other, list):
            other = [other]
        assert all(
            isinstance(o, type(self)) for o in other
        ), "Different Manager objects are being used. This is not allowed"
        sort = kwargs.get("sort", None)
        if sort is None:
            sort = False
        join = kwargs.get("join", "outer")
        ignore_index = kwargs.get("ignore_index", False)
        other_modin_frame = [o._modin_frame for o in other]
        new_modin_frame = self._modin_frame._concat(axis, other_modin_frame, join, sort)
        if ignore_index:
            new_modin_frame.index = pandas.RangeIndex(
                len(self.index) + sum(len(o.index) for o in other)
            )
        return self.__constructor__(new_modin_frame)

    # END Append/Concat/Join

    # Data Management Methods
    def free(self):
        """In the future, this will hopefully trigger a cleanup of this object.
        """
        # TODO create a way to clean up this object.
        return

    # END Data Management Methods

    # To NumPy
    def to_numpy(self):
        """Converts Modin DataFrame to NumPy array.

        Returns:
            NumPy array of the QueryCompiler.
        """
        arr = self._modin_frame.to_numpy()
        ErrorMessage.catch_bugs_and_request_email(
            len(arr) != len(self.index) or len(arr[0]) != len(self.columns)
        )
        return arr

    # END To NumPy

    # Binary operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.

    add = BinaryFunction.register(pandas.DataFrame.add)
    combine = BinaryFunction.register(pandas.DataFrame.combine)
    combine_first = BinaryFunction.register(pandas.DataFrame.combine_first)
    eq = BinaryFunction.register(pandas.DataFrame.eq)
    floordiv = BinaryFunction.register(pandas.DataFrame.floordiv)
    ge = BinaryFunction.register(pandas.DataFrame.ge)
    gt = BinaryFunction.register(pandas.DataFrame.gt)
    le = BinaryFunction.register(pandas.DataFrame.le)
    lt = BinaryFunction.register(pandas.DataFrame.lt)
    mod = BinaryFunction.register(pandas.DataFrame.mod)
    mul = BinaryFunction.register(pandas.DataFrame.mul)
    ne = BinaryFunction.register(pandas.DataFrame.ne)
    pow = BinaryFunction.register(pandas.DataFrame.pow)
    rfloordiv = BinaryFunction.register(pandas.DataFrame.rfloordiv)
    rmod = BinaryFunction.register(pandas.DataFrame.rmod)
    rpow = BinaryFunction.register(pandas.DataFrame.rpow)
    rsub = BinaryFunction.register(pandas.DataFrame.rsub)
    rtruediv = BinaryFunction.register(pandas.DataFrame.rtruediv)
    sub = BinaryFunction.register(pandas.DataFrame.sub)
    truediv = BinaryFunction.register(pandas.DataFrame.truediv)
    __and__ = BinaryFunction.register(pandas.DataFrame.__and__)
    __or__ = BinaryFunction.register(pandas.DataFrame.__or__)
    __rand__ = BinaryFunction.register(pandas.DataFrame.__rand__)
    __ror__ = BinaryFunction.register(pandas.DataFrame.__ror__)
    __rxor__ = BinaryFunction.register(pandas.DataFrame.__rxor__)
    __xor__ = BinaryFunction.register(pandas.DataFrame.__xor__)
    df_update = BinaryFunction.register(
        copy_df_for_func(pandas.DataFrame.update), join_type="left"
    )
    series_update = BinaryFunction.register(
        copy_df_for_func(
            lambda x, y: pandas.Series.update(x.squeeze(axis=1), y.squeeze(axis=1))
        ),
        join_type="left",
    )

    def where(self, cond, other, **kwargs):
        """Gets values from this manager where cond is true else from other.

        Args:
            cond: Condition on which to evaluate values.

        Returns:
            New QueryCompiler with updated data and index.
        """

        assert isinstance(
            cond, type(self)
        ), "Must have the same QueryCompiler subclass to perform this operation"
        if isinstance(other, type(self)):
            # Note: Currently we are doing this with two maps across the entire
            # data. This can be done with a single map, but it will take a
            # modification in the `BlockPartition` class.
            # If this were in one pass it would be ~2x faster.
            # TODO (devin-petersohn) rewrite this to take one pass.
            def where_builder_first_pass(cond, other, **kwargs):
                return cond.where(cond, other, **kwargs)

            first_pass = cond._modin_frame._binary_op(
                where_builder_first_pass, other._modin_frame, join_type="left"
            )

            def where_builder_second_pass(df, new_other, **kwargs):
                return df.where(new_other.eq(True), new_other, **kwargs)

            new_modin_frame = self._modin_frame._binary_op(
                where_builder_second_pass, first_pass, join_type="left"
            )
        # This will be a Series of scalars to be applied based on the condition
        # dataframe.
        else:

            def where_builder_series(df, cond):
                return df.where(cond, other, **kwargs)

            new_modin_frame = self._modin_frame._binary_op(
                where_builder_series, cond._modin_frame, join_type="left"
            )
        return self.__constructor__(new_modin_frame)

    def join(self, *args, **kwargs):
        """Database-style join with another object.

        Returns
        -------
        PandasQueryCompiler
            The joined PandasQueryCompiler

        Note
        ----
        This is not to be confused with `pandas.DataFrame.join` which does an
        index-level join.
        """
        return self.default_to_pandas(pandas.DataFrame.merge, *args, **kwargs)

    # END Inter-Data operations

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

    def reset_index(self, **kwargs):
        """Removes all levels from index and sets a default level_0 index.

        Returns:
            A new QueryCompiler with updated data and reset index.
        """
        drop = kwargs.get("drop", False)
        if not drop:
            new_column_name = (
                self.index.name
                if self.index.name is not None
                else "index"
                if "index" not in self.columns
                else "level_0"
            )
            new_self = self.insert(0, new_column_name, self.index)
        else:
            new_self = self.copy()
        new_self.index = pandas.RangeIndex(len(new_self.index))
        return new_self

    # END Reindex/reset_index

    # Transpose
    # For transpose, we aren't going to immediately copy everything. Since the
    # actual transpose operation is very fast, we will just do it before any
    # operation that gets called on the transposed data. See _prepare_method
    # for how the transpose is applied.
    #
    # Our invariants assume that the blocks are transposed, but not the
    # data inside. Sometimes we have to reverse this transposition of blocks
    # for simplicity of implementation.

    def transpose(self, *args, **kwargs):
        """Transposes this QueryCompiler.

        Returns:
            Transposed new QueryCompiler.
        """
        # Switch the index and columns and transpose the data within the blocks.
        return self.__constructor__(self._modin_frame.transpose())

    # END Transpose

    # MapReduce operations

    def _is_monotonic(self, type=None):
        funcs = {
            "increasing": lambda df: df.is_monotonic_increasing,
            "decreasing": lambda df: df.is_monotonic_decreasing,
        }

        monotonic_fn = funcs.get(type, funcs["increasing"])

        def is_monotonic_map(df):
            df = df.squeeze(axis=1)
            return [monotonic_fn(df), df.iloc[0], df.iloc[len(df) - 1]]

        def is_monotonic_reduce(df):
            df = df.squeeze(axis=1)

            common_case = df[0].all()
            left_edges = df[1]
            right_edges = df[2]

            edges_list = []
            for i in range(len(left_edges)):
                edges_list.extend([left_edges.iloc[i], right_edges.iloc[i]])

            edge_case = monotonic_fn(pandas.Series(edges_list))
            return [common_case and edge_case]

        return MapReduceFunction.register(is_monotonic_map, is_monotonic_reduce)(self)

    def is_monotonic_decreasing(self):
        return self._is_monotonic(type="decreasing")

    is_monotonic = _is_monotonic

    count = MapReduceFunction.register(pandas.DataFrame.count, pandas.DataFrame.sum)
    max = MapReduceFunction.register(pandas.DataFrame.max, pandas.DataFrame.max)
    min = MapReduceFunction.register(pandas.DataFrame.min, pandas.DataFrame.min)
    sum = MapReduceFunction.register(pandas.DataFrame.sum, pandas.DataFrame.sum)
    prod = MapReduceFunction.register(pandas.DataFrame.prod, pandas.DataFrame.prod)
    any = MapReduceFunction.register(pandas.DataFrame.any, pandas.DataFrame.any)
    all = MapReduceFunction.register(pandas.DataFrame.all, pandas.DataFrame.all)
    memory_usage = MapReduceFunction.register(
        pandas.DataFrame.memory_usage,
        lambda x, *args, **kwargs: pandas.DataFrame.sum(x),
        axis=0,
    )
    mean = MapReduceFunction.register(
        lambda df, **kwargs: df.apply(
            lambda x: (x.sum(skipna=kwargs.get("skipna", True)), x.count()),
            axis=kwargs.get("axis", 0),
        ),
        lambda df, **kwargs: df.apply(
            lambda x: x.apply(lambda d: d[0]).sum(skipna=kwargs.get("skipna", True))
            / x.apply(lambda d: d[1]).sum(skipna=kwargs.get("skipna", True)),
            axis=kwargs.get("axis", 0),
        ),
    )

    def value_counts(self, **kwargs):
        """
        Return a QueryCompiler of Series containing counts of unique values.

        Returns
        -------
        PandasQueryCompiler
        """
        if kwargs.get("bins", None) is not None:
            new_modin_frame = self._modin_frame._apply_full_axis(
                0, lambda df: df.squeeze(axis=1).value_counts(**kwargs)
            )
            return self.__constructor__(new_modin_frame)

        def map_func(df, *args, **kwargs):
            return df.squeeze(axis=1).value_counts(**kwargs)

        def reduce_func(df, *args, **kwargs):
            normalize = kwargs.get("normalize", False)
            sort = kwargs.get("sort", True)
            ascending = kwargs.get("ascending", False)
            dropna = kwargs.get("dropna", True)

            try:
                result = df.squeeze(axis=1).groupby(df.index, sort=False).sum()
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            except (ValueError):
                result = df.copy().squeeze(axis=1).groupby(df.index, sort=False).sum()

            if not dropna and np.nan in df.index:
                result = result.append(
                    pandas.Series(
                        [df.squeeze(axis=1).loc[[np.nan]].sum()], index=[np.nan]
                    )
                )
            if normalize:
                result = result / df.squeeze(axis=1).sum()

            result = result.sort_values(ascending=ascending) if sort else result

            # We want to sort both values and indices of the result object.
            # This function will sort indices for equal values.
            def sort_index_for_equal_values(result, ascending):
                """
                Sort indices for equal values of result object.

                Parameters
                ----------
                result : pandas.Series or pandas.DataFrame with one column
                    The object whose indices for equal values is needed to sort.
                ascending : boolean
                    Sort in ascending (if it is True) or descending (if it is False) order.

                Returns
                -------
                pandas.DataFrame
                    A new DataFrame with sorted indices.
                """
                is_range = False
                is_end = False
                i = 0
                new_index = np.empty(len(result), dtype=type(result.index))
                while i < len(result):
                    j = i
                    if i < len(result) - 1:
                        while result[result.index[i]] == result[result.index[i + 1]]:
                            i += 1
                            if is_range is False:
                                is_range = True
                            if i == len(result) - 1:
                                is_end = True
                                break
                    if is_range:
                        k = j
                        for val in sorted(
                            result.index[j : i + 1], reverse=not ascending
                        ):
                            new_index[k] = val
                            k += 1
                        if is_end:
                            break
                        is_range = False
                    else:
                        new_index[j] = result.index[j]
                    i += 1
                return pandas.DataFrame(result, index=new_index)

            return sort_index_for_equal_values(result, ascending)

        return MapReduceFunction.register(map_func, reduce_func, preserve_index=False)(
            self, **kwargs
        )

    # END MapReduce operations

    # Reduction operations
    idxmax = ReductionFunction.register(pandas.DataFrame.idxmax)
    idxmin = ReductionFunction.register(pandas.DataFrame.idxmin)
    median = ReductionFunction.register(pandas.DataFrame.median)
    nunique = ReductionFunction.register(pandas.DataFrame.nunique)
    nlargest = ReductionFunction.register(pandas.DataFrame.nlargest)
    skew = ReductionFunction.register(pandas.DataFrame.skew)
    kurt = ReductionFunction.register(pandas.DataFrame.kurt)
    std = ReductionFunction.register(pandas.DataFrame.std)
    var = ReductionFunction.register(pandas.DataFrame.var)
    sum_min_count = ReductionFunction.register(pandas.DataFrame.sum)
    prod_min_count = ReductionFunction.register(pandas.DataFrame.prod)
    quantile_for_single_value = ReductionFunction.register(pandas.DataFrame.quantile)
    mad = ReductionFunction.register(pandas.DataFrame.mad)
    to_datetime = ReductionFunction.register(
        lambda df, *args, **kwargs: pandas.to_datetime(
            df.squeeze(axis=1), *args, **kwargs
        ),
        axis=1,
    )

    # END Reduction operations

    # Map partitions operations
    # These operations are operations that apply a function to every partition.
    abs = MapFunction.register(pandas.DataFrame.abs, dtypes="copy")
    applymap = MapFunction.register(pandas.DataFrame.applymap)
    conj = MapFunction.register(
        lambda df, *args, **kwargs: pandas.DataFrame(np.conj(df))
    )
    invert = MapFunction.register(pandas.DataFrame.__invert__)
    isin = MapFunction.register(pandas.DataFrame.isin, dtypes=np.bool)
    isna = MapFunction.register(pandas.DataFrame.isna, dtypes=np.bool)
    negative = MapFunction.register(pandas.DataFrame.__neg__)
    notna = MapFunction.register(pandas.DataFrame.notna, dtypes=np.bool)
    round = MapFunction.register(pandas.DataFrame.round)
    series_view = MapFunction.register(
        lambda df, *args, **kwargs: pandas.DataFrame(
            df.squeeze(axis=1).view(*args, **kwargs)
        )
    )
    to_numeric = MapFunction.register(
        lambda df, *args, **kwargs: pandas.DataFrame(
            pandas.to_numeric(df.squeeze(axis=1), *args, **kwargs)
        )
    )

    def repeat(self, repeats):
        def map_fn(df):
            return pandas.DataFrame(df.squeeze(axis=1).repeat(repeats))

        if isinstance(repeats, int) or (is_list_like(repeats) and len(repeats) == 1):
            return MapFunction.register(map_fn, validate_index=True)(self)
        else:
            return self.__constructor__(self._modin_frame._apply_full_axis(0, map_fn))

    # END Map partitions operations

    # String map partitions operations

    str_capitalize = MapFunction.register(_str_map("capitalize"), dtypes="copy")
    str_center = MapFunction.register(_str_map("center"), dtypes="copy")
    str_contains = MapFunction.register(_str_map("contains"), dtypes=np.bool)
    str_count = MapFunction.register(_str_map("count"), dtypes=int)
    str_endswith = MapFunction.register(_str_map("endswith"), dtypes=np.bool)
    str_find = MapFunction.register(_str_map("find"), dtypes="copy")
    str_findall = MapFunction.register(_str_map("findall"), dtypes="copy")
    str_get = MapFunction.register(_str_map("get"), dtypes="copy")
    str_index = MapFunction.register(_str_map("index"), dtypes="copy")
    str_isalnum = MapFunction.register(_str_map("isalnum"), dtypes=np.bool)
    str_isalpha = MapFunction.register(_str_map("isalpha"), dtypes=np.bool)
    str_isdecimal = MapFunction.register(_str_map("isdecimal"), dtypes=np.bool)
    str_isdigit = MapFunction.register(_str_map("isdigit"), dtypes=np.bool)
    str_islower = MapFunction.register(_str_map("islower"), dtypes=np.bool)
    str_isnumeric = MapFunction.register(_str_map("isnumeric"), dtypes=np.bool)
    str_isspace = MapFunction.register(_str_map("isspace"), dtypes=np.bool)
    str_istitle = MapFunction.register(_str_map("istitle"), dtypes=np.bool)
    str_isupper = MapFunction.register(_str_map("isupper"), dtypes=np.bool)
    str_join = MapFunction.register(_str_map("join"), dtypes="copy")
    str_len = MapFunction.register(_str_map("len"), dtypes=int)
    str_ljust = MapFunction.register(_str_map("ljust"), dtypes="copy")
    str_lower = MapFunction.register(_str_map("lower"), dtypes="copy")
    str_lstrip = MapFunction.register(_str_map("lstrip"), dtypes="copy")
    str_match = MapFunction.register(_str_map("match"), dtypes="copy")
    str_normalize = MapFunction.register(_str_map("normalize"), dtypes="copy")
    str_pad = MapFunction.register(_str_map("pad"), dtypes="copy")
    str_partition = MapFunction.register(_str_map("partition"), dtypes="copy")
    str_repeat = MapFunction.register(_str_map("repeat"), dtypes="copy")
    str_replace = MapFunction.register(_str_map("replace"), dtypes="copy")
    str_rfind = MapFunction.register(_str_map("rfind"), dtypes="copy")
    str_rindex = MapFunction.register(_str_map("rindex"), dtypes="copy")
    str_rjust = MapFunction.register(_str_map("rjust"), dtypes="copy")
    str_rpartition = MapFunction.register(_str_map("rpartition"), dtypes="copy")
    str_rsplit = MapFunction.register(_str_map("rsplit"), dtypes="copy")
    str_rstrip = MapFunction.register(_str_map("rstrip"), dtypes="copy")
    str_slice = MapFunction.register(_str_map("slice"), dtypes="copy")
    str_slice_replace = MapFunction.register(_str_map("slice_replace"), dtypes="copy")
    str_split = MapFunction.register(_str_map("split"), dtypes="copy")
    str_startswith = MapFunction.register(_str_map("startswith"), dtypes=np.bool)
    str_strip = MapFunction.register(_str_map("strip"), dtypes="copy")
    str_swapcase = MapFunction.register(_str_map("swapcase"), dtypes="copy")
    str_title = MapFunction.register(_str_map("title"), dtypes="copy")
    str_translate = MapFunction.register(_str_map("translate"), dtypes="copy")
    str_upper = MapFunction.register(_str_map("upper"), dtypes="copy")
    str_wrap = MapFunction.register(_str_map("wrap"), dtypes="copy")
    str_zfill = MapFunction.register(_str_map("zfill"), dtypes="copy")

    # END String map partitions operations

    def unique(self):
        """Return unique values of Series object.

        Returns
        -------
        ndarray
            The unique values returned as a NumPy array.
        """
        new_modin_frame = self._modin_frame._apply_full_axis(
            0, lambda x: x.squeeze(axis=1).unique(), new_columns=self.columns,
        )
        return self.__constructor__(new_modin_frame)

    # Dt map partitions operations

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
    dt_tz = MapFunction.register(_dt_prop_map("tz"))
    dt_freq = MapFunction.register(_dt_prop_map("freq"))
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
    dt_total_seconds = MapFunction.register(_dt_func_map("total_seconds"))
    dt_seconds = MapFunction.register(_dt_prop_map("seconds"))
    dt_days = MapFunction.register(_dt_prop_map("days"))
    dt_microseconds = MapFunction.register(_dt_prop_map("microseconds"))
    dt_nanoseconds = MapFunction.register(_dt_prop_map("nanoseconds"))
    dt_components = MapFunction.register(_dt_prop_map("components"))
    dt_qyear = MapFunction.register(_dt_prop_map("qyear"))
    dt_start_time = MapFunction.register(_dt_prop_map("start_time"))
    dt_end_time = MapFunction.register(_dt_prop_map("end_time"))
    dt_to_timestamp = MapFunction.register(_dt_func_map("to_timestamp"))

    # END Dt map partitions operations

    def astype(self, col_dtypes, **kwargs):
        """Converts columns dtypes to given dtypes.

        Args:
            col_dtypes: Dictionary of {col: dtype,...} where col is the column
                name and dtype is a numpy dtype.

        Returns:
            DataFrame with updated dtypes.
        """
        return self.__constructor__(self._modin_frame.astype(col_dtypes))

    # Column/Row partitions reduce operations

    def first_valid_index(self):
        """Returns index of first non-NaN/NULL value.

        Return:
            Scalar of index name.
        """

        def first_valid_index_builder(df):
            return df.set_axis(
                pandas.RangeIndex(len(df.index)), axis="index", inplace=False
            ).apply(lambda df: df.first_valid_index())

        # We get the minimum from each column, then take the min of that to get
        # first_valid_index. The `to_pandas()` here is just for a single value and
        # `squeeze` will convert it to a scalar.
        first_result = (
            self.__constructor__(
                self._modin_frame._fold_reduce(0, first_valid_index_builder)
            )
            .min(axis=1)
            .to_pandas()
            .squeeze()
        )
        return self.index[first_result]

    def last_valid_index(self):
        """Returns index of last non-NaN/NULL value.

        Return:
            Scalar of index name.
        """

        def last_valid_index_builder(df):
            return df.set_axis(
                pandas.RangeIndex(len(df.index)), axis="index", inplace=False
            ).apply(lambda df: df.last_valid_index())

        # We get the maximum from each column, then take the max of that to get
        # last_valid_index. The `to_pandas()` here is just for a single value and
        # `squeeze` will convert it to a scalar.
        first_result = (
            self.__constructor__(
                self._modin_frame._fold_reduce(0, last_valid_index_builder)
            )
            .max(axis=1)
            .to_pandas()
            .squeeze()
        )
        return self.index[first_result]

    # END Column/Row partitions reduce operations

    # Column/Row partitions reduce operations over select indices
    #
    # These operations result in a reduced dimensionality of data.
    # This will return a new QueryCompiler object which the front end will handle.

    def describe(self, **kwargs):
        """Generates descriptive statistics.

        Returns:
            DataFrame object containing the descriptive statistics of the DataFrame.
        """
        # Use pandas to calculate the correct columns
        empty_df = (
            pandas.DataFrame(columns=self.columns)
            .astype(self.dtypes)
            .describe(**kwargs)
        )

        def describe_builder(df, internal_indices=[]):
            return df.iloc[:, internal_indices].describe(**kwargs)

        return self.__constructor__(
            self._modin_frame._apply_full_axis_select_indices(
                0,
                describe_builder,
                empty_df.columns,
                new_index=empty_df.index,
                new_columns=empty_df.columns,
            )
        )

    # END Column/Row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.

    cummax = FoldFunction.register(pandas.DataFrame.cummax)
    cummin = FoldFunction.register(pandas.DataFrame.cummin)
    cumsum = FoldFunction.register(pandas.DataFrame.cumsum)
    cumprod = FoldFunction.register(pandas.DataFrame.cumprod)
    diff = FoldFunction.register(pandas.DataFrame.diff)

    def clip(self, lower, upper, **kwargs):
        kwargs["upper"] = upper
        kwargs["lower"] = lower
        axis = kwargs.get("axis", 0)
        if is_list_like(lower) or is_list_like(upper):
            new_modin_frame = self._modin_frame._fold(
                axis, lambda df: df.clip(**kwargs)
            )
        else:
            new_modin_frame = self._modin_frame._map(lambda df: df.clip(**kwargs))
        return self.__constructor__(new_modin_frame)

    def dot(self, other, squeeze_self=None, squeeze_other=None):
        """
        Computes the matrix multiplication of self and other.

        Parameters
        ----------
            other : PandasQueryCompiler or NumPy array
                The other query compiler or NumPy array to matrix multiply with self.
            squeeze_self : boolean
                The flag to squeeze self.
            squeeze_other : boolean
                The flag to squeeze other (this flag is applied if other is query compiler).

        Returns
        -------
        PandasQueryCompiler
            A new query compiler that contains result of the matrix multiply.
        """
        if isinstance(other, PandasQueryCompiler):
            other = (
                other.to_pandas().squeeze(axis=1)
                if squeeze_other
                else other.to_pandas()
            )

        def map_func(df, other=other, squeeze_self=squeeze_self):
            result = df.squeeze(axis=1).dot(other) if squeeze_self else df.dot(other)
            if is_list_like(result):
                return pandas.DataFrame(result)
            else:
                return pandas.DataFrame([result])

        num_cols = other.shape[1] if len(other.shape) > 1 else 1
        if len(self.columns) == 1:
            new_index = (
                ["__reduced__"]
                if (len(self.index) == 1 or squeeze_self) and num_cols == 1
                else None
            )
            new_columns = ["__reduced__"] if squeeze_self and num_cols == 1 else None
            axis = 0
        else:
            new_index = self.index
            new_columns = ["__reduced__"] if num_cols == 1 else None
            axis = 1

        new_modin_frame = self._modin_frame._apply_full_axis(
            axis, map_func, new_index=new_index, new_columns=new_columns
        )
        return self.__constructor__(new_modin_frame)

    def nsmallest(self, n, columns=None, keep="first"):
        def map_func(df, n=n, keep=keep, columns=columns):
            if columns is None:
                return pandas.DataFrame(
                    pandas.Series.nsmallest(df.squeeze(axis=1), n=n, keep=keep)
                )
            return pandas.DataFrame.nsmallest(df, n=n, columns=columns, keep=keep)

        if columns is None:
            new_columns = ["__reduced__"]
        else:
            new_columns = self.columns

        new_modin_frame = self._modin_frame._apply_full_axis(
            axis=0, func=map_func, new_columns=new_columns
        )
        return self.__constructor__(new_modin_frame)

    def eval(self, expr, **kwargs):
        """Returns a new QueryCompiler with expr evaluated on columns.

        Args:
            expr: The string expression to evaluate.

        Returns:
            A new QueryCompiler with new columns after applying expr.
        """
        # Make a copy of columns and eval on the copy to determine if result type is
        # series or not
        empty_eval = (
            pandas.DataFrame(columns=self.columns)
            .astype(self.dtypes)
            .eval(expr, inplace=False, **kwargs)
        )
        if isinstance(empty_eval, pandas.Series):
            new_columns = (
                [empty_eval.name] if empty_eval.name is not None else ["__reduced__"]
            )
        else:
            new_columns = empty_eval.columns
        new_modin_frame = self._modin_frame._apply_full_axis(
            1,
            lambda df: pandas.DataFrame(df.eval(expr, inplace=False, **kwargs)),
            new_index=self.index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def mode(self, **kwargs):
        """Returns a new QueryCompiler with modes calculated for each label along given axis.

        Returns:
            A new QueryCompiler with modes calculated.
        """
        axis = kwargs.get("axis", 0)

        def mode_builder(df):
            result = pandas.DataFrame(df.mode(**kwargs))
            # We return a dataframe with the same shape as the input to ensure
            # that all the partitions will be the same shape
            if axis == 0 and len(df) != len(result):
                # Pad rows
                result = result.reindex(index=pandas.RangeIndex(len(df.index)))
            elif axis == 1 and len(df.columns) != len(result.columns):
                # Pad columns
                result = result.reindex(columns=pandas.RangeIndex(len(df.columns)))
            return pandas.DataFrame(result)

        if axis == 0:
            new_index = pandas.RangeIndex(len(self.index))
            new_columns = self.columns
        else:
            new_index = self.index
            new_columns = pandas.RangeIndex(len(self.columns))
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis, mode_builder, new_index=new_index, new_columns=new_columns
        )
        return self.__constructor__(new_modin_frame).dropna(axis=axis, how="all")

    def fillna(self, **kwargs):
        """Replaces NaN values with the method provided.

        Returns:
            A new QueryCompiler with null values filled.
        """
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

    def quantile_for_list_of_values(self, **kwargs):
        """Returns Manager containing quantiles along an axis for numeric columns.

        Returns:
            QueryCompiler containing quantiles of original QueryCompiler along an axis.
        """
        axis = kwargs.get("axis", 0)
        q = kwargs.get("q")
        numeric_only = kwargs.get("numeric_only", True)
        assert isinstance(q, (pandas.Series, np.ndarray, pandas.Index, list))

        if numeric_only:
            new_columns = self._modin_frame._numeric_columns()
        else:
            new_columns = [
                col
                for col, dtype in zip(self.columns, self.dtypes)
                if (is_numeric_dtype(dtype) or is_datetime_or_timedelta_dtype(dtype))
            ]
        if axis == 1:
            query_compiler = self.getitem_column_array(new_columns)
            new_columns = self.index
        else:
            query_compiler = self

        def quantile_builder(df, **kwargs):
            result = df.quantile(**kwargs)
            return result.T if kwargs.get("axis", 0) == 1 else result

        # This took a long time to debug, so here is the rundown of why this is needed.
        # Previously, we were operating on select indices, but that was broken. We were
        # not correctly setting the columns/index. Because of how we compute `to_pandas`
        # and because of the static nature of the index for `axis=1` it is easier to
        # just handle this as the transpose (see `quantile_builder` above for the
        # transpose within the partition) than it is to completely rework other
        # internal methods. Basically we are returning the transpose of the object for
        # correctness and cleanliness of the code.
        if axis == 1:
            q_index = new_columns
            new_columns = pandas.Float64Index(q)
        else:
            q_index = pandas.Float64Index(q)
        new_modin_frame = query_compiler._modin_frame._apply_full_axis(
            axis,
            lambda df: quantile_builder(df, **kwargs),
            new_index=q_index,
            new_columns=new_columns,
            dtypes=np.float64,
        )
        result = self.__constructor__(new_modin_frame)
        return result.transpose() if axis == 1 else result

    def query(self, expr, **kwargs):
        """Query columns of the QueryCompiler with a boolean expression.

        Args:
            expr: Boolean expression to query the columns with.

        Returns:
            QueryCompiler containing the rows where the boolean expression is satisfied.
        """

        def query_builder(df, **kwargs):
            return df.query(expr, inplace=False, **kwargs)

        return self.__constructor__(
            self._modin_frame.filter_full_axis(1, query_builder)
        )

    def rank(self, **kwargs):
        """Computes numerical rank along axis. Equal values are set to the average.

        Returns:
            QueryCompiler containing the ranks of the values along an axis.
        """
        axis = kwargs.get("axis", 0)
        numeric_only = True if axis else kwargs.get("numeric_only", False)
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis,
            lambda df: df.rank(**kwargs),
            new_index=self.index,
            new_columns=self.columns if not numeric_only else None,
            dtypes=np.float64,
        )
        return self.__constructor__(new_modin_frame)

    def sort_index(self, **kwargs):
        """Sorts the data with respect to either the columns or the indices.

        Returns:
            QueryCompiler containing the data sorted by columns or indices.
        """
        axis = kwargs.pop("axis", 0)
        # sort_index can have ascending be None and behaves as if it is False.
        # sort_values cannot have ascending be None. Thus, the following logic is to
        # convert the ascending argument to one that works with sort_values
        ascending = kwargs.pop("ascending", True)
        if ascending is None:
            ascending = False
        kwargs["ascending"] = ascending
        if axis:
            new_columns = pandas.Series(self.columns).sort_values(**kwargs)
            new_index = self.index
        else:
            new_index = pandas.Series(self.index).sort_values(**kwargs)
            new_columns = self.columns
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis,
            lambda df: df.sort_index(axis=axis, **kwargs),
            new_index,
            new_columns,
            dtypes="copy" if axis == 0 else None,
        )
        return self.__constructor__(new_modin_frame)

    # END Map across rows/columns

    # __getitem__ methods
    def getitem_column_array(self, key, numeric=False):
        """Get column data for target labels.

        Args:
            key: Target labels by which to retrieve data.
            numeric: A boolean representing whether or not the key passed in represents
                the numeric index or the named index.

        Returns:
            A new QueryCompiler.
        """
        # Convert to list for type checking
        if numeric:
            new_modin_frame = self._modin_frame.mask(col_numeric_idx=key)
        else:
            new_modin_frame = self._modin_frame.mask(col_indices=key)
        return self.__constructor__(new_modin_frame)

    def getitem_row_array(self, key):
        """Get row data for target labels.

        Args:
            key: Target numeric indices by which to retrieve data.

        Returns:
            A new QueryCompiler.
        """
        return self.__constructor__(self._modin_frame.mask(row_numeric_idx=key))

    def setitem(self, axis, key, value):
        """Set the column defined by `key` to the `value` provided.

        Args:
            key: The column name to set.
            value: The value to set the column to.

        Returns:
             A new QueryCompiler
        """

        def setitem_builder(df, internal_indices=[]):
            df = df.copy()
            if len(internal_indices) == 1:
                if axis == 0:
                    df[df.columns[internal_indices[0]]] = value
                else:
                    df.iloc[internal_indices[0]] = value
            else:
                if axis == 0:
                    df[df.columns[internal_indices]] = value
                else:
                    df.iloc[internal_indices] = value
            return df

        if isinstance(value, type(self)):
            value.columns = [key]
            if axis == 0:
                idx = self.columns.get_indexer_for([key])[0]
                if 0 < idx < len(self.columns) - 1:
                    first_mask = self._modin_frame.mask(
                        col_numeric_idx=list(range(idx))
                    )
                    second_mask = self._modin_frame.mask(
                        col_numeric_idx=list(range(idx + 1, len(self.columns)))
                    )
                    return self.__constructor__(
                        first_mask._concat(
                            1, [value._modin_frame, second_mask], "inner", False
                        )
                    )
                else:
                    mask = self.drop(columns=[key])._modin_frame
                    if idx == 0:
                        return self.__constructor__(
                            value._modin_frame._concat(1, [mask], "inner", False)
                        )
                    else:
                        return self.__constructor__(
                            mask._concat(1, [value._modin_frame], "inner", False)
                        )
            else:
                value = value.transpose()
                idx = self.index.get_indexer_for([key])[0]
                if 0 < idx < len(self.index) - 1:
                    first_mask = self._modin_frame.mask(
                        row_numeric_idx=list(range(idx))
                    )
                    second_mask = self._modin_frame.mask(
                        row_numeric_idx=list(range(idx + 1, len(self.index)))
                    )
                    return self.__constructor__(
                        first_mask._concat(
                            0, [value._modin_frame, second_mask], "inner", False
                        )
                    )
                else:
                    mask = self.drop(index=[key])._modin_frame
                    if idx == 0:
                        return self.__constructor__(
                            value._modin_frame._concat(0, [mask], "inner", False)
                        )
                    else:
                        return self.__constructor__(
                            mask._concat(0, [value._modin_frame], "inner", False)
                        )
        if is_list_like(value):
            new_modin_frame = self._modin_frame._apply_full_axis_select_indices(
                axis,
                setitem_builder,
                [key],
                new_index=self.index,
                new_columns=self.columns,
                keep_remaining=True,
            )
        else:
            new_modin_frame = self._modin_frame._apply_select_indices(
                axis,
                setitem_builder,
                [key],
                new_index=self.index,
                new_columns=self.columns,
                keep_remaining=True,
            )
        return self.__constructor__(new_modin_frame)

    # END __getitem__ methods

    # Drop/Dropna
    # This will change the shape of the resulting data.
    def dropna(self, **kwargs):
        """Returns a new QueryCompiler with null values dropped along given axis.

        Return:
            a new QueryCompiler
        """

        return self.__constructor__(
            self._modin_frame.filter_full_axis(
                kwargs.get("axis", 0) ^ 1,
                lambda df: pandas.DataFrame.dropna(df, **kwargs),
            )
        )

    def drop(self, index=None, columns=None):
        """Remove row data for target index and columns.

        Args:
            index: Target index to drop.
            columns: Target columns to drop.

        Returns:
            A new QueryCompiler.
        """
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

    # END Drop/Dropna

    # Insert
    # This method changes the shape of the resulting data. In Pandas, this
    # operation is always inplace, but this object is immutable, so we just
    # return a new one from here and let the front end handle the inplace
    # update.
    def insert(self, loc, column, value):
        """Insert new column data.

        Args:
            loc: Insertion index.
            column: Column labels to insert.
            value: Dtype object values to insert.

        Returns:
            A new PandasQueryCompiler with new data inserted.
        """
        if is_list_like(value):
            # TODO make work with another querycompiler object as `value`.
            # This will require aligning the indices with a `reindex` and ensuring that
            # the data is partitioned identically.
            if isinstance(value, pandas.Series):
                value = value.reindex(self.index)
            else:
                value = list(value)

        def insert(df, internal_indices=[]):
            internal_idx = int(internal_indices[0])
            df.insert(internal_idx, column, value)
            return df

        new_modin_frame = self._modin_frame._apply_full_axis_select_indices(
            0,
            insert,
            numeric_indices=[loc],
            keep_remaining=True,
            new_index=self.index,
            new_columns=self.columns.insert(loc, column),
        )
        return self.__constructor__(new_modin_frame)

    # END Insert

    # UDF (apply and agg) methods
    # There is a wide range of behaviors that are supported, so a lot of the
    # logic can get a bit convoluted.
    def apply(self, func, axis, *args, **kwargs):
        """Apply func across given axis.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
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
        """Apply func passed as str across given axis in elementwise manner.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        assert isinstance(func, str)
        kwargs["axis"] = axis
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis, lambda df: getattr(df, func)(**kwargs)
        )
        return self.__constructor__(new_modin_frame)

    def _dict_func(self, func, axis, *args, **kwargs):
        """Apply function to certain indices across given axis.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        if "axis" not in kwargs:
            kwargs["axis"] = axis

        def dict_apply_builder(df, func_dict={}):
            # Sometimes `apply` can return a `Series`, but we require that internally
            # all objects are `DataFrame`s.
            return pandas.DataFrame(df.apply(func_dict, *args, **kwargs))

        return self.__constructor__(
            self._modin_frame._apply_full_axis_select_indices(
                axis, dict_apply_builder, func, keep_remaining=False
            )
        )

    def _list_like_func(self, func, axis, *args, **kwargs):
        """Apply list-like function across given axis.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
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
        new_modin_frame = self._modin_frame._apply_full_axis(
            axis,
            lambda df: pandas.DataFrame(df.apply(func, axis, *args, **kwargs)),
            new_index=new_index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def _callable_func(self, func, axis, *args, **kwargs):
        """Apply callable functions across given axis.

        Args:
            func: The functions to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new PandasQueryCompiler.
        """
        if isinstance(pandas.DataFrame().apply(func), pandas.Series):
            new_modin_frame = self._modin_frame._fold_reduce(
                axis, lambda df: df.apply(func, axis=axis, *args, **kwargs)
            )
        else:
            new_modin_frame = self._modin_frame._apply_full_axis(
                axis, lambda df: df.apply(func, axis=axis, *args, **kwargs)
            )
        return self.__constructor__(new_modin_frame)

    # END UDF

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.

    groupby_count = GroupbyReduceFunction.register(
        lambda df, **kwargs: df.count(**kwargs), lambda df, **kwargs: df.sum(**kwargs)
    )
    groupby_any = GroupbyReduceFunction.register(
        lambda df, **kwargs: df.any(**kwargs), lambda df, **kwargs: df.any(**kwargs)
    )
    groupby_min = GroupbyReduceFunction.register(
        lambda df, **kwargs: df.min(**kwargs), lambda df, **kwargs: df.min(**kwargs)
    )
    groupby_prod = GroupbyReduceFunction.register(
        lambda df, **kwargs: df.prod(**kwargs), lambda df, **kwargs: df.prod(**kwargs)
    )
    groupby_max = GroupbyReduceFunction.register(
        lambda df, **kwargs: df.max(**kwargs), lambda df, **kwargs: df.max(**kwargs)
    )
    groupby_all = GroupbyReduceFunction.register(
        lambda df, **kwargs: df.all(**kwargs), lambda df, **kwargs: df.all(**kwargs)
    )
    groupby_sum = GroupbyReduceFunction.register(
        lambda df, **kwargs: df.sum(**kwargs), lambda df, **kwargs: df.sum(**kwargs)
    )
    groupby_size = GroupbyReduceFunction.register(
        lambda df, **kwargs: pandas.DataFrame(df.size()), lambda df, **kwargs: df.sum()
    )

    def groupby_agg(self, by, axis, agg_func, groupby_args, agg_args, drop=False):
        as_index = groupby_args.get("as_index", True)

        def groupby_agg_builder(df):
            # Set `as_index` to True to track the metadata of the grouping object
            # It is used to make sure that between phases we are constructing the
            # right index and placing columns in the correct order.
            groupby_args["as_index"] = True

            def compute_groupby(df):
                grouped_df = df.groupby(by=by, axis=axis, **groupby_args)
                try:
                    result = agg_func(grouped_df, **agg_args)
                # This happens when the partition is filled with non-numeric data and a
                # numeric operation is done. We need to build the index here to avoid
                # issues with extracting the index.
                except (DataError, TypeError):
                    result = pandas.DataFrame(index=grouped_df.size().index)
                return result

            try:
                return compute_groupby(df)
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            except (ValueError, KeyError):
                return compute_groupby(df.copy())

        new_modin_frame = self._modin_frame._apply_full_axis(
            axis, lambda df: groupby_agg_builder(df)
        )
        result = self.__constructor__(new_modin_frame)

        # that means that exception in `compute_groupby` was raised
        # in every partition, so we also should raise it
        if len(result.columns) == 0 and len(self.columns) != 0:
            # determening type of raised exception by applying `aggfunc`
            # to empty DataFrame
            try:
                agg_func(
                    pandas.DataFrame(index=[1], columns=[1]).groupby(level=0),
                    **agg_args
                )
            except Exception as e:
                raise type(e)("No numeric types to aggregate.")

        # Reset `as_index` because it was edited inplace.
        groupby_args["as_index"] = as_index
        if as_index:
            return result
        else:
            if result.index.name is None or result.index.name in result.columns:
                drop = False
            return result.reset_index(drop=not drop)

    # END Manual Partitioning methods

    # Get_dummies
    def get_dummies(self, columns, **kwargs):
        """Convert categorical variables to dummy variables for certain columns.

        Args:
            columns: The columns to convert.

        Returns:
            A new QueryCompiler.
        """
        # `columns` as None does not mean all columns, by default it means only
        # non-numeric columns.
        if columns is None:
            columns = [c for c in self.columns if not is_numeric_dtype(self.dtypes[c])]
            # If we aren't computing any dummies, there is no need for any
            # remote compute.
            if len(columns) == 0:
                return self.copy()
        elif not is_list_like(columns):
            columns = [columns]

        # In some cases, we are mapping across all of the data. It is more
        # efficient if we are mapping over all of the data to do it this way
        # than it would be to reuse the code for specific columns.
        if len(columns) == len(self.columns):
            new_modin_frame = self._modin_frame._apply_full_axis(
                0, lambda df: pandas.get_dummies(df, **kwargs), new_index=self.index
            )
            untouched_frame = None
        else:
            new_modin_frame = self._modin_frame.mask(
                col_indices=columns
            )._apply_full_axis(
                0, lambda df: pandas.get_dummies(df, **kwargs), new_index=self.index
            )
            untouched_frame = self.drop(columns=columns)
        # If we mapped over all the data we are done. If not, we need to
        # prepend the `new_modin_frame` with the raw data from the columns that were
        # not selected.
        if len(columns) != len(self.columns):
            new_modin_frame = untouched_frame._modin_frame._concat(
                1, [new_modin_frame], how="left", sort=False
            )
        return self.__constructor__(new_modin_frame)

    # END Get_dummies

    # Indexing
    def view(self, index=None, columns=None):
        return self.__constructor__(
            self._modin_frame.mask(row_numeric_idx=index, col_numeric_idx=columns)
        )

    def write_items(self, row_numeric_index, col_numeric_index, broadcasted_items):
        def iloc_mut(partition, row_internal_indices, col_internal_indices, item):
            partition = partition.copy()
            partition.iloc[row_internal_indices, col_internal_indices] = item
            return partition

        new_modin_frame = self._modin_frame._apply_select_indices(
            axis=None,
            func=iloc_mut,
            row_indices=row_numeric_index,
            col_indices=col_numeric_index,
            new_index=self.index,
            new_columns=self.columns,
            keep_remaining=True,
            item_to_distribute=broadcasted_items,
        )
        return self.__constructor__(new_modin_frame)
