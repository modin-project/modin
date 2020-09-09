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
import abc

from modin.data_management.functions.default_methods import (
    DataFrameDefault,
    SeriesDefault,
    DateTimeDefault,
    StrDefault,
    BinaryDefault,
    AnyDefault,
    ResampleDefault,
    RollingDefault,
    CatDefault,
    GroupByDefault,
)

from pandas.core.dtypes.common import is_scalar
import pandas
import numpy as np


def _get_axis(axis):
    def axis_getter(self):
        return self.to_pandas().axes[axis]

    return axis_getter


def _set_axis(axis):
    def axis_setter(self, labels):
        new_qc = DataFrameDefault.register("set_axis")(self, axis=axis, labels=labels)
        self.__dict__.update(new_qc.__dict__)

    return axis_setter


class BaseQueryCompiler(abc.ABC):
    """Abstract Class that handles the queries to Modin dataframes.

    Note: See the Abstract Methods and Fields section immediately below this
        for a list of requirements for subclassing this object.
    """

    @abc.abstractmethod
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
        BaseQueryCompiler
            The result of the `pandas_op`, converted back to BaseQueryCompiler
        """
        pass

    # Abstract Methods and Fields: Must implement in children classes
    # In some cases, there you may be able to use the same implementation for
    # some of these abstract methods, but for the sake of generality they are
    # treated differently.

    lazy_execution = False

    # Metadata modification abstract methods
    def add_prefix(self, prefix, axis=1):
        if axis:
            return DataFrameDefault.register("add_prefix")(self, prefix=prefix)
        else:
            return SeriesDefault.register("add_prefix")(self, prefix=prefix)

    def add_suffix(self, suffix, axis=1):
        if axis:
            return DataFrameDefault.register("add_suffix")(self, suffix=suffix)
        else:
            return SeriesDefault.register("add_suffix")(self, suffix=suffix)

    # END Metadata modification abstract methods

    # Abstract copy
    # For copy, we don't want a situation where we modify the metadata of the
    # copies if we end up modifying something here. We copy all of the metadata
    # to prevent that.
    def copy(self):
        return DataFrameDefault.register("copy")(self)

    # END Abstract copy

    # Abstract join and append helper functions

    def concat(self, axis, other, **kwargs):
        """Concatenates two objects together.

        Args:
            axis: The axis index object to join (0 for columns, 1 for index).
            other: The other_index to concat with.

        Returns:
            Concatenated objects.
        """
        concat_join = ["inner", "outer"]

        def concat(df, axis, other, **kwargs):
            kwargs.pop("join_axes", None)
            ignore_index = kwargs.get("ignore_index", False)
            if kwargs.get("join", "outer") in concat_join:
                if not isinstance(other, list):
                    other = [other]
                other = [df] + other
                result = pandas.concat(other, axis=axis, **kwargs)
            else:
                if isinstance(other, (list, np.ndarray)) and len(other) == 1:
                    other = other[0]
                how = kwargs.pop("join", None)
                ignore_index = kwargs.pop("ignore_index", None)
                kwargs["how"] = how
                result = df.join(other, **kwargs)
            if ignore_index:
                if axis == 0:
                    result = result.reset_index(drop=True)
                else:
                    result.columns = pandas.RangeIndex(len(result.columns))
            return result

        return DataFrameDefault.register(concat)(self, axis=axis, other=other, **kwargs)

    # END Abstract join and append helper functions

    # Data Management Methods
    @abc.abstractmethod
    def free(self):
        """In the future, this will hopefully trigger a cleanup of this object."""
        # TODO create a way to clean up this object.
        pass

    # END Data Management Methods

    # To/From Pandas
    @abc.abstractmethod
    def to_pandas(self):
        """Converts Modin DataFrame to Pandas DataFrame.

        Returns:
            Pandas DataFrame of the QueryCompiler.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_pandas(cls, df, data_cls):
        """Improve simple Pandas DataFrame to an advanced and superior Modin DataFrame.

        Parameters
        ----------
        df: pandas.DataFrame
            The pandas DataFrame to convert from.
        data_cls :
            Modin DataFrame object to convert to.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Pandas DataFrame.
        """
        pass

    # END To/From Pandas

    # From Arrow
    @classmethod
    @abc.abstractmethod
    def from_arrow(cls, at, data_cls):
        """Improve simple Arrow Table to an advanced and superior Modin DataFrame.

        Parameters
        ----------
        at : Arrow Table
            The Arrow Table to convert from.
        data_cls :
            Modin DataFrame object to convert to.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Pandas DataFrame.
        """
        pass

    # END From Arrow

    # To NumPy
    def to_numpy(self):
        """Converts Modin DataFrame to NumPy array.

        Returns:
            NumPy array of the QueryCompiler.
        """
        return DataFrameDefault.register("to_numpy")(self)

    # END To NumPy

    # Abstract inter-data operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.

    def add(self, other, **kwargs):
        return BinaryDefault.register("add")(self, other=other, **kwargs)

    def combine(self, other, **kwargs):
        return BinaryDefault.register("combine")(self, other=other, **kwargs)

    def combine_first(self, other, **kwargs):
        return BinaryDefault.register("combine_first")(self, other=other, **kwargs)

    def eq(self, other, **kwargs):
        return BinaryDefault.register("eq")(self, other=other, **kwargs)

    def floordiv(self, other, **kwargs):
        return BinaryDefault.register("floordiv")(self, other=other, **kwargs)

    def ge(self, other, **kwargs):
        return BinaryDefault.register("ge")(self, other=other, **kwargs)

    def gt(self, other, **kwargs):
        return BinaryDefault.register("gt")(self, other=other, **kwargs)

    def le(self, other, **kwargs):
        return BinaryDefault.register("le")(self, other=other, **kwargs)

    def lt(self, other, **kwargs):
        return BinaryDefault.register("lt")(self, other=other, **kwargs)

    def mod(self, other, **kwargs):
        return BinaryDefault.register("mod")(self, other=other, **kwargs)

    def mul(self, other, **kwargs):
        return BinaryDefault.register("mul")(self, other=other, **kwargs)

    def dot(self, other, **kwargs):
        return BinaryDefault.register("dot")(self, other=other, **kwargs)

    def ne(self, other, **kwargs):
        return BinaryDefault.register("ne")(self, other=other, **kwargs)

    def pow(self, other, **kwargs):
        return BinaryDefault.register("pow")(self, other=other, **kwargs)

    def rfloordiv(self, other, **kwargs):
        return BinaryDefault.register("rfloordiv")(self, other=other, **kwargs)

    def rmod(self, other, **kwargs):
        return BinaryDefault.register("rmod")(self, other=other, **kwargs)

    def rpow(self, other, **kwargs):
        return BinaryDefault.register("rpow")(self, other=other, **kwargs)

    def rsub(self, other, **kwargs):
        return BinaryDefault.register("rsub")(self, other=other, **kwargs)

    def rtruediv(self, other, **kwargs):
        return BinaryDefault.register("rtruediv")(self, other=other, **kwargs)

    def sub(self, other, **kwargs):
        return BinaryDefault.register("sub")(self, other=other, **kwargs)

    def truediv(self, other, **kwargs):
        return BinaryDefault.register("truediv")(self, other=other, **kwargs)

    def __and__(self, other, **kwargs):
        return BinaryDefault.register("__and__")(self, other=other, **kwargs)

    def __or__(self, other, **kwargs):
        return BinaryDefault.register("__or__")(self, other=other, **kwargs)

    def __rand__(self, other, **kwargs):
        return BinaryDefault.register("__rand__")(self, other=other, **kwargs)

    def __ror__(self, other, **kwargs):
        return BinaryDefault.register("__ror__")(self, other=other, **kwargs)

    def __rxor__(self, other, **kwargs):
        return BinaryDefault.register("__rxor__")(self, other=other, **kwargs)

    def __xor__(self, other, **kwargs):
        return BinaryDefault.register("__xor__")(self, other=other, **kwargs)

    def df_update(self, other, **kwargs):
        return BinaryDefault.register("update", inplace=True)(
            self, other=other, **kwargs
        )

    def series_update(self, other, **kwargs):
        return BinaryDefault.register("update", inplace=True)(
            self, other=other, squeeze_self=True, squeeze_other=True, **kwargs
        )

    def clip(self, lower, upper, **kwargs):
        return DataFrameDefault.register("clip")(
            self, lower=lower, upper=upper, **kwargs
        )

    def where(self, cond, other, **kwargs):
        """Gets values from this manager where cond is true else from other.

        Args:
            cond: Condition on which to evaluate values.

        Returns:
            New QueryCompiler with updated data and index.
        """
        return DataFrameDefault.register("where")(
            self, cond=cond, other=other, **kwargs
        )

    def merge(self, right, **kwargs):
        """
        Merge DataFrame or named Series objects with a database-style join.

        Parameters
        ----------
        right : BaseQueryCompiler
            The query compiler of the right DataFrame to merge with.

        Returns
        -------
        BaseQueryCompiler
            A new query compiler that contains result of the merge.

        Notes
        -----
        See pd.merge or pd.DataFrame.merge for more info on kwargs.
        """
        return DataFrameDefault.register("merge")(self, right=right, **kwargs)

    def join(self, right, **kwargs):
        """
        Join columns of another DataFrame.

        Parameters
        ----------
        right : BaseQueryCompiler
            The query compiler of the right DataFrame to join with.

        Returns
        -------
        BaseQueryCompiler
            A new query compiler that contains result of the join.

        Notes
        -----
        See pd.DataFrame.join for more info on kwargs.
        """
        return DataFrameDefault.register("join")(self, right, **kwargs)

    # END Abstract inter-data operations

    # Abstract Transpose
    def transpose(self, *args, **kwargs):
        """Transposes this QueryCompiler.

        Returns:
            Transposed new QueryCompiler.
        """
        return DataFrameDefault.register("transpose")(self, *args, **kwargs)

    def columnarize(self):
        """
        Transposes this QueryCompiler if it has a single row but multiple columns.

        This method should be called for QueryCompilers representing a Series object,
        i.e. self.is_series_like() should be True.

        Returns
        -------
        BaseQueryCompiler
            Transposed new QueryCompiler or self.
        """
        if len(self.columns) != 1 or (
            len(self.index) == 1 and self.index[0] == "__reduced__"
        ):
            return self.transpose()
        return self

    def is_series_like(self):
        """Return True if QueryCompiler has a single column or row"""
        return len(self.columns) == 1 or len(self.index) == 1

    # END Abstract Transpose

    # Abstract reindex/reset_index (may shuffle data)
    def reindex(self, axis, labels, **kwargs):
        """Fits a new index for this Manger.

        Args:
            axis: The axis index object to target the reindex on.
            labels: New labels to conform 'axis' on to.

        Returns:
            New QueryCompiler with updated data and new index.
        """
        return DataFrameDefault.register("reindex")(
            self, axis=axis, labels=labels, **kwargs
        )

    def reset_index(self, **kwargs):
        """Removes all levels from index and sets a default level_0 index.

        Returns:
            New QueryCompiler with updated data and reset index.
        """
        return DataFrameDefault.register("reset_index")(self, **kwargs)

    # END Abstract reindex/reset_index

    # Full Reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.

    def is_monotonic(self):
        """Return boolean if values in the object are monotonic_increasing.

        Returns
        -------
            bool
        """
        return SeriesDefault.register("is_monotonic")(self)

    def is_monotonic_decreasing(self):
        """Return boolean if values in the object are monotonic_decreasing.

        Returns
        -------
            bool
        """
        return SeriesDefault.register("is_monotonic_decreasing")(self)

    def count(self, **kwargs):
        """Counts the number of non-NaN objects for each column or row.

        Return:
            Pandas series containing counts of non-NaN objects from each column or row.
        """
        return DataFrameDefault.register("count")(self, **kwargs)

    def max(self, **kwargs):
        """Returns the maximum value for each column or row.

        Return:
            Pandas series with the maximum values from each column or row.
        """
        return DataFrameDefault.register("max")(self, **kwargs)

    def mean(self, **kwargs):
        """Returns the mean for each numerical column or row.

        Return:
            Pandas series containing the mean from each numerical column or row.
        """
        return DataFrameDefault.register("mean")(self, **kwargs)

    def min(self, **kwargs):
        """Returns the minimum from each column or row.

        Return:
            Pandas series with the minimum value from each column or row.
        """
        return DataFrameDefault.register("min")(self, **kwargs)

    def prod(self, **kwargs):
        """Returns the product of each numerical column or row.

        Return:
            Pandas series with the product of each numerical column or row.
        """
        return DataFrameDefault.register("prod")(self, **kwargs)

    def sum(self, **kwargs):
        """Returns the sum of each numerical column or row.

        Return:
            Pandas series with the sum of each numerical column or row.
        """
        return DataFrameDefault.register("sum")(self, **kwargs)

    def to_datetime(self, *args, **kwargs):
        return SeriesDefault.register("to_datetime", obj_type=pandas)(
            self, *args, **kwargs
        )

    # END Abstract full Reduce operations

    # Abstract map partitions operations
    # These operations are operations that apply a function to every partition.
    def abs(self):
        return DataFrameDefault.register("abs")(self)

    def applymap(self, func):
        return DataFrameDefault.register("applymap")(self, func=func)

    def conj(self, **kwargs):
        """
        Return the complex conjugate, element-wise.

        The complex conjugate of a complex number is obtained
        by changing the sign of its imaginary part.
        """

        def conj(df, *args, **kwargs):
            return pandas.DataFrame(np.conj(df))

        return DataFrameDefault.register(conj)(self, **kwargs)

    def isin(self, **kwargs):
        return DataFrameDefault.register("isin")(self, **kwargs)

    def isna(self):
        return DataFrameDefault.register("isna")(self)

    def negative(self, **kwargs):
        return DataFrameDefault.register("__neg__")(self, **kwargs)

    def notna(self):
        return DataFrameDefault.register("notna")(self)

    def round(self, **kwargs):
        return DataFrameDefault.register("round")(self, **kwargs)

    def replace(self, **kwargs):
        return DataFrameDefault.register("replace")(self, **kwargs)

    def series_view(self, **kwargs):
        return SeriesDefault.register("view")(self, **kwargs)

    def to_numeric(self, *args, **kwargs):
        return SeriesDefault.register("to_numeric", obj_type=pandas)(
            self, *args, **kwargs
        )

    def unique(self, **kwargs):
        return SeriesDefault.register("unique")(self, **kwargs)

    def searchsorted(self, **kwargs):
        return SeriesDefault.register("searchsorted")(self, **kwargs)

    # END Abstract map partitions operations

    def value_counts(self, **kwargs):
        return SeriesDefault.register("value_counts")(self, **kwargs)

    def stack(self, level, dropna):
        return DataFrameDefault.register("stack")(self, level=level, dropna=dropna)

    # Abstract map partitions across select indices
    def astype(self, col_dtypes, **kwargs):
        """Converts columns dtypes to given dtypes.

        Args:
            col_dtypes: Dictionary of {col: dtype,...} where col is the column
                name and dtype is a numpy dtype.

        Returns:
            DataFrame with updated dtypes.
        """
        return DataFrameDefault.register("astype")(self, dtype=col_dtypes, **kwargs)

    @property
    def dtypes(self):
        return self.to_pandas().dtypes

    # END Abstract map partitions across select indices

    # Abstract column/row partitions reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    def all(self, **kwargs):
        """Returns whether all the elements are true, potentially over an axis.

        Return:
            Pandas Series containing boolean values or boolean.
        """
        return DataFrameDefault.register("all")(self, **kwargs)

    def any(self, **kwargs):
        """Returns whether any the elements are true, potentially over an axis.

        Return:
            Pandas Series containing boolean values or boolean.
        """
        return DataFrameDefault.register("any")(self, **kwargs)

    def first_valid_index(self):
        """Returns index of first non-NaN/NULL value.

        Return:
            Scalar of index name.
        """
        return AnyDefault.register("first_valid_index")(self).to_pandas().squeeze()

    def idxmax(self, **kwargs):
        """Returns the first occurance of the maximum over requested axis.

        Returns:
            Series containing the maximum of each column or axis.
        """
        return DataFrameDefault.register("idxmax")(self, **kwargs)

    def idxmin(self, **kwargs):
        """Returns the first occurance of the minimum over requested axis.

        Returns:
            Series containing the minimum of each column or axis.
        """
        return DataFrameDefault.register("idxmin")(self, **kwargs)

    def last_valid_index(self):
        """Returns index of last non-NaN/NULL value.

        Return:
            Scalar of index name.
        """
        return AnyDefault.register("last_valid_index")(self).to_pandas().squeeze()

    def median(self, **kwargs):
        """Returns median of each column or row.

        Returns:
            Series containing the median of each column or row.
        """
        return DataFrameDefault.register("median")(self, **kwargs)

    def memory_usage(self, **kwargs):
        """Returns the memory usage of each column.

        Returns:
            Series containing the memory usage of each column.
        """
        return DataFrameDefault.register("memory_usage")(self, **kwargs)

    def nunique(self, **kwargs):
        """Returns the number of unique items over each column or row.

        Returns:
            Series of ints indexed by column or index names.
        """
        return DataFrameDefault.register("nunique")(self, **kwargs)

    def quantile_for_single_value(self, **kwargs):
        """Returns quantile of each column or row.

        Returns:
            Series containing the quantile of each column or row.
        """
        return DataFrameDefault.register("quantile")(self, **kwargs)

    def skew(self, **kwargs):
        """Returns skew of each column or row.

        Returns:
            Series containing the skew of each column or row.
        """
        return DataFrameDefault.register("skew")(self, **kwargs)

    def sem(self, **kwargs):
        """
        Returns standard deviation of the mean over requested axis.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing the standard deviation of the mean over requested axis.
        """
        return DataFrameDefault.register("sem")(self, **kwargs)

    def std(self, **kwargs):
        """Returns standard deviation of each column or row.

        Returns:
            Series containing the standard deviation of each column or row.
        """
        return DataFrameDefault.register("std")(self, **kwargs)

    def var(self, **kwargs):
        """Returns variance of each column or row.

        Returns:
            Series containing the variance of each column or row.
        """
        return DataFrameDefault.register("var")(self, **kwargs)

    # END Abstract column/row partitions reduce operations

    # Abstract column/row partitions reduce operations over select indices
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    def describe(self, **kwargs):
        """Generates descriptive statistics.

        Returns:
            DataFrame object containing the descriptive statistics of the DataFrame.
        """
        return DataFrameDefault.register("describe")(self, **kwargs)

    # END Abstract column/row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    def cumsum(self, **kwargs):
        return DataFrameDefault.register("cumsum")(self, **kwargs)

    def cummax(self, **kwargs):
        return DataFrameDefault.register("cummax")(self, **kwargs)

    def cummin(self, **kwargs):
        return DataFrameDefault.register("cummin")(self, **kwargs)

    def cumprod(self, **kwargs):
        return DataFrameDefault.register("cumprod")(self, **kwargs)

    def diff(self, **kwargs):
        return DataFrameDefault.register("diff")(self, **kwargs)

    def dropna(self, **kwargs):
        """Returns a new QueryCompiler with null values dropped along given axis.
        Return:
            New QueryCompiler
        """
        return DataFrameDefault.register("dropna")(self, **kwargs)

    def nlargest(self, n=5, columns=None, keep="first"):
        if columns is None:
            return SeriesDefault.register("nlargest")(self, n=n, keep=keep)
        else:
            return DataFrameDefault.register("nlargest")(
                self, n=n, columns=columns, keep=keep
            )

    def nsmallest(self, n=5, columns=None, keep="first"):
        if columns is None:
            return SeriesDefault.register("nsmallest")(self, n=n, keep=keep)
        else:
            return DataFrameDefault.register("nsmallest")(
                self, n=n, columns=columns, keep=keep
            )

    def eval(self, expr, **kwargs):
        """Returns a new QueryCompiler with expr evaluated on columns.

        Args:
            expr: The string expression to evaluate.

        Returns:
            A new QueryCompiler with new columns after applying expr.
        """
        return DataFrameDefault.register("eval")(self, expr=expr, **kwargs)

    def mode(self, **kwargs):
        """Returns a new QueryCompiler with modes calculated for each label along given axis.

        Returns:
            A new QueryCompiler with modes calculated.
        """
        return DataFrameDefault.register("mode")(self, **kwargs)

    def fillna(self, **kwargs):
        """Replaces NaN values with the method provided.

        Returns:
            A new QueryCompiler with null values filled.
        """
        return DataFrameDefault.register("fillna")(self, **kwargs)

    def query(self, expr, **kwargs):
        """Query columns of the QueryCompiler with a boolean expression.

        Args:
            expr: Boolean expression to query the columns with.

        Returns:
            QueryCompiler containing the rows where the boolean expression is satisfied.
        """
        return DataFrameDefault.register("query")(self, expr=expr, **kwargs)

    def rank(self, **kwargs):
        """Computes numerical rank along axis. Equal values are set to the average.

        Returns:
            QueryCompiler containing the ranks of the values along an axis.
        """
        return DataFrameDefault.register("rank")(self, **kwargs)

    def sort_index(self, **kwargs):
        """Sorts the data with respect to either the columns or the indices.

        Returns:
            QueryCompiler containing the data sorted by columns or indices.
        """
        return DataFrameDefault.register("sort_index")(self, **kwargs)

    def melt(self, *args, **kwargs):
        return DataFrameDefault.register("melt")(self, *args, **kwargs)

    def sort_columns_by_row_values(self, rows, ascending=True, **kwargs):
        return DataFrameDefault.register("sort_values")(
            self, by=rows, axis=1, ascending=ascending, **kwargs
        )

    def sort_rows_by_column_values(self, rows, ascending=True, **kwargs):
        return DataFrameDefault.register("sort_values")(
            self, by=rows, axis=0, ascending=ascending, **kwargs
        )

    # END Abstract map across rows/columns

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    def quantile_for_list_of_values(self, **kwargs):
        """Returns Manager containing quantiles along an axis for numeric columns.

        Returns:
            QueryCompiler containing quantiles of original QueryCompiler along an axis.
        """
        return DataFrameDefault.register("quantile")(self, **kwargs)

    # END Abstract map across rows/columns

    # Abstract __getitem__ methods
    def getitem_column_array(self, key, numeric=False):
        """Get column data for target labels.

        Args:
            key: Target labels by which to retrieve data.
            numeric: A boolean representing whether or not the key passed in represents
                the numeric index or the named index.

        Returns:
            A new Query Compiler.
        """

        def get_column(df, key):
            if numeric:
                return df.iloc[:, key]
            else:
                return df[key]

        return DataFrameDefault.register(get_column)(self, key=key)

    def getitem_row_array(self, key):
        """Get row data for target labels.

        Args:
            key: Target numeric indices by which to retrieve data.

        Returns:
            A new Query Compiler.
        """

        def get_row(df, key):
            return df.iloc[key]

        return DataFrameDefault.register(get_row)(self, key=key)

    # END Abstract __getitem__ methods

    # Abstract insert
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
            A new QueryCompiler with new data inserted.
        """
        return DataFrameDefault.register("insert", inplace=True)(
            self, loc=loc, column=column, value=value
        )

    # END Abstract insert

    # Abstract drop
    def drop(self, index=None, columns=None):
        """Remove row data for target index and columns.

        Args:
            index: Target index to drop.
            columns: Target columns to drop.

        Returns:
            A new QueryCompiler.
        """
        if index is None and columns is None:
            return self
        else:
            return DataFrameDefault.register("drop")(self, index=index, columns=columns)

    # END drop

    # UDF (apply and agg) methods
    # There is a wide range of behaviors that are supported, so a lot of the
    # logic can get a bit convoluted.
    def apply(self, func, axis, *args, **kwargs):
        """Apply func across given axis.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new QueryCompiler.
        """
        return DataFrameDefault.register("apply")(
            self, func=func, axis=axis, *args, **kwargs
        )

    # END UDF

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.

    def groupby_count(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        """Perform a groupby count.

        Parameters
        ----------
        by : BaseQueryCompiler
            The query compiler object to groupby.
        axis : 0 or 1
            The axis to groupby. Must be 0 currently.
        groupby_args : dict
            The arguments for the groupby component.
        map_args : dict
            The arguments for the `map_func`.
        reduce_args : dict
            The arguments for `reduce_func`.
        numeric_only : bool
            Whether to drop non-numeric columns.
        drop : bool
            Whether the data in `by` was dropped.

        Returns
        -------
        BaseQueryCompiler
        """
        return GroupByDefault.register("groupby_count")(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    def groupby_any(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        """Perform a groupby any.

        Parameters
        ----------
        by : BaseQueryCompiler
            The query compiler object to groupby.
        axis : 0 or 1
            The axis to groupby. Must be 0 currently.
        groupby_args : dict
            The arguments for the groupby component.
        map_args : dict
            The arguments for the `map_func`.
        reduce_args : dict
            The arguments for `reduce_func`.
        numeric_only : bool
            Whether to drop non-numeric columns.
        drop : bool
            Whether the data in `by` was dropped.

        Returns
        -------
        BaseQueryCompiler
        """
        return GroupByDefault.register("groupby_any")(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    def groupby_min(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        """Perform a groupby min.

        Parameters
        ----------
        by : BaseQueryCompiler
            The query compiler object to groupby.
        axis : 0 or 1
            The axis to groupby. Must be 0 currently.
        groupby_args : dict
            The arguments for the groupby component.
        map_args : dict
            The arguments for the `map_func`.
        reduce_args : dict
            The arguments for `reduce_func`.
        numeric_only : bool
            Whether to drop non-numeric columns.
        drop : bool
            Whether the data in `by` was dropped.

        Returns
        -------
        BaseQueryCompiler
        """
        return GroupByDefault.register("groupby_min")(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    def groupby_prod(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        """Perform a groupby prod.

        Parameters
        ----------
        by : BaseQueryCompiler
            The query compiler object to groupby.
        axis : 0 or 1
            The axis to groupby. Must be 0 currently.
        groupby_args : dict
            The arguments for the groupby component.
        map_args : dict
            The arguments for the `map_func`.
        reduce_args : dict
            The arguments for `reduce_func`.
        numeric_only : bool
            Whether to drop non-numeric columns.
        drop : bool
            Whether the data in `by` was dropped.

        Returns
        -------
        BaseQueryCompiler
        """
        return GroupByDefault.register("groupby_prod")(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    def groupby_max(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        """Perform a groupby max.

        Parameters
        ----------
        by : BaseQueryCompiler
            The query compiler object to groupby.
        axis : 0 or 1
            The axis to groupby. Must be 0 currently.
        groupby_args : dict
            The arguments for the groupby component.
        map_args : dict
            The arguments for the `map_func`.
        reduce_args : dict
            The arguments for `reduce_func`.
        numeric_only : bool
            Whether to drop non-numeric columns.
        drop : bool
            Whether the data in `by` was dropped.

        Returns
        -------
        BaseQueryCompiler
        """
        return GroupByDefault.register("groupby_max")(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    def groupby_all(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        """Perform a groupby all.

        Parameters
        ----------
        by : BaseQueryCompiler
            The query compiler object to groupby.
        axis : 0 or 1
            The axis to groupby. Must be 0 currently.
        groupby_args : dict
            The arguments for the groupby component.
        map_args : dict
            The arguments for the `map_func`.
        reduce_args : dict
            The arguments for `reduce_func`.
        numeric_only : bool
            Whether to drop non-numeric columns.
        drop : bool
            Whether the data in `by` was dropped.

        Returns
        -------
        BaseQueryCompiler
        """
        return GroupByDefault.register("groupby_all")(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    def groupby_sum(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        """Perform a groupby sum.

        Parameters
        ----------
        by : BaseQueryCompiler
            The query compiler object to groupby.
        axis : 0 or 1
            The axis to groupby. Must be 0 currently.
        groupby_args : dict
            The arguments for the groupby component.
        map_args : dict
            The arguments for the `map_func`.
        reduce_args : dict
            The arguments for `reduce_func`.
        numeric_only : bool
            Whether to drop non-numeric columns.
        drop : bool
            Whether the data in `by` was dropped.

        Returns
        -------
        BaseQueryCompiler
        """
        return GroupByDefault.register("groupby_sum")(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    def groupby_size(
        self,
        by,
        axis,
        groupby_args,
        map_args,
        reduce_args=None,
        numeric_only=True,
        drop=False,
    ):
        """Perform a groupby size.

        Parameters
        ----------
        by : BaseQueryCompiler
            The query compiler object to groupby.
        axis : 0 or 1
            The axis to groupby. Must be 0 currently.
        groupby_args : dict
            The arguments for the groupby component.
        map_args : dict
            The arguments for the `map_func`.
        reduce_args : dict
            The arguments for `reduce_func`.
        numeric_only : bool
            Whether to drop non-numeric columns.
        drop : bool
            Whether the data in `by` was dropped.

        Returns
        -------
        BaseQueryCompiler
        """
        return GroupByDefault.register("groupby_size")(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
        )

    def groupby_agg(self, by, axis, agg_func, groupby_args, agg_args, drop=False):
        return GroupByDefault.register("groupby_agg")(
            self,
            by=by,
            axis=axis,
            agg_func=agg_func,
            groupby_args=groupby_args,
            agg_args=agg_args,
            drop=drop,
        )

    def groupby_dict_agg(self, by, func_dict, groupby_args, agg_args, drop=False):
        return GroupByDefault.register("groupby_dict_agg")(
            self,
            by=by,
            func_dict=func_dict,
            groupby_args=groupby_args,
            agg_args=agg_args,
            drop=drop,
        )

    # END Manual Partitioning methods

    def unstack(self, level, fill_value):
        return DataFrameDefault.register("unstack")(
            self, level=level, fill_value=fill_value
        )

    def pivot(self, index, columns, values):
        return DataFrameDefault.register("pivot")(
            self, index=index, columns=columns, values=values
        )

    def get_dummies(self, columns, **kwargs):
        """Convert categorical variables to dummy variables for certain columns.

        Args:
            columns: The columns to convert.

        Returns:
            A new QueryCompiler.
        """

        def get_dummies(df, columns, **kwargs):
            return pandas.get_dummies(df, columns=columns, **kwargs)

        return DataFrameDefault.register(get_dummies)(self, columns=columns, **kwargs)

    def repeat(self, repeats):
        """
        Repeat elements of a Series.

        Returns a new Series where each element of the current Series
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Series.

        Returns
        -------
        Series
            Newly created Series with repeated elements.
        """
        return SeriesDefault.register("repeat")(self, repeats=repeats)

    # Indexing

    index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    def view(self, index=None, columns=None):
        index = [] if index is None else index
        columns = [] if columns is None else columns

        def applyier(df):
            return df.iloc[index, columns]

        return DataFrameDefault.register(applyier)(self)

    def setitem(self, axis, key, value):
        def setitem(df, axis, key, value):
            if is_scalar(key) and isinstance(value, pandas.DataFrame):
                value = value.squeeze()
            if not axis:
                df[key] = value
            else:
                df.loc[key] = value
            return df

        return DataFrameDefault.register(setitem)(self, axis=axis, key=key, value=value)

    def write_items(self, row_numeric_index, col_numeric_index, broadcasted_items):
        def write_items(df, broadcasted_items):
            if isinstance(df.iloc[row_numeric_index, col_numeric_index], pandas.Series):
                broadcasted_items = broadcasted_items.squeeze()
            df.iloc[
                list(row_numeric_index), list(col_numeric_index)
            ] = broadcasted_items
            return df

        return DataFrameDefault.register(write_items)(
            self, broadcasted_items=broadcasted_items
        )

    # END Abstract methods for QueryCompiler

    @property
    def __constructor__(self):
        """By default, constructor method will invoke an init."""
        return type(self)

    # __delitem__
    # This will change the shape of the resulting data.
    def delitem(self, key):
        return self.drop(columns=[key])

    # END __delitem__

    def has_multiindex(self, axis=0):
        """
        Check if specified axis is indexed by MultiIndex.

        Parameters
        ----------
        axis : 0 or 1, default 0
            The axis to check (0 - index, 1 - columns).

        Returns
        -------
        bool
            True if index at specified axis is MultiIndex and False otherwise.
        """
        if axis == 0:
            return isinstance(self.index, pandas.MultiIndex)
        assert axis == 1
        return isinstance(self.columns, pandas.MultiIndex)

    # DateTime methods

    dt_ceil = DateTimeDefault.register("dt_ceil")
    dt_components = DateTimeDefault.register("dt_components")
    dt_date = DateTimeDefault.register("dt_date")
    dt_day = DateTimeDefault.register("dt_day")
    dt_day_name = DateTimeDefault.register("dt_day_name")
    dt_dayofweek = DateTimeDefault.register("dt_dayofweek")
    dt_dayofyear = DateTimeDefault.register("dt_dayofyear")
    dt_days = DateTimeDefault.register("dt_days")
    dt_days_in_month = DateTimeDefault.register("dt_days_in_month")
    dt_daysinmonth = DateTimeDefault.register("dt_daysinmonth")
    dt_end_time = DateTimeDefault.register("dt_end_time")
    dt_floor = DateTimeDefault.register("dt_floor")
    dt_freq = DateTimeDefault.register("dt_freq")
    dt_hour = DateTimeDefault.register("dt_hour")
    dt_is_leap_year = DateTimeDefault.register("dt_is_leap_year")
    dt_is_month_end = DateTimeDefault.register("dt_is_month_end")
    dt_is_month_start = DateTimeDefault.register("dt_is_month_start")
    dt_is_quarter_end = DateTimeDefault.register("dt_is_quarter_end")
    dt_is_quarter_start = DateTimeDefault.register("dt_is_quarter_start")
    dt_is_year_end = DateTimeDefault.register("dt_is_year_end")
    dt_is_year_start = DateTimeDefault.register("dt_is_year_start")
    dt_microsecond = DateTimeDefault.register("dt_microsecond")
    dt_microseconds = DateTimeDefault.register("dt_microseconds")
    dt_minute = DateTimeDefault.register("dt_minute")
    dt_month = DateTimeDefault.register("dt_month")
    dt_month_name = DateTimeDefault.register("dt_month_name")
    dt_nanosecond = DateTimeDefault.register("dt_nanosecond")
    dt_nanoseconds = DateTimeDefault.register("dt_nanoseconds")
    dt_normalize = DateTimeDefault.register("dt_normalize")
    dt_quarter = DateTimeDefault.register("dt_quarter")
    dt_qyear = DateTimeDefault.register("dt_qyear")
    dt_round = DateTimeDefault.register("dt_round")
    dt_second = DateTimeDefault.register("dt_second")
    dt_seconds = DateTimeDefault.register("dt_seconds")
    dt_start_time = DateTimeDefault.register("dt_start_time")
    dt_strftime = DateTimeDefault.register("dt_strftime")
    dt_time = DateTimeDefault.register("dt_time")
    dt_timetz = DateTimeDefault.register("dt_timetz")
    dt_to_period = DateTimeDefault.register("dt_to_period")
    dt_to_pydatetime = DateTimeDefault.register("dt_to_pydatetime")
    dt_to_pytimedelta = DateTimeDefault.register("dt_to_pytimedelta")
    dt_to_timestamp = DateTimeDefault.register("dt_to_timestamp")
    dt_total_seconds = DateTimeDefault.register("dt_total_seconds")
    dt_tz = DateTimeDefault.register("dt_tz")
    dt_tz_convert = DateTimeDefault.register("dt_tz_convert")
    dt_tz_localize = DateTimeDefault.register("dt_tz_localize")
    dt_week = DateTimeDefault.register("dt_week")
    dt_weekday = DateTimeDefault.register("dt_weekday")
    dt_weekofyear = DateTimeDefault.register("dt_weekofyear")
    dt_year = DateTimeDefault.register("dt_year")

    # End of DateTime methods

    # Resample methods

    resample_agg_df = ResampleDefault.register("resample_agg_df")
    resample_agg_ser = ResampleDefault.register("resample_agg_ser")
    resample_app_df = ResampleDefault.register("resample_app_df")
    resample_app_ser = ResampleDefault.register("resample_app_ser")
    resample_asfreq = ResampleDefault.register("resample_asfreq")
    resample_backfill = ResampleDefault.register("resample_backfill")
    resample_bfill = ResampleDefault.register("resample_bfill")
    resample_count = ResampleDefault.register("resample_count")
    resample_ffill = ResampleDefault.register("resample_ffill")
    resample_fillna = ResampleDefault.register("resample_fillna")
    resample_first = ResampleDefault.register("resample_first")
    resample_get_group = ResampleDefault.register("resample_get_group")
    resample_interpolate = ResampleDefault.register("resample_interpolate")
    resample_last = ResampleDefault.register("resample_last")
    resample_max = ResampleDefault.register("resample_max")
    resample_mean = ResampleDefault.register("resample_mean")
    resample_median = ResampleDefault.register("resample_median")
    resample_min = ResampleDefault.register("resample_min")
    resample_nearest = ResampleDefault.register("resample_nearest")
    resample_nunique = ResampleDefault.register("resample_nunique")
    resample_ohlc_df = ResampleDefault.register("resample_ohlc_df")
    resample_ohlc_ser = ResampleDefault.register("resample_ohlc_ser")
    resample_pad = ResampleDefault.register("resample_pad")
    resample_pipe = ResampleDefault.register("resample_pipe")
    resample_prod = ResampleDefault.register("resample_prod")
    resample_quantile = ResampleDefault.register("resample_quantile")
    resample_sem = ResampleDefault.register("resample_sem")
    resample_size = ResampleDefault.register("resample_size")
    resample_std = ResampleDefault.register("resample_std")
    resample_sum = ResampleDefault.register("resample_sum")
    resample_transform = ResampleDefault.register("resample_transform")
    resample_var = ResampleDefault.register("resample_var")

    # End of Resample methods

    # Str methods

    str_capitalize = StrDefault.register("str_capitalize")
    str_center = StrDefault.register("str_center")
    str_contains = StrDefault.register("str_contains")
    str_count = StrDefault.register("str_count")
    str_endswith = StrDefault.register("str_endswith")
    str_find = StrDefault.register("str_find")
    str_findall = StrDefault.register("str_findall")
    str_get = StrDefault.register("str_get")
    str_index = StrDefault.register("str_index")
    str_isalnum = StrDefault.register("str_isalnum")
    str_isalpha = StrDefault.register("str_isalpha")
    str_isdecimal = StrDefault.register("str_isdecimal")
    str_isdigit = StrDefault.register("str_isdigit")
    str_islower = StrDefault.register("str_islower")
    str_isnumeric = StrDefault.register("str_isnumeric")
    str_isspace = StrDefault.register("str_isspace")
    str_istitle = StrDefault.register("str_istitle")
    str_isupper = StrDefault.register("str_isupper")
    str_join = StrDefault.register("str_join")
    str_len = StrDefault.register("str_len")
    str_ljust = StrDefault.register("str_ljust")
    str_lower = StrDefault.register("str_lower")
    str_lstrip = StrDefault.register("str_lstrip")
    str_match = StrDefault.register("str_match")
    str_normalize = StrDefault.register("str_normalize")
    str_pad = StrDefault.register("str_pad")
    str_partition = StrDefault.register("str_partition")
    str_repeat = StrDefault.register("str_repeat")
    str_replace = StrDefault.register("str_replace")
    str_rfind = StrDefault.register("str_rfind")
    str_rindex = StrDefault.register("str_rindex")
    str_rjust = StrDefault.register("str_rjust")
    str_rpartition = StrDefault.register("str_rpartition")
    str_rsplit = StrDefault.register("str_rsplit")
    str_rstrip = StrDefault.register("str_rstrip")
    str_slice = StrDefault.register("str_slice")
    str_slice_replace = StrDefault.register("str_slice_replace")
    str_split = StrDefault.register("str_split")
    str_startswith = StrDefault.register("str_startswith")
    str_strip = StrDefault.register("str_strip")
    str_swapcase = StrDefault.register("str_swapcase")
    str_title = StrDefault.register("str_title")
    str_translate = StrDefault.register("str_translate")
    str_upper = StrDefault.register("str_upper")
    str_wrap = StrDefault.register("str_wrap")
    str_zfill = StrDefault.register("str_zfill")

    # End of Str methods

    # Rolling methods

    rolling_aggregate = RollingDefault.register("rolling_aggregate")
    rolling_apply = RollingDefault.register("rolling_apply")
    rolling_corr = RollingDefault.register("rolling_corr")
    rolling_count = RollingDefault.register("rolling_count")
    rolling_cov = RollingDefault.register("rolling_cov")
    rolling_kurt = RollingDefault.register("rolling_kurt")
    rolling_max = RollingDefault.register("rolling_max")
    rolling_mean = RollingDefault.register("rolling_mean")
    rolling_median = RollingDefault.register("rolling_median")
    rolling_min = RollingDefault.register("rolling_min")
    rolling_quantile = RollingDefault.register("rolling_quantile")
    rolling_skew = RollingDefault.register("rolling_skew")
    rolling_std = RollingDefault.register("rolling_std")
    rolling_sum = RollingDefault.register("rolling_sum")
    rolling_var = RollingDefault.register("rolling_var")

    # End of Rolling methods

    # Window methods

    window_mean = RollingDefault.register("window_mean")
    window_std = RollingDefault.register("window_std")
    window_sum = RollingDefault.register("window_sum")
    window_var = RollingDefault.register("window_var")

    # End of Window methods

    # Categories methods

    cat_codes = CatDefault.register("cat_codes")

    # End of Categories methods

    # DataFrame methods

    invert = DataFrameDefault.register("__invert__")
    mad = DataFrameDefault.register("mad")
    kurt = DataFrameDefault.register("kurt")
    sum_min_count = DataFrameDefault.register("sum")
    prod_min_count = DataFrameDefault.register("prod")

    # End of DataFrame methods
