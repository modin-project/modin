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
    ResampleDefault,
    RollingDefault,
    CatDefault,
    GroupByDefault,
)
from modin.error_message import ErrorMessage

from pandas.core.dtypes.common import is_scalar
import pandas.core.resample
import pandas
import numpy as np
from typing import List, Hashable


def _get_axis(axis):
    def axis_getter(self):
        ErrorMessage.default_to_pandas(f"DataFrame.get_axis({axis})")
        return self.to_pandas().axes[axis]

    return axis_getter


def _set_axis(axis):
    def axis_setter(self, labels):
        new_qc = DataFrameDefault.register(pandas.DataFrame.set_axis)(
            self, axis=axis, labels=labels
        )
        self.__dict__.update(new_qc.__dict__)

    return axis_setter


class BaseQueryCompiler(abc.ABC):
    """Abstract Class that handles the queries to Modin dataframes.

    Note: See the Abstract Methods and Fields section immediately below this
        for a list of requirements for subclassing this object.
    """

    @abc.abstractmethod
    def default_to_pandas(self, pandas_op, *args, **kwargs):
        """
        Default to pandas behavior.

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
            return DataFrameDefault.register(pandas.DataFrame.add_prefix)(
                self, prefix=prefix
            )
        else:
            return SeriesDefault.register(pandas.Series.add_prefix)(self, prefix=prefix)

    def add_suffix(self, suffix, axis=1):
        if axis:
            return DataFrameDefault.register(pandas.DataFrame.add_suffix)(
                self, suffix=suffix
            )
        else:
            return SeriesDefault.register(pandas.Series.add_suffix)(self, suffix=suffix)

    # END Metadata modification abstract methods

    # Abstract copy
    # For copy, we don't want a situation where we modify the metadata of the
    # copies if we end up modifying something here. We copy all of the metadata
    # to prevent that.
    def copy(self):
        return DataFrameDefault.register(pandas.DataFrame.copy)(self)

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
                ignore_index = kwargs.pop("ignore_index", None)
                kwargs["how"] = kwargs.pop("join", None)
                result = df.join(other, rsuffix="r_", **kwargs)
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

    @abc.abstractmethod
    def finalize(self):
        """Finalize constructing the dataframe calling all deferred functions which were used to build it."""
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

    def to_numpy(self, **kwargs):
        """
        Converts Modin DataFrame to NumPy array.

        Returns
        -------
            NumPy array of the QueryCompiler.
        """
        return DataFrameDefault.register(pandas.DataFrame.to_numpy)(self, **kwargs)

    # END To NumPy

    # Abstract inter-data operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.

    def add(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.add)(self, other=other, **kwargs)

    def combine(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.combine)(
            self, other=other, **kwargs
        )

    def combine_first(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.combine_first)(
            self, other=other, **kwargs
        )

    def eq(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.eq)(self, other=other, **kwargs)

    def floordiv(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.floordiv)(
            self, other=other, **kwargs
        )

    def ge(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.ge)(self, other=other, **kwargs)

    def gt(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.gt)(self, other=other, **kwargs)

    def le(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.le)(self, other=other, **kwargs)

    def lt(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.lt)(self, other=other, **kwargs)

    def mod(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.mod)(self, other=other, **kwargs)

    def mul(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.mul)(self, other=other, **kwargs)

    def corr(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.corr)(self, **kwargs)

    def cov(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.cov)(self, **kwargs)

    def dot(self, other, **kwargs):
        if kwargs.get("squeeze_self", False):
            applyier = pandas.Series.dot
        else:
            applyier = pandas.DataFrame.dot
        return BinaryDefault.register(applyier)(self, other=other, **kwargs)

    def ne(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.ne)(self, other=other, **kwargs)

    def pow(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.pow)(self, other=other, **kwargs)

    def rfloordiv(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.rfloordiv)(
            self, other=other, **kwargs
        )

    def rmod(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.rmod)(
            self, other=other, **kwargs
        )

    def rpow(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.rpow)(
            self, other=other, **kwargs
        )

    def rsub(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.rsub)(
            self, other=other, **kwargs
        )

    def rtruediv(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.rtruediv)(
            self, other=other, **kwargs
        )

    def sub(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.sub)(self, other=other, **kwargs)

    def truediv(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.truediv)(
            self, other=other, **kwargs
        )

    def __and__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__and__)(
            self, other=other, **kwargs
        )

    def __or__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__or__)(
            self, other=other, **kwargs
        )

    def __rand__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__rand__)(
            self, other=other, **kwargs
        )

    def __ror__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__ror__)(
            self, other=other, **kwargs
        )

    def __rxor__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__rxor__)(
            self, other=other, **kwargs
        )

    def __xor__(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.__xor__)(
            self, other=other, **kwargs
        )

    def df_update(self, other, **kwargs):
        return BinaryDefault.register(pandas.DataFrame.update, inplace=True)(
            self, other=other, **kwargs
        )

    def series_update(self, other, **kwargs):
        return BinaryDefault.register(pandas.Series.update, inplace=True)(
            self, other=other, squeeze_self=True, squeeze_other=True, **kwargs
        )

    def clip(self, lower, upper, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.clip)(
            self, lower=lower, upper=upper, **kwargs
        )

    def where(self, cond, other, **kwargs):
        """Gets values from this manager where cond is true else from other.

        Args:
            cond: Condition on which to evaluate values.

        Returns:
            New QueryCompiler with updated data and index.
        """
        return DataFrameDefault.register(pandas.DataFrame.where)(
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
        return DataFrameDefault.register(pandas.DataFrame.merge)(
            self, right=right, **kwargs
        )

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
        return DataFrameDefault.register(pandas.DataFrame.join)(self, right, **kwargs)

    # END Abstract inter-data operations

    # Abstract Transpose
    def transpose(self, *args, **kwargs):
        """Transposes this QueryCompiler.

        Returns:
            Transposed new QueryCompiler.
        """
        return DataFrameDefault.register(pandas.DataFrame.transpose)(
            self, *args, **kwargs
        )

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
        return DataFrameDefault.register(pandas.DataFrame.reindex)(
            self, axis=axis, labels=labels, **kwargs
        )

    def reset_index(self, **kwargs):
        """Removes all levels from index and sets a default level_0 index.

        Returns:
            New QueryCompiler with updated data and reset index.
        """
        return DataFrameDefault.register(pandas.DataFrame.reset_index)(self, **kwargs)

    def set_index_from_columns(
        self, keys: List[Hashable], drop: bool = True, append: bool = False
    ):
        """Create new row labels from a list of columns.

        Parameters
        ----------
        keys : list of hashable
            The list of column names that will become the new index.
        drop : boolean
            Whether or not to drop the columns provided in the `keys` argument.
        append : boolean
            Whether or not to add the columns in `keys` as new levels appended to the
            existing index.

        Returns
        -------
        PandasQueryCompiler
            A new QueryCompiler with updated index.
        """
        return DataFrameDefault.register(pandas.DataFrame.set_index)(
            self, keys=keys, drop=drop, append=append
        )

    # END Abstract reindex/reset_index

    # Full Reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.

    def is_monotonic_increasing(self):
        """Return boolean if values in the object are monotonic_increasing.

        Returns
        -------
            bool
        """
        return SeriesDefault.register(pandas.Series.is_monotonic_increasing)(self)

    def is_monotonic_decreasing(self):
        """Return boolean if values in the object are monotonic_decreasing.

        Returns
        -------
            bool
        """
        return SeriesDefault.register(pandas.Series.is_monotonic_decreasing)(self)

    def count(self, **kwargs):
        """Counts the number of non-NaN objects for each column or row.

        Return:
            Pandas series containing counts of non-NaN objects from each column or row.
        """
        return DataFrameDefault.register(pandas.DataFrame.count)(self, **kwargs)

    def max(self, **kwargs):
        """Returns the maximum value for each column or row.

        Return:
            Pandas series with the maximum values from each column or row.
        """
        return DataFrameDefault.register(pandas.DataFrame.max)(self, **kwargs)

    def mean(self, **kwargs):
        """Returns the mean for each numerical column or row.

        Return:
            Pandas series containing the mean from each numerical column or row.
        """
        return DataFrameDefault.register(pandas.DataFrame.mean)(self, **kwargs)

    def min(self, **kwargs):
        """Returns the minimum from each column or row.

        Return:
            Pandas series with the minimum value from each column or row.
        """
        return DataFrameDefault.register(pandas.DataFrame.min)(self, **kwargs)

    def prod(self, **kwargs):
        """Returns the product of each numerical column or row.

        Return:
            Pandas series with the product of each numerical column or row.
        """
        return DataFrameDefault.register(pandas.DataFrame.prod)(self, **kwargs)

    def sum(self, **kwargs):
        """Returns the sum of each numerical column or row.

        Return:
            Pandas series with the sum of each numerical column or row.
        """
        return DataFrameDefault.register(pandas.DataFrame.sum)(self, **kwargs)

    def to_datetime(self, *args, **kwargs):
        return SeriesDefault.register(pandas.to_datetime)(self, *args, **kwargs)

    # END Abstract full Reduce operations

    # Abstract map partitions operations
    # These operations are operations that apply a function to every partition.
    def abs(self):
        return DataFrameDefault.register(pandas.DataFrame.abs)(self)

    def applymap(self, func):
        return DataFrameDefault.register(pandas.DataFrame.applymap)(self, func=func)

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
        return DataFrameDefault.register(pandas.DataFrame.isin)(self, **kwargs)

    def isna(self):
        return DataFrameDefault.register(pandas.DataFrame.isna)(self)

    def negative(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.__neg__)(self, **kwargs)

    def notna(self):
        return DataFrameDefault.register(pandas.DataFrame.notna)(self)

    def round(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.round)(self, **kwargs)

    def replace(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.replace)(self, **kwargs)

    def series_view(self, **kwargs):
        return SeriesDefault.register(pandas.Series.view)(self, **kwargs)

    def to_numeric(self, *args, **kwargs):
        return SeriesDefault.register(pandas.to_numeric)(self, *args, **kwargs)

    def unique(self, **kwargs):
        return SeriesDefault.register(pandas.Series.unique)(self, **kwargs)

    def searchsorted(self, **kwargs):
        return SeriesDefault.register(pandas.Series.searchsorted)(self, **kwargs)

    # END Abstract map partitions operations

    def value_counts(self, **kwargs):
        return SeriesDefault.register(pandas.Series.value_counts)(self, **kwargs)

    def stack(self, level, dropna):
        return DataFrameDefault.register(pandas.DataFrame.stack)(
            self, level=level, dropna=dropna
        )

    # Abstract map partitions across select indices
    def astype(self, col_dtypes, **kwargs):
        """Converts columns dtypes to given dtypes.

        Args:
            col_dtypes: Dictionary of {col: dtype,...} where col is the column
                name and dtype is a numpy dtype.

        Returns:
            DataFrame with updated dtypes.
        """
        return DataFrameDefault.register(pandas.DataFrame.astype)(
            self, dtype=col_dtypes, **kwargs
        )

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
        return DataFrameDefault.register(pandas.DataFrame.all)(self, **kwargs)

    def any(self, **kwargs):
        """Returns whether any the elements are true, potentially over an axis.

        Return:
            Pandas Series containing boolean values or boolean.
        """
        return DataFrameDefault.register(pandas.DataFrame.any)(self, **kwargs)

    def first_valid_index(self):
        """Returns index of first non-NaN/NULL value.

        Return:
            Scalar of index name.
        """
        return (
            DataFrameDefault.register(pandas.DataFrame.first_valid_index)(self)
            .to_pandas()
            .squeeze()
        )

    def idxmax(self, **kwargs):
        """Returns the first occurance of the maximum over requested axis.

        Returns:
            Series containing the maximum of each column or axis.
        """
        return DataFrameDefault.register(pandas.DataFrame.idxmax)(self, **kwargs)

    def idxmin(self, **kwargs):
        """Returns the first occurance of the minimum over requested axis.

        Returns:
            Series containing the minimum of each column or axis.
        """
        return DataFrameDefault.register(pandas.DataFrame.idxmin)(self, **kwargs)

    def last_valid_index(self):
        """Returns index of last non-NaN/NULL value.

        Return:
            Scalar of index name.
        """
        return (
            DataFrameDefault.register(pandas.DataFrame.last_valid_index)(self)
            .to_pandas()
            .squeeze()
        )

    def median(self, **kwargs):
        """Returns median of each column or row.

        Returns:
            Series containing the median of each column or row.
        """
        return DataFrameDefault.register(pandas.DataFrame.median)(self, **kwargs)

    def memory_usage(self, **kwargs):
        """Returns the memory usage of each column.

        Returns:
            Series containing the memory usage of each column.
        """
        return DataFrameDefault.register(pandas.DataFrame.memory_usage)(self, **kwargs)

    def nunique(self, **kwargs):
        """Returns the number of unique items over each column or row.

        Returns:
            Series of ints indexed by column or index names.
        """
        return DataFrameDefault.register(pandas.DataFrame.nunique)(self, **kwargs)

    def quantile_for_single_value(self, **kwargs):
        """Returns quantile of each column or row.

        Returns:
            Series containing the quantile of each column or row.
        """
        return DataFrameDefault.register(pandas.DataFrame.quantile)(self, **kwargs)

    def skew(self, **kwargs):
        """Returns skew of each column or row.

        Returns:
            Series containing the skew of each column or row.
        """
        return DataFrameDefault.register(pandas.DataFrame.skew)(self, **kwargs)

    def sem(self, **kwargs):
        """
        Returns standard deviation of the mean over requested axis.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing the standard deviation of the mean over requested axis.
        """
        return DataFrameDefault.register(pandas.DataFrame.sem)(self, **kwargs)

    def std(self, **kwargs):
        """Returns standard deviation of each column or row.

        Returns:
            Series containing the standard deviation of each column or row.
        """
        return DataFrameDefault.register(pandas.DataFrame.std)(self, **kwargs)

    def var(self, **kwargs):
        """Returns variance of each column or row.

        Returns:
            Series containing the variance of each column or row.
        """
        return DataFrameDefault.register(pandas.DataFrame.var)(self, **kwargs)

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
        return DataFrameDefault.register(pandas.DataFrame.describe)(self, **kwargs)

    # END Abstract column/row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    def cumsum(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.cumsum)(self, **kwargs)

    def cummax(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.cummax)(self, **kwargs)

    def cummin(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.cummin)(self, **kwargs)

    def cumprod(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.cumprod)(self, **kwargs)

    def diff(self, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.diff)(self, **kwargs)

    def dropna(self, **kwargs):
        """Returns a new QueryCompiler with null values dropped along given axis.
        Return:
            New QueryCompiler
        """
        return DataFrameDefault.register(pandas.DataFrame.dropna)(self, **kwargs)

    def nlargest(self, n=5, columns=None, keep="first"):
        if columns is None:
            return SeriesDefault.register(pandas.Series.nlargest)(self, n=n, keep=keep)
        else:
            return DataFrameDefault.register(pandas.DataFrame.nlargest)(
                self, n=n, columns=columns, keep=keep
            )

    def nsmallest(self, n=5, columns=None, keep="first"):
        if columns is None:
            return SeriesDefault.register(pandas.Series.nsmallest)(self, n=n, keep=keep)
        else:
            return DataFrameDefault.register(pandas.DataFrame.nsmallest)(
                self, n=n, columns=columns, keep=keep
            )

    def eval(self, expr, **kwargs):
        """Returns a new QueryCompiler with expr evaluated on columns.

        Args:
            expr: The string expression to evaluate.

        Returns:
            A new QueryCompiler with new columns after applying expr.
        """
        return DataFrameDefault.register(pandas.DataFrame.eval)(
            self, expr=expr, **kwargs
        )

    def mode(self, **kwargs):
        """Returns a new QueryCompiler with modes calculated for each label along given axis.

        Returns:
            A new QueryCompiler with modes calculated.
        """
        return DataFrameDefault.register(pandas.DataFrame.mode)(self, **kwargs)

    def fillna(self, **kwargs):
        """Replaces NaN values with the method provided.

        Returns:
            A new QueryCompiler with null values filled.
        """
        return DataFrameDefault.register(pandas.DataFrame.fillna)(self, **kwargs)

    def query(self, expr, **kwargs):
        """Query columns of the QueryCompiler with a boolean expression.

        Args:
            expr: Boolean expression to query the columns with.

        Returns:
            QueryCompiler containing the rows where the boolean expression is satisfied.
        """
        return DataFrameDefault.register(pandas.DataFrame.query)(
            self, expr=expr, **kwargs
        )

    def rank(self, **kwargs):
        """Computes numerical rank along axis. Equal values are set to the average.

        Returns:
            QueryCompiler containing the ranks of the values along an axis.
        """
        return DataFrameDefault.register(pandas.DataFrame.rank)(self, **kwargs)

    def sort_index(self, **kwargs):
        """Sorts the data with respect to either the columns or the indices.

        Returns:
            QueryCompiler containing the data sorted by columns or indices.
        """
        return DataFrameDefault.register(pandas.DataFrame.sort_index)(self, **kwargs)

    def melt(self, *args, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.melt)(self, *args, **kwargs)

    def sort_columns_by_row_values(self, rows, ascending=True, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.sort_values)(
            self, by=rows, axis=1, ascending=ascending, **kwargs
        )

    def sort_rows_by_column_values(self, rows, ascending=True, **kwargs):
        return DataFrameDefault.register(pandas.DataFrame.sort_values)(
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
        return DataFrameDefault.register(pandas.DataFrame.quantile)(self, **kwargs)

    # END Abstract map across rows/columns

    # Abstract __getitem__ methods
    def getitem_array(self, key):
        """
        Get column or row data specified by key.
        Parameters
        ----------
        key : BaseQueryCompiler, numpy.ndarray, pandas.Index or list
            Target numeric indices or labels by which to retrieve data.
        Returns
        -------
        BaseQueryCompiler
            A new Query Compiler.
        """

        def getitem_array(df, key):
            return df[key]

        return DataFrameDefault.register(getitem_array)(self, key)

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
        return DataFrameDefault.register(pandas.DataFrame.insert, inplace=True)(
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
            return DataFrameDefault.register(pandas.DataFrame.drop)(
                self, index=index, columns=columns
            )

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
        return DataFrameDefault.register(pandas.DataFrame.apply)(
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
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.count)(
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
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.any)(
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
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.min)(
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
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.prod)(
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
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.max)(
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
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.all)(
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
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.sum)(
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
        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.size)(
            self,
            by=by,
            axis=axis,
            groupby_args=groupby_args,
            map_args=map_args,
            reduce_args=reduce_args,
            numeric_only=numeric_only,
            drop=drop,
            method="size",
        )

    def groupby_agg(
        self,
        by,
        is_multi_by,
        axis,
        agg_func,
        agg_args,
        agg_kwargs,
        groupby_kwargs,
        drop=False,
    ):
        if isinstance(by, type(self)) and len(by.columns) == 1:
            by = by.columns[0] if drop else by.to_pandas().squeeze()
        elif isinstance(by, type(self)):
            by = list(by.columns)

        return GroupByDefault.register(pandas.core.groupby.DataFrameGroupBy.aggregate)(
            self,
            by=by,
            is_multi_by=is_multi_by,
            axis=axis,
            agg_func=agg_func,
            groupby_args=groupby_kwargs,
            agg_args=agg_kwargs,
            drop=drop,
        )

    # END Manual Partitioning methods

    def unstack(self, level, fill_value):
        return DataFrameDefault.register(pandas.DataFrame.unstack)(
            self, level=level, fill_value=fill_value
        )

    def pivot(self, index, columns, values):
        return DataFrameDefault.register(pandas.DataFrame.pivot)(
            self, index=index, columns=columns, values=values
        )

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
        return DataFrameDefault.register(pandas.DataFrame.pivot_table)(
            self,
            index=index,
            values=values,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=fill_value,
            margins=margins,
            dropna=dropna,
            margins_name=margins_name,
            observed=observed,
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
        return SeriesDefault.register(pandas.Series.repeat)(self, repeats=repeats)

    # Indexing

    index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    def get_axis(self, axis):
        """
        Return index labels of the specified axis.

        Parameters
        ----------
        axis: int,
            Axis to return labels on.

        Returns
        -------
            Index
        """
        return self.index if axis == 0 else self.columns

    def view(self, index=None, columns=None):
        index = [] if index is None else index
        columns = [] if columns is None else columns

        def applyier(df):
            return df.iloc[index, columns]

        return DataFrameDefault.register(applyier)(self)

    def insert_item(self, axis, loc, value, how="inner", replace=False):
        """
        Insert new column/row defined by `value` at the specified `loc`

        Parameters
        ----------
        axis: int, axis to insert along
        loc: int, position to insert `value`
        value: BaseQueryCompiler, value to insert
        how : str,
            The type of join to join to make.
        replace: bool (default False),
            Whether to insert item at `loc` or to replace item at `loc`.

        Returns
        -------
            A new BaseQueryCompiler
        """
        assert isinstance(value, type(self))

        def mask(idx):
            if len(idx) == len(self.get_axis(axis)):
                return self
            return (
                self.getitem_column_array(idx, numeric=True)
                if axis
                else self.getitem_row_array(idx)
            )

        if 0 <= loc < len(self.get_axis(axis)):
            first_mask = mask(list(range(loc)))
            second_mask_loc = loc + 1 if replace else loc
            second_mask = mask(list(range(second_mask_loc, len(self.get_axis(axis)))))
            return first_mask.concat(axis, [value, second_mask], join=how, sort=False)
        else:
            return self.concat(axis, [value], join=how, sort=False)

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

    def get_index_name(self, axis=0):
        """
        Get index name of specified axis.

        Parameters
        ----------
        axis: int (default 0),
            Axis to return index name on.

        Returns
        -------
        hashable
            Index name, None for MultiIndex.
        """
        return self.get_axis(axis).name

    def set_index_name(self, name, axis=0):
        """
        Set index name for the specified axis.

        Parameters
        ----------
        name: hashable,
            New index name.
        axis: int (default 0),
            Axis to set name along.
        """
        self.get_axis(axis).name = name

    def get_index_names(self, axis=0):
        """
        Get index names of specified axis.

        Parameters
        ----------
        axis: int (default 0),
            Axis to return index names on.

        Returns
        -------
        list
            Index names.
        """
        return self.get_axis(axis).names

    def set_index_names(self, names, axis=0):
        """
        Set index names for the specified axis.

        Parameters
        ----------
        names: list,
            New index names.
        axis: int (default 0),
            Axis to set names along.
        """
        self.get_axis(axis).names = names

    # DateTime methods

    dt_ceil = DateTimeDefault.register(pandas.Series.dt.ceil)
    dt_components = DateTimeDefault.register(pandas.Series.dt.components)
    dt_date = DateTimeDefault.register(pandas.Series.dt.date)
    dt_day = DateTimeDefault.register(pandas.Series.dt.day)
    dt_day_name = DateTimeDefault.register(pandas.Series.dt.day_name)
    dt_dayofweek = DateTimeDefault.register(pandas.Series.dt.dayofweek)
    dt_dayofyear = DateTimeDefault.register(pandas.Series.dt.dayofyear)
    dt_days = DateTimeDefault.register(pandas.Series.dt.days)
    dt_days_in_month = DateTimeDefault.register(pandas.Series.dt.days_in_month)
    dt_daysinmonth = DateTimeDefault.register(pandas.Series.dt.daysinmonth)
    dt_end_time = DateTimeDefault.register(pandas.Series.dt.end_time)
    dt_floor = DateTimeDefault.register(pandas.Series.dt.floor)
    dt_freq = DateTimeDefault.register(pandas.Series.dt.freq)
    dt_hour = DateTimeDefault.register(pandas.Series.dt.hour)
    dt_is_leap_year = DateTimeDefault.register(pandas.Series.dt.is_leap_year)
    dt_is_month_end = DateTimeDefault.register(pandas.Series.dt.is_month_end)
    dt_is_month_start = DateTimeDefault.register(pandas.Series.dt.is_month_start)
    dt_is_quarter_end = DateTimeDefault.register(pandas.Series.dt.is_quarter_end)
    dt_is_quarter_start = DateTimeDefault.register(pandas.Series.dt.is_quarter_start)
    dt_is_year_end = DateTimeDefault.register(pandas.Series.dt.is_year_end)
    dt_is_year_start = DateTimeDefault.register(pandas.Series.dt.is_year_start)
    dt_microsecond = DateTimeDefault.register(pandas.Series.dt.microsecond)
    dt_microseconds = DateTimeDefault.register(pandas.Series.dt.microseconds)
    dt_minute = DateTimeDefault.register(pandas.Series.dt.minute)
    dt_month = DateTimeDefault.register(pandas.Series.dt.month)
    dt_month_name = DateTimeDefault.register(pandas.Series.dt.month_name)
    dt_nanosecond = DateTimeDefault.register(pandas.Series.dt.nanosecond)
    dt_nanoseconds = DateTimeDefault.register(pandas.Series.dt.nanoseconds)
    dt_normalize = DateTimeDefault.register(pandas.Series.dt.normalize)
    dt_quarter = DateTimeDefault.register(pandas.Series.dt.quarter)
    dt_qyear = DateTimeDefault.register(pandas.Series.dt.qyear)
    dt_round = DateTimeDefault.register(pandas.Series.dt.round)
    dt_second = DateTimeDefault.register(pandas.Series.dt.second)
    dt_seconds = DateTimeDefault.register(pandas.Series.dt.seconds)
    dt_start_time = DateTimeDefault.register(pandas.Series.dt.start_time)
    dt_strftime = DateTimeDefault.register(pandas.Series.dt.strftime)
    dt_time = DateTimeDefault.register(pandas.Series.dt.time)
    dt_timetz = DateTimeDefault.register(pandas.Series.dt.timetz)
    dt_to_period = DateTimeDefault.register(pandas.Series.dt.to_period)
    dt_to_pydatetime = DateTimeDefault.register(pandas.Series.dt.to_pydatetime)
    dt_to_pytimedelta = DateTimeDefault.register(pandas.Series.dt.to_pytimedelta)
    dt_to_timestamp = DateTimeDefault.register(pandas.Series.dt.to_timestamp)
    dt_total_seconds = DateTimeDefault.register(pandas.Series.dt.total_seconds)
    dt_tz = DateTimeDefault.register(pandas.Series.dt.tz)
    dt_tz_convert = DateTimeDefault.register(pandas.Series.dt.tz_convert)
    dt_tz_localize = DateTimeDefault.register(pandas.Series.dt.tz_localize)
    dt_week = DateTimeDefault.register(pandas.Series.dt.week)
    dt_weekday = DateTimeDefault.register(pandas.Series.dt.weekday)
    dt_weekofyear = DateTimeDefault.register(pandas.Series.dt.weekofyear)
    dt_year = DateTimeDefault.register(pandas.Series.dt.year)

    # End of DateTime methods

    # Resample methods

    resample_agg_df = ResampleDefault.register(pandas.core.resample.Resampler.aggregate)
    resample_agg_ser = ResampleDefault.register(
        pandas.core.resample.Resampler.aggregate, squeeze_self=True
    )
    resample_app_df = ResampleDefault.register(pandas.core.resample.Resampler.apply)
    resample_app_ser = ResampleDefault.register(
        pandas.core.resample.Resampler.apply, squeeze_self=True
    )
    resample_asfreq = ResampleDefault.register(pandas.core.resample.Resampler.asfreq)
    resample_backfill = ResampleDefault.register(
        pandas.core.resample.Resampler.backfill
    )
    resample_bfill = ResampleDefault.register(pandas.core.resample.Resampler.bfill)
    resample_count = ResampleDefault.register(pandas.core.resample.Resampler.count)
    resample_ffill = ResampleDefault.register(pandas.core.resample.Resampler.ffill)
    resample_fillna = ResampleDefault.register(pandas.core.resample.Resampler.fillna)
    resample_first = ResampleDefault.register(pandas.core.resample.Resampler.first)
    resample_get_group = ResampleDefault.register(
        pandas.core.resample.Resampler.get_group
    )
    resample_interpolate = ResampleDefault.register(
        pandas.core.resample.Resampler.interpolate
    )
    resample_last = ResampleDefault.register(pandas.core.resample.Resampler.last)
    resample_max = ResampleDefault.register(pandas.core.resample.Resampler.max)
    resample_mean = ResampleDefault.register(pandas.core.resample.Resampler.mean)
    resample_median = ResampleDefault.register(pandas.core.resample.Resampler.median)
    resample_min = ResampleDefault.register(pandas.core.resample.Resampler.min)
    resample_nearest = ResampleDefault.register(pandas.core.resample.Resampler.nearest)
    resample_nunique = ResampleDefault.register(pandas.core.resample.Resampler.nunique)
    resample_ohlc_df = ResampleDefault.register(pandas.core.resample.Resampler.ohlc)
    resample_ohlc_ser = ResampleDefault.register(
        pandas.core.resample.Resampler.ohlc, squeeze_self=True
    )
    resample_pad = ResampleDefault.register(pandas.core.resample.Resampler.pad)
    resample_pipe = ResampleDefault.register(pandas.core.resample.Resampler.pipe)
    resample_prod = ResampleDefault.register(pandas.core.resample.Resampler.prod)
    resample_quantile = ResampleDefault.register(
        pandas.core.resample.Resampler.quantile
    )
    resample_sem = ResampleDefault.register(pandas.core.resample.Resampler.sem)
    resample_size = ResampleDefault.register(pandas.core.resample.Resampler.size)
    resample_std = ResampleDefault.register(pandas.core.resample.Resampler.std)
    resample_sum = ResampleDefault.register(pandas.core.resample.Resampler.sum)
    resample_transform = ResampleDefault.register(
        pandas.core.resample.Resampler.transform
    )
    resample_var = ResampleDefault.register(pandas.core.resample.Resampler.var)

    # End of Resample methods

    # Str methods

    str_capitalize = StrDefault.register(pandas.Series.str.capitalize)
    str_center = StrDefault.register(pandas.Series.str.center)
    str_contains = StrDefault.register(pandas.Series.str.contains)
    str_count = StrDefault.register(pandas.Series.str.count)
    str_endswith = StrDefault.register(pandas.Series.str.endswith)
    str_find = StrDefault.register(pandas.Series.str.find)
    str_findall = StrDefault.register(pandas.Series.str.findall)
    str_get = StrDefault.register(pandas.Series.str.get)
    str_index = StrDefault.register(pandas.Series.str.index)
    str_isalnum = StrDefault.register(pandas.Series.str.isalnum)
    str_isalpha = StrDefault.register(pandas.Series.str.isalpha)
    str_isdecimal = StrDefault.register(pandas.Series.str.isdecimal)
    str_isdigit = StrDefault.register(pandas.Series.str.isdigit)
    str_islower = StrDefault.register(pandas.Series.str.islower)
    str_isnumeric = StrDefault.register(pandas.Series.str.isnumeric)
    str_isspace = StrDefault.register(pandas.Series.str.isspace)
    str_istitle = StrDefault.register(pandas.Series.str.istitle)
    str_isupper = StrDefault.register(pandas.Series.str.isupper)
    str_join = StrDefault.register(pandas.Series.str.join)
    str_len = StrDefault.register(pandas.Series.str.len)
    str_ljust = StrDefault.register(pandas.Series.str.ljust)
    str_lower = StrDefault.register(pandas.Series.str.lower)
    str_lstrip = StrDefault.register(pandas.Series.str.lstrip)
    str_match = StrDefault.register(pandas.Series.str.match)
    str_normalize = StrDefault.register(pandas.Series.str.normalize)
    str_pad = StrDefault.register(pandas.Series.str.pad)
    str_partition = StrDefault.register(pandas.Series.str.partition)
    str_repeat = StrDefault.register(pandas.Series.str.repeat)
    str_replace = StrDefault.register(pandas.Series.str.replace)
    str_rfind = StrDefault.register(pandas.Series.str.rfind)
    str_rindex = StrDefault.register(pandas.Series.str.rindex)
    str_rjust = StrDefault.register(pandas.Series.str.rjust)
    str_rpartition = StrDefault.register(pandas.Series.str.rpartition)
    str_rsplit = StrDefault.register(pandas.Series.str.rsplit)
    str_rstrip = StrDefault.register(pandas.Series.str.rstrip)
    str_slice = StrDefault.register(pandas.Series.str.slice)
    str_slice_replace = StrDefault.register(pandas.Series.str.slice_replace)
    str_split = StrDefault.register(pandas.Series.str.split)
    str_startswith = StrDefault.register(pandas.Series.str.startswith)
    str_strip = StrDefault.register(pandas.Series.str.strip)
    str_swapcase = StrDefault.register(pandas.Series.str.swapcase)
    str_title = StrDefault.register(pandas.Series.str.title)
    str_translate = StrDefault.register(pandas.Series.str.translate)
    str_upper = StrDefault.register(pandas.Series.str.upper)
    str_wrap = StrDefault.register(pandas.Series.str.wrap)
    str_zfill = StrDefault.register(pandas.Series.str.zfill)

    # End of Str methods

    # Rolling methods

    rolling_aggregate = RollingDefault.register(
        pandas.core.window.rolling.Rolling.aggregate
    )
    rolling_apply = RollingDefault.register(pandas.core.window.rolling.Rolling.apply)
    rolling_corr = RollingDefault.register(pandas.core.window.rolling.Rolling.corr)
    rolling_count = RollingDefault.register(pandas.core.window.rolling.Rolling.count)
    rolling_cov = RollingDefault.register(pandas.core.window.rolling.Rolling.cov)
    rolling_kurt = RollingDefault.register(pandas.core.window.rolling.Rolling.kurt)
    rolling_max = RollingDefault.register(pandas.core.window.rolling.Rolling.max)
    rolling_mean = RollingDefault.register(pandas.core.window.rolling.Rolling.mean)
    rolling_median = RollingDefault.register(pandas.core.window.rolling.Rolling.median)
    rolling_min = RollingDefault.register(pandas.core.window.rolling.Rolling.min)
    rolling_quantile = RollingDefault.register(
        pandas.core.window.rolling.Rolling.quantile
    )
    rolling_skew = RollingDefault.register(pandas.core.window.rolling.Rolling.skew)
    rolling_std = RollingDefault.register(pandas.core.window.rolling.Rolling.std)
    rolling_sum = RollingDefault.register(pandas.core.window.rolling.Rolling.sum)
    rolling_var = RollingDefault.register(pandas.core.window.rolling.Rolling.var)

    # End of Rolling methods

    # Window methods

    window_mean = RollingDefault.register(pandas.core.window.Window.mean)
    window_std = RollingDefault.register(pandas.core.window.Window.std)
    window_sum = RollingDefault.register(pandas.core.window.Window.sum)
    window_var = RollingDefault.register(pandas.core.window.Window.var)

    # End of Window methods

    # Categories methods

    cat_codes = CatDefault.register(pandas.Series.cat.codes)

    # End of Categories methods

    # DataFrame methods

    invert = DataFrameDefault.register(pandas.DataFrame.__invert__)
    mad = DataFrameDefault.register(pandas.DataFrame.mad)
    kurt = DataFrameDefault.register(pandas.DataFrame.kurt)
    sum_min_count = DataFrameDefault.register(pandas.DataFrame.sum)
    prod_min_count = DataFrameDefault.register(pandas.DataFrame.prod)
    compare = DataFrameDefault.register(pandas.DataFrame.compare)

    # End of DataFrame methods
