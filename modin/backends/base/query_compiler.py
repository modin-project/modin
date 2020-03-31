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


class BaseQueryCompiler(abc.ABC):
    """Abstract Class that handles the queries to Modin dataframes.

    Note: See the Abstract Methods and Fields section immediately below this
        for a list of requirements for subclassing this object.
    """

    # Abstract Methods and Fields: Must implement in children classes
    # In some cases, there you may be able to use the same implementation for
    # some of these abstract methods, but for the sake of generality they are
    # treated differently.

    # Metadata modification abstract methods
    @abc.abstractmethod
    def add_prefix(self, prefix, axis=1):
        pass

    @abc.abstractmethod
    def add_suffix(self, suffix, axis=1):
        pass

    # END Metadata modification abstract methods

    # Abstract copy
    # For copy, we don't want a situation where we modify the metadata of the
    # copies if we end up modifying something here. We copy all of the metadata
    # to prevent that.
    @abc.abstractmethod
    def copy(self):
        pass

    # END Abstract copy

    # Abstract join and append helper functions

    @abc.abstractmethod
    def concat(self, axis, other, **kwargs):
        """Concatenates two objects together.

        Args:
            axis: The axis index object to join (0 for columns, 1 for index).
            other: The other_index to concat with.

        Returns:
            Concatenated objects.
        """
        pass

    # END Abstract join and append helper functions

    # Data Management Methods
    @abc.abstractmethod
    def free(self):
        """In the future, this will hopefully trigger a cleanup of this object.
        """
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
    def from_pandas(cls, df, block_partitions_cls):
        """Improve simple Pandas DataFrame to an advanced and superior Modin DataFrame.

        Args:
            cls: DataManger object to convert the DataFrame to.
            df: Pandas DataFrame object.
            block_partitions_cls: BlockParitions object to store partitions

        Returns:
            Returns QueryCompiler containing data from the Pandas DataFrame.
        """
        pass

    # END To/From Pandas

    # To NumPy
    @abc.abstractmethod
    def to_numpy(self):
        """Converts Modin DataFrame to NumPy DataFrame.

        Returns:
            NumPy Array of the QueryCompiler.
        """
        pass

    # END To NumPy

    # Abstract inter-data operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.

    @abc.abstractmethod
    def add(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def combine(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def combine_first(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def eq(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def floordiv(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def ge(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def gt(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def le(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def lt(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def mod(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def mul(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def ne(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def pow(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def rfloordiv(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def rmod(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def rpow(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def rsub(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def rtruediv(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def sub(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def truediv(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def __and__(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def __or__(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def __rand__(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def __ror__(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def __rxor__(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def __xor__(self, other, **kwargs):
        pass

    @abc.abstractmethod
    def update(self, other, **kwargs):
        """Uses other manager to update corresponding values in this manager.

        Args:
            other: The other manager.

        Returns:
            New QueryCompiler with updated data and index.
        """
        pass

    @abc.abstractmethod
    def clip(self, lower, upper, **kwargs):
        pass

    @abc.abstractmethod
    def where(self, cond, other, **kwargs):
        """Gets values from this manager where cond is true else from other.

        Args:
            cond: Condition on which to evaluate values.

        Returns:
            New QueryCompiler with updated data and index.
        """
        pass

    # END Abstract inter-data operations

    # Abstract Transpose
    @abc.abstractmethod
    def transpose(self, *args, **kwargs):
        """Transposes this QueryCompiler.

        Returns:
            Transposed new QueryCompiler.
        """
        pass

    # END Abstract Transpose

    # Abstract reindex/reset_index (may shuffle data)
    @abc.abstractmethod
    def reindex(self, axis, labels, **kwargs):
        """Fits a new index for this Manger.

        Args:
            axis: The axis index object to target the reindex on.
            labels: New labels to conform 'axis' on to.

        Returns:
            New QueryCompiler with updated data and new index.
        """
        pass

    @abc.abstractmethod
    def reset_index(self, **kwargs):
        """Removes all levels from index and sets a default level_0 index.

        Returns:
            New QueryCompiler with updated data and reset index.
        """
        pass

    # END Abstract reindex/reset_index

    # Full Reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.

    @abc.abstractmethod
    def count(self, **kwargs):
        """Counts the number of non-NaN objects for each column or row.

        Return:
            Pandas series containing counts of non-NaN objects from each column or row.
        """
        pass

    @abc.abstractmethod
    def max(self, **kwargs):
        """Returns the maximum value for each column or row.

        Return:
            Pandas series with the maximum values from each column or row.
        """
        pass

    @abc.abstractmethod
    def mean(self, **kwargs):
        """Returns the mean for each numerical column or row.

        Return:
            Pandas series containing the mean from each numerical column or row.
        """
        pass

    @abc.abstractmethod
    def min(self, **kwargs):
        """Returns the minimum from each column or row.

        Return:
            Pandas series with the minimum value from each column or row.
        """
        pass

    @abc.abstractmethod
    def prod(self, **kwargs):
        """Returns the product of each numerical column or row.

        Return:
            Pandas series with the product of each numerical column or row.
        """
        pass

    @abc.abstractmethod
    def sum(self, **kwargs):
        """Returns the sum of each numerical column or row.

        Return:
            Pandas series with the sum of each numerical column or row.
        """
        pass

    # END Abstract full Reduce operations

    # Abstract map partitions operations
    # These operations are operations that apply a function to every partition.
    @abc.abstractmethod
    def abs(self):
        pass

    @abc.abstractmethod
    def applymap(self, func):
        pass

    @abc.abstractmethod
    def isin(self, **kwargs):
        pass

    @abc.abstractmethod
    def isna(self):
        pass

    @abc.abstractmethod
    def negative(self, **kwargs):
        pass

    @abc.abstractmethod
    def notna(self):
        pass

    @abc.abstractmethod
    def round(self, **kwargs):
        pass

    # END Abstract map partitions operations

    # Abstract map partitions across select indices
    @abc.abstractmethod
    def astype(self, col_dtypes, **kwargs):
        """Converts columns dtypes to given dtypes.

        Args:
            col_dtypes: Dictionary of {col: dtype,...} where col is the column
                name and dtype is a numpy dtype.

        Returns:
            DataFrame with updated dtypes.
        """
        pass

    # END Abstract map partitions across select indices

    # Abstract column/row partitions reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    @abc.abstractmethod
    def all(self, **kwargs):
        """Returns whether all the elements are true, potentially over an axis.

        Return:
            Pandas Series containing boolean values or boolean.
        """
        pass

    @abc.abstractmethod
    def any(self, **kwargs):
        """Returns whether any the elements are true, potentially over an axis.

        Return:
            Pandas Series containing boolean values or boolean.
        """
        pass

    @abc.abstractmethod
    def first_valid_index(self):
        """Returns index of first non-NaN/NULL value.

        Return:
            Scalar of index name.
        """
        pass

    @abc.abstractmethod
    def idxmax(self, **kwargs):
        """Returns the first occurance of the maximum over requested axis.

        Returns:
            Series containing the maximum of each column or axis.
        """
        pass

    @abc.abstractmethod
    def idxmin(self, **kwargs):
        """Returns the first occurance of the minimum over requested axis.

        Returns:
            Series containing the minimum of each column or axis.
        """
        pass

    @abc.abstractmethod
    def last_valid_index(self):
        """Returns index of last non-NaN/NULL value.

        Return:
            Scalar of index name.
        """
        pass

    @abc.abstractmethod
    def median(self, **kwargs):
        """Returns median of each column or row.

        Returns:
            Series containing the median of each column or row.
        """
        pass

    @abc.abstractmethod
    def memory_usage(self, **kwargs):
        """Returns the memory usage of each column.

        Returns:
            Series containing the memory usage of each column.
        """
        pass

    @abc.abstractmethod
    def nunique(self, **kwargs):
        """Returns the number of unique items over each column or row.

        Returns:
            Series of ints indexed by column or index names.
        """
        pass

    @abc.abstractmethod
    def quantile_for_single_value(self, **kwargs):
        """Returns quantile of each column or row.

        Returns:
            Series containing the quantile of each column or row.
        """
        pass

    @abc.abstractmethod
    def skew(self, **kwargs):
        """Returns skew of each column or row.

        Returns:
            Series containing the skew of each column or row.
        """
        pass

    @abc.abstractmethod
    def std(self, **kwargs):
        """Returns standard deviation of each column or row.

        Returns:
            Series containing the standard deviation of each column or row.
        """
        pass

    @abc.abstractmethod
    def var(self, **kwargs):
        """Returns variance of each column or row.

        Returns:
            Series containing the variance of each column or row.
        """
        pass

    # END Abstract column/row partitions reduce operations

    # Abstract column/row partitions reduce operations over select indices
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.
    @abc.abstractmethod
    def describe(self, **kwargs):
        """Generates descriptive statistics.

        Returns:
            DataFrame object containing the descriptive statistics of the DataFrame.
        """
        pass

    # END Abstract column/row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    @abc.abstractmethod
    def cumsum(self, **kwargs):
        pass

    @abc.abstractmethod
    def cummax(self, **kwargs):
        pass

    @abc.abstractmethod
    def cummin(self, **kwargs):
        pass

    @abc.abstractmethod
    def cumprod(self, **kwargs):
        pass

    @abc.abstractmethod
    def diff(self, **kwargs):
        pass

    @abc.abstractmethod
    def dropna(self, **kwargs):
        """Returns a new QueryCompiler with null values dropped along given axis.
        Return:
            New QueryCompiler
        """
        pass

    @abc.abstractmethod
    def eval(self, expr, **kwargs):
        """Returns a new QueryCompiler with expr evaluated on columns.

        Args:
            expr: The string expression to evaluate.

        Returns:
            A new QueryCompiler with new columns after applying expr.
        """
        pass

    @abc.abstractmethod
    def mode(self, **kwargs):
        """Returns a new QueryCompiler with modes calculated for each label along given axis.

        Returns:
            A new QueryCompiler with modes calculated.
        """
        pass

    @abc.abstractmethod
    def fillna(self, **kwargs):
        """Replaces NaN values with the method provided.

        Returns:
            A new QueryCompiler with null values filled.
        """
        pass

    @abc.abstractmethod
    def query(self, expr, **kwargs):
        """Query columns of the QueryCompiler with a boolean expression.

        Args:
            expr: Boolean expression to query the columns with.

        Returns:
            QueryCompiler containing the rows where the boolean expression is satisfied.
        """
        pass

    @abc.abstractmethod
    def rank(self, **kwargs):
        """Computes numerical rank along axis. Equal values are set to the average.

        Returns:
            QueryCompiler containing the ranks of the values along an axis.
        """
        pass

    @abc.abstractmethod
    def sort_index(self, **kwargs):
        """Sorts the data with respect to either the columns or the indices.

        Returns:
            QueryCompiler containing the data sorted by columns or indices.
        """
        pass

    # END Abstract map across rows/columns

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    @abc.abstractmethod
    def quantile_for_list_of_values(self, **kwargs):
        """Returns Manager containing quantiles along an axis for numeric columns.

        Returns:
            QueryCompiler containing quantiles of original QueryCompiler along an axis.
        """
        pass

    # END Abstract map across rows/columns

    # Abstract head/tail/front/back
    @abc.abstractmethod
    def head(self, n):
        """Returns the first n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            QueryCompiler containing the first n rows of the original QueryCompiler.
        """
        pass

    @abc.abstractmethod
    def tail(self, n):
        """Returns the last n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            QueryCompiler containing the last n rows of the original QueryCompiler.
        """
        pass

    @abc.abstractmethod
    def front(self, n):
        """Returns the first n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            QueryCompiler containing the first n columns of the original QueryCompiler.
        """
        pass

    @abc.abstractmethod
    def back(self, n):
        """Returns the last n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            QueryCompiler containing the last n columns of the original QueryCompiler.
        """
        pass

    # END head/tail/front/back

    # Abstract __getitem__ methods
    @abc.abstractmethod
    def getitem_column_array(self, key):
        """Get column data for target labels.

        Args:
            key: Target labels by which to retrieve data.

        Returns:
            A new Query Compiler.
        """
        pass

    @abc.abstractmethod
    def getitem_row_array(self, key):
        """Get row data for target labels.

        Args:
            key: Target numeric indices by which to retrieve data.

        Returns:
            A new Query Compiler.
        """
        pass

    # END Abstract __getitem__ methods

    # Abstract insert
    # This method changes the shape of the resulting data. In Pandas, this
    # operation is always inplace, but this object is immutable, so we just
    # return a new one from here and let the front end handle the inplace
    # update.
    @abc.abstractmethod
    def insert(self, loc, column, value):
        """Insert new column data.

        Args:
            loc: Insertion index.
            column: Column labels to insert.
            value: Dtype object values to insert.

        Returns:
            A new QueryCompiler with new data inserted.
        """
        pass

    # END Abstract insert

    # Abstract drop
    @abc.abstractmethod
    def drop(self, index=None, columns=None):
        """Remove row data for target index and columns.

        Args:
            index: Target index to drop.
            columns: Target columns to drop.

        Returns:
            A new QueryCompiler.
        """
        pass

    # END drop

    # UDF (apply and agg) methods
    # There is a wide range of behaviors that are supported, so a lot of the
    # logic can get a bit convoluted.
    @abc.abstractmethod
    def apply(self, func, axis, *args, **kwargs):
        """Apply func across given axis.

        Args:
            func: The function to apply.
            axis: Target axis to apply the function along.

        Returns:
            A new QueryCompiler.
        """
        pass

    # END UDF

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.
    @abc.abstractmethod
    def groupby_agg(self, by, axis, agg_func, groupby_args, agg_args):
        pass

    @abc.abstractmethod
    def groupby_reduce(
        self,
        by,
        axis,
        groupby_args,
        map_func,
        map_args,
        reduce_func=None,
        reduce_args=None,
        numeric_only=True,
    ):
        pass

    # END Manual Partitioning methods

    @abc.abstractmethod
    def get_dummies(self, columns, **kwargs):
        """Convert categorical variables to dummy variables for certain columns.

        Args:
            columns: The columns to convert.

        Returns:
            A new QueryCompiler.
        """
        pass

    # Indexing
    @abc.abstractmethod
    def view(self, index=None, columns=None):
        pass

    @abc.abstractmethod
    def write_items(self, row_numeric_index, col_numeric_index, broadcasted_items):
        pass

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
