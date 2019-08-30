NOT_IMPLEMENTED_MESSAGE = "Must be implemented in children classes"


class BaseQueryCompiler(object):
    """Abstract Class that handles the queries to Modin dataframes.

    Note: See the Abstract Methods and Fields section immediately below this
        for a list of requirements for subclassing this object.
    """

    # Abstract Methods and Fields: Must implement in children classes
    # In some cases, there you may be able to use the same implementation for
    # some of these abstract methods, but for the sake of generality they are
    # treated differently.

    # Metadata modification abstract methods
    def add_prefix(self, prefix, axis=1):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def add_suffix(self, suffix, axis=1):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END Metadata modification abstract methods

    # Abstract copy
    # For copy, we don't want a situation where we modify the metadata of the
    # copies if we end up modifying something here. We copy all of the metadata
    # to prevent that.
    def copy(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

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
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END Abstract join and append helper functions

    # Data Management Methods
    def free(self):
        """In the future, this will hopefully trigger a cleanup of this object.
        """
        # TODO create a way to clean up this object.
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END Data Management Methods

    # To/From Pandas
    def to_pandas(self):
        """Converts Modin DataFrame to Pandas DataFrame.

        Returns:
            Pandas DataFrame of the QueryCompiler.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def from_pandas(cls, df, block_partitions_cls):
        """Improve simple Pandas DataFrame to an advanced and superior Modin DataFrame.

        Args:
            cls: DataManger object to convert the DataFrame to.
            df: Pandas DataFrame object.
            block_partitions_cls: BlockParitions object to store partitions

        Returns:
            Returns QueryCompiler containing data from the Pandas DataFrame.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END To/From Pandas

    # To NumPy
    def to_numpy(self):
        """Converts Modin DataFrame to NumPy DataFrame.

        Returns:
            NumPy Array of the QueryCompiler.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END To NumPy

    # Abstract inter-data operations (e.g. add, sub)
    # These operations require two DataFrames and will change the shape of the
    # data if the index objects don't match. An outer join + op is performed,
    # such that columns/rows that don't have an index on the other DataFrame
    # result in NaN values.

    def binary_op(self, op, other, **kwargs):
        """Perform an operation between two objects.

        Note: The list of operations is as follows:
            - add
            - eq
            - floordiv
            - ge
            - gt
            - le
            - lt
            - mod
            - mul
            - ne
            - pow
            - rfloordiv
            - rmod
            - rpow
            - rsub
            - rtruediv
            - sub
            - truediv
            - __and__
            - __or__
            - __xor__
        Args:
            op: The operation. See list of operations above
            other: The object to operate against.

        Returns:
            A new QueryCompiler object.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def clip(self, lower, upper, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def update(self, other, **kwargs):
        """Uses other manager to update corresponding values in this manager.

        Args:
            other: The other manager.

        Returns:
            New QueryCompiler with updated data and index.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def where(self, cond, other, **kwargs):
        """Gets values from this manager where cond is true else from other.

        Args:
            cond: Condition on which to evaluate values.

        Returns:
            New QueryCompiler with updated data and index.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END Abstract inter-data operations

    # Abstract Transpose
    def transpose(self, *args, **kwargs):
        """Transposes this QueryCompiler.

        Returns:
            Transposed new QueryCompiler.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

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
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def reset_index(self, **kwargs):
        """Removes all levels from index and sets a default level_0 index.

        Returns:
            New QueryCompiler with updated data and reset index.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END Abstract reindex/reset_index

    # Full Reduce operations
    #
    # These operations result in a reduced dimensionality of data.
    # Currently, this means a Pandas Series will be returned, but in the future
    # we will implement a Distributed Series, and this will be returned
    # instead.

    def count(self, **kwargs):
        """Counts the number of non-NaN objects for each column or row.

        Return:
            Pandas series containing counts of non-NaN objects from each column or row.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def max(self, **kwargs):
        """Returns the maximum value for each column or row.

        Return:
            Pandas series with the maximum values from each column or row.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def mean(self, **kwargs):
        """Returns the mean for each numerical column or row.

        Return:
            Pandas series containing the mean from each numerical column or row.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def min(self, **kwargs):
        """Returns the minimum from each column or row.

        Return:
            Pandas series with the minimum value from each column or row.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def prod(self, **kwargs):
        """Returns the product of each numerical column or row.

        Return:
            Pandas series with the product of each numerical column or row.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def sum(self, **kwargs):
        """Returns the sum of each numerical column or row.

        Return:
            Pandas series with the sum of each numerical column or row.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END Abstract full Reduce operations

    # Abstract map partitions operations
    # These operations are operations that apply a function to every partition.
    def abs(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def applymap(self, func):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def isin(self, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def isna(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def isnull(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def negative(self, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def notna(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def notnull(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def round(self, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END Abstract map partitions operations

    # Abstract map partitions across select indices
    def astype(self, col_dtypes, **kwargs):
        """Converts columns dtypes to given dtypes.

        Args:
            col_dtypes: Dictionary of {col: dtype,...} where col is the column
                name and dtype is a numpy dtype.

        Returns:
            DataFrame with updated dtypes.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

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
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def any(self, **kwargs):
        """Returns whether any the elements are true, potentially over an axis.

        Return:
            Pandas Series containing boolean values or boolean.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def first_valid_index(self):
        """Returns index of first non-NaN/NULL value.

        Return:
            Scalar of index name.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def idxmax(self, **kwargs):
        """Returns the first occurance of the maximum over requested axis.

        Returns:
            Series containing the maximum of each column or axis.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def idxmin(self, **kwargs):
        """Returns the first occurance of the minimum over requested axis.

        Returns:
            Series containing the minimum of each column or axis.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def last_valid_index(self):
        """Returns index of last non-NaN/NULL value.

        Return:
            Scalar of index name.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def median(self, **kwargs):
        """Returns median of each column or row.

        Returns:
            Series containing the median of each column or row.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def memory_usage(self, **kwargs):
        """Returns the memory usage of each column.

        Returns:
            Series containing the memory usage of each column.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def nunique(self, **kwargs):
        """Returns the number of unique items over each column or row.

        Returns:
            Series of ints indexed by column or index names.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def quantile_for_single_value(self, **kwargs):
        """Returns quantile of each column or row.

        Returns:
            Series containing the quantile of each column or row.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def skew(self, **kwargs):
        """Returns skew of each column or row.

        Returns:
            Series containing the skew of each column or row.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def std(self, **kwargs):
        """Returns standard deviation of each column or row.

        Returns:
            Series containing the standard deviation of each column or row.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def to_datetime(self, **kwargs):
        """Converts the Manager to a Series of DateTime objects.

        Returns:
            Series of DateTime objects.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def var(self, **kwargs):
        """Returns variance of each column or row.

        Returns:
            Series containing the variance of each column or row.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

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
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END Abstract column/row partitions reduce operations over select indices

    # Map across rows/columns
    # These operations require some global knowledge of the full column/row
    # that is being operated on. This means that we have to put all of that
    # data in the same place.
    def cumsum(self, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def cummax(self, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def cummin(self, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def cumprod(self, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def diff(self, **kwargs):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def dropna(self, **kwargs):
        """Returns a new QueryCompiler with null values dropped along given axis.
        Return:
            New QueryCompiler
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def eval(self, expr, **kwargs):
        """Returns a new QueryCompiler with expr evaluated on columns.

        Args:
            expr: The string expression to evaluate.

        Returns:
            A new QueryCompiler with new columns after applying expr.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def mode(self, **kwargs):
        """Returns a new QueryCompiler with modes calculated for each label along given axis.

        Returns:
            A new QueryCompiler with modes calculated.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def fillna(self, **kwargs):
        """Replaces NaN values with the method provided.

        Returns:
            A new QueryCompiler with null values filled.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def query(self, expr, **kwargs):
        """Query columns of the QueryCompiler with a boolean expression.

        Args:
            expr: Boolean expression to query the columns with.

        Returns:
            QueryCompiler containing the rows where the boolean expression is satisfied.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def rank(self, **kwargs):
        """Computes numerical rank along axis. Equal values are set to the average.

        Returns:
            QueryCompiler containing the ranks of the values along an axis.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def sort_index(self, **kwargs):
        """Sorts the data with respect to either the columns or the indices.

        Returns:
            QueryCompiler containing the data sorted by columns or indices.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

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
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END Abstract map across rows/columns

    # Abstract head/tail/front/back
    def head(self, n):
        """Returns the first n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            QueryCompiler containing the first n rows of the original QueryCompiler.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def tail(self, n):
        """Returns the last n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            QueryCompiler containing the last n rows of the original QueryCompiler.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def front(self, n):
        """Returns the first n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            QueryCompiler containing the first n columns of the original QueryCompiler.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def back(self, n):
        """Returns the last n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            QueryCompiler containing the last n columns of the original QueryCompiler.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END head/tail/front/back

    # Abstract __getitem__ methods
    def getitem_column_array(self, key):
        """Get column data for target labels.

        Args:
            key: Target labels by which to retrieve data.

        Returns:
            A new Query Compiler.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def getitem_row_array(self, key):
        """Get row data for target labels.

        Args:
            key: Target numeric indices by which to retrieve data.

        Returns:
            A new Query Compiler.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

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
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

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
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

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
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END UDF

    # Manual Partitioning methods (e.g. merge, groupby)
    # These methods require some sort of manual partitioning due to their
    # nature. They require certain data to exist on the same partition, and
    # after the shuffle, there should be only a local map required.
    def groupby_agg(self, by, axis, agg_func, groupby_args, agg_args):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

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
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # END Manual Partitioning methods

    def get_dummies(self, columns, **kwargs):
        """Convert categorical variables to dummy variables for certain columns.

        Args:
            columns: The columns to convert.

        Returns:
            A new QueryCompiler.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    # Indexing
    def view(self, index=None, columns=None):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def squeeze(self, ndim=0, axis=None):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def write_items(self, row_numeric_index, col_numeric_index, broadcasted_items):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

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
