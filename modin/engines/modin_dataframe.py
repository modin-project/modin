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

from abc import ABC


class ModinDataframe(ABC):
    """An abstract class that represents the Parent class for any DataFrame class.

    This class is intended to specify the behaviours that a DataFrame must implement.

    For more details about how these methods were chosen, please refer to this
    (http://www.vldb.org/pvldb/vol13/p2033-petersohn.pdf) paper, which specifies
    a DataFrame algebra that this class exposes.

    """

    def mask(
        self, row_indices=None, row_positions=None, col_indices=None, col_positions=None
    ):
        """Allows users to perform selection and projection on the row and column number (positional notation),
        in addition to the row and column labels (named notation)

        Parameters
        ----------
        row_indices : list of hashable
            The row labels to extract.
        row_positions : list of int
            The row indices to extract.
        col_indices : list of hashable
            The column labels to extract.
        col_positions : list of int
            The column indices to extract.

        Returns
        -------
        ModinDataframe
             A new ModinDataframe from the mask provided.
        """
        pass

    def filter_by_types(self, types):
        """Allows the user to specify a type or set of types by which to filter the columns.

        Parameters
        ----------
        types: hashable or list of hashables
            The types to filter columns by.

        Returns
        -------
        ModinDataframe
             A new ModinDataframe from the filter provided.
        """
        pass

    def map(self, axis, function, result_schema=None):
        """Applies a user-defined function row- wise (or column-wise if axis=1).

        Notes
        -----
            This does not change the number of rows.

            The user-defined function may increase the number of columns (rows if axis=1),
                but it should not remove or drop columns and each invocation of the function
                must generate the same number of new columns.

        Parameters
        ----------
            axis: int
                The axis to map over.
            function: callable
                The function to map across the dataframe.
            result_schema: list of dtypes
                List of data types that represent the types of the output dataframe.

        Returns
        -------
        ModinDataframe
             A new ModinDataframe with the map applied.
        """
        pass

    def filter(self, axis, condition):
        """Filter data based on the function provided along the specified axis.

        Parameters
        ----------
            axis: int
                The axis to filter over.
            condition: callable
                The function to use for the filter. This function should filter the
                data itself.

        Returns
        -------
        ModinDataframe
             A new ModinDataframe from the filter provided.
        """
        pass

    def explode(self, axis, function, result_schema=None):
        """Explode data based on the function provided along the specified axis.

        Notes
        -----
            Only one axis can be expanded at a time.

            The user-defined function may increase the number of rows (columns if axis=1),
                    but it should not remove or drop rows.

        Parameters
        ----------
            axis: int
                The axis to expand over.
            function: callable
                The function to use to expand the data. This function should accept one
                row/column, and return multiple.
            result_schema: list of dtypes
                List of data types that represent the types of the output dataframe.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the specified axis expanded.
        """
        pass

    def window(self, axis, function, window_size, result_schema=None):
        """Applies a user-defined function over a sliding window along the specified axis.

        Notes
        -----
            The shapes of the output and input dataframes must match. The user-defined function
                recieves window_size arguments and must return the same number of outputs.

            The user-defined function may only access values in the same column (row if axis=1).

        Parameters
        ----------
            axis: int
                The axis to slide over.
            function: callable
                The sliding window function to apply over the data.
            window_size: int
                The number of row/columns to pass to the function.
                (The size of the sliding window).
            result_schema: list of dtypes
                List of data types that represent the types of the output dataframe.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the function applied over windows of the specified axis.
        """
        pass

    def window_reduction(self, axis, reduction_fn, window_size, result_schema=None):
        """Applies a sliding window operator that acts as a GROUPBY on each window,
        which reduces down to a single row (column) per window.

        Notes
        -----
            The user-defined reduction function must reduce each window’s column
                (row if axis=1) down to a single value.

        Parameters
        ----------
            axis: int
                The axis to slide over.
            reduction_fn: callable
                The reduction function to apply over the data.
            window_size: int
                The number of row/columns to pass to the function.
                (The size of the sliding window).
            result_schema: list of dtypes
                List of data types that represent the types of the output dataframe.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the reduction function applied over windows of the specified
                axis.
        """
        pass

    def groupby(self, axis, by, operator, result_schema=None):
        """Generates groups based on values in the input column(s) and performs
        the specified operation (e.g. aggregation or backfill) on the groups.

        Notes
        -----
            No communication between groups is allowed in this algebra implementation.

            The number of rows (columns if axis=1) returned by the user-defined function
                passed to the groupby may be at most the number of rows in the group, and
                may be as small as a single row.

            Unlike the pandas API, an intermediate “GROUP BY” object is not present in this
                algebra implementation.

        Parameters
        ----------
            axis: int
                The axis to apply the grouping over.
            by: string or list of strings
                One or more column labels to use for grouping.
            operator: callable
                The operation to carry out on each of the groups. The operator is another
                algebraic operator with its own user-defined function parameter, depending
                on the output desired by the user.
            result_schema: list of dtypes
                List of data types that represent the types of the output dataframe.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe containing the groupings specified, with the operator
                applied to each group.
        """
        pass

    def reduction(self, axis, function, tree_reduce=False, result_schema=None):
        """Performs a user-defined per-column aggregation, where each column reduces
        down to a single value.

        Notes
        -----
            The user-defined function must reduce to a single value.

        Parameters
        ----------
            axis: int
                The axis to perform the reduction over.
            function: callable
                The reduction function to apply to each column.
            tree_reduce: boolean
                Flag to signal to the compiler that the function
                can be applied using a tree reduction (e.g. max or sum).
                Set this flag to False for functions that need to look at
                the entire column at once to perform their reduction (e.g. median).
            result_schema: list of dtypes
                List of data types that represent the types of the output dataframe.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the same columns as the previous, with only a single row.
        """
        pass

    def infer_types(self, columns_list):
        """Determines the compatible type shared by all values in the specified columns,
        and converts all values to that type.

        Parameters
        ----------
            columns_list: list of strings
                List of column labels to infer and induce types over.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the inferred schema.
        """
        pass

    def join(self, axis, condition, other, join_type):
        """Joins this dataframe with the other.

        Notes
        -----
            During the join, this dataframe is considered the left, while the other is
                treated as the right.

            Only inner joins, left outer, right outer, and full outer joins are currently supported.
                Support for other join types (e.g. natural join) may be implemented in the future.

        Parameters
        ----------
            axis: int
                The axis to perform the join on.
            condition: callable
                Function that determines which rows should be joined. The condition can be a
                simple equality, e.g. "left.col1 == right.col1" or can be arbitrarily complex.
            other: ModinDataframe
                The other data to join with, i.e. the right dataframe.
            join_type: string
                The type of join to perform.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe that is the result of applying the specified join over the two
            dataframes.
        """
        pass

    def concat(self, axis, others):
        """Appends the rows of identical column labels from multiple dataframes.

        Notes
        -----
            The concat operator incurs fixed overheads, and so this algebra places no
                limit to the number of dataframes that may be concatenated in this way.

        Parameters
        ----------
            axis: int
                The axis on which to perform the concatenation.
            others: ModinDataframe or list of ModinDataframes
                The other ModinDataframe(s) to concatenate.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe that is the result of concatenating the dataframes over the
            specified axis.
        """
        pass

    def transpose(self):
        """Swaps the row and column axes.

        Notes
        -----
            Transposing a dataframe is expensive, and thus, while the axes are swapped
                logically immediately, the physical swap does not occur until absolutely necessary,
                which helps motivate the axis argument to the other operators in this algebra.

            This operation explicitly manipulates dataframe metadata.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the row and column axes swapped.
        """
        pass

    def to_labels(self, column_labels):
        """Replaces the row labels with one or more columns of data. When multiple column
        labels are specified, a heirarchical set of labels is created, ordered by the ordering
        of labels in the input.

        Notes
        -----
            This operation explicitly manipulates dataframe metadata.

        Parameters
        ----------
            column_labels: string or list of strings
                Column label(s) to use as the new row labels.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the row labels replaced by the specified columns.
        """
        pass

    def from_labels(self):
        """Moves the row labels into the data at position 0, and sets the row labels
        to the positional notation. In the case that the dataframe has hierarchical labels, all label
        “levels” are inserted into the dataframe in the order they occur in the labels, with the outermost
        being in position 0.

        Notes
        -----
            This operation explicitly manipulates dataframe metadata.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the row labels moved into the data.
        """
        pass

    def sort_by(self, axis, columns, ascending=True):
        """Logically reorders the dataframe’s rows (columns if axis=1) by the lexicographical
        order of the data in a column or set of columns.

        Parameters
        ----------
            axis: int
                The axis to perform the sort over.
            columns: string or list of strings
                Column label(s) to use to determine lexicographical ordering.
            ascending: boolean
                Whether to sort in ascending or descending order.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe sorted into lexicographical order by the specified column(s).
        """
        pass
