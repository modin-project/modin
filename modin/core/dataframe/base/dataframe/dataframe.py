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

"""
Module contains class ModinDataframe.

ModinDataframe is a parent abstract class for any dataframe class.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Hashable, List, Optional, Union

from modin.core.dataframe.base.dataframe.utils import Axis, JoinType


class ModinDataframe(ABC):
    """
    An abstract class that represents the Parent class for any Dataframe class.

    This class is intended to specify the behaviors that a Dataframe must implement.

    For more details about how these methods were chosen, please refer to this
    (https://people.eecs.berkeley.edu/~totemtang/paper/Modin.pdf) paper, which specifies
    a Dataframe algebra that this class exposes.
    """

    @abstractmethod
    def take_2d_labels_or_positional(
        self,
        row_labels: Optional[List[Hashable]] = None,
        row_positions: Optional[List[int]] = None,
        col_labels: Optional[List[Hashable]] = None,
        col_positions: Optional[List[int]] = None,
    ) -> "ModinDataframe":
        """
        Mask rows and columns in the dataframe.

        Allow users to perform selection and projection on the row and column labels (named notation),
        in addition to the row and column number (positional notation).

        Parameters
        ----------
        row_labels : list of hashable, optional
            The row labels to extract.
        row_positions : list of int, optional
            The row positions to extract.
        col_labels : list of hashable, optional
            The column labels to extract.
        col_positions : list of int, optional
            The column positions to extract.

        Returns
        -------
        ModinDataframe
             A new ModinDataframe from the mask provided.

        Notes
        -----
        If both `row_labels` and `row_positions` are provided, a ValueError is raised.
        The same rule applies for `col_labels` and `col_positions`.
        """
        pass

    @abstractmethod
    def filter_by_types(self, types: List[Hashable]) -> "ModinDataframe":
        """
        Allow the user to specify a type or set of types by which to filter the columns.

        Parameters
        ----------
        types : list of hashables
            The types to filter columns by.

        Returns
        -------
        ModinDataframe
             A new ModinDataframe with only the columns whose dtypes appear in `types`.
        """
        pass

    @abstractmethod
    def map(
        self,
        function: Callable,
        axis: Optional[Union[int, Axis]] = None,
        dtypes: Optional[str] = None,
        new_columns: Optional[List[Hashable]] = None,
    ) -> "ModinDataframe":
        """
        Apply a user-defined function row-wise if `axis`=0, column-wise if `axis`=1, and cell-wise if `axis` is None.

        Parameters
        ----------
        function : callable(row|col|cell) -> row|col|cell
            The function to map across the dataframe.
        axis : int or modin.core.dataframe.base.utils.Axis, optional
            The axis to map over.
        dtypes : str, optional
            The data types for the result. This is an optimization
            because there are functions that always result in a particular data
            type, and this allows us to avoid (re)computing it.
        new_columns : List[Hashable], optional
            New column labels of the result, its length has to be identical
            to the older columns. If not specified, old column labels are preserved.

        Returns
        -------
        ModinDataframe
             A new ModinDataframe with the map applied.

        Notes
        -----
        This does not change the shape of the dataframe.
        """
        pass

    @abstractmethod
    def filter(self, axis: Union[int, Axis], condition: Callable) -> "ModinDataframe":
        """
        Filter data based on the function provided along the specified axis.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to filter over.
        condition : callable(row|col) -> bool
            The function to use for the filter. This function should filter the
            data itself. It accepts either a row or column (depending on the axis argument) and
            returns True to keep the row/col, and False to drop it.

        Returns
        -------
        ModinDataframe
             A new ModinDataframe filtered by content according to the filter provided by condition.
        """
        pass

    @abstractmethod
    def explode(
        self,
        axis: Union[int, Axis],
        function: Callable,
        result_schema: Optional[Dict[Hashable, type]] = None,
    ) -> "ModinDataframe":
        """
        Explode data based on the function provided along the specified axis.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to expand over.
        function : callable
            The function to use to expand the data. This function should accept one
            row/column, and return multiple.
        result_schema : dictionary, optional
            Mapping from column labels to data types that represents the types of the output dataframe.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the specified axis expanded.

        Notes
        -----
        Only one axis can be expanded at a time.

        The user-defined function may increase the number of rows (columns if axis=1),
        but it should not remove or drop rows.
        """
        pass

    @abstractmethod
    def window(
        self,
        axis: Union[int, Axis],
        reduce_fn: Callable,
        window_size: int,
        result_schema: Optional[Dict[Hashable, type]] = None,
    ) -> "ModinDataframe":
        """
        Apply a sliding window operator that acts as a GROUPBY on each window, reducing each window to a single row (column).

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to slide over.
        reduce_fn : callable(rowgroup|colgroup) -> row|col
            The reduce function to apply over the data.
        window_size : int
            The number of row/columns to pass to the function.
            (The size of the sliding window).
        result_schema : dictionary, optional
            Mapping from column labels to data types that represents the types of the output dataframe.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the reduce function applied over windows of the specified
            axis.

        Notes
        -----
        The user-defined reduce function must reduce each window's column
        (row if axis=1) down to a single value.
        """
        pass

    @abstractmethod
    def groupby(
        self,
        axis: Union[int, Axis],
        by: Union[str, List[str]],
        operator: Callable,
        result_schema: Optional[Dict[Hashable, type]] = None,
    ) -> "ModinDataframe":
        """
        Generate groups based on values in the input column(s) and perform the specified operation on each.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to apply the grouping over.
        by : string or list of strings
            One or more column labels to use for grouping.
        operator : callable
            The operation to carry out on each of the groups. The operator is another
            algebraic operator with its own user-defined function parameter, depending
            on the output desired by the user.
        result_schema : dictionary, optional
            Mapping from column labels to data types that represents the types of the output dataframe.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe containing the groupings specified, with the operator
            applied to each group.

        Notes
        -----
        No communication between groups is allowed in this algebra implementation.

        The number of rows (columns if axis=1) returned by the user-defined function
        passed to the groupby may be at most the number of rows in the group, and
        may be as small as a single row.

        Unlike the pandas API, an intermediate "GROUP BY" object is not present in this
        algebra implementation.
        """
        pass

    @abstractmethod
    def reduce(
        self,
        axis: Union[int, Axis],
        function: Callable,
        dtypes: Optional[str] = None,
    ) -> "ModinDataframe":
        """
        Perform a user-defined aggregation on the specified axis, where the axis reduces down to a singleton.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to perform the reduce over.
        function : callable(row|col) -> single value
            The reduce function to apply to each column.
        dtypes : str, optional
            The data types for the result. This is an optimization
            because there are functions that always result in a particular data
            type, and this allows us to avoid (re)computing it.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the same columns as the previous, with only a single row.

        Notes
        -----
        The user-defined function must reduce to a single value.
        """
        pass

    @abstractmethod
    def tree_reduce(
        self,
        axis: Union[int, Axis],
        map_func: Callable,
        reduce_func: Optional[Callable] = None,
        dtypes: Optional[str] = None,
    ) -> "ModinDataframe":
        """
        Perform a user-defined aggregation on the specified axis, where the axis reduces down to a singleton using a tree-reduce computation pattern.

        The map function is applied first over multiple partitions of a column, and then the reduce
        function (if specified, otherwise the map function is applied again) is applied to the
        results to produce a single value.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to perform the tree reduce over.
        map_func : callable(row|col) -> row|col|single value
            The map function to apply to each column.
        reduce_func : callable(row|col) -> single value, optional
            The reduce function to apply to the results of the map function.
        dtypes : str, optional
            The data types for the result. This is an optimization
            because there are functions that always result in a particular data
            type, and this allows us to avoid (re)computing it.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the same columns as the previous, with only a single row.

        Notes
        -----
        The user-defined function must reduce to a single value.

        If the user-defined function requires access to the entire column, please use reduce instead.
        """
        pass

    @abstractmethod
    def infer_types(self, columns_list: List[str]) -> "ModinDataframe":
        """
        Determine the compatible type shared by all values in the specified columns, and coerce them to that type.

        Parameters
        ----------
        columns_list : list of strings
            List of column labels to infer and induce types over.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the inferred schema.
        """
        pass

    @abstractmethod
    def join(
        self,
        axis: Union[int, Axis],
        condition: Callable,
        other: "ModinDataframe",
        join_type: Union[str, JoinType],
    ) -> "ModinDataframe":
        """
        Join this dataframe with the other.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to perform the join on.
        condition : callable
            Function that determines which rows should be joined. The condition can be a
            simple equality, e.g. "left.col1 == right.col1" or can be arbitrarily complex.
        other : ModinDataframe
            The other data to join with, i.e. the right dataframe.
        join_type : string  {"inner", "left", "right", "outer"} or modin.core.dataframe.base.utils.JoinType
            The type of join to perform.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe that is the result of applying the specified join over the two
            dataframes.

        Notes
        -----
        During the join, this dataframe is considered the left, while the other is
        treated as the right.

        Only inner joins, left outer, right outer, and full outer joins are currently supported.
        Support for other join types (e.g. natural join) may be implemented in the future.
        """
        pass

    @abstractmethod
    def concat(
        self,
        axis: Union[int, Axis],
        others: Union["ModinDataframe", List["ModinDataframe"]],
    ) -> "ModinDataframe":
        """
        Append rows/columns along the specified axis from multiple dataframes.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis on which to perform the concatenation.
        others : ModinDataframe or list of ModinDataframes
            The other ModinDataframe(s) to concatenate.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe that is the result of concatenating the dataframes over the
            specified axis.

        Notes
        -----
        The concat operator incurs fixed overheads, and so this algebra places no
        limit to the number of dataframes that may be concatenated in this way.
        """
        pass

    @abstractmethod
    def transpose(self) -> "ModinDataframe":
        """
        Swap the row and column axes.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the row and column axes swapped.

        Notes
        -----
        Transposing a dataframe is expensive, and so it is performed lazily. The axes are swapped
        logically immediately, but the physical swap does not occur until absolutely necessary,
        which helps motivate the axis argument to the other operators in this algebra.
        """
        pass

    @abstractmethod
    def to_labels(self, column_labels: Union[str, List[str]]) -> "ModinDataframe":
        """
        Replace the row labels with one or more columns of data.

        Parameters
        ----------
        column_labels : string or list of strings
            Column label(s) to use as the new row labels.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the row labels replaced by the specified columns.

        Notes
        -----
        When multiple column labels are specified, a hierarchical set of labels is created, ordered by the ordering
        of labels in the input.
        """
        pass

    @abstractmethod
    def from_labels(self) -> "ModinDataframe":
        """
        Move the row labels into the data at position 0, and sets the row labels to the positional notation.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the row labels moved into the data.

        Notes
        -----
        In the case that the dataframe has hierarchical labels, all label "levels" are inserted into the dataframe
        in the order they occur in the labels, with the outermost being in position 0.
        """
        pass

    @abstractmethod
    def rename(
        self,
        new_row_labels: Optional[Union[Dict[Hashable, Hashable], Callable]] = None,
        new_col_labels: Optional[Union[Dict[Hashable, Hashable], Callable]] = None,
    ) -> "ModinDataframe":
        """
        Replace the row and column labels with the specified new labels.

        Parameters
        ----------
        new_row_labels : dictionary or callable, optional
            Mapping or callable that relates old row labels to new labels.
        new_col_labels : dictionary or callable, optional
            Mapping or callable that relates old col labels to new labels.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe with the new row and column labels.
        """
        pass

    @abstractmethod
    def sort_by(
        self,
        axis: Union[int, Axis],
        labels: Union[str, List[str]],
        ascending: bool = True,
    ) -> "ModinDataframe":
        """
        Logically reorder rows (columns if axis=1) lexicographically by the data in a column or set of columns.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to perform the sort over.
        labels : string or list of strings
            Column (row if axis=1) label(s) to use to determine lexicographical ordering. If multiple
            columns (rows if axis=1) are provided, the sort is performed on the first column (row if axis=1),
            with ties broken by the other columns (rows if axis=1) provided.
        ascending : boolean, default: True
            Whether to sort in ascending or descending order.

        Returns
        -------
        ModinDataframe
            A new ModinDataframe sorted into lexicographical order by the specified column(s).
        """
        pass
