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

"""Module houses ``DataFrame`` class, that is distributed version of ``polars.DataFrame``."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Sequence

import numpy as np
import pandas
import polars
from pandas.core.dtypes.common import is_list_like

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.pandas import DataFrame as ModinPandasDataFrame
from modin.pandas import Series as ModinPandasSeries
from modin.pandas.io import from_pandas
from modin.polars.base import BasePolarsDataset

if TYPE_CHECKING:
    from modin.polars import Series
    from modin.polars.groupby import GroupBy
    from modin.polars.lazyframe import LazyFrame


class DataFrame(BasePolarsDataset):

    def __init__(
        self,
        data=None,
        schema=None,
        *,
        schema_overrides=None,
        strict=True,
        orient=None,
        infer_schema_length=100,
        nan_to_null=False,
        _query_compiler=None,
    ) -> None:
        """
        Constructor for DataFrame object.

        Args:
            data: Data to be converted to DataFrame.
            schema: Schema of the data.
            schema_overrides: Schema overrides.
            strict: Whether to enforce strict schema.
            orient: Orientation of the data.
            infer_schema_length: Length of the data to infer schema.
            nan_to_null: Whether to convert NaNs to nulls.
            _query_compiler: Query compiler to use.
        """
        if _query_compiler is None:
            if isinstance(data, (ModinPandasDataFrame, ModinPandasSeries)):
                self._query_compiler: BaseQueryCompiler = data._query_compiler.copy()
            else:
                self._query_compiler: BaseQueryCompiler = from_pandas(
                    polars.DataFrame(
                        data=data,
                        schema=schema,
                        schema_overrides=schema_overrides,
                        strict=strict,
                        orient=orient,
                        infer_schema_length=infer_schema_length,
                        nan_to_null=nan_to_null,
                    ).to_pandas()
                )._query_compiler
        else:
            self._query_compiler: BaseQueryCompiler = _query_compiler

    def __getitem__(self, item):
        """
        Get item from DataFrame.

        Args:
            item: Column to get.

        Returns:
            Series or DataFrame with the column.
        """
        if is_list_like(item):
            missing = [i for i in item if i not in self.columns]
            if len(missing) > 0:
                raise polars.exceptions.ColumnNotFoundError(missing[0])
            return self.__constructor__(
                _query_compiler=self._query_compiler.getitem_array(item)
            )
        else:
            if item not in self.columns:
                raise polars.exceptions.ColumnNotFoundError(item)
            from .series import Series

            return Series(_query_compiler=self._query_compiler.getitem_array([item]))

    def _copy(self):
        return self.__constructor__(_query_compiler=self._query_compiler.copy())

    def _to_polars(self) -> polars.DataFrame:
        """
        Convert the DataFrame to Polars format.

        Returns:
            Polars representation of the DataFrame.
        """
        return polars.from_pandas(self._query_compiler.to_pandas())

    def _get_columns(self):
        """
        Get columns of the DataFrame.

        Returns:
            List of columns.
        """
        return list(self._query_compiler.columns)

    def _set_columns(self, new_columns):
        """
        Set columns of the DataFrame.

        Args:
            new_columns: New columns to set.
        """
        new_query_compiler = self._query_compiler.copy()
        new_query_compiler.columns = new_columns
        self._query_compiler = new_query_compiler

    columns = property(_get_columns, _set_columns)

    _sorted_columns_cache = None

    def _get_sorted_columns(self):
        if self._sorted_columns_cache is None:
            self._sorted_columns_cache = [False] * len(self.columns)
        return self._sorted_columns_cache

    def _set_sorted_columns(self, value):
        self._sorted_columns_cache = value

    _sorted_columns = property(_get_sorted_columns, _set_sorted_columns)

    @property
    def dtypes(self):
        """
        Get dtypes of the DataFrame.

        Returns:
            List of dtypes.
        """
        return polars.from_pandas(
            pandas.DataFrame(columns=self.columns).astype(self._query_compiler.dtypes)
        ).dtypes

    @property
    def flags(self):
        """
        Get flags of the DataFrame.

        Returns:
            List of flags.
        """
        # TODO: Add flags support
        return []

    @property
    def height(self):
        """
        Get height of the DataFrame.

        Returns:
            Number of rows in the DataFrame.
        """
        return len(self._query_compiler.index)

    @property
    def schema(self):
        """
        Get schema of the DataFrame.

        Returns:
            OrderedDict of column names and dtypes.
        """
        return OrderedDict(zip(self.columns, self.dtypes, strict=True))

    @property
    def shape(self):
        """
        Get shape of the DataFrame.

        Returns:
            Tuple of (height, width
        """
        return self.height, self.width

    @property
    def width(self):
        """
        Get width of the DataFrame.

        Returns:
            Number of columns in the DataFrame.
        """
        return len(self.columns)

    def __repr__(self):
        """
        Get string representation of the DataFrame.

        Returns:
            String representation of the DataFrame.
        """
        return repr(polars.from_pandas(self._query_compiler.to_pandas()))

    def max(self, axis=None):
        """
        Get the maximum value in each column.

        Args:
            axis: Axis to get the maximum value on.

        Returns:
            DataFrame with the maximum values.
        """
        if axis is None or axis == 0:
            return self.__constructor__(
                _query_compiler=self._query_compiler.max(axis=0)
            )
        else:
            return self.max_horizontal()

    def max_horizontal(self):
        """
        Get the maximum value in each row.

        Returns:
            DataFrame with the maximum values.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.max(axis=1))

    def _convert_non_numeric_to_null(self):
        """
        Convert non-numeric columns to null.

        Returns:
            DataFrame with non-numeric columns converted to null.
        """
        non_numeric_cols = [
            c
            for c, t in zip(self.columns, self.dtypes, strict=True)
            if not t.is_numeric()
        ]
        if len(non_numeric_cols) > 0:
            return self.__constructor__(
                _query_compiler=self._query_compiler.write_items(
                    slice(None),
                    [self.columns.index(c) for c in non_numeric_cols],
                    pandas.NA,
                    need_columns_reindex=False,
                ).astype({c: self._query_compiler.dtypes[c] for c in non_numeric_cols})
            )
        return self._copy()

    def mean(self, *, axis=None, null_strategy="ignore"):
        """
        Get the mean of each column.

        Args:
            axis: Axis to get the mean on.
            null_strategy: Strategy to handle null values.

        Returns:
            DataFrame with the mean of each column or row.
        """
        # TODO: this converts non numeric columns to numeric
        obj = self._convert_non_numeric_to_null()
        if axis is None or axis == 0:
            return self.__constructor__(
                _query_compiler=obj._query_compiler.mean(
                    axis=0,
                    skipna=True if null_strategy == "ignore" else False,
                )
            )
        else:
            return obj.mean_horizontal(
                ignore_nulls=True if null_strategy == "ignore" else False
            )

    def median(self) -> "DataFrame":
        """
        Get the median of each column.

        Returns:
            DataFrame with the median of each column.
        """
        return self.__constructor__(
            _query_compiler=self._convert_non_numeric_to_null()._query_compiler.median(
                0
            )
        )

    def mean_horizontal(self, *, ignore_nulls: bool = True):
        """
        Get the mean of each row.

        Args:
            ignore_nulls: Whether to ignore null values.

        Returns:
            DataFrame with the mean of each row.
        """
        obj = self._convert_non_numeric_to_null()
        return self.__constructor__(
            _query_compiler=obj._query_compiler.mean(axis=1, skipna=ignore_nulls)
        )

    def min(self, axis=None):
        """
        Get the minimum value in each column.

        Args:
            axis: Axis to get the minimum value on.

        Returns:
            DataFrame with the minimum values of each row or column.
        """
        if axis is None or axis == 0:
            return self.__constructor__(
                _query_compiler=self._query_compiler.min(axis=0)
            )
        else:
            return self.max_horizontal()

    def min_horizontal(self):
        """
        Get the minimum value in each row.

        Returns:
            DataFrame with the minimum values of each row.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.min(axis=1))

    def product(self):
        """
        Get the product of each column.

        Returns:
            DataFrame with the product of each column.
        """
        obj = self._convert_non_numeric_to_null()
        return self.__constructor__(_query_compiler=obj._query_compiler.prod(axis=0))

    def quantile(self, quantile: float, interpolation="nearest"):
        """
        Get the quantile of each column.

        Args:
            quantile: Quantile to get.
            interpolation: Interpolation method.

        Returns:
            DataFrame with the quantile of each column.
        """
        obj = self._convert_non_numeric_to_null()
        # TODO: interpolation support
        return self.__constructor__(
            _query_compiler=obj._query_compiler.quantile_for_single_value(quantile)
        )

    def std(self, ddof: int = 1):
        """
        Get the standard deviation of each column.

        Args:
            ddof: Delta degrees of freedom.

        Returns:
            DataFrame with the standard deviation of each column
        """
        obj = self._convert_non_numeric_to_null()
        return self.__constructor__(_query_compiler=obj._query_compiler.std(ddof=ddof))

    def sum(self, axis: int | None = None, null_strategy="ignore"):
        """
        Get the sum of each column.

        Args:
            axis: Axis to get the sum on.
            null_strategy: Strategy to handle null values.

        Returns:
            DataFrame with the sum of each column or row.
        """
        obj = self._convert_non_numeric_to_null()
        if axis is None or axis == 0:
            return self.__constructor__(
                _query_compiler=obj._query_compiler.sum(
                    axis=0,
                    skipna=True if null_strategy == "ignore" else False,
                )
            )
        else:
            return obj.sum_horizontal(
                ignore_nulls=True if null_strategy == "ignore" else False
            )

    def sum_horizontal(self, *, ignore_nulls: bool = True):
        """
        Get the sum of each row.

        Args:
            ignore_nulls: Whether to ignore null values.

        Returns:
            DataFrame with the sum of each row.
        """
        # TODO: if there are strings in the row, polars will append numeric values
        # this behavior may not be intended so doing this instead (for now)
        obj = self._convert_non_numeric_to_null()
        return self.__constructor__(
            _query_compiler=obj._query_compiler.sum(axis=1, skipna=ignore_nulls)
        )

    def var(self, ddof: int = 1):
        """
        Get the variance of each column.

        Args:
            ddof: Delta degrees of freedom.

        Returns:
            DataFrame with the variance of each column.
        """
        obj = self._convert_non_numeric_to_null()
        return self.__constructor__(_query_compiler=obj._query_compiler.var(ddof=ddof))

    def approx_n_unique(self):
        """
        Get the approximate number of unique values in each column.

        Returns:
            DataFrame with the approximate number of unique values in each column.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.nunique())

    def describe(self, percentiles: Sequence[float] | float = (0.25, 0.5, 0.75)):
        """
        Get the descriptive statistics of each column.

        Args:
            percentiles: Percentiles to get.

        Returns:
            DataFrame with the descriptive statistics of each column.
        """
        return self.__constructor__(
            self.__constructor__(
                _query_compiler=self._query_compiler.describe(
                    percentiles=np.array(percentiles)
                ).astype(
                    {
                        k: str
                        for k, v in zip(self.columns, self.dtypes, strict=True)
                        if v == polars.String
                    }
                )
            )
            .to_pandas()
            .loc[
                [
                    "count",
                    # "null_count",  TODO: support null_count in describe
                    "mean",
                    "std",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max",
                ]
            ]
            .reset_index()
            .rename({"index": "statistic"})
        )

    def estimated_size(self, unit="b"):
        """
        Get the estimated amount of memory used by the DataFrame.

        Args:
            unit: Unit of the memory size.

        Returns:
            DataFrame with the extimated memory usage.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.memory_usage())

    def glimpse(
        self,
        *,
        max_items_per_column: int = 10,
        max_colname_length: int = 50,
        return_as_string: bool = False,
    ) -> str | None:
        raise NotImplementedError("not yet")

    def n_unique(self, subset=None) -> int:
        """
        Get the number of unique values in each column.

        Args:
            subset: Columns to get the number of unique values for.

        Returns:
            Number of unique values in each column.
        """
        if subset is not None:
            raise NotImplementedError("not yet")
        return (
            self.is_unique()._query_compiler.sum(axis=0).to_pandas().squeeze(axis=None)
        )

    def null_count(self) -> "DataFrame":
        """
        Get the number of null values in each column.

        Returns:
            DataFrame with the number of null values in each column.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.isna().sum(axis=0)
        )

    def to_pandas(self):
        """
        Convert the DataFrame to Pandas format.

        Returns:
            modin.pandas representation of the DataFrame.
        """
        return ModinPandasDataFrame(query_compiler=self._query_compiler.copy())

    def group_by(
        self,
        *by,
        maintain_order: bool = False,
        **named_by,
    ) -> "GroupBy":
        """
        Group the DataFrame by the given columns.

        Args:
            by: Columns to group by.
            maintain_order: Whether to maintain the order of the groups.
            named_by: Named columns to group by.

        Returns:
            GroupBy object.
        """
        from modin.polars.groupby import GroupBy

        return GroupBy(self, *by, maintain_order=maintain_order, **named_by)

    def drop(self, *columns, strict: bool = True) -> "DataFrame":
        """
        Drop the given columns.

        Args:
            columns: Columns to drop.
            strict: Whether to raise an error if a column is not found.

        Returns:
            DataFrame with the columns dropped.
        """
        if strict:
            for c in columns:
                if c not in self.columns:
                    raise KeyError(c)
        columns = list(columns) if not isinstance(columns[0], list) else columns[0]
        return self.__constructor__(_query_compiler=self._query_compiler.drop(columns))

    def drop_in_place(self, name: str) -> "DataFrame":
        """
        Drop the given column in place and return the dropped column.

        Args:
            name: Column to drop.

        Returns:
            The column that was dropped from the DataFrame.
        """
        col_to_return = self[name]
        self._query_compiler = self._query_compiler.drop([name])
        return col_to_return

    def get_column(self, name: str) -> "Series":
        """
        Get the column by name.

        Args:
            name: Name of the column to get.

        Returns:
            Series with the column.
        """
        return self[name]

    def get_column_index(self, name: str) -> int:
        """
        Find the index of the column by name.

        Args:
            name: Name of the column to find.

        Returns:
            Index of the column.
        """
        return self.columns.index(name)

    def get_columns(self) -> list["Series"]:
        """
        Get the columns of the DataFrame.

        Returns:
            List of Series with the columns.
        """
        return [self[name] for name in self.columns]

    def group_by_dynamic(
        self,
        index_column,
        *,
        every,
        period,
        offset,
        truncate,
        include_boundaries,
        closed,
        label,
        group_by,
        start_by,
        check_sorted,
    ):
        raise NotImplementedError("not yet")

    def hstack(self, columns, *, inplace: bool = False) -> "DataFrame":
        """
        Stack the given columns horizontally.

        Args:
            columns: Columns to stack.
            inplace: Whether to stack the columns in place.

        Returns:
            DataFrame with the columns stacked horizontally.
        """
        if isinstance(columns, DataFrame):
            columns = columns.get_columns()
        result_query_compiler = self._query_compiler.concat(
            axis=1, other=[c._query_compiler for c in columns]
        )
        if inplace:
            self._query_compiler = result_query_compiler
            return self
        return self.__constructor__(_query_compiler=result_query_compiler)

    def insert_column(self, index: int, column: "Series") -> "DataFrame":
        """
        Insert the given column at the given index.

        Args:
            index: Index to insert the column at.
            column: Column to insert.
            name: Name of the column to insert.

        Returns:
            DataFrame with the column inserted.
        """
        return self.__constructor__(
            self._query_compiler.insert(index, column.name, column._query_compiler)
        )

    def item(self, row: int | None = None, column: str | int | None = None) -> Any:
        """
        Get the value at the given row and column.

        Args:
            row: Row to get the value from.
            column: Column to get the value from.

        Returns:
            Value at the given row and column.
        """
        if row is None:
            row = 0
        if column is None:
            column = 0
        if isinstance(column, str):
            column = self.columns.index(column)
        return (
            self._query_compiler.take_2d_labels(row, column)
            .to_pandas()
            .squeeze(axis=None)
        )

    def iter_columns(self) -> Iterator["Series"]:
        """
        Iterate over the columns of the DataFrame.

        Returns:
            Iterator over the columns.
        """
        return iter(self.get_columns())

    def iter_rows(
        self,
        *,
        named: bool = False,
        buffer_size: int = 512,
    ) -> Iterator[tuple[Any]] | Iterator[dict[str, Any]]:
        """
        Iterate over the rows of the DataFrame.

        Returns:
            Iterator over the rows.
        """
        raise NotImplementedError("not yet")

    def iter_slices(
        self,
        n_rows: int = 10000,
    ) -> Iterator["DataFrame"]:
        """
        Iterate over the slices of the DataFrame.

        Args:
            n_rows: Number of rows in each slice.

        Returns:
            Iterator over the slices.
        """
        raise NotImplementedError("not yet")

    def join(
        self,
        other: "DataFrame",
        on: str | list[str] | None = None,
        how: str = "inner",
        *,
        left_on: str | list[str] | None = None,
        right_on: str | list[str] | None = None,
        suffix: str = "_right",
        validate="m:m",
        join_nulls: bool = False,
        coalesce: bool | None = None,
    ) -> "DataFrame":
        """
        Join the DataFrame with another DataFrame.

        Args:
            other: DataFrame to join with.
            on: Column to join on.
            how: How to join the DataFrames.

        Returns:
            Joined DataFrame.
        """
        if how == "full":
            how = "outer"
        elif how == "cross":
            raise NotImplementedError("not yet")
        elif how == "semi":
            how = "right"
        elif how == "anti":
            raise NotImplementedError("not yet")
        return self.__constructor__(
            _query_compiler=self._query_compiler.merge(
                other._query_compiler,
                on=on,
                how=how,
                suffixes=("", suffix),
                left_on=left_on,
                right_on=right_on,
            )
        )

    def join_asof(
        self,
        other: "DataFrame",
        *,
        left_on: str | None = None,
        right_on: str | None = None,
        on: str | None = None,
        by_left: str | Sequence[str] | None = None,
        by_right: str | Sequence[str] | None = None,
        by: str | Sequence[str] | None = None,
        strategy: str = "backward",
        suffix: str = "_right",
        tolerance: str,
    ) -> "DataFrame":
        """
        Join the DataFrame with another DataFrame using asof logic.

        Args:
            other: DataFrame to join with.
            left_on: Column to join on in the left DataFrame.
            right_on: Column to join on in the right DataFrame.
            on: Column to join on in both DataFrames.
            by_left: Columns to join on in the left DataFrame.
            by_right: Columns to join on in the right DataFrame.
            by: Columns to join on in both DataFrames.
            strategy: Strategy to use for the join.
            suffix: Suffix to add to the columns.
            tolerance: Tolerance for the join.

        Returns:
            Joined DataFrame.
        """
        if on is not None and left_on is None and right_on is None:
            left_on = right_on = on
        if by is not None and by_left is None and by_right is None:
            by_left = by_right = by
        return self.__constructor__(
            _query_compiler=self._query_compiler.merge_asof(
                other._query_compiler,
                left_on=left_on,
                right_on=right_on,
                left_by=by_left,
                right_by=by_right,
                direction=strategy,
                suffixes=("", suffix),
                tolerance=tolerance,
            )
        )

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        variable_name: str | None = None,
        value_name: str | None = None,
    ) -> "DataFrame":
        """
        Melt the DataFrame.

        Args:
            id_vars: Columns to keep.
            value_vars: Columns to melt.
            variable_name: Name of the variable column.
            value_name: Name of the value column.

        Returns:
            Melted DataFrame.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name=variable_name,
                value_name=value_name,
            )
        )

    def merge_sorted(self, other: "DataFrame", on: str | list[str]) -> "DataFrame":
        # TODO: support natural join + sort
        raise NotImplementedError("not yet")

    def partition_by(
        self,
        by,
        *more_by,
        maintain_order: bool = True,
        include_key: bool = True,
        as_dict: bool = False,
    ) -> list["DataFrame"] | dict[Any, "DataFrame"]:
        """
        Partition the DataFrame by the given columns.

        Args:
            by: Columns to partition by.
            more_by: Additional columns to partition by.
            maintain_order: Whether to maintain the order of the partitions.
            include_key: Whether to include the partition key.
            as_dict: Whether to return the partitions as a dictionary.

        Returns:
            List of DataFrames or dictionary of DataFrames.
        """
        if isinstance(by, str):
            by = [by, *more_by]
        elif isinstance(by, list):
            by = [*by, *more_by]
        if as_dict:
            return {
                k: self.__constructor__(v)
                for k, v in self.to_pandas()
                .groupby(by, as_index=not include_key)
                .groups
            }
        else:
            return [
                self.__constructor__(g)
                for g in self.to_pandas().groupby(by, as_index=not include_key)
            ]

    def pipe(self, function, *args, **kwargs) -> Any:
        return function(self, *args, **kwargs)

    def pivot(
        self,
        *,
        values,
        index,
        columns,
        aggregate_function=None,
        maintain_order: bool = True,
        sort_columns: bool = False,
        separator: str = "_",
    ) -> "DataFrame":
        """
        Pivot the DataFrame.

        Args:
            values: Values to pivot.
            index: Index columns.
            columns: Columns to pivot.
            aggregate_function: Function to aggregate the values.
            maintain_order: Whether to maintain the order of the pivot.
            sort_columns: Whether to sort the columns.
            separator: Separator for the columns.

        Returns:
            Pivoted DataFrame.
        """
        # TODO: handle maintain_order, sort_columns, separator
        return self.__constructor__(
            _query_compiler=self._query_compiler.pivot(
                values=values,
                index=index,
                columns=columns,
                agg=aggregate_function,
            )
        )

    def rechunk(self) -> "DataFrame":
        """
        Rechunk the DataFrame into the given number of partitions.

        Returns:
            Rechunked DataFrame.
        """
        return self._copy()

    def rename(self, mapping: dict[str, str] | callable) -> "DataFrame":
        """
        Rename the columns of the DataFrame.

        Args:
            mapping: Mapping of old names to new names.

        Returns:
            DataFrame with the columns renamed.
        """
        if callable(mapping):
            mapping = {c: mapping(c) for c in self.columns}
        # TODO: add a query compiler method for `rename`
        new_columns = {c: mapping.get(c, c) for c in self.columns}
        new_obj = self._copy()
        new_obj.columns = new_columns
        return new_obj

    def replace_column(self, index: int, column: "Series") -> "DataFrame":
        """
        Replace the column at the given index with the new column.

        Args:
            index: Index of the column to replace.
            column: New column to replace with.

        Returns:
            DataFrame with the column replaced.
        """
        self._query_compiler = self._query_compiler.drop([self.columns[index]]).insert(
            index,
            column.name,
            column._query_compiler,
        )
        return self

    def reverse(self) -> "DataFrame":
        """
        Reverse the DataFrame.

        Returns:
            Reversed DataFrame.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.getitem_row_array(
                slice(None, None, -1)
            )
        )

    def rolling(self, index_column, *, period, offset, closed, group_by, check_sorted):
        raise NotImplementedError("not yet")

    def row(
        self, index: int | None = None, *, by_predicate=None, named: bool = False
    ) -> tuple[Any] | dict[str, Any]:
        """
        Get the row at the given index.

        Args:
            index: Index of the row to get.
            by_predicate: Predicate to get the row by.
            named: Whether to return the row as a dictionary.

        Returns:
            Row at the given index.
        """
        if index is not None:
            if named:
                return dict(self.to_pandas().iloc[index])
            else:
                return tuple(self.to_pandas().iloc[index])
        else:
            # TODO: support expressions
            raise NotImplementedError("not yet")

    def rows(self, *, named: bool = False) -> list[tuple[Any]] | list[dict[str, Any]]:
        raise NotImplementedError("not yet")

    def rows_by_key(
        self,
        key: Any,
        *,
        named: bool = False,
        include_key: bool = False,
        unique: bool = False,
    ) -> dict[Any, Iterable[Any]]:
        raise NotImplementedError("not yet")

    def select(self, *exprs, **named_exprs) -> "DataFrame":
        # TODO: support expressions
        raise NotImplementedError("not yet")

    def select_seq(self, *exprs, **named_exprs) -> "DataFrame":
        # TODO: support expressions
        raise NotImplementedError("not yet")

    def set_sorted(
        self, column: str | Iterable[str], *more_columns: str, descending: bool = False
    ) -> "DataFrame":
        """
        Set the columns to be sorted.

        Args:
            column: Column to sort by.
            more_columns: Additional columns to sort by.
            descending: Whether to sort in descending order.

        Returns:
            DataFrame with the columns sorted.
        """
        if len(more_columns) > 0:
            if isinstance(column, Iterable):
                column = [*column, *more_columns]
            else:
                column = [column, *more_columns]
        if isinstance(column, str):
            column = [column]
        new_sorted_columns = [c in column for c in self.columns]
        obj = self._copy()
        obj._sorted_columns = new_sorted_columns
        return obj

    def sql(self, query: str, *, table_name: str = "self") -> "DataFrame":
        raise NotImplementedError("not yet")

    def to_series(self, index: int = 0) -> "Series":
        """
        Convert the DataFrame at index provided to a Series.

        Args:
            index: Index of the column to convert to a Series.

        Returns:
            Series representation of the DataFrame at index provided.
        """
        return self[self.columns[index]]

    def transpose(
        self,
        *,
        include_header: bool = False,
        header_name: str = "column",
        column_names: str | Sequence[str] | None = None,
    ) -> "DataFrame":
        """
        Transpose the DataFrame.

        Args:
            include_header: Whether to include a header.
            header_name: Name of the header.
            column_names: Names of the columns.

        Returns:
            Transposed DataFrame.
        """
        result = self.__constructor__(_query_compiler=self._query_compiler.transpose())
        if column_names is not None:
            result.columns = column_names
        elif include_header:
            result.columns = [f"{header_name}_{i}" for i in range(result.width)]
        return result

    def unnest(self, columns, *more_columns) -> "DataFrame":
        """
        Unnest the given columns.

        Args:
            columns: Columns to unnest.
            more_columns: Additional columns to unnest.

        Returns:
            DataFrame with the columns unnested.
        """
        raise NotImplementedError("not yet")

    def unstack(
        self,
        step: int,
        how: str = "vertical",
        columns=None,
        fill_values: list[Any] | None = None,
    ):
        """
        Unstack the DataFrame.

        Args:
            step: Step to unstack by.
            how: How to unstack the DataFrame.
            columns: Columns to unstack.
            fill_values: Values to fill the unstacked DataFrame with.

        Returns:
            Unstacked DataFrame.
        """
        raise NotImplementedError("not yet")

    def update(
        self,
        other: "DataFrame",
        on: str | Sequence[str] | None = None,
        how: Literal["left", "inner", "full"] = "left",
        *,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        include_nulls: bool = False,
    ) -> "DataFrame":
        """
        Update the DataFrame with another DataFrame.

        Args:
            other: DataFrame to update with.
            on: Column to update on.
            how: How to update the DataFrame.

        Returns:
            Updated DataFrame.
        """
        raise NotImplementedError("not yet")

    def upsample(
        self,
        time_column: str,
        *,
        every: str,
        offset: str | None = None,
        group_by: str | Sequence[str] | None = None,
        maintain_order: bool = False,
    ) -> "DataFrame":
        raise NotImplementedError("not yet")

    def vstack(self, other: "DataFrame", *, in_place: bool = False) -> "DataFrame":
        """
        Stack the given DataFrame vertically.

        Args:
            other: DataFrame to stack.
            in_place: Whether to stack the DataFrames in place.

        Returns:
            Stacked DataFrame.
        """
        if in_place:
            self._query_compiler = self._query_compiler.concat(
                axis=0, other=other._query_compiler
            )
            return self
        else:
            return self.__constructor__(
                _query_compiler=self._query_compiler.concat(
                    axis=0, other=other._query_compiler
                )
            )

    def with_columns(self, *exprs, **named_exprs) -> "DataFrame":
        # TODO: support expressions
        raise NotImplementedError("not yet")

    def with_columns_seq(self, *exprs, **named_exprs) -> "DataFrame":
        # TODO: support expressions
        raise NotImplementedError("not yet")

    def with_row_index(self, name: str = "index", offset: int = 0) -> "DataFrame":
        """
        Add a row index to the DataFrame.

        Args:
            name: Name of the row index.
            offset: Offset for the row index.

        Returns:
            DataFrame with the row index added.
        """
        if offset != 0:
            obj = self._copy()
            obj.index = obj.index + offset
        result = self.__constructor__(
            _query_compiler=self._query_compiler.reset_index(drop=False)
        )
        result.columns = [name, *self.columns]
        return result

    with_row_count = with_row_index

    def map_rows(
        self, function: callable, return_dtype=None, *, inference_size: int = 256
    ) -> "DataFrame":
        """
        Apply the given function to the DataFrame.

        Args:
            function: Function to apply.
            return_dtype: Return type of the function.
            inference_size: Size of the inference.

        Returns:
            DataFrame with the function applied.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.apply(function, axis=1)
        )

    def corr(self, **kwargs: Any) -> "DataFrame":
        """
        Compute the correlation of the DataFrame.

        Returns:
            DataFrame with the correlation.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.corr(**kwargs))

    def lazy(self) -> "LazyFrame":
        """
        Convert the DataFrame to a lazy DataFrame.

        Returns:
            Lazy DataFrame.
        """
        raise NotImplementedError("not yet")

    @classmethod
    def deserialize(cls, source) -> "DataFrame":
        """
        Deserialize the DataFrame.

        Args:
            source: Source to deserialize.

        Returns:
            Deserialized DataFrame.
        """
        return cls(polars.DataFrame.deserialize(source))

    def serialize(self, file=None) -> str | None:
        """
        Serialize the DataFrame.

        Args:
            file: File to serialize to.

        Returns:
            Serialized DataFrame.
        """
        return polars.from_pandas(self._query_compiler.to_pandas()).serialize(file)

    @property
    def style(self):
        """
        Create a Great Table for styling.

        Returns:
            GreatTable object.
        """
        return self._to_polars().style

    def to_dict(
        self, *, as_series: bool = True
    ) -> dict[str, "Series"] | dict[str, list[Any]]:
        """
        Convert the DataFrame to a dictionary representation.

        Args:
            as_series: Whether to convert the columns to Series.

        Returns:
            Dictionary representation of the DataFrame.
        """
        if as_series:
            return {name: self[name] for name in self.columns}
        else:
            return polars.from_pandas(self._query_compiler.to_pandas()).to_dict(
                as_series=as_series
            )

    def to_dicts(self) -> list[dict[str, Any]]:
        """
        Convert the DataFrame to a list of dictionaries.

        Returns:
            List of dictionaries.
        """
        return self._to_polars().to_dicts()

    def to_init_repr(self, n: int = 1000) -> str:
        """
        Get the string representation of the DataFrame for initialization.

        Returns:
            String representation of the DataFrame for initialization.
        """
        return self._to_polars().to_init_repr(n)

    def to_struct(self, name: str = "") -> "Series":
        """
        Convert the DataFrame to a struct.

        Args:
            name: Name of the struct.

        Returns:
            Series representation of the DataFrame as a struct.
        """
        raise NotImplementedError("not yet")

    def unpivot(
        self,
        on,
        *,
        index,
        variable_name: str | None = None,
        value_name: str | None = None,
    ) -> "DataFrame":
        """
        Unpivot a DataFrame from wide to long format.

        Args:
            on: Columns to unpivot.
            index: Columns to keep.
            variable_name: Name of the variable column.
            value_name: Name of the value column.

        Returns:
            Unpivoted DataFrame.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.melt(
                on=on,
                index=index,
                var_name=variable_name,
                value_name=value_name,
            )
        )

    write_avro = write_clipboard = write_csv = write_database = write_delta = (
        write_excel
    ) = write_ipc = write_ipc_stream = write_json = write_ndjson = write_parquet = (
        write_parquet_partitioned
    ) = lambda *args, **kwargs: (_ for _ in ()).throw(NotImplementedError("not yet"))

    def clear(self, n: int = 0) -> "DataFrame":
        """
        Create an empty (n=0) or null filled (n>0) DataFrame.

        Args:
            n: Number of rows to create.

        Returns:
            Empty or null filled DataFrame.
        """
        return self.__constructor__(polars.DataFrame(schema=self.schema).clear(n=n))

    def collect_schema(self) -> dict[str, str]:
        """
        Collect the schema of the DataFrame.

        Returns:
            Dictionary of the schema.
        """
        return self.schema

    def fold(self, operation: callable) -> "Series":
        """
        Fold the DataFrame.

        Args:
            operation: Operation to fold the DataFrame with.

        Returns:
            Series with the folded DataFrame.
        """
        raise NotImplementedError("not yet")

    def hash_rows(
        self,
        seed: int = 0,
        seed_1: int | None = None,
        seed_2: int | None = None,
        seed_3: int | None = None,
    ) -> "Series":
        raise NotImplementedError("not yet")
