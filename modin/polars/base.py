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

"""Implement DataFrame/Series public API as polars does."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import polars

from modin.core.storage_formats import BaseQueryCompiler

if TYPE_CHECKING:
    import numpy as np

    from modin.polars import DataFrame, Series


class BasePolarsDataset:

    _query_compiler: BaseQueryCompiler

    @property
    def __constructor__(self):
        """
        DataFrame constructor.

        Returns:
            Constructor of the DataFrame
        """
        return type(self)

    def __eq__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.eq(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __ne__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.ne(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __add__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.add(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __sub__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.sub(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __mul__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.mul(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __truediv__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.truediv(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __floordiv__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.floordiv(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __mod__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.mod(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __pow__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.pow(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __and__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.__and__(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __or__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.__or__(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __xor__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.__xor__(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __lt__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.lt(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __le__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.le(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __gt__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.gt(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __ge__(self, other) -> "BasePolarsDataset":
        return self.__constructor__(
            _query_compiler=self._query_compiler.ge(
                other._query_compiler if isinstance(other, BasePolarsDataset) else other
            )
        )

    def __invert__(self) -> "BasePolarsDataset":
        return self.__constructor__(_query_compiler=self._query_compiler.invert())

    def __neg__(self) -> "BasePolarsDataset":
        return self.__constructor__(_query_compiler=self._query_compiler.negative())

    def __abs__(self) -> "BasePolarsDataset":
        return self.__constructor__(_query_compiler=self._query_compiler.abs())

    def is_duplicated(self):
        """
        Determine whether each row is a duplicate in the DataFrame.

        Returns:
            DataFrame with True for each duplicate row, and False for unique rows.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.duplicated(keep=False)
        )

    def is_empty(self) -> bool:
        """
        Determine whether the DataFrame is empty.

        Returns:
            True if the DataFrame is empty, False otherwise
        """
        return self.height == 0

    def is_unique(self):
        """
        Determine whether each row is unique in the DataFrame.

        Returns:
            DataFrame with True for each unique row, and False for duplicate rows.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.duplicated(keep=False).invert()
        )

    def n_chunks(self, strategy: str = "first") -> int | list[int]:
        raise NotImplementedError("not yet")

    def to_arrow(self):
        """
        Convert the DataFrame to Arrow format.

        Returns:
            Arrow representation of the DataFrame.
        """
        return polars.from_pandas(self._query_compiler.to_pandas()).to_arrow()

    def to_jax(self, device=None):
        """
        Convert the DataFrame to JAX format.

        Args:
            device: The device to use.

        Returns:
            JAX representation of the DataFrame.
        """
        return polars.from_pandas(self._query_compiler.to_pandas()).to_jax(
            device=device
        )

    def to_numpy(
        self,
        *,
        writable: bool = False,
        allow_copy: bool = True,
        use_pyarrow: bool | None = None,
        zero_copy_only: bool | None = None,
    ) -> "np.ndarray":
        """
        Convert the DataFrame to a NumPy representation.

        Args:
            writable: Whether the NumPy array should be writable.
            allow_copy: Whether to allow copying the data.
            use_pyarrow: Whether to use PyArrow for conversion.
            zero_copy_only: Whether to use zero-copy conversion only.

        Returns:
            NumPy representation of the DataFrame.
        """
        return polars.from_pandas(self._query_compiler.to_pandas()).to_numpy(
            writable=writable,
            allow_copy=allow_copy,
            use_pyarrow=use_pyarrow,
            zero_copy_only=zero_copy_only,
        )

    def to_torch(self):
        """
        Convert the DataFrame to PyTorch format.

        Returns:
            PyTorch representation of the DataFrame.
        """
        return polars.from_pandas(self._query_compiler.to_pandas()).to_torch()

    def bottom_k(
        self,
        k: int,
        *,
        by,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] | None = None,
        maintain_order: bool | None = None,
    ) -> "BasePolarsDataset":
        raise NotImplementedError("not yet")

    def cast(self, dtypes, *, strict: bool = True) -> "BasePolarsDataset":
        """
        Cast the DataFrame to the given dtypes.

        Args:
            dtypes: Dtypes to cast the DataFrame to.
            strict: Whether to enforce strict casting.

        Returns:
            DataFrame with the new dtypes.
        """
        # TODO: support strict
        return self.__constructor__(_query_compiler=self._query_compiler.astype(dtypes))

    def clone(self) -> "BasePolarsDataset":
        """
        Clone the DataFrame.

        Returns:
            Cloned DataFrame.
        """
        return self.copy()

    def drop_nulls(self, subset=None):
        """
        Drop the rows with null values.

        Args:
            subset: Columns to consider for null values.

        Returns:
            DataFrame with the rows with null values dropped.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.dropna(subset=subset, how="any")
        )

    def explode(self, columns: str, *more_columns: str) -> "BasePolarsDataset":
        """
        Explode the given columns to long format.

        Args:
            columns: Columns to explode.
            more_columns: Additional columns to explode.

        Returns:
            DataFrame with the columns exploded.
        """
        if len(more_columns) > 0:
            columns = [columns, *more_columns]
        return self.__constructor__(
            _query_compiler=self._query_compiler.explode(columns)
        )

    def extend(self, other: "BasePolarsDataset") -> "BasePolarsDataset":
        """
        Extend the DataFrame with another DataFrame.

        Args:
            other: DataFrame to extend with.

        Returns:
            Extended DataFrame for convenience. DataFrame is modified in place.
        """
        self._query_compiler = self._query_compiler.concat(
            axis=0, other=other._query_compiler
        )
        return self

    def fill_nan(self, value):
        """
        Fill NaN values with the given value.

        Args:
            value: Value to fill NaN values with.

        Returns:
            DataFrame with NaN values filled.
        """
        # TODO: Handle null values differently than nan.
        return self.__constructor__(_query_compiler=self._query_compiler.fillna(value))

    def fill_null(
        self,
        value: Any | None = None,
        strategy: str | None = None,
        limit: int | None = None,
        *,
        matches_supertype: bool = True,
    ) -> "BasePolarsDataset":
        """
        Fill null values with the given value or strategy.

        Args:
            value: Value to fill null values with.
            strategy: Strategy to fill null values with.
            limit: Maximum number of null values to fill.
            matches_supertype: Whether the value matches the supertype.

        Returns:
            DataFrame with null values filled.
        """
        if strategy == "forward":
            strategy = "ffill"
        elif strategy == "backward":
            strategy = "bfill"
        elif strategy in ["min", "max", "mean"]:
            value = getattr(self, strategy)()._query_compiler
            strategy = None
        elif strategy == "zero":
            strategy = None
            value = 0
        elif strategy == "one":
            strategy = None
            value = 1
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return self.__constructor__(
            _query_compiler=self._query_compiler.fillna(
                value=value, method=strategy, limit=limit
            )
        )

    def filter(self, *predicates, **constraints: Any) -> "BasePolarsDataset":
        predicates = predicates[0]
        for p in predicates[1:]:
            predicates = predicates & p
        if constraints:
            raise NotImplementedError("Named constraints are not supported")
        return self.__constructor__(
            _query_compiler=self._query_compiler.getitem_array(
                predicates._query_compiler
            )
        )

    def gather_every(self, n: int, offset: int = 0) -> "BasePolarsDataset":
        """
        Gather every nth row of the DataFrame.

        Args:
            n: Number of rows to gather.
            offset: Offset to start gathering from.

        Returns:
            DataFrame with every nth row gathered.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.getitem_row_array(
                slice(offset, None, n)
            )
        )

    def head(self, n: int = 5) -> "BasePolarsDataset":
        """
        Get the first n rows of the DataFrame.

        Args:
            n: Number of rows to get.

        Returns:
            DataFrame with the first n rows.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.getitem_row_array(slice(0, n))
        )

    def limit(self, n: int = 10) -> "BasePolarsDataset":
        """
        Limit the DataFrame to the first n rows.

        Args:
            n: Number of rows to limit to.

        Returns:
            DataFrame with the first n rows.
        """
        return self.head(n)

    def interpolate(self) -> "BasePolarsDataset":
        """
        Interpolate values the DataFrame using a linear method.

        Returns:
            DataFrame with the interpolated values.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.interpolate())

    def sample(
        self,
        n: int | "Series" | None = None,
        *,
        fraction: float | "Series" | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> "BasePolarsDataset":
        """
        Sample the DataFrame.

        Args:
            n: Number of rows to sample.
            fraction: Fraction of rows to sample.
            with_replacement: Whether to sample with replacement.
            shuffle: Whether to shuffle the rows.
            seed: Seed for the random number generator.

        Returns:
            Sampled DataFrame.
        """
        return self.__constructor__(
            _query_compiler=self.to_pandas()
            .sample(n=n, frac=fraction, replace=with_replacement, random_state=seed)
            ._query_compiler
        )

    def shift(self, n: int = 1, *, fill_value=None) -> "DataFrame":
        raise NotImplementedError("not yet")

    def shrink_to_fit(self) -> "DataFrame":
        """
        Shrink the DataFrame to fit in memory.

        Returns:
            A copy of the DataFrame.
        """
        return self.copy()

    def slice(self, offset: int, length: int) -> "DataFrame":
        """
        Slice the DataFrame.

        Args:
            offset: Offset to start the slice from.
            length: Length of the slice.

        Returns:
            Sliced DataFrame.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.getitem_row_array(
                slice(offset, offset + length)
            )
        )

    def sort(
        self,
        by,
        *more_by,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] | None = None,
        multithreaded: bool = True,
        maintain_order: bool = False,
    ) -> "DataFrame":
        """
        Sort the DataFrame.

        Args:
            by: Column to sort by.
            more_by: Additional columns to sort by.
            descending: Whether to sort in descending order.
            nulls_last: Whether to sort null values last.
            multithreaded: Whether to use multiple threads.
            maintain_order: Whether to maintain the order of the DataFrame.

        Returns:
            Sorted DataFrame.
        """
        # TODO: support expressions in by
        if len(more_by) > 0:
            by = [by, *more_by]
        return self.__constructor__(
            _query_compiler=self._query_compiler.sort_rows_by_column_values(
                by=by,
                reverse=descending,
                nulls_first=None if nulls_last is None else not nulls_last,
            )
        )

    def tail(self, n: int = 5) -> "DataFrame":
        """
        Get the last n rows of the DataFrame.

        Args:
            n: Number of rows to get.

        Returns:
            DataFrame with the last n rows.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.getitem_row_array(slice(-n, None))
        )

    def to_dummies(
        self,
        columns: str | Sequence[str] | None = None,
        *,
        separator: str = "_",
        drop_first: bool = False,
    ) -> "DataFrame":
        """
        Convert the columns to dummy variables.

        Args:
            columns: Columns to convert to dummy variables.
            separator: Separator for the dummy variables.
            drop_first: Whether to drop the first dummy variable.

        Returns:
            DataFrame with the columns converted to dummy variables.
        """
        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
        else:
            columns = self.columns
        result = self.__constructor__(
            _query_compiler=self._query_compiler.get_dummies(columns)
        )
        if separator != "_":
            result.columns = [
                c.replace(separator, "_") if separator in c else c
                for c in result.columns
            ]
        if drop_first:
            columns_to_drop = [
                next(
                    result_col
                    for result_col in result.columns
                    if result_col.startswith(c)
                )
                for c in columns
            ]
            return result.drop(columns_to_drop)
        else:
            return result

    def top_k(
        self,
        k: int,
        *,
        by,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] | None = None,
        maintain_order: bool | None = None,
    ) -> "DataFrame":
        raise NotImplementedError("not yet")

    def unique(self, subset=None, *, keep="any", maintain_order: bool = False):
        """
        Get the unique values in each column.

        Args:
            subset: Columns to consider for unique values.
            keep: Strategy to keep unique values.
            maintain_order: Whether to maintain the order of the unique values.

        Returns:
            DataFrame with the unique values in each column.
        """
        if keep == "none" or keep == "last":
            # TODO: support keep="none"
            raise NotImplementedError("not yet")
        return self.__constructor__(
            _query_compiler=self._query_compiler.unique(subset=subset)
        )

    def equals(self, other: "BasePolarsDataset", *, null_equal: bool = True) -> bool:
        """
        Determine whether the DataFrame is equal to another DataFrame.

        Args:
            other: DataFrame to compare with.

        Returns:
            True if the DataFrames are equal, False otherwise.
        """
        return (
            isinstance(other, type(self))
            and self._query_compiler.equals(other._query_compiler)
            and (
                null_equal
                or (
                    not self.to_pandas().isna().any(axis=None)
                    and not other.to_pandas().isna().any(axis=None)
                )
            )
        )

    @property
    def plot(self):
        return polars.from_pandas(self._query_compiler.to_pandas()).plot

    def count(self):
        """
        Get the number of non-null values in each column.

        Returns:
            DataFrame with the counts.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.count(axis=0))
