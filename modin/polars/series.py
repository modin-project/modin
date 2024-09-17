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

"""Module houses `Series` class, that is distributed version of `polars.Series`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas
import polars
from polars._utils.various import no_default

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.pandas import Series as ModinPandasSeries
from modin.pandas.io import from_pandas
from modin.polars.base import BasePolarsDataset

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from polars import PolarsDataType

    from modin.polars import DataFrame


class Series(BasePolarsDataset):
    def __init__(
        self,
        name: str | "ArrayLike" | None = None,
        values: "ArrayLike" | None = None,
        dtype: "PolarsDataType | None" = None,
        *,
        strict: "bool" = True,
        nan_to_null: "bool" = False,
        dtype_if_empty: "PolarsDataType" = polars.Null,
        _query_compiler: BaseQueryCompiler | None = None,
    ) -> None:
        if _query_compiler is None:
            if isinstance(values, ModinPandasSeries):
                self._query_compiler = values._query_compiler.copy()
            else:
                self._query_compiler: BaseQueryCompiler = from_pandas(
                    polars.Series(
                        name=name,
                        values=values,
                        dtype=dtype,
                        strict=strict,
                        nan_to_null=nan_to_null,
                        dtype_if_empty=dtype_if_empty,
                    )
                    .to_pandas()
                    .to_frame()
                )._query_compiler
        else:
            self._query_compiler: BaseQueryCompiler = _query_compiler

    def __repr__(self):
        return repr(
            polars.from_pandas(self._query_compiler.to_pandas().squeeze(axis=1))
        )

    _sorted = False
    _descending = None

    def to_pandas(self) -> ModinPandasSeries:
        return ModinPandasSeries(query_compiler=self._query_compiler)

    def arg_max(self) -> int:
        """
        Get the index of the maximum value.

        Returns:
            Index of the maximum value.
        """
        return self.to_pandas().argmax()

    def arg_min(self) -> int:
        """
        Get the index of the minimum value.

        Returns:
            Index of the minimum value.
        """
        return self.to_pandas().argmin()

    def implode(self) -> "Series":
        """
        Aggregate values into a list.

        Returns:
            Imploded Series.
        """
        raise NotImplementedError("not yet")

    def max(self) -> Any:
        """
        Get the maximum value.

        Returns:
            Maximum value.
        """
        return self.to_pandas().max()

    def min(self) -> Any:
        """
        Get the minimum value.

        Returns:
            Minimum value.
        """
        return self.to_pandas().min()

    def mean(self) -> Any:
        """
        Get the mean value.

        Returns:
            Mean value.
        """
        return self.to_pandas().mean()

    def median(self) -> Any:
        """
        Get the median value.

        Returns:
            Median value.
        """
        return self.to_pandas().median()

    def mode(self) -> Any:
        """
        Get the mode value.

        Returns:
            Mode value.
        """
        return self.to_pandas().mode()

    def nan_max(self) -> Any:
        """
        Get the maximum value, ignoring NaN values.

        Returns:
            Maximum value.
        """
        return self.to_pandas().max(skipna=True)

    def nan_min(self) -> Any:
        """
        Get the minimum value, ignoring NaN values.

        Returns:
            Minimum value.
        """
        return self.to_pandas().min(skipna=True)

    def product(self) -> Any:
        """
        Get the product of all values.

        Returns:
            Product of all values.
        """
        return self.to_pandas().product()

    def quantile(self, quantile: float, interpolation: str = "nearest") -> float | None:
        """
        Get the quantile value.

        Args:
            quantile: Quantile to calculate.
            interpolation: Interpolation method.

        Returns:
            Quantile value.
        """
        return self.to_pandas().quantile(quantile, interpolation=interpolation)

    def std(self, ddof: int = 1) -> float:
        """
        Get the standard deviation.

        Args:
            ddof: Delta Degrees of Freedom.

        Returns:
            Standard deviation.
        """
        return self.to_pandas().std(ddof=ddof)

    def sum(self) -> Any:
        """
        Get the sum of all values.

        Returns:
            Sum of all values.
        """
        return self.to_pandas().sum()

    def var(self, ddof: int = 1) -> float:
        """
        Get the variance.

        Args:
            ddof: Delta Degrees of Freedom.

        Returns:
            Variance.
        """
        return self.to_pandas().var(ddof=ddof)

    @property
    def arr(self) -> polars.series.array.ArrayNameSpace:
        """
        Get the underlying array.

        Returns:
            Underlying array.
        """
        return polars.from_pandas(self._query_compiler.to_pandas().squeeze(axis=1)).arr

    @property
    def dtype(self) -> polars.datatypes.DataType:
        """
        Get the data type.

        Returns:
            Data type.
        """
        return polars.from_pandas(
            pandas.Series().astype(self._query_compiler.dtypes.iloc[0])
        ).dtype

    @property
    def name(self) -> str:
        """
        Get the name.

        Returns:
            Name.
        """
        return self._query_compiler.columns[0]

    @property
    def shape(self) -> tuple[int]:
        """
        Get the shape.

        Returns:
            Shape.
        """
        return (len(self._query_compiler.index),)

    flags = []

    @property
    def bin(self):
        raise NotImplementedError("not yet")

    def all(self) -> bool:
        """
        Check if all values are True.

        Returns:
            True if all values are True, False otherwise.
        """
        return self.to_pandas().all()

    def any(self) -> bool:
        """
        Check if any value is True.

        Returns:
            True if any value is True, False otherwise.
        """
        return self.to_pandas().any()

    def not_(self) -> "Series":
        """
        Negate the values.

        Returns:
            Negated Series.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.invert())

    @property
    def cat(self):
        raise NotImplementedError("not yet")

    def abs(self) -> "Series":
        """
        Get the absolute values.

        Returns:
            Absolute values Series.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.abs())

    def arccos(self) -> "Series":
        """
        Get the arc cosine values.

        Returns:
            Arc cosine values Series.
        """
        raise NotImplementedError("not yet")

    def arccosh(self) -> "Series":
        """
        Get the hyperbolic arc cosine values.

        Returns:
            Hyperbolic arc cosine values Series.
        """
        raise NotImplementedError("not yet")

    def arcsin(self) -> "Series":
        """
        Get the arc sine values.

        Returns:
            Arc sine values Series.
        """
        raise NotImplementedError("not yet")

    def arcsinh(self) -> "Series":
        """
        Get the hyperbolic arc sine values.

        Returns:
            Hyperbolic arc sine values Series.
        """
        raise NotImplementedError("not yet")

    def arctan(self) -> "Series":
        """
        Get the arc tangent values.

        Returns:
            Arc tangent values Series.
        """
        raise NotImplementedError("not yet")

    def arctanh(self) -> "Series":
        """
        Get the hyperbolic arc tangent values.

        Returns:
            Hyperbolic arc tangent values Series.
        """
        raise NotImplementedError("not yet")

    def arg_true(self) -> "Series":
        """
        Get the index of the first True value.

        Returns:
            Index of the first True value.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.reset_index(drop=False)
            .getitem_array(self._query_compiler)
            .getitem_column_array(0, numeric=True)
        ).rename(self.name)

    def arg_unique(self) -> "Series":
        """
        Get the index of the first unique value.

        Returns:
            Index of the first unique value.
        """
        raise NotImplementedError("not yet")

    def cbrt(self) -> "Series":
        """
        Get the cube root values.

        Returns:
            Cube root values Series.
        """
        raise NotImplementedError("not yet")

    def cos(self) -> "Series":
        """
        Get the cosine values.

        Returns:
            Cosine values Series.
        """
        raise NotImplementedError("not yet")

    def cosh(self) -> "Series":
        """
        Get the hyperbolic cosine values.

        Returns:
            Hyperbolic cosine values Series.
        """
        raise NotImplementedError("not yet")

    def cot(self) -> "Series":
        """
        Get the cotangent values.

        Returns:
            Cotangent values Series.
        """
        raise NotImplementedError("not yet")

    def cum_count(self) -> "Series":
        """
        Get the cumulative count values.

        Returns:
            Cumulative count values Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.isna().cumsum()
        )

    def cum_max(self) -> "Series":
        """
        Get the cumulative maximum values.

        Returns:
            Cumulative maximum values Series.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.cummax())

    def cum_min(self) -> "Series":
        """
        Get the cumulative minimum values.

        Returns:
            Cumulative minimum values Series.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.cummin())

    def cum_prod(self) -> "Series":
        """
        Get the cumulative product values.

        Returns:
            Cumulative product values Series.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.cumprod())

    def cum_sum(self) -> "Series":
        """
        Get the cumulative sum values.

        Returns:
            Cumulative sum values Series.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.cumsum())

    def cumulative_eval(
        self, expr, min_periods: int = 1, *, parallel: bool = False
    ) -> "Series":
        """
        Get the cumulative evaluation values.

        Args:
            expr: Expression to evaluate.
            min_periods: Minimum number of periods.

        Returns:
            Cumulative evaluation values Series.
        """
        raise NotImplementedError("not yet")

    def diff(self, n: int = 1, null_behavior: str = "ignore") -> "Series":
        """
        Calculate the first discrete difference between shifted items.

        Args:
            n: Number of periods to shift.
            null_behavior: Null behavior.

        Returns:
            Difference values Series.
        """
        raise NotImplementedError("not yet")

    def dot(self, other) -> int | float | None:
        """
        Calculate the dot product.

        Args:
            other: Other Series.

        Returns:
            Dot product.
        """
        if isinstance(other, Series):
            other = other.to_pandas()
        return self.to_pandas().dot(other)

    def entropy(
        self, base: float = 2.718281828459045, *, normalize: bool = False
    ) -> float:
        """
        Calculate the entropy.

        Args:
            base: Logarithm base.
            normalize: Normalize the entropy.

        Returns:
            Entropy.
        """
        raise NotImplementedError("not yet")

    def ewm_mean(
        self,
        com: int | None = None,
        span: int | None = None,
        half_life: int | None = None,
        alpha: float | None = None,
        *,
        adjust: bool = True,
        min_periods: int = 1,
        ignore_nulls: bool | None = None,
    ) -> "Series":
        """
        Calculate the exponential weighted mean.

        Args:
            com: Center of mass.
            span: Span.

        Returns:
            Exponential weighted mean Series.
        """
        return self.__constructor__(
            self.to_pandas()
            .ewm(
                com=com,
                span=span,
                halflife=half_life,
                alpha=alpha,
                adjust=adjust,
                min_periods=min_periods,
                ignore_na=ignore_nulls,
            )
            .mean()
        )

    def ewm_mean_by(self, by, *, half_life: int | None = None) -> "Series":
        """
        Calculate the exponential weighted mean by group.

        Args:
            by: Grouping Series.

        Returns:
            Exponential weighted mean Series.
        """
        raise NotImplementedError("not yet")

    def ewm_std(
        self,
        com: int | None = None,
        span: int | None = None,
        half_life: int | None = None,
        alpha: float | None = None,
        *,
        adjust: bool = True,
        min_periods: int = 1,
        ignore_nulls: bool | None = None,
    ) -> "Series":
        """
        Calculate the exponential weighted standard deviation.

        Args:
            com: Center of mass.
            span: Span.

        Returns:
            Exponential weighted standard deviation Series.
        """
        return self.__constructor__(
            self.to_pandas()
            .ewm(
                com=com,
                span=span,
                halflife=half_life,
                alpha=alpha,
                adjust=adjust,
                min_periods=min_periods,
                ignore_na=ignore_nulls,
            )
            .std()
        )

    def ewm_var(
        self,
        com: int | None = None,
        span: int | None = None,
        half_life: int | None = None,
        alpha: float | None = None,
        *,
        adjust: bool = True,
        min_periods: int = 1,
        ignore_nulls: bool | None = None,
    ) -> "Series":
        """
        Calculate the exponential weighted variance.

        Args:
            com: Center of mass.
            span: Span.

        Returns:
            Exponential weighted variance Series.
        """
        return self.__constructor__(
            self.to_pandas()
            .ewm(
                com=com,
                span=span,
                halflife=half_life,
                alpha=alpha,
                adjust=adjust,
                min_periods=min_periods,
                ignore_na=ignore_nulls,
            )
            .var()
        )

    def exp(self) -> "Series":
        """
        Calculate the exponential values.

        Returns:
            Exponential values Series.
        """
        return self.__constructor__(self.to_pandas().exp())

    def hash(
        self,
        seed: int = 0,
        seed_1: int | None = None,
        seed_2: int | None = None,
        seed_3: int | None = None,
    ) -> "Series":
        """
        Calculate the hash values.

        Args:
            seed: Seed.
            seed_1: Seed 1.
            seed_2: Seed 2.
            seed_3: Seed 3.

        Returns:
            Hash values Series.
        """
        raise NotImplementedError("not yet")

    def hist(
        self,
        bins: list[float] | None = None,
        *,
        bin_count: int | None = None,
        include_category: bool = True,
        include_breakpoint: bool = True,
    ) -> "Series":
        """
        Calculate the histogram.

        Args:
            bins: Bins.
            bin_count: Bin count.

        Returns:
            Histogram Series.
        """
        raise NotImplementedError("not yet")

    def is_between(self, lower_bound, upper_bound, closed: str = "both") -> "Series":
        """
        Check if values are between the bounds.

        Args:
            lower_bound: Lower bound.
            upper_bound: Upper bound.
            closed: Closed bounds.

        Returns:
            Boolean Series.
        """
        raise NotImplementedError("not yet")

    def kurtosis(self, *, fisher: bool = True, bias: bool = True) -> float | None:
        """
        Calculate the kurtosis.

        Args:
            fisher: Fisher method.
            bias: Bias method.

        Returns:
            Kurtosis.
        """
        return self.to_pandas().kurtosis(fisher=fisher, bias=bias)

    def log(self, base: float = 2.718281828459045) -> "Series":
        """
        Calculate the logarithm values.

        Args:
            base: Logarithm base.

        Returns:
            Logarithm values Series.
        """
        raise NotImplementedError("not yet")

    def log10(self) -> "Series":
        """
        Calculate the base 10 logarithm values.

        Returns:
            Base 10 logarithm values Series.
        """
        return self.log(10)

    def log1p(self) -> "Series":
        """
        Calculate the natural logarithm of 1 plus the values.

        Returns:
            Natural logarithm of 1 plus the values Series.
        """
        raise NotImplementedError("not yet")

    def replace(
        self,
        mapping: dict[Any, Any],
        *,
        default: Any = None,
        return_dtype=None,
    ) -> "Series":
        """
        Map values to other values.

        Args:
            mapping: Mapping.

        Returns:
            Mapped Series.
        """
        return self.__constructor__(
            self.to_pandas().apply(lambda x: mapping.get(x, default))
        )

    def pct_change(self, n: int = 1) -> "Series":
        """
        Calculate the percentage change.

        Args:
            n: Number of periods to shift.

        Returns:
            Percentage change Series.
        """
        return self.__constructor__(self.to_pandas().pct_change(n))

    def peak_max(self) -> "Series":
        """
        Get the peak maximum values.

        Returns:
            Peak maximum values Series.
        """
        return self.__eq__(self.max())

    def peak_min(self) -> "Series":
        """
        Get the peak minimum values.

        Returns:
            Peak minimum values Series.
        """
        return self.__eq__(self.min())

    def rank(
        self,
        method: str = "average",
        *,
        descending: bool = False,
        seed: int | None = None,
    ) -> "Series":
        """
        Calculate the rank.

        Args:
            method: Rank method.

        Returns:
            Rank Series.
        """
        # TODO: support seed
        if method not in ["average", "min", "max", "first", "dense"]:
            raise ValueError(f"method {method} not supported")
        return self.__constructor__(
            self.to_pandas().rank(method=method, ascending=not descending)
        )

    def rolling_map(
        self,
        function: callable,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int = 1,
        *,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling function.

        Args:
            function: Function to apply.
            window_size: Window size.

        Returns:
            Applied Series.
        """
        if weights is not None:
            raise NotImplementedError("not yet")
        return self.__constructor__(
            self.to_pandas()
            .rolling(window=window_size, min_periods=min_periods, center=center)
            .apply(function)
        )

    def rolling_max(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int = 1,
        *,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling maximum function.

        Args:
            function: Function to apply.
            window_size: Window size.

        Returns:
            Applied Series.
        """
        if weights is not None:
            raise NotImplementedError("not yet")
        return self.__constructor__(
            self.to_pandas()
            .rolling(window=window_size, min_periods=min_periods, center=center)
            .max()
        )

    def rolling_mean(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int = 1,
        *,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling mean function.

        Args:
            function: Function to apply.
            window_size: Window size.

        Returns:
            Applied Series.
        """
        if weights is not None:
            raise NotImplementedError("not yet")
        return self.__constructor__(
            self.to_pandas()
            .rolling(window=window_size, min_periods=min_periods, center=center)
            .mean()
        )

    def rolling_median(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int = 1,
        *,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling median function.

        Args:
            function: Function to apply.
            window_size: Window size.

        Returns:
            Applied Series.
        """
        if weights is not None:
            raise NotImplementedError("not yet")
        return self.__constructor__(
            self.to_pandas()
            .rolling(window=window_size, min_periods=min_periods, center=center)
            .median()
        )

    def rolling_min(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int = 1,
        *,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling minimum function.

        Args:
            function: Function to apply.
            window_size: Window size.

        Returns:
            Applied Series.
        """
        if weights is not None:
            raise NotImplementedError("not yet")
        return self.__constructor__(
            self.to_pandas()
            .rolling(window=window_size, min_periods=min_periods, center=center)
            .min()
        )

    def rolling_quantile(
        self,
        window_size: int,
        quantile: float,
        interpolation: str = "nearest",
        weights: list[float] | None = None,
        min_periods: int = 1,
        *,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling quantile function.

        Args:
            function: Function to apply.
            window_size: Window size.

        Returns:
            Applied Series.
        """
        if weights is not None:
            raise NotImplementedError("not yet")
        return self.__constructor__(
            self.to_pandas()
            .rolling(window=window_size, min_periods=min_periods, center=center)
            .quantile(quantile, interpolation=interpolation)
        )

    def rolling_skew(self, window_size: int, *, bias: bool = False) -> "Series":
        """
        Apply a rolling skewness function.

        Args:
            function: Function to apply.
            window_size: Window size.

        Returns:
            Applied Series.
        """
        return self.__constructor__(self.to_pandas().rolling(window=window_size).skew())

    def rolling_std(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int = 1,
        *,
        center: bool = False,
        ddof: int = 1,
    ) -> "Series":
        """
        Apply a rolling standard deviation function.

        Args:
            function: Function to apply.
            window_size: Window size.

        Returns:
            Applied Series.
        """
        if weights is not None:
            raise NotImplementedError("not yet")
        return self.__constructor__(
            self.to_pandas()
            .rolling(window=window_size, min_periods=min_periods, center=center)
            .std(ddof=ddof)
        )

    def rolling_sum(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int = 1,
        *,
        center: bool = False,
    ) -> "Series":
        """
        Apply a rolling sum function.

        Args:
            function: Function to apply.
            window_size: Window size.

        Returns:
            Applied Series.
        """
        if weights is not None:
            raise NotImplementedError("not yet")
        return self.__constructor__(
            self.to_pandas()
            .rolling(window=window_size, min_periods=min_periods, center=center)
            .sum()
        )

    def rolling_var(
        self,
        window_size: int,
        weights: list[float] | None = None,
        min_periods: int = 1,
        *,
        center: bool = False,
        ddof: int = 1,
    ) -> "Series":
        """
        Apply a rolling variance function.

        Args:
            function: Function to apply.
            window_size: Window size.

        Returns:
            Applied Series.
        """
        if weights is not None:
            raise NotImplementedError("not yet")
        return self.__constructor__(
            self.to_pandas()
            .rolling(window=window_size, min_periods=min_periods, center=center)
            .var(ddof=ddof)
        )

    def search_sorted(self, element, side: str = "any") -> int | "Series":
        """
        Search for the element in the sorted Series.

        Args:
            element: Element to search.
            side: Side to search.

        Returns:
            Index of the element.
        """
        if side == "any":
            side = "left"
        return self.__constructor__(self.to_pandas().searchsorted(element, side=side))

    def sign(self) -> "Series":
        """
        Get the sign values.

        Returns:
            Sign values Series.
        """
        return self.__lt__(0).__mul__(-1).__add__(self.__gt__(0))

    def sin(self) -> "Series":
        """
        Get the sine values.

        Returns:
            Sine values Series.
        """
        raise NotImplementedError("not yet")

    def sinh(self) -> "Series":
        """
        Get the hyperbolic sine values.

        Returns:
            Hyperbolic sine values Series.
        """
        raise NotImplementedError("not yet")

    def skew(self, *, bias: bool = True) -> float:
        """
        Calculate the skewness.

        Args:
            bias: Bias method.

        Returns:
            Skewness.
        """
        return self.to_pandas().skew()

    def sqrt(self) -> "Series":
        """
        Get the square root values.

        Returns:
            Square root values Series.
        """
        return self.__constructor__(self.to_pandas().sqrt())

    def tan(self) -> "Series":
        """
        Get the tangent values.

        Returns:
            Tangent values Series.
        """
        raise NotImplementedError("not yet")

    def tanh(self) -> "Series":
        """
        Get the hyperbolic tangent values.

        Returns:
            Hyperbolic tangent values Series.
        """
        raise NotImplementedError("not yet")

    def chunk_lengths(self) -> list[int]:
        """
        Get the chunk lengths.

        Returns:
            Chunk lengths.
        """
        raise NotImplementedError("not yet")

    def describe(
        self,
        percentiles: Sequence[float] | float | None = (0.25, 0.5, 0.75),
        interpolation: str = "nearest",
    ):
        """
        Generate descriptive statistics.

        Args:
            percentiles: Percentiles to calculate.

        Returns:
            Descriptive statistics.
        """
        return self.to_pandas().describe(percentiles=percentiles)

    def estimated_size(self) -> int:
        """
        Get the estimated size.

        Returns:
            Estimated size.
        """
        return self.to_pandas().memory_usage(index=False)

    def has_nulls(self) -> bool:
        """
        Check if there are null values.

        Returns:
            True if there are null values, False otherwise.
        """
        return self.to_pandas().isnull().any()

    has_validity = has_nulls

    def is_finite(self) -> "Series":
        """
        Check if the values are finite.

        Returns:
            True if the values are finite, False otherwise.
        """
        return self.__ne__(np.inf)

    def is_first_distinct(self) -> "Series":
        """
        Check if the values are the first occurrence.

        Returns:
            True if the values are the first occurrence, False otherwise.
        """
        raise NotImplementedError("not yet")

    def is_in(self, other: "Series" | list[Any]) -> "Series":
        """
        Check if the values are in the other Series.

        Args:
            other: Other Series.

        Returns:
            True if the values are in the other Series, False otherwise.
        """
        return self.__constructor__(self.to_pandas().isin(other))

    def is_infinite(self) -> "Series":
        """
        Check if the values are infinite.

        Returns:
            True if the values are infinite, False otherwise.
        """
        return self.__eq__(np.inf)

    def is_last_distinct(self) -> "Series":
        """
        Check if the values are the last occurrence.

        Returns:
            True if the values are the last occurrence, False otherwise.
        """
        raise NotImplementedError("not yet")

    def is_nan(self) -> "Series":
        """
        Check if the values are NaN.

        Returns:
            True if the values are NaN, False otherwise.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.isna())

    def is_not_nan(self) -> "Series":
        """
        Check if the values are not NaN.

        Returns:
            True if the values are not NaN, False otherwise.
        """
        return self.__constructor__(_query_compiler=self._query_compiler.notna())

    def is_not_null(self) -> "Series":
        """
        Check if the values are not null.

        Returns:
            True if the values are not null, False otherwise.
        """
        return self.is_not_nan()

    def is_null(self) -> "Series":
        """
        Check if the values are null.

        Returns:
            True if the values are null, False otherwise.
        """
        return self.is_nan()

    def is_sorted(
        self,
        *,
        descending: bool = False,
        nulls_last: bool = False,
    ) -> bool:
        """
        Check if the values are sorted.

        Args:
            descending: Descending order.

        Returns:
            True if the values are sorted, False otherwise.
        """
        return (
            self.to_pandas().is_monotonic_increasing
            if not descending
            else self.to_pandas().is_monotonic_decreasing
        )

    def len(self) -> int:
        """
        Get the length of the values.

        Returns:
            Length of the values Series.
        """
        return len(self.to_pandas())

    def lower_bound(self) -> "Series":
        """
        Get the lower bound values.

        Returns:
            Lower bound values Series.
        """
        raise NotImplementedError("not yet")

    def null_count(self) -> int:
        """
        Get the number of null values.

        Returns:
            Number of null values.
        """
        return self.to_pandas().isnull().sum()

    def unique_counts(self) -> "Series":
        """
        Get the unique counts.

        Returns:
            Unique counts.
        """
        return self.__constructor__(values=self.to_pandas().value_counts())

    def upper_bound(self) -> "Series":
        """
        Get the upper bound values.

        Returns:
            Upper bound values Series.
        """
        raise NotImplementedError("not yet")

    def value_counts(
        self, *, sort: bool = False, parallel: bool = False, name: str = "count"
    ) -> "DataFrame":
        """
        Get the value counts.

        Returns:
            Value counts.
        """
        from modin.polars import DataFrame

        return DataFrame(
            self.to_pandas().value_counts(sort=sort).reset_index(drop=False, names=name)
        )

    def to_frame(self, name: str | None = None) -> "DataFrame":
        """
        Convert the Series to a DataFrame.

        Args:
            name: Name of the Series.

        Returns:
            DataFrame representation of the Series.
        """
        from modin.polars import DataFrame

        return DataFrame(_query_compiler=self._query_compiler).rename({self.name: name})

    def to_init_repr(self, n: int = 1000) -> str:
        """
        Convert Series to instantiatable string representation.

        Args:
            n: First n elements.

        Returns:
            Instantiatable string representation.
        """
        return polars.from_pandas(
            self.slice(0, n)._query_compiler.to_pandas()
        ).to_init_repr()

    @property
    def list(self):
        # TODO: implement list object
        #  https://docs.pola.rs/api/python/stable/reference/series/list.html
        raise NotImplementedError("not yet")

    def alias(self, name: str) -> "Series":
        """
        Rename the Series.

        Args:
            name: New name.

        Returns:
            Renamed Series.
        """
        return self.to_frame(name).to_series()

    def append(self, other: "Series") -> "Series":
        """
        Append another Series.

        Args:
            other: Other Series.

        Returns:
            Appended Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.concat(0, other._query_compiler)
        )

    def arg_sort(
        self, *, descending: bool = False, nulls_last: bool = False
    ) -> "Series":
        """
        Get the sorted indices.

        Args:
            descending: Descending order.

        Returns:
            Sorted indices Series.
        """
        # TODO: implement nulls_last
        result = self.__constructor__(values=self.to_pandas().argsort())
        if descending:
            return result.reverse()
        else:
            return result

    def ceil(self) -> "Series":
        """
        Get the ceiling values.

        Returns:
            Ceiling values Series.
        """
        raise NotImplementedError("not yet")

    def clear(self, n: int = 0) -> "Series":
        """
        Create an empty copy of the current Series, with zero to ‘n’ elements.

        Args:
            n: Number of elements.

        Returns:
            Series will n nulls.
        """
        raise NotImplementedError("not yet")

    def clip(self, lower_bound=None, upper_bound=None) -> "Series":
        """
        Clip the values.

        Args:
            lower_bound: Lower bound.
            upper_bound: Upper bound.

        Returns:
            Clipped values Series.
        """
        return self.__constructor__(
            values=self.to_pandas().clip(lower_bound, upper_bound)
        )

    def cut(
        self,
        breaks: Sequence[float],
        *,
        labels: list[str] | None = None,
        break_point_label: str = "breakpoint",
        left_closed: bool = False,
        include_breaks: bool = False,
        as_series: bool = True,
    ) -> "BasePolarsDataset":
        raise NotImplementedError("not yet")

    def extend_constant(self, value) -> "Series":
        """
        Extend the Series with a constant value.

        Args:
            value: Constant value.

        Returns:
            Extended Series.
        """
        raise NotImplementedError("not yet")

    def floor(self) -> "BasePolarsDataset":
        return self.__floordiv__(1)

    def gather(self, indices) -> "Series":
        """
        Gather values by indices.

        Args:
            indices: Indices.

        Returns:
            Gathered Series.
        """
        return self.__constructor__(
            values=self.to_pandas().iloc[
                (
                    indices._query_compiler
                    if hasattr(indices, "_query_compiler")
                    else indices
                )
            ]
        )

    def interpolate_by(self, by) -> "Series":
        """
        Interpolate values by group.

        Args:
            by: Grouping Series.

        Returns:
            Interpolated Series.
        """
        raise NotImplementedError("not yet")

    def item(self, index: int | None = None) -> Any:
        """
        Get the item at the index.

        Args:
            index: Index.

        Returns:
            Item at the index.
        """
        return self.to_pandas().iloc[index]

    def new_from_index(self, index: int, length: int) -> "Series":
        """
        Create a new Series from the index.

        Args:
            index: Index.
            length: Length.

        Returns:
            New Series.
        """
        raise NotImplementedError("not yet")

    def qcut(
        self,
        quantiles: Sequence[float] | int,
        *,
        labels: Sequence[str] | None = None,
        left_closed: bool = False,
        allow_duplicates: bool = False,
        include_breaks: bool = False,
        break_point_label: str = "breakpoint",
        category_labels: str = "category",
        as_series: bool = True,
    ) -> "Series" | "DataFrame":
        """
        Bin continuous values into discrete categories based on quantiles.

        Args:
            quantiles: Number of quantiles or sequence of quantiles.
            labels: Labels for the resulting bins.
            left_closed: Whether the intervals are left-closed.
            allow_duplicates: Whether to allow duplicate intervals.
            include_breaks: Whether to include the breaks in the result.
            break_point_label: Label for the break points.
            category_labels: Label for the categories.
            as_series: Whether to return a Series.

        Returns:
            Binned Series.
        """
        raise NotImplementedError("not yet")

    def rechunk(self, *, in_place: bool = False) -> "Series":
        """
        Rechunk the Series.

        Args:
            in_place: In-place operation.

        Returns:
            Rechunked Series.
        """
        raise NotImplementedError("not yet")

    rename = alias

    def reshape(self, dimensions, nested_type) -> "Series":
        """
        Reshape the Series.

        Args:
            dimensions: Dimensions.
            nested_type: Nested type.

        Returns:
            Reshaped Series.
        """
        raise NotImplementedError("not yet")

    def reverse(self) -> "Series":
        """
        Reverse the Series.

        Returns:
            Reversed Series.
        """
        return self.__constructor__(values=self.to_pandas().iloc[::-1])

    def rle(self) -> "Series":
        """
        Run-length encode the Series.

        Returns:
            Run-length encoded Series.
        """
        raise NotImplementedError("not yet")

    def rle_id(self) -> "Series":
        """
        Run-length encode the Series with IDs.

        Returns:
            Run-length encoded Series with IDs.
        """
        raise NotImplementedError("not yet")

    def round(self, decimals: int = 0) -> "Series":
        """
        Round the values.

        Args:
            decimals: Number of decimals.

        Returns:
            Rounded values Series.
        """
        return self.__constructor__(values=self.to_pandas().round(decimals))

    def round_sig_figs(self, digits: int) -> "Series":
        """
        Round the values to significant figures.

        Args:
            digits: Number of significant figures.

        Returns:
            Rounded values Series.
        """
        raise NotImplementedError("not yet")

    def scatter(self, indices, values) -> "Series":
        """
        Scatter values by indices.

        Args:
            indices: Indices.
            values: Values.

        Returns:
            Scattered Series.
        """
        raise NotImplementedError("not yet")

    def set(self, filter: "Series", value: int | float | str | bool | None) -> "Series":
        """
        Set values by filter.

        Args:
            filter: Filter.
            value: Value.

        Returns:
            Set Series.
        """
        raise NotImplementedError("not yet")

    def shrink_dtype(self) -> "Series":
        """
        Shrink the data type.

        Returns:
            Shrunk Series.
        """
        raise NotImplementedError("not yet")

    def shuffle(self, seed: int | None = None) -> "Series":
        """
        Shuffle the Series.

        Args:
            seed: Seed.

        Returns:
            Shuffled Series.
        """
        raise NotImplementedError("not yet")

    def zip_with(self, mask: "Series", other: "Series") -> "Series":
        """
        Zip the Series with another Series.

        Args:
            mask: Mask Series.
            other: Other Series.

        Returns:
            Zipped Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.where(
                mask._query_compiler, other._query_compiler
            )
        )

    def map_elements(
        self,
        function: callable,
        return_dtype=None,
        *,
        skip_nulls: bool = True,
    ) -> "Series":
        """
        Map the elements.

        Args:
            function: Function to apply.

        Returns:
            Mapped Series.
        """
        if return_dtype is not None or skip_nulls is False:
            ErrorMessage.warn(
                "`return_dtype` and `skip_nulls=False` are not supported yet"
            )
        return self.__constructor__(values=self.to_pandas().apply(function))

    def reinterpret(self, *, signed: bool = True) -> "Series":
        """
        Reinterpret the data type of the series as signed or unsigned.

        Args:
            signed: If True, reinterpret as signed, otherwise as unsigned.

        Returns:
            Reinterpreted Series.
        """
        raise NotImplementedError("not yet")

    def set_sorted(self, *, descending: bool = False) -> "Series":
        """
        Set the Series as sorted.

        Args:
            descending: Descending order.

        Returns:
            Sorted Series.
        """
        self._sorted = True
        self._descending = descending
        return self

    def to_physical(self) -> "Series":
        """
        Convert the Series to physical.

        Returns:
            Physical Series.
        """
        raise NotImplementedError("not yet")

    def get_chunks(self) -> list["Series"]:
        """
        Get the chunks.

        Returns:
            Chunks.
        """
        raise NotImplementedError("not yet")

    @property
    def str(self):
        # TODO: implement str object
        #  https://docs.pola.rs/api/python/stable/reference/series/string.html
        raise NotImplementedError("not yet")

    @property
    def struct(self):
        # TODO: implement struct object
        #  https://docs.pola.rs/api/python/stable/reference/series/struct.html
        raise NotImplementedError("not yet")

    @property
    def dt(self):
        # TODO: implement dt object
        #  https://docs.pola.rs/api/python/stable/reference/series/temporal.html
        raise NotImplementedError("not yet")

    def __len__(self) -> int:
        """
        Get the length of the Series.
        """
        return self.len()

    def __matmul__(self, other) -> "Series":
        """
        Matrix multiplication.

        Args:
            other: Other Series.

        Returns:
            Matrix multiplication Series.
        """
        raise NotImplementedError("not yet")

    def __radd__(self, other) -> "Series":
        """
        Right addition.

        Args:
            other: Other Series.

        Returns:
            Added Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.radd(other, axis=0)
        )

    def __rand__(self, other) -> "Series":
        """
        Right and.

        Args:
            other: Other Series.

        Returns:
            And Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.__rand__(other, axis=0)
        )

    def __rfloordiv__(self, other) -> "Series":
        """
        Right floor division.

        Args:
            other: Other Series.

        Returns:
            Floored Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.rfloordiv(other, axis=0)
        )

    def __rmatmul__(self, other) -> "Series":
        """
        Right matrix multiplication.

        Args:
            other: Other Series.

        Returns:
            Matrix multiplication Series.
        """
        raise NotImplementedError("not yet")

    def __rmod__(self, other) -> "Series":
        """
        Right modulo.

        Args:
            other: Other Series.

        Returns:
            Modulo Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.rmod(other, axis=0)
        )

    def __rmul__(self, other) -> "Series":
        """
        Right multiplication.

        Args:
            other: Other Series.

        Returns:
            Multiplied Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.rmul(other, axis=0)
        )

    def __ror__(self, other) -> "Series":
        """
        Right or.

        Args:
            other: Other Series.

        Returns:
            Or Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.__ror__(other, axis=0)
        )

    def __rpow__(self, other) -> "Series":
        """
        Right power.

        Args:
            other: Other Series.

        Returns:
            Powered Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.rpow(other, axis=0)
        )

    def __rsub__(self, other) -> "Series":
        """
        Right subtraction.

        Args:
            other: Other Series.

        Returns:
            Subtracted Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.rsub(other, axis=0)
        )

    def __rtruediv__(self, other) -> "Series":
        """
        Right true division.

        Args:
            other: Other Series.

        Returns:
            Divided Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.rtruediv(other, axis=0)
        )

    def __rxor__(self, other) -> "Series":
        """
        Right xor.

        Args:
            other: Other Series.

        Returns:
            Xor Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.__rxor__(other, axis=0)
        )

    def eq(self, other) -> "Series":
        """
        Check if the values are equal to the other Series.

        Args:
            other: Other Series.

        Returns:
            Boolean Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.eq(other._query_compiler)
        )

    def eq_missing(self, other) -> "Series":
        """
        Check if the values are equal to the other Series, including missing values.

        Args:
            other: Other Series.

        Returns:
            Boolean Series.
        """
        raise NotImplementedError("not yet")

    def ge(self, other) -> "Series":
        """
        Check if the values are greater than or equal to the other Series.

        Args:
            other: Other Series.

        Returns:
            Boolean Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.ge(other._query_compiler)
        )

    def gt(self, other) -> "Series":
        """
        Check if the values are greater than the other Series.

        Args:
            other: Other Series.

        Returns:
            Boolean Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.gt(other._query_compiler)
        )

    def le(self, other) -> "Series":
        """
        Check if the values are less than or equal to the other Series.

        Args:
            other: Other Series.

        Returns:
            Boolean Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.le(other._query_compiler)
        )

    def lt(self, other) -> "Series":
        """
        Check if the values are less than the other Series.

        Args:
            other: Other Series.

        Returns:
            Boolean Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.lt(other._query_compiler)
        )

    def n_unique(self) -> int:
        """
        Get the number of unique values.

        Returns:
            Number of unique values.
        """
        return self._query_compiler.nunique().to_pandas().squeeze(axis=None)

    def ne(self, other) -> "Series":
        """
        Check if the values are not equal to the other Series.

        Args:
            other: Other Series.

        Returns:
            Boolean Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.ne(other._query_compiler)
        )

    def ne_missing(self, other) -> "Series":
        """
        Check if the values are not equal to the other Series, including missing values.

        Args:
            other: Other Series.

        Returns:
            Boolean Series.
        """
        raise NotImplementedError("not yet")

    def pow(self, exponent) -> "Series":
        """
        Raise the values to the power of the exponent.

        Args:
            exponent: Exponent.

        Returns:
            Powered Series.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.pow(exponent, axis=0)
        )

    def replace_strict(
        self, old, new=no_default, *, default=no_default, return_dtype=None
    ) -> "Series":
        """
        Replace values strictly.

        Args:
            old: Old values.
            new: New values.
            default: Default value.

        Returns:
            Replaced Series.
        """
        raise NotImplementedError("not yet")

    def to_list(self) -> list:
        """
        Convert the Series to a list.

        Returns:
            List representation of the Series.
        """
        return self._to_polars().tolist()

    def drop_nans(self) -> "Series":
        """
        Drop NaN values.

        Returns:
            Series without NaN values.
        """
        return self.__constructor__(
            _query_compiler=self._query_compiler.dropna(how="any")
        )
