from __future__ import annotations

import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import numpy as np
from pandas.api.types import is_extension_array_dtype

import modin.dataframe_api_standard
import modin.pandas as pd
from modin.dataframe_api_standard.utils import validate_comparand

if TYPE_CHECKING:
    from dataframe_api import Column as ColumnT
    from dataframe_api.typing import DType, NullType, Scalar

    from modin.dataframe_api_standard.dataframe_object import DataFrame
else:
    ColumnT = object


NUMPY_MAPPING = {
    "Int64": "int64",
    "Int32": "int32",
    "Int16": "int16",
    "Int8": "int8",
    "UInt64": "uint64",
    "UInt32": "uint32",
    "UInt16": "uint16",
    "UInt8": "uint8",
    "boolean": "bool",
    "Float64": "float64",
    "Float32": "float32",
}


class Column(ColumnT):
    def __init__(
        self,
        series: pd.Series[Any],
        *,
        df: DataFrame | None,
        api_version: str,
        is_persisted: bool = False,
    ) -> None:
        """Parameters
        ----------
        df
            DataFrame this column originates from.
        """

        self._name = series.name
        assert self._name is not None
        self._series = series
        self._api_version = api_version
        self._df = df
        self._is_persisted = is_persisted
        assert is_persisted ^ (df is not None)

    def _to_scalar(self, value: Any) -> Scalar:
        from modin.dataframe_api_standard.scalar_object import Scalar

        return Scalar(
            value,
            api_version=self._api_version,
            df=self._df,
            is_persisted=self._is_persisted,
        )

    def __repr__(self) -> str:  # pragma: no cover
        header = f" Standard Column (api_version={self._api_version}) "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Add `.column` to see native output         |\n"
            + "└"
            + "─" * length
            + "┘\n"
        )

    def __iter__(self) -> NoReturn:
        msg = ""
        raise NotImplementedError(msg)

    def _from_series(self, series: pd.Series) -> Column:
        return Column(
            series.reset_index(drop=True).rename(series.name),
            api_version=self._api_version,
            df=self._df,
            is_persisted=self._is_persisted,
        )

    def _materialise(self) -> pd.Series:
        if not self._is_persisted:
            msg = "Column is not persisted, please call `.persist()` first.\nNote: `persist` forces computation, use it with care, only when you need to,\nand as late and little as possible."
            raise RuntimeError(
                msg,
            )
        return self.column

    # In the standard
    def __column_namespace__(
        self,
    ) -> modin.dataframe_api_standard.Namespace:
        return modin.dataframe_api_standard.Namespace(
            api_version=self._api_version,
        )

    def persist(self) -> Column:
        if self._is_persisted:
            warnings.warn(
                "Calling `.persist` on Column that was already persisted",
                UserWarning,
                stacklevel=2,
            )
        return Column(
            self.column,
            df=None,
            api_version=self._api_version,
            is_persisted=True,
        )

    @property
    def name(self) -> str:
        return self._name  # type: ignore[no-any-return]

    @property
    def column(self) -> pd.Series[Any]:
        return self._series

    @property
    def dtype(self) -> DType:
        return modin.dataframe_api_standard.map_pandas_dtype_to_standard_dtype(
            self._series.dtype,
        )

    @property
    def parent_dataframe(self) -> DataFrame | None:
        return self._df

    def take(self, indices: Column) -> Column:
        return self._from_series(self.column.iloc[indices.column])

    def filter(self, mask: Column) -> Column:
        ser = self.column
        return self._from_series(ser.loc[mask.column])

    def get_value(self, row_number: int) -> Any:
        ser = self.column
        return self._to_scalar(
            ser.iloc[row_number],
        )

    def slice_rows(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
    ) -> Column:
        return self._from_series(self.column.iloc[start:stop:step])

    # Binary comparisons

    def __eq__(self, other: Column | Any) -> Column:  # type: ignore[override]
        other = validate_comparand(self, other)
        ser = self.column
        return self._from_series((ser == other).rename(ser.name))

    def __ne__(self, other: Column | Any) -> Column:  # type: ignore[override]
        other = validate_comparand(self, other)
        ser = self.column
        return self._from_series((ser != other).rename(ser.name))

    def __ge__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        ser = self.column
        return self._from_series((ser >= other).rename(ser.name))

    def __gt__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        ser = self.column
        return self._from_series((ser > other).rename(ser.name))

    def __le__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        ser = self.column
        return self._from_series((ser <= other).rename(ser.name))

    def __lt__(self, other: Column | Any) -> Column:
        other = validate_comparand(self, other)
        ser = self.column
        return self._from_series((ser < other).rename(ser.name))

    def __and__(self, other: Column | bool | Scalar) -> Column:
        ser = self.column
        other = validate_comparand(self, other)
        return self._from_series((ser & other).rename(ser.name))

    def __rand__(self, other: Column | Any) -> Column:
        return self.__and__(other)

    def __or__(self, other: Column | bool | Scalar) -> Column:
        ser = self.column
        other = validate_comparand(self, other)
        return self._from_series((ser | other).rename(ser.name))

    def __ror__(self, other: Column | Any) -> Column:
        return self.__or__(other)

    def __add__(self, other: Column | Any) -> Column:
        ser = self.column
        other = validate_comparand(self, other)
        return self._from_series((ser + other).rename(ser.name))

    def __radd__(self, other: Column | Any) -> Column:
        return self.__add__(other)

    def __sub__(self, other: Column | Any) -> Column:
        ser = self.column
        other = validate_comparand(self, other)
        return self._from_series((ser - other).rename(ser.name))

    def __rsub__(self, other: Column | Any) -> Column:
        return -1 * self.__sub__(other)

    def __mul__(self, other: Column | Any) -> Column:
        ser = self.column
        other = validate_comparand(self, other)
        return self._from_series((ser * other).rename(ser.name))

    def __rmul__(self, other: Column | Any) -> Column:
        return self.__mul__(other)

    def __truediv__(self, other: Column | Any) -> Column:
        ser = self.column
        other = validate_comparand(self, other)
        return self._from_series((ser / other).rename(ser.name))

    def __rtruediv__(self, other: Column | Any) -> Column:
        raise NotImplementedError

    def __floordiv__(self, other: Column | Any) -> Column:
        ser = self.column
        other = validate_comparand(self, other)
        return self._from_series((ser // other).rename(ser.name))

    def __rfloordiv__(self, other: Column | Any) -> Column:
        raise NotImplementedError

    def __pow__(self, other: Column | Any) -> Column:
        ser = self.column
        other = validate_comparand(self, other)
        return self._from_series((ser**other).rename(ser.name))

    def __rpow__(self, other: Column | Any) -> Column:  # pragma: no cover
        raise NotImplementedError

    def __mod__(self, other: Column | Any) -> Column:
        ser = self.column
        other = validate_comparand(self, other)
        return self._from_series((ser % other).rename(ser.name))

    def __rmod__(self, other: Column | Any) -> Column:  # pragma: no cover
        raise NotImplementedError

    def __divmod__(self, other: Column | Any) -> tuple[Column, Column]:
        quotient = self // other
        remainder = self - quotient * other
        return quotient, remainder

    # Unary

    def __invert__(self: Column) -> Column:
        ser = self.column
        return self._from_series(~ser)

    # Reductions

    def any(self, *, skip_nulls: bool | Scalar = True) -> Scalar:
        ser = self.column
        return self._to_scalar(ser.any())

    def all(self, *, skip_nulls: bool | Scalar = True) -> Scalar:
        ser = self.column
        return self._to_scalar(ser.all())

    def min(self, *, skip_nulls: bool | Scalar = True) -> Any:
        ser = self.column
        return self._to_scalar(ser.min())

    def max(self, *, skip_nulls: bool | Scalar = True) -> Any:
        ser = self.column
        return self._to_scalar(ser.max())

    def sum(self, *, skip_nulls: bool | Scalar = True) -> Any:
        ser = self.column
        return self._to_scalar(ser.sum())

    def prod(self, *, skip_nulls: bool | Scalar = True) -> Any:
        ser = self.column
        return self._to_scalar(ser.prod())

    def median(self, *, skip_nulls: bool | Scalar = True) -> Any:
        ser = self.column
        return self._to_scalar(ser.median())

    def mean(self, *, skip_nulls: bool | Scalar = True) -> Any:
        ser = self.column
        return self._to_scalar(ser.mean())

    def std(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> Any:
        ser = self.column
        return self._to_scalar(
            ser.std(ddof=correction),
        )

    def var(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> Any:
        ser = self.column
        return self._to_scalar(
            ser.var(ddof=correction),
        )

    def len(self) -> Scalar:
        return self._to_scalar(len(self._series))

    def n_unique(
        self,
        *,
        skip_nulls: bool = True,
    ) -> Scalar:
        ser = self.column
        return self._to_scalar(
            ser.nunique(),
        )

    # Transformations

    def is_null(self) -> Column:
        ser = self.column
        return self._from_series(ser.isna())

    def is_nan(self) -> Column:
        ser = self.column
        if is_extension_array_dtype(ser.dtype):
            return self._from_series((ser != ser).fillna(False))  # noqa: PLR0124
        return self._from_series(ser.isna())

    def sort(
        self,
        *,
        ascending: bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> Column:
        ser = self.column
        if ascending:
            return self._from_series(ser.sort_values().rename(self.name))
        return self._from_series(ser.sort_values().rename(self.name)[::-1])

    def is_in(self, values: Column) -> Column:
        ser = self.column
        return self._from_series(ser.isin(values.column))

    def sorted_indices(
        self,
        *,
        ascending: bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> Column:
        ser = self.column
        if ascending:
            return self._from_series(ser.sort_values().index.to_series(name=self.name))
        return self._from_series(
            ser.sort_values().index.to_series(name=self.name)[::-1]
        )

    def unique_indices(
        self,
        *,
        skip_nulls: bool | Scalar = True,
    ) -> Column:  # pragma: no cover
        msg = "not yet supported"
        raise NotImplementedError(msg)

    def fill_nan(self, value: float | NullType | Scalar) -> Column:
        ser = self.column.copy()
        if is_extension_array_dtype(ser.dtype):
            if self.__column_namespace__().is_null(value):
                ser[np.isnan(ser).fillna(False).to_numpy(bool)] = pd.NA
            else:
                ser[np.isnan(ser).fillna(False).to_numpy(bool)] = value
        else:
            if self.__column_namespace__().is_null(value):
                ser[np.isnan(ser).fillna(False).to_numpy(bool)] = np.nan
            else:
                ser[np.isnan(ser).fillna(False).to_numpy(bool)] = value
        return self._from_series(ser)

    def fill_null(
        self,
        value: Any,
    ) -> Column:
        value = validate_comparand(self, value)
        ser = self.column.copy()
        if is_extension_array_dtype(ser.dtype):
            # Mask should include NA values, but not NaN ones
            mask = ser.isna() & (~(ser != ser).fillna(False))  # noqa: PLR0124
            ser = ser.where(~mask, value)
        else:
            ser = ser.fillna(value)
        return self._from_series(ser.rename(self.name))

    def cumulative_sum(self, *, skip_nulls: bool | Scalar = True) -> Column:
        ser = self.column
        return self._from_series(ser.cumsum())

    def cumulative_prod(self, *, skip_nulls: bool | Scalar = True) -> Column:
        ser = self.column
        return self._from_series(ser.cumprod())

    def cumulative_max(self, *, skip_nulls: bool | Scalar = True) -> Column:
        ser = self.column
        return self._from_series(ser.cummax())

    def cumulative_min(self, *, skip_nulls: bool | Scalar = True) -> Column:
        ser = self.column
        return self._from_series(ser.cummin())

    def rename(self, name: str | Scalar) -> Column:
        ser = self.column
        return self._from_series(ser.rename(name))

    def shift(self, offset: int | Scalar) -> Column:
        ser = self.column
        return self._from_series(ser.shift(offset))

    # Conversions

    def to_array(self) -> Any:
        ser = self._materialise()
        return ser.to_numpy(
            dtype=NUMPY_MAPPING.get(self.column.dtype.name, self.column.dtype.name),
        )

    def cast(self, dtype: DType) -> Column:
        ser = self.column
        pandas_dtype = modin.dataframe_api_standard.map_standard_dtype_to_pandas_dtype(
            dtype,
        )
        return self._from_series(ser.astype(pandas_dtype))

    # --- temporal methods ---

    def year(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.year)

    def month(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.month)

    def day(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.day)

    def hour(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.hour)

    def minute(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.minute)

    def second(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.second)

    def microsecond(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.microsecond)

    def nanosecond(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.microsecond * 1000 + ser.dt.nanosecond)

    def iso_weekday(self) -> Column:
        ser = self.column
        return self._from_series(ser.dt.weekday + 1)

    def floor(self, frequency: str) -> Column:
        frequency = (
            frequency.replace("day", "D")
            .replace("hour", "H")
            .replace("minute", "T")
            .replace("second", "S")
            .replace("millisecond", "ms")
            .replace("microsecond", "us")
            .replace("nanosecond", "ns")
        )
        ser = self.column
        return self._from_series(ser.dt.floor(frequency))

    def unix_timestamp(
        self,
        *,
        time_unit: str | Scalar = "s",
    ) -> Column:
        ser = self.column
        if ser.dt.tz is None:
            result = ser - datetime(1970, 1, 1)
        else:  # pragma: no cover (todo: tz-awareness)
            result = ser.dt.tz_convert("UTC").dt.tz_localize(None) - datetime(
                1970, 1, 1
            )
        if time_unit == "s":
            result = pd.Series(
                np.floor(result.dt.total_seconds().astype("float64")),
                name=ser.name,
            )
        elif time_unit == "ms":
            result = pd.Series(
                np.floor(
                    np.floor(result.dt.total_seconds()) * 1000
                    + result.dt.microseconds // 1000,
                ),
                name=ser.name,
            )
        elif time_unit == "us":
            result = pd.Series(
                np.floor(result.dt.total_seconds()) * 1_000_000
                + result.dt.microseconds,
                name=ser.name,
            )
        elif time_unit == "ns":
            result = pd.Series(
                (
                    np.floor(result.dt.total_seconds()).astype("Int64") * 1_000_000
                    + result.dt.microseconds.astype("Int64")
                )
                * 1000
                + result.dt.nanoseconds.astype("Int64"),
                name=ser.name,
            )
        else:  # pragma: no cover
            msg = "Got invalid time_unit"
            raise AssertionError(msg)
        return self._from_series(result)
