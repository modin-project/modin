from __future__ import annotations

import collections
import warnings
from typing import TYPE_CHECKING, Any, Iterator, Literal, NoReturn

import numpy as np
from pandas.api.types import is_extension_array_dtype

import modin.dataframe_api_standard
import modin.pandas as pd
from modin.dataframe_api_standard.utils import validate_comparand

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dataframe_api import DataFrame as DataFrameT
    from dataframe_api.typing import AnyScalar, Column, DType, NullType, Scalar

    from modin.dataframe_api_standard.group_by_object import GroupBy
else:
    DataFrameT = object


class DataFrame(DataFrameT):
    """dataframe object"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        api_version: str,
        is_persisted: bool = False,
    ) -> None:
        self._is_persisted = is_persisted
        self._validate_columns(dataframe.columns)
        self._dataframe = dataframe.reset_index(drop=True)
        self._api_version = api_version

    # Validation helper methods

    def _validate_is_persisted(self) -> pd.DataFrame:
        if not self._is_persisted:
            msg = "Method requires you to call `.persist` first.\n\nNote: `.persist` forces materialisation in lazy libraries and so should be called as late as possible in your pipeline. Use with care."
            raise ValueError(
                msg,
            )
        return self.dataframe

    def __repr__(self) -> str:  # pragma: no cover
        header = f" Standard DataFrame (api_version={self._api_version}) "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Add `.dataframe` to see native output         |\n"
            + "└"
            + "─" * length
            + "┘\n"
        )

    def _validate_columns(self, columns: Sequence[str]) -> None:
        counter = collections.Counter(columns)
        for col, count in counter.items():
            if count > 1:
                msg = f"Expected unique column names, got {col} {count} time(s)"
                raise ValueError(
                    msg,
                )

    def _validate_booleanness(self) -> None:
        if not (
            (self.dataframe.dtypes == "bool") | (self.dataframe.dtypes == "boolean")
        ).all():
            msg = "'any' can only be called on DataFrame where all dtypes are 'bool'"
            raise TypeError(
                msg,
            )

    def _from_dataframe(self, df: pd.DataFrame) -> DataFrame:
        return DataFrame(
            df,
            api_version=self._api_version,
            is_persisted=self._is_persisted,
        )

    # Properties
    @property
    def schema(self) -> dict[str, DType]:
        return {
            column_name: modin.dataframe_api_standard.map_pandas_dtype_to_standard_dtype(
                dtype.name,
            )
            for column_name, dtype in self.dataframe.dtypes.items()
        }

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._dataframe

    @property
    def column_names(self) -> list[str]:
        return self.dataframe.columns.tolist()  # type: ignore[no-any-return]

    # In the Standard

    def __dataframe_namespace__(
        self,
    ) -> modin.dataframe_api_standard.Namespace:
        return modin.dataframe_api_standard.Namespace(
            api_version=self._api_version,
        )

    def iter_columns(self) -> Iterator[Column]:
        return (self.col(col_name) for col_name in self.column_names)

    def col(self, name: str) -> Column:
        from modin.dataframe_api_standard.column_object import Column

        return Column(
            self.dataframe.loc[:, name],
            df=None if self._is_persisted else self,
            api_version=self._api_version,
            is_persisted=self._is_persisted,
        )

    def shape(self) -> tuple[int, int]:
        df = self._validate_is_persisted()
        return df.shape  # type: ignore[no-any-return]

    def group_by(self, *keys: str) -> GroupBy:
        from modin.dataframe_api_standard.group_by_object import GroupBy

        for key in keys:
            if key not in self.column_names:
                msg = f"key {key} not present in DataFrame's columns"
                raise KeyError(msg)
        return GroupBy(self, keys, api_version=self._api_version)

    def select(self, *columns: str) -> DataFrame:
        cols = list(columns)
        if cols and isinstance(cols[0], (list, tuple)):
            msg = f"Expected iterable of column names, but the first element is: {type(cols[0])}"
            raise TypeError(msg)
        return self._from_dataframe(
            self.dataframe.loc[:, list(columns)],
        )

    def take(
        self,
        indices: Column,
    ) -> DataFrame:
        _indices = validate_comparand(self, indices)
        return self._from_dataframe(
            self.dataframe.iloc[_indices.to_list(), :],
        )

    def slice_rows(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
    ) -> DataFrame:
        return self._from_dataframe(self.dataframe.iloc[start:stop:step])

    def filter(
        self,
        mask: Column,
    ) -> DataFrame:
        _mask = validate_comparand(self, mask)
        df = self.dataframe
        df = df.loc[_mask]
        return self._from_dataframe(df)

    def assign(
        self,
        *columns: Column,
    ) -> DataFrame:
        from modin.dataframe_api_standard.column_object import Column

        df = self.dataframe.copy()  # TODO: remove defensive copy with CoW?
        for column in columns:
            if not isinstance(column, Column):
                msg = f"Expected iterable of Column, but the first element is: {type(column)}"
                raise TypeError(msg)
            _series = validate_comparand(self, column)
            df[_series.name] = _series
        return self._from_dataframe(df)

    def drop(self, *labels: str) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.drop(list(labels), axis=1),
        )

    def rename(self, mapping: Mapping[str, str]) -> DataFrame:
        if not isinstance(mapping, collections.abc.Mapping):
            msg = f"Expected Mapping, got: {type(mapping)}"
            raise TypeError(msg)
        return self._from_dataframe(
            self.dataframe.rename(columns=mapping),
        )

    def get_column_names(self) -> list[str]:
        # DO NOT REMOVE
        # This one is used in upstream tests - even if deprecated,
        # just leave it in for backwards compatibility
        return self.dataframe.columns.tolist()  # type: ignore[no-any-return]

    def sort(
        self,
        *keys: str,
        ascending: Sequence[bool] | bool = True,
        nulls_position: Literal["first", "last"] = "last",
    ) -> DataFrame:
        if not keys:
            keys = self.dataframe.columns.tolist()
        df = self.dataframe
        return self._from_dataframe(
            df.sort_values(list(keys), ascending=ascending),
        )

    # Binary operations

    def __eq__(self, other: AnyScalar) -> DataFrame:  # type: ignore[override]
        return self._from_dataframe(self.dataframe.__eq__(other))

    def __ne__(self, other: AnyScalar) -> DataFrame:  # type: ignore[override]
        return self._from_dataframe(self.dataframe.__ne__(other))

    def __ge__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(self.dataframe.__ge__(other))

    def __gt__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(self.dataframe.__gt__(other))

    def __le__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(self.dataframe.__le__(other))

    def __lt__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(self.dataframe.__lt__(other))

    def __and__(self, other: AnyScalar) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.__and__(other),
        )

    def __rand__(self, other: Column | AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self.__and__(_other)

    def __or__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(self.dataframe.__or__(_other))

    def __ror__(self, other: Column | AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self.__or__(_other)

    def __add__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__add__(_other),
        )

    def __radd__(self, other: Column | AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self.__add__(_other)

    def __sub__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__sub__(_other),
        )

    def __rsub__(self, other: Column | AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return -1 * self.__sub__(_other)

    def __mul__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__mul__(_other),
        )

    def __rmul__(self, other: Column | AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self.__mul__(_other)

    def __truediv__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__truediv__(_other),
        )

    def __rtruediv__(self, other: Column | AnyScalar) -> DataFrame:  # pragma: no cover
        _other = validate_comparand(self, other)  # noqa: F841
        raise NotImplementedError

    def __floordiv__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__floordiv__(_other),
        )

    def __rfloordiv__(self, other: Column | AnyScalar) -> DataFrame:  # pragma: no cover
        _other = validate_comparand(self, other)  # noqa: F841
        raise NotImplementedError

    def __pow__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)
        return self._from_dataframe(
            self.dataframe.__pow__(_other),
        )

    def __rpow__(self, other: Column | AnyScalar) -> DataFrame:  # pragma: no cover
        _other = validate_comparand(self, other)  # noqa: F841
        raise NotImplementedError

    def __mod__(self, other: AnyScalar) -> DataFrame:
        _other = validate_comparand(self, other)  # noqa: F841
        return self._from_dataframe(
            self.dataframe.__mod__(other),
        )

    def __rmod__(self, other: Column | AnyScalar) -> DataFrame:  # type: ignore[misc]  # pragma: no cover
        _other = validate_comparand(self, other)  # noqa: F841
        raise NotImplementedError

    def __divmod__(
        self,
        other: DataFrame | AnyScalar,
    ) -> tuple[DataFrame, DataFrame]:
        _other = validate_comparand(self, other)
        quotient, remainder = self.dataframe.__divmod__(_other)
        return self._from_dataframe(quotient), self._from_dataframe(
            remainder,
        )

    # Unary

    def __invert__(self) -> DataFrame:
        self._validate_booleanness()
        return self._from_dataframe(self.dataframe.__invert__())

    def __iter__(self) -> NoReturn:
        raise NotImplementedError

    # Reductions

    def any(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        self._validate_booleanness()
        return self._from_dataframe(
            self.dataframe.any().to_frame().T,
        )

    def all(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        self._validate_booleanness()
        return self._from_dataframe(
            self.dataframe.all().to_frame().T,
        )

    def min(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.min().to_frame().T,
        )

    def max(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.max().to_frame().T,
        )

    def sum(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.sum().to_frame().T,
        )

    def prod(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.prod().to_frame().T,
        )

    def median(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.median().to_frame().T,
        )

    def mean(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.mean().to_frame().T,
        )

    def std(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.std().to_frame().T,
        )

    def var(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        return self._from_dataframe(
            self.dataframe.var().to_frame().T,
        )

    # Transformations

    def is_null(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result: list[pd.Series] = []
        for column in self.dataframe.columns:
            result.append(self.dataframe[column].isna())
        return self._from_dataframe(pd.concat(result, axis=1))

    def is_nan(self) -> DataFrame:
        return self.assign(*[col.is_nan() for col in self.iter_columns()])

    def fill_nan(self, value: float | Scalar | NullType) -> DataFrame:
        _value = validate_comparand(self, value)
        new_cols = {}
        df = self.dataframe
        for col in df.columns:
            ser = df[col].copy()
            if is_extension_array_dtype(ser.dtype):
                if self.__dataframe_namespace__().is_null(_value):
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = pd.NA
                else:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = _value
            else:
                if self.__dataframe_namespace__().is_null(_value):
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = np.nan
                else:
                    ser[np.isnan(ser).fillna(False).to_numpy(bool)] = _value
            new_cols[col] = ser
        df = pd.DataFrame(new_cols)
        return self._from_dataframe(df)

    def fill_null(
        self,
        value: AnyScalar,
        *,
        column_names: list[str] | None = None,
    ) -> DataFrame:
        if column_names is None:
            column_names = self.dataframe.columns.tolist()
        assert isinstance(column_names, list)  # help type checkers
        return self.assign(
            *[
                col.fill_null(value)
                for col in self.iter_columns()
                if col.name in column_names
            ],
        )

    def drop_nulls(
        self,
        *,
        column_names: list[str] | None = None,
    ) -> DataFrame:
        namespace = self.__dataframe_namespace__()
        mask = ~namespace.any_horizontal(
            *[
                self.col(col_name).is_null()
                for col_name in column_names or self.column_names
            ],
        )
        return self.filter(mask)

    # Other

    def join(
        self,
        other: DataFrame,
        *,
        how: Literal["left", "inner", "outer"],
        left_on: str | list[str],
        right_on: str | list[str],
    ) -> DataFrame:
        if how not in ["left", "inner", "outer"]:
            msg = f"Expected 'left', 'inner', 'outer', got: {how}"
            raise ValueError(msg)

        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        if overlap := (set(self.column_names) - set(left_on)).intersection(
            set(other.column_names) - set(right_on),
        ):
            msg = f"Found overlapping columns in join: {overlap}. Please rename columns to avoid this."
            raise ValueError(msg)

        return self._from_dataframe(
            self.dataframe.merge(
                other.dataframe,
                left_on=left_on,
                right_on=right_on,
                how=how,
            ),
        )

    def persist(self) -> DataFrame:
        if self._is_persisted:
            warnings.warn(
                "Calling `.persist` on DataFrame that was already persisted",
                UserWarning,
                stacklevel=2,
            )
        return DataFrame(
            self.dataframe,
            api_version=self._api_version,
            is_persisted=True,
        )

    # Conversion

    def to_array(self, dtype: DType | None = None) -> Any:
        self._validate_is_persisted()
        return self.dataframe.to_numpy()

    def cast(self, dtypes: Mapping[str, DType]) -> DataFrame:
        from modin.dataframe_api_standard import map_standard_dtype_to_pandas_dtype

        df = self._dataframe
        return self._from_dataframe(
            df.astype(
                {
                    col: map_standard_dtype_to_pandas_dtype(dtype)
                    for col, dtype in dtypes.items()
                },
            ),
        )
