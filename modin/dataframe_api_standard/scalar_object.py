from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from modin.dataframe_api_standard.utils import validate_comparand

if TYPE_CHECKING:
    from dataframe_api.typing import DType, Namespace
    from dataframe_api.typing import Scalar as ScalarT

    from modin.dataframe_api_standard.dataframe_object import DataFrame
else:
    ScalarT = object


class Scalar(ScalarT):
    def __init__(
        self,
        value: Any,
        api_version: str,
        df: DataFrame | None,
        *,
        is_persisted: bool = False,
    ) -> None:
        self._value = value
        self._api_version = api_version
        self._df = df
        self._is_persisted = is_persisted
        assert is_persisted ^ (df is not None)

    def __scalar_namespace__(self) -> Namespace:
        from modin.dataframe_api_standard import Namespace

        return Namespace(api_version=self._api_version)

    def _from_scalar(self, scalar: Scalar) -> Scalar:
        return Scalar(
            scalar,
            df=self._df,
            api_version=self._api_version,
            is_persisted=self._is_persisted,
        )

    @property
    def dtype(self) -> DType:  # pragma: no cover  # todo
        msg = "dtype not yet implemented for Scalar"
        raise NotImplementedError(msg)

    @property
    def scalar(self) -> Any:  # pragma: no cover  # todo
        return self._value

    @property
    def parent_dataframe(self) -> Any:  # pragma: no cover  # todo
        return self._df

    def _materialise(self) -> Any:
        if not self._is_persisted:
            msg = "Can't call __bool__ on Scalar. Please use .persist() first."
            raise RuntimeError(msg)
        return self._value

    def persist(self) -> Scalar:
        if self._is_persisted:
            warnings.warn(
                "Calling `.persist` on Scalar that was already persisted",
                UserWarning,
                stacklevel=2,
            )
        return Scalar(
            self._value,
            df=None,
            api_version=self._api_version,
            is_persisted=True,
        )

    def __lt__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__lt__(other))

    def __le__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__le__(other))

    def __eq__(self, other: Any) -> Scalar:  # type: ignore[override]
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__eq__(other))

    def __ne__(self, other: Any) -> Scalar:  # type: ignore[override]
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__ne__(other))

    def __gt__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__gt__(other))

    def __ge__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__ge__(other))

    def __add__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__add__(other))

    def __radd__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(other + self._value)

    def __sub__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__sub__(other))

    def __rsub__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(other - self._value)

    def __mul__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__mul__(other))

    def __rmul__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(other * self._value)

    def __mod__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__mod__(other))

    def __rmod__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(other % self._value)

    def __pow__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__pow__(other))

    def __rpow__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(other**self._value)

    def __floordiv__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__floordiv__(other))

    def __rfloordiv__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(other // self._value)

    def __truediv__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(self._value.__truediv__(other))

    def __rtruediv__(self, other: Any) -> Scalar:
        other = validate_comparand(self, other)
        if other is NotImplemented:
            return NotImplemented
        return self._from_scalar(other / self._value)

    def __neg__(self) -> Scalar:
        return self._from_scalar(self._value.__neg__())

    def __abs__(self) -> Scalar:
        return self._from_scalar(self._value.__abs__())

    def __bool__(self) -> bool:
        return self._materialise().__bool__()  # type: ignore[no-any-return]

    def __int__(self) -> int:
        return self._materialise().__int__()  # type: ignore[no-any-return]

    def __float__(self) -> float:
        return self._materialise().__float__()  # type: ignore[no-any-return]

    def __repr__(self) -> str:  # pragma: no cover
        header = f" Standard Scalar (api_version={self._api_version}) "
        length = len(header)
        return (
            "┌"
            + "─" * length
            + "┐\n"
            + f"|{header}|\n"
            + "| Add `.scalar` to see native output         |\n"
            + "└"
            + "─" * length
            + "┘\n"
        )
