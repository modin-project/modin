from __future__ import annotations

from typing import TYPE_CHECKING, cast

import modin.pandas as pd
from modin.dataframe_api_standard import Namespace
from modin.dataframe_api_standard.dataframe_object import DataFrame

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dataframe_api import Aggregation as AggregationT
    from dataframe_api import GroupBy as GroupByT
    from dataframe_api.typing import NullType, Scalar


else:
    GroupByT = object


class GroupBy(GroupByT):
    def __init__(self, df: DataFrame, keys: Sequence[str], api_version: str) -> None:
        self._df = df.dataframe
        self._is_persisted = df._is_persisted
        self._grouped = self._df.groupby(list(keys), sort=False, as_index=False)
        self._keys = list(keys)
        self._api_version = api_version

    def _validate_result(self, result: pd.DataFrame) -> None:
        failed_columns = self._df.columns.difference(result.columns)
        if len(failed_columns) > 0:  # pragma: no cover
            msg = "Groupby operation could not be performed on columns "
            +f"{failed_columns}. Please drop them before calling group_by."
            raise AssertionError(
                msg,
            )

    def _validate_booleanness(self) -> None:
        if not (
            (self._df.drop(columns=self._keys).dtypes == "bool")
            | (self._df.drop(columns=self._keys).dtypes == "boolean")
        ).all():
            msg = (
                "'function' can only be called on DataFrame where all dtypes are 'bool'"
            )
            raise TypeError(
                msg,
            )

    def _to_dataframe(self, result: pd.DataFrame) -> DataFrame:
        return DataFrame(
            result,
            api_version=self._api_version,
            is_persisted=self._is_persisted,
        )

    def size(self) -> DataFrame:
        return self._to_dataframe(self._grouped.size())

    def any(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        self._validate_booleanness()
        result = self._grouped.any()
        self._validate_result(result)
        return self._to_dataframe(result)

    def all(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        self._validate_booleanness()
        result = self._grouped.all()
        self._validate_result(result)
        return self._to_dataframe(result)

    def min(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.min()
        self._validate_result(result)
        return self._to_dataframe(result)

    def max(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.max()
        self._validate_result(result)
        return self._to_dataframe(result)

    def sum(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.sum()
        self._validate_result(result)
        return self._to_dataframe(result)

    def prod(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.prod()
        self._validate_result(result)
        return self._to_dataframe(result)

    def median(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.median()
        self._validate_result(result)
        return self._to_dataframe(result)

    def mean(self, *, skip_nulls: bool | Scalar = True) -> DataFrame:
        result = self._grouped.mean()
        self._validate_result(result)
        return self._to_dataframe(result)

    def std(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        result = self._grouped.std()
        self._validate_result(result)
        return self._to_dataframe(result)

    def var(
        self,
        *,
        correction: float | Scalar | NullType = 1.0,
        skip_nulls: bool | Scalar = True,
    ) -> DataFrame:
        result = self._grouped.var()
        self._validate_result(result)
        return self._to_dataframe(result)

    def aggregate(
        self,
        *aggregations: AggregationT,
    ) -> DataFrame:
        aggregations = validate_aggregations(*aggregations, keys=self._keys)
        df = self._grouped.agg(
            **{
                aggregation.output_name: resolve_aggregation(  # type: ignore[attr-defined]
                    aggregation,
                )
                for aggregation in aggregations
            },
        )
        return self._to_dataframe(
            df,
        )


def validate_aggregations(
    *aggregations: AggregationT,
    keys: Sequence[str],
) -> tuple[AggregationT, ...]:
    return tuple(
        (
            aggregation
            if aggregation.aggregation != "size"  # type: ignore[attr-defined]
            else aggregation._replace(column_name=keys[0])  # type: ignore[attr-defined]
        )
        for aggregation in aggregations
    )


def resolve_aggregation(aggregation: AggregationT) -> pd.NamedAgg:
    aggregation = cast(Namespace.Aggregation, aggregation)
    return pd.NamedAgg(
        column=aggregation.column_name,
        aggfunc=aggregation.aggregation,
    )
