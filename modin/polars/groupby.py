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

"""Implement GroupBy public API as pandas does."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modin.polars import DataFrame


class GroupBy:

    def __init__(
        self,
        df: "DataFrame",
        *by,
        maintain_order: bool = False,
        **named_by,
    ) -> None:
        self.df = df
        if len(by) == 1:
            self.by = by[0]
        else:
            if all(isinstance(b, str) and b in self.df.columns for b in by):
                self.by = self.df[list(by)]._query_compiler
            elif all(isinstance(b, type(self._df._query_compiler)) for b in by):
                self.by = by
            else:
                raise NotImplementedError("not yet")
        self.named_by = named_by
        self.maintain_order = maintain_order

    def agg(self, *aggs, **named_aggs):
        raise NotImplementedError("not yet")

    def all(self):
        raise NotImplementedError("not yet")

    def map_groups(self, function) -> "DataFrame":
        raise NotImplementedError("not yet")

    apply = map_groups

    def count(self):
        return self.len(name="count")

    def first(self) -> "DataFrame":
        return self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_first(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=True,
                ),
                agg_args=(),
                agg_kwargs={},
                drop=False,
            ).reset_index(drop=False)
        )

    def head(self, n: int = 5):
        return self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_head(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=False,
                ),
                agg_args=(),
                agg_kwargs=dict(n=n),
                drop=False,
            )
        )

    def last(self) -> "DataFrame":
        return self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_last(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=True,
                ),
                agg_args=(),
                agg_kwargs={},
                drop=False,
            ).reset_index(drop=False)
        )

    def len(self, name: str | None = None) -> "DataFrame":
        if name is None:
            name = "len"
        result = self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_size(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=False,
                ),
                agg_args=(),
                agg_kwargs={},
                drop=False,
            )
        )
        result._query_compiler.columns = [
            c if c != "size" else name for c in result.columns
        ]
        return result

    def max(self) -> "DataFrame":
        return self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_max(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=False,
                ),
                agg_args=(),
                agg_kwargs={},
                drop=False,
            )
        )

    def mean(self) -> "DataFrame":
        # TODO: Non numeric columns are dropped, but in Polars they are converted to null
        return self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_mean(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=True,
                ),
                agg_args=(),
                agg_kwargs=dict(numeric_only=True),
                drop=False,
            ).reset_index(drop=False)
        )

    def median(self) -> "DataFrame":
        # TODO: Non numeric columns are dropped, but in Polars they are converted to null
        return self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_median(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=True,
                ),
                agg_args=(),
                agg_kwargs=dict(numeric_only=True),
                drop=False,
            ).reset_index(drop=False)
        )

    def min(self) -> "DataFrame":
        return self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_min(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=False,
                ),
                agg_args=(),
                agg_kwargs={},
                drop=False,
            )
        )

    def n_unique(self) -> "DataFrame":
        return self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_nunique(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=False,
                ),
                agg_args=(),
                agg_kwargs={},
                drop=False,
            )
        )

    def quantile(self, quantile: float, interpolation="nearest") -> "DataFrame":
        # TODO: Non numeric columns are dropped, but in Polars they are converted to null
        # TODO: interpolation types not yet supported
        return self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_quantile(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=True,
                ),
                agg_args=(),
                agg_kwargs=dict(numeric_only=True, q=quantile),
                drop=False,
            ).reset_index(drop=False)
        )

    def sum(self) -> "DataFrame":
        # TODO: Non numeric columns are dropped, but in Polars they are converted to null
        return self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_sum(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=True,
                ),
                agg_args=(),
                agg_kwargs=dict(numeric_only=True),
                drop=False,
            ).reset_index(drop=False)
        )

    def tail(self, n: int = 5):
        return self.df.__constructor__(
            _query_compiler=self.df._query_compiler.groupby_tail(
                self.by,
                axis=0,
                groupby_kwargs=dict(
                    sort=not self.maintain_order,
                    as_index=False,
                ),
                agg_args=(),
                agg_kwargs=dict(n=n),
                drop=False,
            )
        )
