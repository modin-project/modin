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
Dataframe exchange protocol implementation.

See more in https://data-apis.org/dataframe-protocol/latest/index.html.

Notes
-----
- Interpreting a raw pointer (as in ``Buffer.ptr``) is annoying and unsafe to
  do in pure Python. It's more general but definitely less friendly than having
  ``to_arrow`` and ``to_numpy`` methods. So for the buffers which lack
  ``__dlpack__`` (e.g., because the column dtype isn't supported by DLPack),
  this is worth looking at again.
"""

import collections
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
    ProtocolDataframe,
)
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.utils import _inherit_docstrings

from .column import PandasProtocolColumn


@_inherit_docstrings(ProtocolDataframe)
class PandasProtocolDataframe(ProtocolDataframe):
    """
    A data frame class, with only the methods required by the interchange protocol defined.

    Instances of this (private) class are returned from ``modin.pandas.DataFrame.__dataframe__``
    as objects with the methods and attributes defined on this class.

    A "data frame" represents an ordered collection of named columns.
    A column's "name" must be a unique string. Columns may be accessed by name or by position.
    This could be a public data frame class, or an object with the methods and
    attributes defined on this DataFrame class could be returned from the
    ``__dataframe__`` method of a public data frame class in a library adhering
    to the dataframe interchange protocol specification.

    Parameters
    ----------
    df : PandasDataframe
        A ``PandasDataframe`` object.
    nan_as_null : bool, default:False
        A keyword intended for the consumer to tell the producer
        to overwrite null values in the data with ``NaN`` (or ``NaT``).
        This currently has no effect; once support for nullable extension
        dtypes is added, this value should be propagated to columns.
    allow_copy : bool, default: True
        A keyword that defines whether or not the library is allowed
        to make a copy of the data. For example, copying data would be necessary
        if a library supports strided buffers, given that this protocol
        specifies contiguous buffers. Currently, if the flag is set to ``False``
        and a copy is needed, a ``RuntimeError`` will be raised.
    """

    def __init__(
        self,
        df: PandasDataframe,
        nan_as_null: bool = False,
        allow_copy: bool = True,
    ) -> None:
        self._df = df
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):
        return PandasProtocolDataframe(
            self._df, nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        return {"modin.index": self._df.index}

    def num_columns(self) -> int:
        return len(self._df.columns)

    def num_rows(self) -> int:
        return len(self._df.index)

    def num_chunks(self) -> int:
        return self._df._partitions.shape[0]

    def column_names(self) -> Iterable[str]:
        for col in self._df.columns:
            yield col

    def get_column(self, i: int) -> PandasProtocolColumn:
        return PandasProtocolColumn(
            self._df.take_2d_labels_or_positional(
                row_positions=None, col_positions=[i]
            ),
            allow_copy=self._allow_copy,
        )

    def get_column_by_name(self, name: str) -> PandasProtocolColumn:
        return PandasProtocolColumn(
            self._df.take_2d_labels_or_positional(
                row_positions=None, col_labels=[name]
            ),
            allow_copy=self._allow_copy,
        )

    def get_columns(self) -> Iterable[PandasProtocolColumn]:
        for name in self._df.columns:
            yield PandasProtocolColumn(
                self._df.take_2d_labels_or_positional(
                    row_positions=None, col_labels=[name]
                ),
                allow_copy=self._allow_copy,
            )

    def select_columns(self, indices: Sequence[int]) -> "PandasProtocolDataframe":
        if not isinstance(indices, collections.abc.Sequence):
            raise ValueError("`indices` is not a sequence")

        return PandasProtocolDataframe(
            self._df.take_2d_labels_or_positional(
                row_positions=None, col_positions=indices
            ),
            allow_copy=self._allow_copy,
        )

    def select_columns_by_name(self, names: Sequence[str]) -> "PandasProtocolDataframe":
        if not isinstance(names, collections.abc.Sequence):
            raise ValueError("`names` is not a sequence")

        return PandasProtocolDataframe(
            self._df.take_2d_labels_or_positional(row_positions=None, col_labels=names),
            allow_copy=self._allow_copy,
        )

    def get_chunks(
        self, n_chunks: Optional[int] = None
    ) -> Iterable["PandasProtocolDataframe"]:
        cur_n_chunks = self.num_chunks()
        n_rows = self.num_rows()
        if n_chunks is None or n_chunks == cur_n_chunks:
            cum_row_lengths = np.cumsum([0] + self._df.row_lengths)
            for i in range(len(cum_row_lengths) - 1):
                yield PandasProtocolDataframe(
                    self._df.take_2d_labels_or_positional(
                        row_positions=range(cum_row_lengths[i], cum_row_lengths[i + 1]),
                        col_positions=None,
                    ),
                    allow_copy=self._allow_copy,
                )
            return
        if n_chunks % cur_n_chunks != 0:
            raise RuntimeError(
                "The passed `n_chunks` must be a multiple of `self.num_chunks()`."
            )

        if n_chunks > n_rows:
            raise RuntimeError(
                "The passed `n_chunks` value is bigger than `self.num_rows()`."
            )

        chunksize = n_rows // n_chunks
        new_lengths = [chunksize] * n_chunks
        new_lengths[-1] = n_rows % n_chunks + new_lengths[-1]

        new_partitions = self._df._partition_mgr_cls.map_axis_partitions(
            0,
            self._df._partitions,
            lambda df: df,
            keep_partitioning=False,
            lengths=new_lengths,
        )
        new_df = self._df.__constructor__(
            new_partitions,
            self._df.index,
            self._df.columns,
            new_lengths,
            self._df.column_widths,
        )
        cum_row_lengths = np.cumsum([0] + new_df.row_lengths)
        for i in range(len(cum_row_lengths) - 1):
            yield PandasProtocolDataframe(
                new_df.take_2d_labels_or_positional(
                    row_positions=range(cum_row_lengths[i], cum_row_lengths[i + 1]),
                    col_positions=None,
                ),
                allow_copy=self._allow_copy,
            )
