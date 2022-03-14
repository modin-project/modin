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

"""The module houses OmnisciOnNative implementation of the Dataframe class of DataFrame exchange protocol."""

import collections
import numpy as np
import pyarrow as pa
from typing import Optional, Iterable, Sequence, Dict, Any

from modin.experimental.core.execution.native.implementations.omnisci_on_native.dataframe.dataframe import (
    OmnisciOnNativeDataframe,
)
from modin.core.dataframe.base.exchange.dataframe_protocol import ProtocolDataframe
from modin.utils import _inherit_docstrings
from modin.error_message import ErrorMessage
from modin.experimental.core.execution.native.implementations.omnisci_on_native.df_algebra import (
    MaskNode,
    FrameNode,
    TransformNode,
    UnionNode,
)
from .column import OmnisciProtocolColumn


@_inherit_docstrings(ProtocolDataframe)
class OmnisciProtocolDataframe(ProtocolDataframe):
    """
    Implement the DataFrame exchange protocol class for ``OmnisciOnNative`` execution.

    Parameters
    ----------
    df : OmnisciOnNativeDataframe
        DataFrame object that holds the data.
    nan_as_null : bool, default: False
        Whether to overwrite null values in the data with ``NaN``.
    allow_copy : bool, default: True
        Whether allow to doing copy of the underlying data during export flow.
        If a copy or any kind of data transfer/materialization would be required raise ``RuntimeError``.
    """

    def __init__(
        self,
        df: OmnisciOnNativeDataframe,
        nan_as_null: bool = False,
        allow_copy: bool = True,
    ) -> None:
        if nan_as_null:
            raise NotImplementedError(
                "Proccessing of `nan_as_null=True` is not yet supported."
            )

        self._df = df
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    @property
    def metadata(self) -> Dict[str, Any]:
        # TODO: as the frame's index is stored as a separate column inside PyArrow table
        # we may want to return the column's name here instead of materialized index.
        # This will require the internal index column to be visible in the protocol's column
        # accessor methods.
        self._maybe_raise_if_materialize()
        return {"modin.index": self._df.index}

    def num_columns(self) -> int:
        return len(self._df.columns)

    def num_rows(self) -> int:
        self._maybe_raise_if_materialize()
        return len(self._df.index)

    def num_chunks(self) -> int:
        return len(self._chunk_slices) - 1

    __chunk_slices = None

    @property
    def _chunk_slices(self) -> np.ndarray:
        """
        Compute chunks start-stop indices in the underlying PyArrow table.

        Returns
        -------
        np.ndarray
            An array holding start-stop indices of the chunks, for ex. ``[0, 5, 10, 20]``
            describes 3 chunks bound by the following indices:
                chunk1: [0, 5),
                chunk2: [5, 10),
                chunk3: [10, 20).

        Notes
        -----
        Arrow table allows for the columns to be chunked independently, so in order to satisfy
        the protocol's requirement of equally chunked columns, we have to align column chunks
        with the minimal one. For example:
            Originally chunked table:        Aligned table:
            |col0|col1|                       |col0|col1|
            |    |    |                       |    |    |
            |0   |a   |                       |0   |a   |
            |----|b   |                       |----|----|
            |1   |----|                       |1   |b   |
            |2   |c   |                       |----|----|
            |3   |d   |                       |2   |c   |
            |----|----|                       |3   |d   |
            |4   |e   |                       |----|----|
                                              |4   |e   |
        """
        if self.__chunk_slices is None:
            at = self._pyarrow_table
            # What we need to do is to union offsets of all the columns
            col_slices = set({0})
            for col in at.columns:
                col_slices = col_slices.union(
                    np.cumsum([len(chunk) for chunk in col.chunks])
                )
            self.__chunk_slices = np.sort(
                np.fromiter(col_slices, dtype=int, count=len(col_slices))
            )

        return self.__chunk_slices

    def _maybe_raise_if_materialize(self):
        """Raise a ``RuntimeError`` if the way of retrieving the data violates the ``allow_copy`` flag."""
        if not self._allow_copy and not self._is_zero_copy_possible:
            raise RuntimeError("Copy required with 'allow_copy=False'")

    __is_zero_copy_possible = None

    @property
    def _is_zero_copy_possible(self) -> bool:
        """
        Check whether it's possible to retrieve data from the DataFrame zero-copy.

        The 'zero-copy' term also means that no extra computations or data transers
        are needed to access the data.

        Returns
        -------
        bool
        """
        if self.__is_zero_copy_possible is None:
            if self._df._has_arrow_table():
                # If PyArrow table is already materialized then we can
                # retrieve data zero-copy
                self.__is_zero_copy_possible = True
            elif not self._df._can_execute_arrow():
                # When not able to execute the plan via PyArrow means
                # that we have to involve OmniSci, so no zero-copy.
                self.__is_zero_copy_possible = False
            else:
                # Check whether the plan for PyArrow can be executed zero-copy
                self.__is_zero_copy_possible = self._is_zero_copy_arrow_op(self._df._op)
        return self.__is_zero_copy_possible

    @classmethod
    def _is_zero_copy_arrow_op(cls, op) -> bool:
        """
        Check whether the passed node of the delayed computation tree could be executed zero-copy via PyArrow execution.

        Parameters
        ----------
        op : DFAlgNode

        Returns
        -------
        bool
        """
        is_zero_copy_op = False
        if isinstance(op, (FrameNode, TransformNode, UnionNode)):
            # - FrameNode: already materialized PyArrow table
            # - TransformNode: select certain columns of the table, implemented zero-copy (``df._arrow_select``)
            # - UnionNode: concatenate PyArrow tables, implemented zero-copy (``df._arrow_concat``)
            is_zero_copy_op = True
        elif isinstance(op, MaskNode) and (
            isinstance(op.row_positions, slice) or is_range_like(op.row_positions)
        ):
            # Can select rows zero-copy if indexer is a slice-like (``df._arrow_row_slice``)
            is_zero_copy_op = True
        return is_zero_copy_op and all(
            # Walk the computation tree
            cls._is_zero_copy_arrow_op(_op)
            for _op in getattr(op, "inputs", [])
        )

    @property
    def _pyarrow_table(self) -> pa.Table:
        """
        Get PyArrow table representing the DataFrame.

        Returns
        -------
        pyarrow.Table
        """
        self._maybe_raise_if_materialize()

        if not self._df._has_arrow_table():
            self._df._execute()

        at = self._df._partitions[0][0].arrow_table
        assert at is not None
        return at

    def column_names(self) -> Iterable[str]:
        for col in self._df.columns:
            yield col

    def get_column(self, i: int) -> OmnisciProtocolColumn:
        return OmnisciProtocolColumn(
            OmnisciProtocolDataframe(
                self._df.mask(col_positions=[i]),
                allow_copy=self._allow_copy,
            ),
        )

    def get_column_by_name(self, name: str) -> OmnisciProtocolColumn:
        return OmnisciProtocolColumn(
            OmnisciProtocolDataframe(
                self._df.mask(col_labels=[name]),
                allow_copy=self._allow_copy,
            ),
        )

    def get_columns(self) -> Iterable[OmnisciProtocolColumn]:
        for name in self._df.columns:
            yield OmnisciProtocolColumn(
                OmnisciProtocolDataframe(
                    self._df.mask(col_labels=[name]),
                    nan_as_null=self._nan_as_null,
                    allow_copy=self._allow_copy,
                ),
            )

    def select_columns(self, indices: Sequence[int]) -> "OmnisciProtocolDataframe":
        if not isinstance(indices, collections.Sequence):
            raise ValueError("`indices` is not a sequence")

        return OmnisciProtocolDataframe(
            self._df.mask(col_positions=list(indices)),
            nan_as_null=self._nan_as_null,
            allow_copy=self._allow_copy,
        )

    def select_columns_by_name(
        self, names: Sequence[str]
    ) -> "OmnisciProtocolDataframe":
        if not isinstance(names, collections.Sequence):
            raise ValueError("`names` is not a sequence")

        return OmnisciProtocolDataframe(
            self._df.mask(col_labels=list(names)),
            nan_as_null=self._nan_as_null,
            allow_copy=self._allow_copy,
        )

    def get_chunks(
        self, n_chunks: Optional[int] = None
    ) -> Iterable["OmnisciProtocolDataframe"]:
        if n_chunks is None:
            return self._yield_chunks(self._chunk_slices)

        if n_chunks % self.num_chunks() != 0:
            raise RuntimeError(
                "The passed `n_chunks` has to be a multiple of `num_chunks`."
            )

        if n_chunks > self.num_rows():
            raise RuntimeError(
                "The passed `n_chunks` value is bigger than the amout of rows in the frame."
            )

        extra_chunks = n_chunks - self.num_chunks()
        # `._chunk_slices` is a cached property, we don't want to modify the property's
        # array inplace, so doing a copy here
        subdivided_slices = self._chunk_slices.copy()

        # The subdividing behavior is a bit different from "subdividing each chunk",
        # instead it subdivides the biggest chunks first, so overall chunking be as
        # equal as possible
        for _ in range(extra_chunks):
            # 1. Find the biggest chunk
            # 2. Split it in the middle
            biggest_chunk_idx = np.argmax(np.diff(subdivided_slices))
            new_chunk_offset = (
                subdivided_slices[biggest_chunk_idx + 1]
                - subdivided_slices[biggest_chunk_idx]
            ) // 2
            ErrorMessage.catch_bugs_and_request_email(
                failure_condition=new_chunk_offset == 0,
                extra_log="No more chunks to subdivide",
            )
            subdivided_slices = np.insert(
                subdivided_slices,
                biggest_chunk_idx + 1,
                subdivided_slices[biggest_chunk_idx] + new_chunk_offset,
            )

        return self._yield_chunks(subdivided_slices)

    def _yield_chunks(self, chunk_slices) -> "OmnisciProtocolDataframe":
        """
        Yield DataFrame chunks according to the passed offsets.

        Parameters
        ----------
        chunk_slices : list
            Chunking offsets.

        Yields
        ------
        OmnisciProtocolDataframe
        """
        for i in range(len(chunk_slices) - 1):
            yield OmnisciProtocolDataframe(
                df=self._df.mask(
                    row_positions=range(chunk_slices[i], chunk_slices[i + 1])
                ),
                nan_as_null=self._nan_as_null,
                allow_copy=self._allow_copy,
            )
