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

Public API
----------
from_dataframe : construct a DataFrame from an input data frame which
                 implements the exchange protocol.
Notes
-----
- Interpreting a raw pointer (as in ``Buffer.ptr``) is annoying and unsafe to
  do in pure Python. It's more general but definitely less friendly than having
  ``to_arrow`` and ``to_numpy`` methods. So for the buffers which lack
  ``__dlpack__`` (e.g., because the column dtype isn't supported by DLPack),
  this is worth looking at again.
"""

import collections
import numpy as np

from typing import Optional, Iterable, Sequence
from modin.core.dataframe.base.dataframe.dataframe import ModinDataframe
from modin.core.dataframe.base.exchange.dataframe_protocol import ProtocolDataframe

from modin.experimental.core.execution.native.implementations.omnisci_on_native.df_algebra import (
    MaskNode,
    FrameNode,
    TransformNode,
    UnionNode,
)
from .column import OmnisciProtocolColumn


class OmnisciProtocolDataframe(ProtocolDataframe):
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
    df : ModinDataframe
        A ``ModinDataframe`` object.
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
        df: ModinDataframe,
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
    def metadata(self):
        # TODO: as the frame's index is stored as a separate column inside pyarrow table
        # we may want to return the column's name here instead of materialized index.
        # This will require the internal index column to be visible in the protocol's column
        # accessor methods.
        self._maybe_raise_if_materialize()
        return {"index": self._df.index}

    def num_columns(self) -> int:
        """
        Return the number of columns in the DataFrame.

        Returns
        -------
        int
            The number of columns in the DataFrame.
        """
        return len(self._df.columns)

    def num_rows(self) -> int:
        """
        Return the number of rows in the DataFrame, if available.

        Returns
        -------
        int
            The number of rows in the DataFrame.
        """
        if not self._allow_copy and not self._is_zero_copy_possible:
            raise RuntimeError("Copy required with 'allow_copy=False'")
        return len(self._df.index)

    def num_chunks(self) -> int:
        """
        Return the number of chunks the DataFrame consists of.

        Returns
        -------
        int
            The number of chunks the DataFrame consists of.
        """
        return len(self._chunk_slices) - 1

    __chunk_slices = None

    @property
    def _chunk_slices(self):
        """
        Compute chunk start-stop indices in the underlying pyarrow table.

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
    def _is_zero_copy_possible(self):
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
                self.__is_zero_copy_possible = True
            elif not self._df._can_execute_arrow():
                self.__is_zero_copy_possible = False
            else:
                self.__is_zero_copy_possible = self._is_zero_copy_op(self._df.op)
        return self.__is_zero_copy_possible

    @classmethod
    def _is_zero_copy_op(cls, op):
        """
        Check whether the passed node of the delayed computation tree could be executed zero-copy via pyarrow execution.

        Parameters
        ----------
        op : DFAlgNode

        Returns
        -------
        bool
        """
        is_zero_copy_op = False
        if isinstance(op, (FrameNode, TransformNode, UnionNode)):
            is_zero_copy_op = True
        elif isinstance(op, MaskNode) and (
            isinstance(op.row_positions, slice) or is_range_like(op.row_positions)
        ):
            is_zero_copy_op = True
        return is_zero_copy_op and all(
            cls._is_zero_copy_op(_op) for _op in getattr(op, "inputs", [])
        )

    @property
    def _pyarrow_table(self):
        """Get ``pyarrow.Table`` representing the dataframe."""
        self._maybe_raise_if_materialize()

        if not self._df._has_arrow_table():
            self._df._execute()

        at = self._df._partitions[0][0].arrow_table
        assert at is not None
        return at

    def _replace_at(self, at):
        self._df = self._df.from_arrow(at)

    def column_names(self) -> Iterable[str]:
        """
        Return an iterator yielding the column names.

        Yields
        ------
        str
            The name of the column(s).
        """
        for col in self._df.columns:
            yield col

    def get_column(self, i: int) -> OmnisciProtocolColumn:
        """
        Return the column at the indicated position.

        Returns
        -------
        Column
            The column at the indicated position.
        """
        return OmnisciProtocolColumn(
            OmnisciProtocolDataframe(
                self._df.mask(col_positions=[i]),
                allow_copy=self._allow_copy,
            ),
        )

    def get_column_by_name(self, name: str) -> OmnisciProtocolColumn:
        """
        Return the column whose name is the indicated name.

        Returns
        -------
        Column
            The column whose name is the indicated name.
        """
        return OmnisciProtocolColumn(
            OmnisciProtocolDataframe(
                self._df.mask(col_labels=[name]),
                allow_copy=self._allow_copy,
            ),
        )

    def get_columns(self) -> Iterable[OmnisciProtocolColumn]:
        """
        Return an iterator yielding the columns.

        Yields
        ------
        Column
            The ``Column`` object(s).
        """
        for name in self._df.columns:
            yield OmnisciProtocolColumn(
                OmnisciProtocolDataframe(
                    self._df.mask(col_labels=[name]),
                    allow_copy=self._allow_copy,
                ),
            )

    def select_columns(self, indices: Sequence[int]) -> "DataFrame":
        """
        Create a new DataFrame by selecting a subset of columns by index.

        Returns
        -------
        DataFrame
            A new DataFrame with selected a subset of columns by index.
        """
        if not isinstance(indices, collections.Sequence):
            raise ValueError("`indices` is not a sequence")

        return OmnisciProtocolDataframe(
            self._df.mask(col_positions=list(indices)),
            allow_copy=self._allow_copy,
        )

    def select_columns_by_name(self, names: Sequence[str]) -> "DataFrame":
        """
        Create a new DataFrame by selecting a subset of columns by name.

        Returns
        -------
        DataFrame
            A new DataFrame with selected a subset of columns by name.
        """
        if not isinstance(names, collections.Sequence):
            raise ValueError("`names` is not a sequence")

        return OmnisciProtocolDataframe(
            self._df.mask(col_labels=list(names)),
            allow_copy=self._allow_copy,
        )

    def get_chunks(self, n_chunks: Optional[int] = None) -> Iterable["DataFrame"]:
        """
        Return an iterator yielding the chunks.

        By default ``n_chunks=None``, yields the chunks that the data is stored as by the producer.
        If given, ``n_chunks`` must be a multiple of ``self.num_chunks()``,
        meaning the producer must subdivide each chunk before yielding it.

        Parameters
        ----------
        n_chunks : int, optional
            Number of chunks to yield.

        Yields
        ------
        DataFrame
            A ``DataFrame`` object(s).
        """
        if n_chunks is None:
            return self._yield_chunks(self._chunk_slices)

        if n_chunks % self.num_chunks() != 0:
            raise RuntimeError(
                "The passed `n_chunks` has to be a multiple of `num_chunks`."
            )

        extra_chunks = n_chunks - self.num_chunks()
        subdivided_slices = self._chunk_slices.copy()

        for _ in range(extra_chunks):
            # 1. Find the biggest chunk
            # 2. Split it in the middle
            biggest_chunk_idx = np.argmax(np.diff(subdivided_slices))
            new_chunk_offset = (
                subdivided_slices[biggest_chunk_idx + 1]
                - subdivided_slices[biggest_chunk_idx]
            ) // 2
            if new_chunk_offset == 0:
                raise RuntimeError(
                    "The passed `n_chunks` value is bigger than the amout of rows in the frame."
                )
            subdivided_slices = np.insert(
                subdivided_slices,
                biggest_chunk_idx + 1,
                subdivided_slices[biggest_chunk_idx] + new_chunk_offset,
            )

        return self._yield_chunks(subdivided_slices)

    def _yield_chunks(self, chunk_slices):
        """
        Yield dataframe chunks according to the passed chunking.

        Parameters
        ----------
        chunk_slices : list

        Yield
        -----
        DataFrame
        """
        for i in range(len(chunk_slices) - 1):
            yield OmnisciProtocolDataframe(
                df=self._df.mask(
                    row_positions=range(chunk_slices[i], chunk_slices[i + 1])
                ),
                allow_copy=self._allow_copy,
                nan_as_null=self._nan_as_null,
            )
