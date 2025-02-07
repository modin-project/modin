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

from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas

from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
    CategoricalDescription,
    ProtocolColumn,
)
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
    ColumnNullType,
    DTypeKind,
    pandas_dtype_to_arrow_c,
)
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.utils import _inherit_docstrings

from .buffer import PandasProtocolBuffer
from .exception import NoOffsetsBuffer, NoValidityBuffer

_NO_VALIDITY_BUFFER = {
    ColumnNullType.NON_NULLABLE: "This column is non-nullable so does not have a mask",
    ColumnNullType.USE_NAN: "This column uses NaN as null so does not have a separate mask",
    ColumnNullType.USE_SENTINEL: "This column uses a sentinel value so does not have a mask",
}


@_inherit_docstrings(ProtocolColumn)
class PandasProtocolColumn(ProtocolColumn):
    """
    A column object, with only the methods and properties required by the interchange protocol defined.

    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length strings).

    TBD: Arrow has a separate "null" dtype, and has no separate mask concept.
         Instead, it seems to use "children" for both columns with a bit mask,
         and for nested dtypes. Unclear whether this is elegant or confusing.
         This design requires checking the null representation explicitly.
         The Arrow design requires checking:
         1. the ARROW_FLAG_NULLABLE (for sentinel values)
         2. if a column has two children, combined with one of those children
            having a null dtype.
         Making the mask concept explicit seems useful. One null dtype would
         not be enough to cover both bit and byte masks, so that would mean
         even more checking if we did it the Arrow way.
    TBD: there's also the "chunk" concept here, which is implicit in Arrow as
         multiple buffers per array (= column here). Semantically it may make
         sense to have both: chunks were meant for example for lazy evaluation
         of data which doesn't fit in memory, while multiple buffers per column
         could also come from doing a selection operation on a single
         contiguous buffer.
         Given these concepts, one would expect chunks to be all of the same
         size (say a 10,000 row dataframe could have 10 chunks of 1,000 rows),
         while multiple buffers could have data-dependent lengths. Not an issue
         in pandas if one column is backed by a single NumPy array, but in
         Arrow it seems possible.
         Are multiple chunks *and* multiple buffers per column necessary for
         the purposes of this interchange protocol, or must producers either
         reuse the chunk concept for this or copy the data?

    Parameters
    ----------
    column : PandasDataframe
        A ``PandasDataframe`` object.
    allow_copy : bool, default: True
        A keyword that defines whether or not the library is allowed
        to make a copy of the data. For example, copying data would be necessary
        if a library supports strided buffers, given that this protocol
        specifies contiguous buffers. Currently, if the flag is set to ``False``
        and a copy is needed, a ``RuntimeError`` will be raised.

    Notes
    -----
    This Column object can only be produced by ``__dataframe__``,
    so doesn't need its own version or ``__column__`` protocol.
    """

    def __init__(self, column: PandasDataframe, allow_copy: bool = True) -> None:
        if not isinstance(column, PandasDataframe):
            raise NotImplementedError(f"Columns of type {type(column)} not handled yet")

        self._col = column
        self._allow_copy = allow_copy

    def size(self) -> int:
        return len(self._col.index)

    @property
    def offset(self) -> int:
        return 0

    @cached_property
    def dtype(self) -> Tuple[DTypeKind, int, str, str]:
        dtype = self._col.dtypes.iloc[0]

        if isinstance(dtype, pandas.CategoricalDtype):
            pandas_series = self._col.to_pandas().squeeze(axis=1)
            codes = pandas_series.values.codes
            (
                _,
                bitwidth,
                c_arrow_dtype_f_str,
                _,
            ) = self._dtype_from_primitive_pandas_dtype(codes.dtype)
            dtype_cache = (
                DTypeKind.CATEGORICAL,
                bitwidth,
                c_arrow_dtype_f_str,
                "=",
            )
        elif pandas.api.types.is_string_dtype(dtype):
            dtype_cache = (DTypeKind.STRING, 8, pandas_dtype_to_arrow_c(dtype), "=")
        else:
            dtype_cache = self._dtype_from_primitive_pandas_dtype(dtype)

        return dtype_cache

    def _dtype_from_primitive_pandas_dtype(
        self, dtype
    ) -> Tuple[DTypeKind, int, str, str]:
        """
        Deduce dtype specific for the protocol from pandas dtype.

        See `self.dtype` for details.

        Parameters
        ----------
        dtype : any
            A pandas dtype.

        Returns
        -------
        tuple
        """
        _np_kinds = {
            "i": DTypeKind.INT,
            "u": DTypeKind.UINT,
            "f": DTypeKind.FLOAT,
            "b": DTypeKind.BOOL,
            "M": DTypeKind.DATETIME,
        }
        kind = _np_kinds.get(dtype.kind, None)
        if kind is None:
            raise NotImplementedError(
                f"Data type {dtype} not supported by the dataframe exchange protocol"
            )
        return (
            kind,
            dtype.itemsize * 8,
            pandas_dtype_to_arrow_c(dtype),
            dtype.byteorder,
        )

    @property
    def describe_categorical(self) -> CategoricalDescription:
        if self.dtype[0] != DTypeKind.CATEGORICAL:
            raise TypeError(
                "`describe_categorical only works on a column with "
                + "categorical dtype!"
            )

        pandas_series = self._col.to_pandas().squeeze(axis=1)
        cat_frame = type(self._col).from_pandas(
            pandas.DataFrame({"cat": pandas_series.cat.categories})
        )
        return {
            "is_ordered": pandas_series.cat.ordered,
            "is_dictionary": True,
            "categories": PandasProtocolColumn(cat_frame, self._allow_copy),
        }

    @property
    def describe_null(self) -> Tuple[int, Any]:
        nulls = {
            DTypeKind.FLOAT: (ColumnNullType.USE_NAN, None),
            DTypeKind.DATETIME: (ColumnNullType.USE_NAN, None),
            DTypeKind.INT: (ColumnNullType.NON_NULLABLE, None),
            DTypeKind.UINT: (ColumnNullType.NON_NULLABLE, None),
            DTypeKind.BOOL: (ColumnNullType.NON_NULLABLE, None),
            # Null values for categoricals are stored as `-1` sentinel values
            # in the category date (e.g., `col.values.codes` is int8 np.ndarray)
            DTypeKind.CATEGORICAL: (ColumnNullType.USE_SENTINEL, -1),
            # follow Arrow in using 1 as valid value and 0 for missing/null value
            DTypeKind.STRING: (ColumnNullType.USE_BYTEMASK, 0),
        }

        kind = self.dtype[0]
        try:
            null, value = nulls[kind]
        except KeyError:
            raise NotImplementedError(f"Data type {kind} not yet supported")

        return null, value

    @cached_property
    def null_count(self) -> int:

        def map_func(df):
            return df.isna()

        def reduce_func(df):
            return pandas.DataFrame(df.sum())

        intermediate_df = self._col.tree_reduce(0, map_func, reduce_func)
        # Set ``pandas.RangeIndex(1)`` to index and column labels because
        # 1) We internally use `MODIN_UNNAMED_SERIES_LABEL` for labels of a reduced axis
        # 2) The return value of `reduce_func` is a pandas DataFrame with
        # index and column labels set to ``pandas.RangeIndex(1)``
        # 3) We further use `to_pandas().squeeze()` to get an integer value of the null count.
        # Otherwise, we get mismatching internal and external indices for both axes
        intermediate_df.index = pandas.RangeIndex(1)
        intermediate_df.columns = pandas.RangeIndex(1)
        return intermediate_df.to_pandas().squeeze(axis=1).item()

    @property
    def metadata(self) -> Dict[str, Any]:
        return {"modin.index": self._col.index}

    def num_chunks(self) -> int:
        return self._col._partitions.shape[0]

    def get_chunks(
        self, n_chunks: Optional[int] = None
    ) -> Iterable["PandasProtocolColumn"]:
        cur_n_chunks = self.num_chunks()
        n_rows = self.size()
        if n_chunks is None or n_chunks == cur_n_chunks:
            cum_row_lengths = np.cumsum([0] + self._col.row_lengths)
            for i in range(len(cum_row_lengths) - 1):
                yield PandasProtocolColumn(
                    self._col.take_2d_labels_or_positional(
                        row_positions=range(cum_row_lengths[i], cum_row_lengths[i + 1]),
                        col_positions=None,
                    ),
                    allow_copy=self._col._allow_copy,
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

        new_partitions = self._col._partition_mgr_cls.map_axis_partitions(
            0,
            self._col._partitions,
            lambda df: df,
            keep_partitioning=False,
            lengths=new_lengths,
        )
        new_df = self._col.__constructor__(
            new_partitions,
            self._col.index,
            self._col.columns,
            new_lengths,
            self._col.column_widths,
        )
        cum_row_lengths = np.cumsum([0] + new_df.row_lengths)
        for i in range(len(cum_row_lengths) - 1):
            yield PandasProtocolColumn(
                new_df.take_2d_labels_or_positional(
                    row_positions=range(cum_row_lengths[i], cum_row_lengths[i + 1]),
                    col_positions=None,
                ),
                allow_copy=self._allow_copy,
            )

    def get_buffers(self) -> Dict[str, Any]:
        buffers = {}
        buffers["data"] = self._get_data_buffer()
        try:
            buffers["validity"] = self._get_validity_buffer()
        except NoValidityBuffer:
            buffers["validity"] = None

        try:
            buffers["offsets"] = self._get_offsets_buffer()
        except NoOffsetsBuffer:
            buffers["offsets"] = None

        return buffers

    _data_buffer_cache = None

    def _get_data_buffer(
        self,
    ) -> Tuple[PandasProtocolBuffer, Any]:  # Any is for self.dtype tuple
        """
        Return the buffer containing the data and the buffer's associated dtype.

        Returns
        -------
        tuple
            The data buffer.
        """
        if self._data_buffer_cache is not None:
            return self._data_buffer_cache

        dtype = self.dtype
        if dtype[0] in (
            DTypeKind.INT,
            DTypeKind.UINT,
            DTypeKind.FLOAT,
            DTypeKind.BOOL,
            DTypeKind.DATETIME,
        ):
            buffer = PandasProtocolBuffer(
                self._col.to_numpy().flatten(), allow_copy=self._allow_copy
            )
        elif dtype[0] == DTypeKind.CATEGORICAL:
            pandas_series = self._col.to_pandas().squeeze(axis=1)
            codes = pandas_series.values.codes
            buffer = PandasProtocolBuffer(codes, allow_copy=self._allow_copy)
            dtype = self._dtype_from_primitive_pandas_dtype(codes.dtype)
        elif dtype[0] == DTypeKind.STRING:
            # Marshal the strings from a NumPy object array into a byte array
            buf = self._col.to_numpy().flatten()
            b = bytearray()

            # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
            for i in range(buf.size):
                if type(buf[i]) is str:
                    b.extend(buf[i].encode(encoding="utf-8"))

            # Convert the byte array to a pandas "buffer" using a NumPy array as the backing store
            buffer = PandasProtocolBuffer(np.frombuffer(b, dtype="uint8"))

            # Define the dtype for the returned buffer
            dtype = (
                DTypeKind.STRING,
                8,
                "u",
                "=",
            )  # note: currently only support native endianness
        else:
            raise NotImplementedError(f"Data type {self._col.dtype[0]} not handled yet")

        self._data_buffer_cache = (buffer, dtype)
        return self._data_buffer_cache

    _validity_buffer_cache = None

    def _get_validity_buffer(self) -> Tuple[PandasProtocolBuffer, Any]:
        """
        Get the validity buffer.

        The buffer contains the mask values indicating
        missing data and the buffer's associated dtype.

        Returns
        -------
        tuple
            The validity buffer.

        Raises
        ------
        ``NoValidityBuffer`` if null representation is not a bit or byte mask.
        """
        if self._validity_buffer_cache is not None:
            return self._validity_buffer_cache

        null, invalid = self.describe_null

        if self.dtype[0] == DTypeKind.STRING:
            # For now, have the mask array be comprised of bytes, rather than a bit array
            buf = self._col.to_numpy().flatten()

            # Determine the encoding for valid values
            valid = invalid == 0
            invalid = not valid

            mask = np.empty(shape=(len(buf),), dtype=np.bool_)
            for i, obj in enumerate(buf):
                mask[i] = valid if isinstance(obj, str) else invalid

            # Convert the mask array to a Pandas "buffer" using a NumPy array as the backing store
            buffer = PandasProtocolBuffer(mask)

            # Define the dtype of the returned buffer
            dtype = (DTypeKind.BOOL, 8, "b", "=")

            self._validity_buffer_cache = (buffer, dtype)
            return self._validity_buffer_cache

        try:
            msg = _NO_VALIDITY_BUFFER[null]
        except KeyError:
            raise NotImplementedError("See self.describe_null")

        raise NoValidityBuffer(msg)

    _offsets_buffer_cache = None

    def _get_offsets_buffer(self) -> Tuple[PandasProtocolBuffer, Any]:
        """
        Get the offsets buffer.

        The buffer contains the offset values for variable-size binary data
        (e.g., variable-length strings) and the buffer's associated dtype.

        Returns
        -------
        tuple
            The offsets buffer.

        Raises
        ------
        ``NoOffsetsBuffer`` if the data buffer does not have an associated offsets buffer.
        """
        if self._offsets_buffer_cache is not None:
            return self._offsets_buffer_cache

        if self.dtype[0] == DTypeKind.STRING:
            # For each string, we need to manually determine the next offset
            values = self._col.to_numpy().flatten()
            ptr = 0
            offsets = [ptr] + [None] * len(values)
            for i, v in enumerate(values):
                # For missing values (in this case, `np.nan` values), we don't increment the pointer)
                if type(v) is str:
                    b = v.encode(encoding="utf-8")
                    ptr += len(b)

                offsets[i + 1] = ptr

            # Convert the list of offsets to a NumPy array of signed 64-bit integers (note: Arrow allows the offsets array to be either `int32` or `int64`; here, we default to the latter)
            buf = np.asarray(offsets, dtype="int64")

            # Convert the offsets to a Pandas "buffer" using the NumPy array as the backing store
            buffer = PandasProtocolBuffer(buf)

            # Assemble the buffer dtype info
            dtype = (
                DTypeKind.INT,
                64,
                "l",
                "=",
            )  # note: currently only support native endianness
        else:
            raise NoOffsetsBuffer(
                "This column has a fixed-length dtype so does not have an offsets buffer"
            )

        self._offsets_buffer_cache = (buffer, dtype)
        return self._offsets_buffer_cache
