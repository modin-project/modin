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

"""The module houses OmnisciOnNative implementation of the Column class of DataFrame exchange protocol."""

import pyarrow as pa
import pandas
import numpy as np
from typing import Any, Optional, Tuple, Dict, Iterable
from math import ceil

from modin.core.dataframe.base.exchange.dataframe_protocol.utils import (
    DTypeKind,
    ColumnNullType,
    ArrowCTypes,
    Endianness,
    pandas_dtype_to_arrow_c,
    raise_copy_alert,
)
from modin.core.dataframe.base.exchange.dataframe_protocol.dataframe import (
    ProtocolColumn,
)
from modin.utils import _inherit_docstrings
from .buffer import OmnisciProtocolBuffer
from .utils import arrow_dtype_to_arrow_c, arrow_types_map


@_inherit_docstrings(ProtocolColumn)
class OmnisciProtocolColumn(ProtocolColumn):
    """
    Wrapper of ``OmnisciProtocolDataframe`` holding a single column.

    The Column object wraps a ``ProtocolDataframe`` to ease referencing original
    Modin DataFrame with no materialization of PyArrow table where possible.
    ``ProtocolDataframe`` also already implements methods like chunking and ``allow_copy``
    checks, so we can just forward calls for the methods to ``ProtocolDataFrame`` without
    reimplementing them.

    Parameters
    ----------
    column : OmnisciProtocolDataframe
        DataFrame protocol object holding a PyArrow table with a single column.

    Notes
    -----
    The object could be modified inplace due to either casting PyArrow buffers to a new dtype
    or combining physical chunks into a single congingous buffer:
    ``_propagate_dtype``, ``_cast_at``, ``_combine_chunks`` - the methods replace the wrapped
    ``OmnisciProtocolDataframe`` object with the new one holding the modified PyArrow table.
    """

    def __init__(self, column: "OmnisciProtocolDataframe") -> None:
        self._col = column

    @property
    def size(self) -> int:
        return self._col.num_rows()

    @property
    def offset(self) -> int:
        # The offset may change if it would require to cast buffers as the casted ones
        # no longer depend on their parent tables. So materializing buffers
        # before returning the offset
        self._materialize_actual_buffers()
        return self._pyarrow_table.column(0).chunks[0].offset

    @property
    def dtype(self) -> Tuple[DTypeKind, int, str, str]:
        dtype = self._pandas_dtype

        if pandas.api.types.is_bool_dtype(dtype):
            return (DTypeKind.BOOL, 1, ArrowCTypes.BOOL, Endianness.NATIVE)
        elif pandas.api.types.is_datetime64_dtype(
            dtype
        ) or pandas.api.types.is_categorical_dtype(dtype):
            # We can't fully describe an actual underlying type's metadata from pandas dtype,
            # use a `._arrow_dtype` for missing parts of information like datetime resulution,
            # dictionary metadata, etc?...
            return self._dtype_from_pyarrow(self._arrow_dtype)
        elif pandas.api.types.is_string_dtype(dtype):
            return (
                DTypeKind.STRING,
                8,
                pandas_dtype_to_arrow_c(dtype),
                Endianness.NATIVE,
            )
        else:
            return self._dtype_from_primitive_numpy(dtype)

    def _dtype_from_pyarrow(self, dtype):
        """
        Build protocol dtype from PyArrow type.

        Parameters
        ----------
        dtype : pyarrow.DataType
            Data type to convert from.

        Returns
        -------
        tuple(DTypeKind, bitwidth: int, format_str: str, edianess: str)
        """
        kind = None
        if (
            pa.types.is_timestamp(dtype)
            or pa.types.is_date(dtype)
            or pa.types.is_time(dtype)
        ):
            kind = DTypeKind.DATETIME
            bit_width = dtype.bit_width
        elif pa.types.is_dictionary(dtype):
            kind = DTypeKind.CATEGORICAL
            bit_width = dtype.bit_width
        elif pa.types.is_string(dtype):
            kind = DTypeKind.STRING
            bit_width = 8
        elif pa.types.is_boolean(dtype):
            kind = DTypeKind.BOOL
            bit_width = dtype.bit_width

        if kind is not None:
            return (kind, bit_width, arrow_dtype_to_arrow_c(dtype), Endianness.NATIVE)
        else:
            return self._dtype_from_primitive_numpy(np.dtype(dtype.to_pandas_dtype()))

    def _dtype_from_primitive_numpy(
        self, dtype: np.dtype
    ) -> Tuple[DTypeKind, int, str, str]:
        """
        Build protocol dtype from primitive pandas dtype.

        Parameters
        ----------
        dtype : np.dtype
            Data type to convert from.

        Returns
        -------
        tuple(DTypeKind, bitwidth: int, format_str: str, edianess: str)
        """
        np_kinds = {
            "i": DTypeKind.INT,
            "u": DTypeKind.UINT,
            "f": DTypeKind.FLOAT,
            "b": DTypeKind.BOOL,
        }
        kind = np_kinds.get(dtype.kind, None)
        if kind is None:
            raise NotImplementedError(
                f"Data type {dtype} not supported by exchange protocol"
            )
        return (
            kind,
            dtype.itemsize * 8,
            pandas_dtype_to_arrow_c(dtype),
            dtype.byteorder,
        )

    @property
    def describe_categorical(self) -> Dict[str, Any]:
        dtype = self._pandas_dtype

        if dtype != "category":
            raise RuntimeError(
                "`describe_categorical only works on a column with "
                + "categorical dtype!"
            )

        ordered = dtype.ordered

        # Category codes may change during materialization flow, so trigger
        # materialization before returning the codes
        self._materialize_actual_buffers()

        # Although we can retrieve codes from pandas dtype, they're unsynced with
        # the actual PyArrow data most of the time. So getting the mapping directly
        # from the materialized PyArrow table.
        col = self._pyarrow_table.column(0)
        if len(col.chunks) > 1:
            if not self._col._allow_copy:
                raise_copy_alert(
                    copy_reason="physical chunks combining due to contiguous buffer materialization"
                )
            col = col.combine_chunks()

        col = col.chunks[0]
        mapping = dict(enumerate(col.dictionary.tolist()))

        return {
            "is_ordered": ordered,
            "is_dictionary": True,
            "mapping": mapping,
        }

    @property
    def describe_null(self) -> Tuple[ColumnNullType, Any]:
        null_buffer = self._pyarrow_table.column(0).chunks[0].buffers()[0]
        if null_buffer is None:
            return (ColumnNullType.NON_NULLABLE, None)
        else:
            return (ColumnNullType.USE_BITMASK, 0)

    @property
    def null_count(self) -> int:
        return self._pyarrow_table.column(0).null_count

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._col.metadata

    @property
    def _pandas_dtype(self) -> np.dtype:
        """
        Get column's dtype representation in Modin DataFrame.

        Returns
        -------
        numpy.dtype
        """
        return self._col._df.dtypes.iloc[0]

    @property
    def _arrow_dtype(self) -> pa.DataType:
        """
        Get column's dtype representation in underlying PyArrow table.

        Returns
        -------
        pyarrow.DataType
        """
        return self._pyarrow_table.column(0).type

    @property
    def _pyarrow_table(self) -> pa.Table:
        """
        Get PyArrow table representing the column.

        Returns
        -------
        pyarrow.Table
        """
        return self._col._pyarrow_table

    def num_chunks(self) -> int:
        return self._col.num_chunks()

    def get_chunks(self, n_chunks: Optional[int] = None) -> Iterable["Column"]:
        for chunk in self._col.get_chunks(n_chunks):
            yield OmnisciProtocolColumn(chunk)

    def get_buffers(self) -> Dict[str, Any]:
        self._materialize_actual_buffers()
        at = self._pyarrow_table
        pyarrow_array = at.column(0).chunks[0]

        result = dict()
        result["data"] = self._get_data_buffer(pyarrow_array)
        result["validity"] = self._get_validity_buffer(pyarrow_array)
        result["offsets"] = self._get_offsets_buffer(pyarrow_array)

        return result

    def _materialize_actual_buffers(self):
        """
        Materialize PyArrow table's buffers that can be zero-copy returned to a consumer, if they aren't already materialized.

        Besides materializing PyArrow table itself (if there were some delayed computations)
        the function also may do the following if required:
        1. Propagate external dtypes to the PyArrow table. For example,
            if ``self.dtype`` is a string kind, but internal PyArrow dtype is a dictionary
            (if the table were just exported from OmniSci), then the dictionary will be casted
            to string dtype.
        2. Combine physical chunks of PyArrow table into a single contiguous buffer.
        """
        if self.num_chunks() != 1:
            if not self._col._allow_copy:
                raise_copy_alert(
                    copy_reason="physical chunks combining due to contiguous buffer materialization"
                )
            self._combine_chunks()

        external_dtype = self.dtype
        internal_dtype = self._dtype_from_pyarrow(self._arrow_dtype)

        if external_dtype[0] != internal_dtype[0]:
            self._propagate_dtype(external_dtype)

    def _get_buffer_size(self, bit_width: int, is_offset_buffer: bool = False) -> int:
        """
        Compute buffer's size in bytes for the current chunk.

        Parameters
        ----------
        bit_width : int
            Bit width of the underlying data type.
        is_offset_buffer : bool, default: False
            Whether the buffer describes offsets.

        Returns
        -------
        int
            Number of bytes to read from the start of the buffer + offset to retrieve the whole chunk.
        """
        # Offset buffer always has ``size + 1`` elements in it as it describes slices bounds
        elements_in_buffer = self.size + 1 if is_offset_buffer else self.size
        result = ceil((bit_width * elements_in_buffer) / 8)
        # For a bitmask, if the chunk started in the middle of the byte then we need to
        # read one extra byte from the buffer to retrieve the chunk's tail in the last byte. Example:
        # Bitmask of 3 bytes, the chunk offset is 3 elements and its size is 16
        # |* * * * * * * *|* * * * * * * *|* * * * * * * *|
        #      ^- the chunk starts here      ^- the chunk ends here
        # Although ``ceil(bit_width * elements_in_buffer / 8)`` gives us '2 bytes',
        # the chunk is located in 3 bytes, that's why we assume the chunk's buffer size
        # to be 'result += 1' in this case:
        if bit_width == 1 and self.offset % 8 + self.size > result * 8:
            result += 1
        return result

    def _get_data_buffer(
        self, arr: pa.Array
    ) -> Tuple[OmnisciProtocolBuffer, Tuple[DTypeKind, int, str, str]]:
        """
        Get column's data buffer.

        Parameters
        ----------
        arr : pa.Array
            PyArrow array holding column's data.

        Returns
        -------
        tuple
            Tuple of ``OmnisciProtocolBuffer`` and protocol dtype representation of the buffer's underlying data.
        """
        if self.dtype[0] == DTypeKind.CATEGORICAL:
            # For dictionary data the buffer has to return categories codes
            arr = arr.indices

        arrow_type = self._dtype_from_pyarrow(arr.type)
        buff_size = (
            self._get_buffer_size(bit_width=arrow_type[1])
            if self.dtype[0] != DTypeKind.STRING
            # We don't chunk string buffers as it would require modifying offset values,
            # so just return the whole data buffer for every chunk.
            else None
        )

        return (
            # According to the Arrow's memory layout, the data buffer is always present
            # at the last position of `.buffers()`:
            # https://arrow.apache.org/docs/format/Columnar.html#buffer-listing-for-each-layout
            OmnisciProtocolBuffer(arr.buffers()[-1], buff_size),
            arrow_type,
        )

    def _get_validity_buffer(
        self, arr: pa.Array
    ) -> Optional[Tuple[OmnisciProtocolBuffer, Tuple[DTypeKind, int, str, str]]]:
        """
        Get column's validity buffer.

        Parameters
        ----------
        arr : pa.Array
            PyArrow array holding column's data.

        Returns
        -------
        tuple or None
            Tuple of ``OmnisciProtocolBuffer`` and protocol dtype representation of the buffer's underlying data.
            None if column is non-nullable (``self.describe_null == ColumnNullType.NON_NULLABLE``).
        """
        # According to the Arrow's memory layout, the validity buffer is always present at zero position:
        # https://arrow.apache.org/docs/format/Columnar.html#buffer-listing-for-each-layout
        validity_buffer = arr.buffers()[0]
        if validity_buffer is None:
            return None

        # If exist, validity buffer is always a bit-mask.
        data_size = self._get_buffer_size(bit_width=1)
        return (
            OmnisciProtocolBuffer(validity_buffer, data_size),
            (DTypeKind.BOOL, 1, ArrowCTypes.BOOL, Endianness.NATIVE),
        )

    def _get_offsets_buffer(
        self, arr: pa.Array
    ) -> Optional[Tuple[OmnisciProtocolBuffer, Tuple[DTypeKind, int, str, str]]]:
        """
        Get column's offsets buffer.

        Parameters
        ----------
        arr : pa.Array
            PyArrow array holding column's data.

        Returns
        -------
        tuple or None
            Tuple of ``OmnisciProtocolBuffer`` and protocol dtype representation of the buffer's underlying data.
            None if the column's dtype is fixed-size.
        """
        buffs = arr.buffers()
        # According to the Arrow's memory layout, the offsets buffer is always at the second position
        # of `.buffers()` if present. Considering the support of only Primitive, Variable-length binary,
        # and Dict-encoded types from the layout table, we can assume that there's no offsets buffer
        # if there are fewer than 3 buffers available.
        # https://arrow.apache.org/docs/format/Columnar.html#buffer-listing-for-each-layout
        if len(buffs) < 3:
            return None

        offset_buff = buffs[1]
        # According to Arrow's data layout, the offset buffer type is "int32"
        dtype = self._dtype_from_primitive_numpy(np.dtype("int32"))
        return (
            OmnisciProtocolBuffer(
                offset_buff,
                self._get_buffer_size(bit_width=dtype[1], is_offset_buffer=True),
            ),
            dtype,
        )

    def _propagate_dtype(self, dtype: Tuple[DTypeKind, int, str, str]):
        """
        Propagate `dtype` to the underlying PyArrow table.

        Modifies the column object inplace by replacing underlying PyArrow table with
        the casted one.

        Parameters
        ----------
        dtype : tuple
            Data type conforming protocol dtypes format to cast underlying PyArrow table.
        """
        if not self._col._allow_copy:
            raise_copy_alert(
                copy_reason="casting to align pandas and PyArrow data types"
            )

        kind, bit_width, format_str, _ = dtype
        arrow_type = None

        if kind in arrow_types_map:
            arrow_type = arrow_types_map[kind].get(bit_width, None)
        elif kind == DTypeKind.DATETIME:
            arrow_type = pa.timestamp("ns")
        elif kind == DTypeKind.CATEGORICAL:
            index_type = arrow_types_map[DTypeKind.INT].get(bit_width, None)
            if index_type is not None:
                arrow_type = pa.dictionary(
                    index_type=index_type,
                    # There is no way to deduce an actual value type, so casting to a string
                    # as it's the most common one
                    value_type=pa.string(),
                )

        if arrow_type is None:
            raise NotImplementedError(f"Propagation for type {dtype} is not supported.")

        at = self._pyarrow_table
        schema_to_cast = at.schema
        field = at.schema[0]

        schema_to_cast = schema_to_cast.set(
            0, pa.field(field.name, arrow_type, field.nullable)
        )

        # TODO: currently, each column chunk casts its buffers independently which results
        # in an `N_CHUNKS - 1` amount of redundant casts. We can make the PyArrow table
        # being shared across all the chunks, so the cast being triggered in a single chunk
        # propagate to all of them.
        self._cast_at(schema_to_cast)

    def _cast_at(self, new_schema: pa.Schema):
        """
        Cast underlying PyArrow table with the passed schema.

        Parameters
        ----------
        new_schema : pyarrow.Schema
            New schema to cast the table.

        Notes
        -----
        This method modifies the column inplace by replacing the wrapped ``OmnisciProtocolDataframe``
        with the new one holding the casted PyArrow table.
        """
        casted_at = self._pyarrow_table.cast(new_schema)
        self._col = type(self._col)(
            self._col._df.from_arrow(casted_at),
            self._col._nan_as_null,
            self._col._allow_copy,
        )

    def _combine_chunks(self):
        """
        Combine physical chunks of underlying PyArrow table.

        Notes
        -----
        This method modifies the column inplace by replacing the wrapped ``OmnisciProtocolDataframe``
        with the new one holding PyArrow table with the column's data placed in a single contingous buffer.
        """
        contiguous_at = self._pyarrow_table.combine_chunks()
        self._col = type(self._col)(
            self._col._df.from_arrow(contiguous_at),
            self._col._nan_as_null,
            self._col._allow_copy,
        )
