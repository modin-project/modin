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

import pyarrow as pa
import pandas
import numpy as np

from typing import Any, Optional, Tuple, Dict, Iterable
from math import ceil
from modin.core.dataframe.base.exchange.dataframe_protocol.utils import (
    DTypeKind,
    ColumnNullType,
    pandas_dtype_to_arrow_c,
)
from modin.core.dataframe.base.exchange.dataframe_protocol.dataframe import (
    ProtocolColumn,
)
from .buffer import OmnisciProtocolBuffer
from .utils import arrow_dtype_to_arrow_c


class OmnisciProtocolColumn(ProtocolColumn):
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
    column : DataFrame
        A ``DataFrame`` object.
    allow_copy : bool, default: True
        A keyword that defines whether or not the library is allowed
        to make a copy of the data. For example, copying data would be necessary
        if a library supports strided buffers, given that this protocol
        specifies contiguous buffers. Currently, if the flag is set to ``False``
        and a copy is needed, a ``RuntimeError`` will be raised.
    offset : int, default: 0
        The offset of the first element

    Notes
    -----
    This Column object can only be produced by ``__dataframe__``,
    so doesn't need its own version or ``__column__`` protocol.
    """

    def __init__(self, column: "DataFrame") -> None:
        """
        Note: doesn't deal with extension arrays yet, just assume a regular
        Series/ndarray for now.
        """
        self._col = column

    @property
    def size(self) -> int:
        """
        Size of the column, in elements.

        Corresponds to DataFrame.num_rows() if column is a single chunk;
        equal to size of this current chunk otherwise.

        Returns
        -------
        int
            Size of the column, in elements.
        """
        return self._col.num_rows()

    @property
    def offset(self) -> int:
        """
        Get the offset of first element.

        May be > 0 if using chunks; for example for a column
        with N chunks of equal size M (only the last chunk may be shorter),
        ``offset = n * M``, ``n = 0 .. N-1``.

        Returns
        -------
        int
            The offset of first element.
        """
        # The offset may change if it would require to cast buffers as the casted ones
        # no longer depend on their parent tables. So materializing casted buffers
        # before accessing the offset
        self._materialize_actual_buffers()
        return self._pyarrow_table.column(0).chunks[0].offset

    @property
    def dtype(self) -> Tuple[DTypeKind, int, str, str]:
        """
        Dtype description as a tuple ``(kind, bit-width, format string, endianness)``, where

        * Kind : DTypeKind
        * Bit-width : the number of bits as an integer
        * Format string : data type description format string in Apache Arrow C
                        Data Interface format.
        * Endianness : current only native endianness (``=``) is supported

        Kind :

            - INT = 0 # infer
            - UINT = 1 # infer
            - FLOAT = 2 # infer
            - BOOL = 20 # infer
            - STRING = 21 # infer?
            - DATETIME = 22 # have to materialize to deduce resolution (always should be ns???)
            - CATEGORICAL = 23 # not implemented error

        Notes
        -----
        - Kind specifiers are aligned with DLPack where possible
          (hence the jump to 20, leave enough room for future extension).
        - Masks must be specified as boolean with either bit width 1 (for bit masks)
          or 8 (for byte masks).
        - Dtype width in bits was preferred over bytes
        - Endianness isn't too useful, but included now in case in the future
          we need to support non-native endianness
        - Went with Apache Arrow format strings over NumPy format strings
          because they're more complete from a dataframe perspective
        - Format strings are mostly useful for datetime specification, and for categoricals.
        - For categoricals, the format string describes the type of the categorical
          in the data buffer. In case of a separate encoding of the categorical
          (e.g. an integer to string mapping), this can be derived from ``self.describe_categorical``.
        - Data types not included: complex, Arrow-style null, binary, decimal,
          and nested (list, struct, map, union) dtypes.
        """
        dtype = self._pandas_dtype

        if pandas.api.types.is_datetime64_dtype(dtype):
            return self._dtype_from_pyarrow(self._arrow_dtype)
        elif pandas.api.types.is_categorical_dtype(dtype):
            return (
                DTypeKind.CATEGORICAL,
                32,
                pandas_dtype_to_arrow_c(np.dtype("int32")),
                "=",
            )
        elif pandas.api.types.is_string_dtype(dtype):
            return (DTypeKind.STRING, 8, pandas_dtype_to_arrow_c(dtype), "=")
        else:
            return self._dtype_from_primitive_pandas(dtype)

    def _dtype_from_pyarrow(self, dtype):
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

        if kind is not None:
            return (kind, bit_width, arrow_dtype_to_arrow_c(dtype), "=")
        else:
            return self._dtype_from_primitive_pandas(np.dtype(dtype.to_pandas_dtype()))

    def _dtype_from_primitive_pandas(self, dtype) -> Tuple[DTypeKind, int, str, str]:
        """
        See `self.dtype` for details.
        """
        _np_kinds = {
            "i": DTypeKind.INT,
            "u": DTypeKind.UINT,
            "f": DTypeKind.FLOAT,
            "b": DTypeKind.BOOL,
        }
        kind = _np_kinds.get(dtype.kind, None)
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
        """
        If the dtype is categorical, there are two options:
        - There are only values in the data buffer.
        - There is a separate dictionary-style encoding for categorical values.

        TBD: are there any other in-memory representations that are needed?

        Returns
        -------
        dict
            Content of returned dict:
            - "is_ordered" : bool, whether the ordering of dictionary indices is
                             semantically meaningful.
            - "is_dictionary" : bool, whether a dictionary-style mapping of
                                categorical values to other objects exists
            - "mapping" : dict, Python-level only (e.g. ``{int: str}``).
                          None if not a dictionary-style categorical.

        Raises
        ------
        ``RuntimeError`` if the dtype is not categorical.
        """
        dtype = self._pandas_dtype

        if dtype != "category":
            raise RuntimeError(
                f"Column 'dtype' has to be categorical to be able to dectribe categiries, met: {dtype}"
            )

        ordered = dtype.ordered
        mapping = {index: value for index, value in enumerate(dtype.categories)}

        return {
            "is_ordered": ordered,
            "is_dictionary": True,
            "mapping": mapping,
        }

    @property
    def describe_null(self) -> Tuple[ColumnNullType, Any]:
        """
        Return the missing value (or "null") representation the column dtype uses.

        Return as a tuple ``(kind, value)``.

        * Kind:
            - 0 : non-nullable
            - 1 : NaN/NaT
            - 2 : sentinel value
            - 3 : bit mask
            - 4 : byte mask
        * Value : if kind is "sentinel value", the actual value. If kind is a bit
          mask or a byte mask, the value (0 or 1) indicating a missing value. None
          otherwise.

        Returns
        -------
        tuple
            ``(kind, value)``.
        """
        null_buffer = self._pyarrow_table.column(0).chunks[0].buffers()[0]
        if null_buffer is None:
            return (ColumnNullType.NON_NULLABLE, None)
        else:
            return (ColumnNullType.USE_BITMASK, 0)

    @property
    def null_count(self) -> int:
        """
        Number of null elements, if known.
        Note: Arrow uses -1 to indicate "unknown", but None seems cleaner.
        """
        ncount = self._pyarrow_table.column(0).null_count
        return ncount if ncount >= 0 else None

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        The metadata for the column. See `DataFrame.metadata` for more details.
        """
        return {}

    @property
    def _pandas_dtype(self):
        return self._col._df.dtypes.iloc[0]

    @property
    def _arrow_dtype(self):
        return self._pyarrow_table.column(0).type

    @property
    def _pyarrow_table(self):
        return self._col._pyarrow_table

    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.

        Returns
        -------
        int
           The number of chunks the column consists of.
        """
        return self._col.num_chunks()

    def get_chunks(self, n_chunks: Optional[int] = None) -> Iterable["Column"]:
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
        for chunk in self._col.get_chunks(n_chunks):
            yield OmnisciProtocolColumn(chunk)

    def get_buffers(self) -> Dict[str, Any]:
        """
        Return a dictionary containing the underlying buffers.

        Returns
        -------
        dict
            - "data": a two-element tuple whose first element is a buffer
              containing the data and whose second element is the data buffer's associated dtype.
            - "validity": a two-element tuple whose first element is a buffer
              containing mask values indicating missing data and
              whose second element is the mask value buffer's
              associated dtype. None if the null representation is not a bit or byte mask.
            - "offsets": a two-element tuple whose first element is a buffer
              containing the offset values for variable-size binary data
              (e.g., variable-length strings) and whose second element is the offsets
              buffer's associated dtype. None if the data buffer does not have
              an associated offsets buffer.
        """
        if self.num_chunks() != 1:
            raise NotImplementedError()

        self._materialize_actual_buffers()
        at = self._pyarrow_table
        pyarrow_array = at.column(0).chunks[0]

        result = dict()
        result["data"] = self._get_data_buffer(pyarrow_array)
        result["validity"] = self._get_validity_buffer(pyarrow_array)
        result["offsets"] = self._get_offsets_buffer(pyarrow_array)

        return result

    def _materialize_actual_buffers(self):
        external_dtype = self.dtype
        internal_dtype = self._dtype_from_pyarrow(self._arrow_dtype)

        if external_dtype[0] != internal_dtype[0]:
            self._propagate_dtype(external_dtype)

    def _get_buffer_size(self, bit_width, is_offset_buffer=False):
        elements_in_buffer = self.size + 1 if is_offset_buffer else self.size
        return ceil((bit_width * elements_in_buffer) / 8)

    def _get_data_buffer(self, arr):
        if self.dtype[0] == DTypeKind.CATEGORICAL:
            arr = arr.indices

        arrow_type = self._dtype_from_pyarrow(arr.type)

        buff_size = (
            self._get_buffer_size(bit_width=arrow_type[1])
            if self.dtype[0] != DTypeKind.STRING
            else None
        )

        return (
            OmnisciProtocolBuffer(arr.buffers()[-1], buff_size),
            arrow_type,
        )

    def _get_validity_buffer(self, arr):
        validity_buffer = arr.buffers()[0]
        if validity_buffer is None:
            return validity_buffer

        data_size = self._get_buffer_size(bit_width=1)
        if self.offset % 8 + self.size > data_size * 8:
            data_size += 1

        return (
            OmnisciProtocolBuffer(validity_buffer, data_size),
            self._dtype_from_primitive_pandas(np.dtype("uint8")),
        )

    def _get_offsets_buffer(self, arr):
        buffs = arr.buffers()
        if len(buffs) < 3:
            return None

        offset_buff = buffs[1]

        dtype = self._dtype_from_primitive_pandas(np.dtype("int32"))
        return (
            OmnisciProtocolBuffer(
                offset_buff,
                self._get_buffer_size(bit_width=dtype[1], is_offset_buffer=True),
            ),
            dtype,
        )

    def _propagate_dtype(self, dtype):
        if not self._col._allow_copy:
            raise RuntimeError("Copy required with 'allow_copy=False' flag")

        arrow_types_map = {
            DTypeKind.BOOL: {8: pa.bool_()},
            DTypeKind.INT: {
                8: pa.int8(),
                16: pa.int16(),
                32: pa.int32(),
                64: pa.int64(),
            },
            DTypeKind.UINT: {
                8: pa.uint8(),
                16: pa.uint16(),
                32: pa.uint32(),
                64: pa.uint64(),
            },
            DTypeKind.FLOAT: {16: pa.float16(), 32: pa.float32(), 64: pa.float64()},
            DTypeKind.STRING: {8: pa.string()},
        }
        kind, bit_width, format_str, _ = dtype
        arrow_type = None

        if kind in arrow_types_map:
            arrow_type = arrow_types_map[kind].get(bit_width, None)
        elif kind == DTypeKind.DATETIME:
            arrow_type = pa.timestamp("ns")
        elif kind == DTypeKind.CATEGORICAL:
            arrow_type = pa.dictionary(
                index_type=arrow_types_map[DTypeKind.INT][bit_width],
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
        # in an `NCHUNKS - 1` amount of redundant casts. We can make the pyarrow table
        # being shared across all the chunks, so the cast being triggered in a single chunk
        # propagate to all of them.
        self._col._replace_at(at.cast(schema_to_cast))
