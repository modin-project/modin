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

from typing import Any, Optional, Tuple, Dict, Iterable
import numpy as np
import pandas

import modin.pandas as pd
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from .utils import DTypeKind
from .buffer import Buffer


class Column(object):
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
    offset : int, default: 0
        The offset of the first element.

    Notes
    -----
    This Column object can only be produced by ``__dataframe__``,
    so doesn't need its own version or ``__column__`` protocol.
    """

    def __init__(
        self, column: PandasDataframe, allow_copy: bool = True, offset: int = 0
    ) -> None:
        if not isinstance(column, PandasDataframe):
            raise NotImplementedError(
                "Columns of type {} not handled " "yet".format(type(column))
            )

        # Store the column as a private attribute
        self._col = column
        self._allow_copy = allow_copy
        self._offset = offset

    @property
    def size(self) -> int:
        """
        Size of the column, in elements.

        Corresponds to `DataFrame.num_rows()` if column is a single chunk;
        equal to size of this current chunk otherwise.

        Returns
        -------
        int
            Size of the column, in elements.
        """
        return len(self._col.index)

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
        return self._offset

    @property
    def dtype(self) -> Tuple[DTypeKind, int, str, str]:
        """
        Dtype description as a tuple ``(kind, bit-width, format string, endianness)``.

        * Kind : DTypeKind
        * Bit-width : the number of bits as an integer
        * Format string : data type description format string in Apache Arrow C
                        Data Interface format.
        * Endianness : current only native endianness (``=``) is supported

        Returns
        -------
        tuple
            ``(kind, bit-width, format string, endianness)``.

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
        dtype = self._col.dtypes[0]

        # For now, assume that, if the column dtype is 'O' (i.e., `object`), then we have an array of strings
        if not isinstance(dtype, pd.CategoricalDtype) and dtype.kind == "O":
            return (DTypeKind.STRING, 8, "u", "=")

        return self._dtype_from_pandas_dtype(dtype)

    def _dtype_from_pandas_dtype(self, dtype) -> Tuple[DTypeKind, int, str, str]:
        """
        Deduce dtype from pandas dtype.

        See `self.dtype` for details.

        Parameters
        ----------
        dtype : any
            A pandas dtype.

        Returns
        -------
        tuple
        """
        # Note: 'c' (complex) not handled yet (not in array spec v1).
        #       'b', 'B' (bytes), 'S', 'a', (old-style string) 'V' (void) not handled
        #       datetime and timedelta both map to datetime (is timedelta handled?)
        _k = DTypeKind
        _np_kinds = {
            "i": _k.INT,
            "u": _k.UINT,
            "f": _k.FLOAT,
            "b": _k.BOOL,
            "U": _k.STRING,
            "M": _k.DATETIME,
            "m": _k.DATETIME,
        }
        kind = _np_kinds.get(dtype.kind, None)
        if kind is None:
            # Not a NumPy dtype. Check if it's a categorical maybe
            if isinstance(dtype, pd.CategoricalDtype):
                # 23 matches CATEGORICAL type in DTypeKind
                kind = 23
            else:
                raise ValueError(
                    f"Data type {dtype} not supported by exchange protocol"
                )

        if kind not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL, _k.CATEGORICAL, _k.STRING):
            raise NotImplementedError(f"Data type {dtype} not handled yet")

        bitwidth = dtype.itemsize * 8
        format_str = dtype.str
        endianness = dtype.byteorder if not kind == _k.CATEGORICAL else "="
        return (kind, bitwidth, format_str, endianness)

    @property
    def describe_categorical(self) -> Dict[str, Any]:
        """
        If the dtype is categorical, there are two options.

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
        if not self.dtype[0] == DTypeKind.CATEGORICAL:
            raise TypeError(
                "`describe_categorical only works on a column with "
                "categorical dtype!"
            )

        cat_dtype = self._col.dtypes[0]
        ordered = cat_dtype.ordered
        is_dictionary = True
        # NOTE: this shows the children approach is better, transforming
        # `categories` to a "mapping" dict is inefficient
        # codes = self._col.values.codes  # ndarray, length `self.size`
        # categories.values is ndarray of length n_categories
        categories = cat_dtype.categories
        mapping = {ix: val for ix, val in enumerate(categories)}
        return {"is_ordered": ordered, "is_dictionary": is_dictionary, "mapping": mapping}

    @property
    def describe_null(self) -> Tuple[int, Any]:
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
        _k = DTypeKind
        kind = self.dtype[0]
        value = None
        if kind == _k.FLOAT:
            null = 1  # np.nan
        elif kind == _k.DATETIME:
            null = 1  # np.datetime64('NaT')
        elif kind in (_k.INT, _k.UINT, _k.BOOL):
            # TODO: check if extension dtypes are used once support for them is
            #       implemented in this protocol code
            null = 0  # integer and boolean dtypes are non-nullable
        elif kind == _k.CATEGORICAL:
            # Null values for categoricals are stored as `-1` sentinel values
            # in the category date (e.g., `col.values.codes` is int8 np.ndarray)
            null = 2
            value = -1
        elif kind == _k.STRING:
            null = 4
            value = (
                0  # follow Arrow in using 1 as valid value and 0 for missing/null value
            )
        else:
            raise NotImplementedError(f"Data type {kind} not yet supported")

        return null, value

    _null_count_cache = None

    # TODO: since python 3.9:
    # @cached_property
    @property
    def null_count(self) -> int:
        """
        Get number of null elements, if known.

        Returns
        -------
        int

        Notes
        -----
        Arrow uses -1 to indicate "unknown", but None seems cleaner.
        """
        if self._null_count_cache is not None:
            return self._null_count_cache

        def map_func(df):
            return df.isna()

        def reduce_func(df):
            return pandas.DataFrame(df.sum())

        intermediate_df = self._col.tree_reduce(0, map_func, reduce_func)
        intermediate_df.index = pandas.RangeIndex(1)
        intermediate_df.columns = pandas.RangeIndex(1)
        self._null_count_cache = intermediate_df.to_pandas().squeeze()
        return self._null_count_cache

    # TODO: ``What should we return???``, remove before the changes are merged
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get the metadata for the column.

        See `DataFrame.metadata` for more details.

        Returns
        -------
        dict
        """
        return {}

    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.

        Returns
        -------
        int
           The number of chunks the column consists of.
        """
        return self._col._partitions.shape[0]

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
        offset = 0
        if n_chunks is None:
            for length in self._col._row_lengths:
                yield Column(
                    PandasDataframe(
                        self._col.mask(row_positions=range(length), col_positions=None),
                        allow_copy=self._col._allow_copy,
                        offset=offset,
                    )
                )
                offset += length
        else:
            new_row_lengths = self.num_rows() // n_chunks
            if self.num_rows() % n_chunks:
                # TODO: raise exception in this case
                new_row_lengths += 1

            new_partitions = self._col._partition_mgr_cls.map_axis_partitions(
                0,
                self._col._partitions,
                lambda df: df,
                keep_partitioning=False,
                lengths=new_row_lengths,
            )
            new_df = self._col.__constructor__(
                new_partitions,
                self._col.index,
                self._col.columns,
                new_row_lengths,
                self._col._column_widths,
            )
            for length in new_df._row_lengths:
                yield Column(
                    PandasDataframe(
                        self._col.mask(row_positions=range(length), col_positions=None),
                        allow_copy=self._allow_copy,
                        offset=offset,
                    )
                )
                offset += length

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
        buffers = {}
        buffers["data"] = self._get_data_buffer()
        try:
            buffers["validity"] = self._get_validity_buffer()
        except Exception:
            buffers["validity"] = None

        try:
            buffers["offsets"] = self._get_offsets_buffer()
        except Exception:
            buffers["offsets"] = None

        return buffers

    def _get_data_buffer(self) -> Tuple[Buffer, Any]:  # Any is for self.dtype tuple
        """
        Return the buffer containing the data and the buffer's associated dtype.

        Returns
        -------
        tuple
            The data buffer.
        """
        _k = DTypeKind
        dtype = self.dtype
        if dtype[0] in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
            buffer = Buffer(self._col.to_numpy().flatten(), allow_copy=self._allow_copy)
            dtype = dtype
        elif dtype[0] == _k.CATEGORICAL:
            pandas_series = self._col.to_pandas().squeeze(axis=1)
            codes = pandas_series.values.codes
            buffer = Buffer(codes, allow_copy=self._allow_copy)
            dtype = self._dtype_from_pandas_dtype(codes.dtype)
        elif dtype[0] == _k.STRING:
            # Marshal the strings from a NumPy object array into a byte array
            buf = self._col.to_numpy().flatten()
            b = bytearray()

            # TODO: this for-loop is slow; can be implemented in Cython/C/C++ later
            for i in range(buf.size):
                if type(buf[i]) == str:
                    b.extend(buf[i].encode(encoding="utf-8"))

            # Convert the byte array to a pandas "buffer" using a NumPy array as the backing store
            buffer = Buffer(np.frombuffer(b, dtype="uint8"))

            # Define the dtype for the returned buffer
            dtype = (
                _k.STRING,
                8,
                "u",
                "=",
            )  # note: currently only support native endianness
        else:
            raise NotImplementedError(f"Data type {self._col.dtype[0]} not handled yet")

        return buffer, dtype

    def _get_validity_buffer(self) -> Tuple[Buffer, Any]:
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
        ``RuntimeError`` if null representation is not a bit or byte mask.
        """
        null, invalid = self.describe_null

        _k = DTypeKind
        if self.dtype[0] == _k.STRING:
            # For now, have the mask array be comprised of bytes, rather than a bit array
            buf = self._col.to_numpy().flatten()
            mask = []

            # Determine the encoding for valid values
            if invalid == 0:
                valid = 1
            else:
                valid = 0

            for i in range(buf.size):
                if type(buf[i]) == str:
                    v = valid
                else:
                    v = invalid

                mask.append(v)

            # Convert the mask array to a Pandas "buffer" using a NumPy array as the backing store
            buffer = Buffer(np.asarray(mask, dtype="uint8"))

            # Define the dtype of the returned buffer
            dtype = (_k.UINT, 8, "C", "=")

            return buffer, dtype

        if null == 0:
            msg = "This column is non-nullable so does not have a mask"
        elif null == 1:
            msg = "This column uses NaN as null so does not have a separate mask"
        else:
            raise NotImplementedError("See self.describe_null")

        raise RuntimeError(msg)

    def _get_offsets_buffer(self) -> Tuple[Buffer, Any]:
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
        ``RuntimeError`` if the data buffer does not have an associated offsets buffer.
        """
        _k = DTypeKind
        if self.dtype[0] == _k.STRING:
            # For each string, we need to manually determine the next offset
            values = self._col.to_numpy().flatten()
            ptr = 0
            offsets = [ptr]
            for v in values:
                # For missing values (in this case, `np.nan` values), we don't increment the pointer)
                if type(v) == str:
                    b = v.encode(encoding="utf-8")
                    ptr += len(b)

                offsets.append(ptr)

            # Convert the list of offsets to a NumPy array of signed 64-bit integers (note: Arrow allows the offsets array to be either `int32` or `int64`; here, we default to the latter)
            buf = np.asarray(offsets, dtype="int64")

            # Convert the offsets to a Pandas "buffer" using the NumPy array as the backing store
            buffer = Buffer(buf)

            # Assemble the buffer dtype info
            dtype = (
                _k.INT,
                64,
                "l",
                "=",
            )  # note: currently only support native endianness
        else:
            raise RuntimeError(
                "This column has a fixed-length dtype so does not have an offsets buffer"
            )

        return buffer, dtype
