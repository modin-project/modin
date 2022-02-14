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

import enum
import collections
import ctypes
from typing import Any, Optional, Tuple, Dict, Iterable, Sequence
import numpy as np
import pandas

import modin.pandas as pd
from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe
from modin.pandas.utils import from_pandas

# A typing protocol could be added later
# to let Mypy validate code using `from_dataframe` better.
DataFrameObject = Any
ColumnObject = Any


def from_dataframe(df: DataFrameObject, allow_copy: bool = True) -> "DataFrame":
    """
    Construct a ``DataFrame`` from ``df`` if it supports ``__dataframe__``.

    Parameters
    ----------
    df : DataFrameObject
        An object to create a DataFrame from.
    allow_copy : bool, default: True
        A keyword that defines whether or not the library is allowed
        to make a copy of the data. For example, copying data would be necessary
        if a library supports strided buffers, given that this protocol
        specifies contiguous buffers. Currently, if the flag is set to ``False``
        and a copy is needed, a ``RuntimeError`` will be raised.

    Notes
    -----
    Not all cases are handled yet, only ones that can be implemented with
    only pandas. Later, we need to implement/test support for categoricals,
    bit/byte masks, chunk handling, etc.
    """
    # NOTE: commented out for roundtrip testing
    # if isinstance(df, pandas.DataFrame):
    #     return df

    if not hasattr(df, "__dataframe__"):
        raise ValueError("`df` does not support __dataframe__")

    def _get_pandas_df(df):
        # We need a dict of columns here, with each column being a numpy array (at
        # least for now, deal with non-numpy dtypes later).
        columns = dict()
        _k = DTypeKind
        _buffers = []  # hold on to buffers, keeps memory alive
        for name in df.column_names():
            if not isinstance(name, str):
                raise ValueError(f"Column {name} is not a string")
            if name in columns:
                raise ValueError(f"Column {name} is not unique")
            col = df.get_column_by_name(name)
            if col.dtype[0] in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
                # Simple numerical or bool dtype, turn into numpy array
                columns[name], _buf = convert_column_to_ndarray(col)
            elif col.dtype[0] == _k.CATEGORICAL:
                columns[name], _buf = convert_categorical_column(col)
            elif col.dtype[0] == _k.STRING:
                columns[name], _buf = convert_string_column(col)
            else:
                raise NotImplementedError(f"Data type {col.dtype[0]} not handled yet")

            _buffers.append(_buf)

        pandas_df = pandas.DataFrame(columns)
        pandas_df._buffers = _buffers

    pandas_dfs = []
    for chunk in df.get_chunks():
        pandas_df = _get_pandas_df(chunk)
        pandas_dfs.append(pandas_df)
    pandas_df = pandas.concat(pandas_dfs, axis=0)
    modin_frame = from_pandas(pandas_df)._query_compiler._modin_frame
    return modin_frame


class DTypeKind(enum.IntEnum):
    """Enum for data types."""

    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


def convert_column_to_ndarray(col: ColumnObject) -> np.ndarray:
    """
    Convert an int, uint, float or bool column to a NumPy array.

    Parameters
    ----------
    col : ColumnObject
        A column to convert to a NumPy array from.

    Returns
    -------
    np.ndarray
        NumPy array.
    """
    if col.offset != 0:
        raise NotImplementedError("column.offset > 0 not handled yet")

    if col.describe_null[0] not in (0, 1):
        raise NotImplementedError(
            "Null values represented as masks or " "sentinel values not handled yet"
        )

    _buffer, _dtype = col.get_buffers()["data"]
    return buffer_to_ndarray(_buffer, _dtype), _buffer


def buffer_to_ndarray(_buffer, _dtype) -> np.ndarray:
    """
    Convert a ``Buffer`` object to a NumPy array.

    Parameters
    ----------
    col : Buffer
        A buffer to convert to a NumPy array from.
    _dtype : any
        A dtype object.

    Returns
    -------
    np.ndarray
        NumPy array.
    """
    # Handle the dtype
    kind = _dtype[0]
    bitwidth = _dtype[1]
    _k = DTypeKind
    if _dtype[0] not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
        raise RuntimeError("Not a boolean, integer or floating-point dtype")

    _ints = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
    _uints = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    _floats = {32: np.float32, 64: np.float64}
    _np_dtypes = {0: _ints, 1: _uints, 2: _floats, 20: {8: bool}}
    column_dtype = _np_dtypes[kind][bitwidth]

    # No DLPack yet, so need to construct a new ndarray from the data pointer
    # and size in the buffer plus the dtype on the column
    ctypes_type = np.ctypeslib.as_ctypes_type(column_dtype)
    data_pointer = ctypes.cast(_buffer.ptr, ctypes.POINTER(ctypes_type))

    # NOTE: `x` does not own its memory, so the caller of this function must
    #       either make a copy or hold on to a reference of the column or
    #       buffer! (not done yet, this is pretty awful ...)
    x = np.ctypeslib.as_array(data_pointer, shape=(_buffer.bufsize // (bitwidth // 8),))

    return x


def convert_categorical_column(col: ColumnObject) -> pandas.Series:
    """
    Convert a categorical column to a pandas Series instance.

    Parameters
    ----------
    col : ColumnObject
        A column to convert to to a pandas Series instance from.

    Returns
    -------
    pandas.Series
        A pandas Series instance.
    """
    ordered, is_dict, mapping = col.describe_categorical
    if not is_dict:
        raise NotImplementedError("Non-dictionary categoricals not supported yet")

    # If you want to cheat for testing (can't use `_col` in real-world code):
    #    categories = col._col.values.categories.values
    #    codes = col._col.values.codes
    categories = np.asarray(list(mapping.values()))
    codes_buffer, codes_dtype = col.get_buffers()["data"]
    codes = buffer_to_ndarray(codes_buffer, codes_dtype)
    values = categories[codes]

    # Seems like Pandas can only construct with non-null values, so need to
    # null out the nulls later
    cat = pandas.Categorical(values, categories=categories, ordered=ordered)
    series = pandas.Series(cat)
    null_kind = col.describe_null[0]
    if null_kind == 2:  # sentinel value
        sentinel = col.describe_null[1]
        series[codes == sentinel] = np.nan
    else:
        raise NotImplementedError(
            "Only categorical columns with sentinel " "value supported at the moment"
        )

    return series, codes_buffer


def convert_string_column(col: ColumnObject) -> np.ndarray:
    """
    Convert a string column to a NumPy array.

    Parameters
    ----------
    col : ColumnObject
        A string column to convert to a NumPy array from.

    Returns
    -------
    np.ndarray
        NumPy array object.
    """
    # Retrieve the data buffers
    buffers = col.get_buffers()

    # Retrieve the data buffer containing the UTF-8 code units
    dbuffer, bdtype = buffers["data"]

    # Retrieve the offsets buffer containing the index offsets demarcating the beginning and end of each string
    obuffer, odtype = buffers["offsets"]

    # Retrieve the mask buffer indicating the presence of missing values
    mbuffer, mdtype = buffers["validity"]

    # Retrieve the missing value encoding
    null_kind, null_value = col.describe_null

    # Convert the buffers to NumPy arrays
    dt = (
        DTypeKind.UINT,
        8,
        None,
        None,
    )  # note: in order to go from STRING to an equivalent ndarray, we claim that the buffer is uint8 (i.e., a byte array)
    dbuf = buffer_to_ndarray(dbuffer, dt)

    obuf = buffer_to_ndarray(obuffer, odtype)
    mbuf = buffer_to_ndarray(mbuffer, mdtype)

    # Assemble the strings from the code units
    str_list = []
    for i in range(obuf.size - 1):
        # Check for missing values
        if null_kind == 3:  # bit mask
            v = mbuf[i / 8]
            if null_value == 1:
                v = ~v

            if v & (1 << (i % 8)):
                str_list.append(np.nan)
                continue

        elif null_kind == 4 and mbuf[i] == null_value:  # byte mask
            str_list.append(np.nan)
            continue

        # Extract a range of code units
        units = dbuf[obuf[i] : obuf[i + 1]]

        # Convert the list of code units to bytes
        b = bytes(units)

        # Create the string
        s = b.decode(encoding="utf-8")

        # Add to our list of strings
        str_list.append(s)

    # Convert the string list to a NumPy array
    return np.asarray(str_list, dtype="object"), buffers


# Implementation of interchange protocol
# --------------------------------------


class Buffer:
    """
    Data in the buffer is guaranteed to be contiguous in memory.

    Note that there is no dtype attribute present, a buffer can be thought of
    as simply a block of memory. However, if the column that the buffer is
    attached to has a dtype that's supported by DLPack and ``__dlpack__`` is
    implemented, then that dtype information will be contained in the return
    value from ``__dlpack__``.

    This distinction is useful to support both (a) data exchange via DLPack on a
    buffer and (b) dtypes like variable-length strings which do not have a
    fixed number of bytes per element.

    Parameters
    ----------
    x : np.ndarray
        Data to be held by ``Buffer``.
    allow_copy : bool, default: True
        A keyword that defines whether or not the library is allowed
        to make a copy of the data. For example, copying data would be necessary
        if a library supports strided buffers, given that this protocol
        specifies contiguous buffers. Currently, if the flag is set to ``False``
        and a copy is needed, a ``RuntimeError`` will be raised.
    """

    def __init__(self, x: np.ndarray, allow_copy: bool = True) -> None:
        """
        Handle only regular columns (= numpy arrays) for now.
        """
        if not x.strides == (x.dtype.itemsize,):
            # The protocol does not support strided buffers, so a copy is
            # necessary. If that's not allowed, we need to raise an exception.
            if allow_copy:
                x = x.copy()
            else:
                raise RuntimeError(
                    "Exports cannot be zero-copy in the case "
                    "of a non-contiguous buffer"
                )

        # Store the numpy array in which the data resides as a private
        # attribute, so we can use it to retrieve the public attributes
        self._x = x

    @property
    def bufsize(self) -> int:
        """
        Buffer size in bytes.
        """
        return self._x.size * self._x.dtype.itemsize

    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.
        """
        return self._x.__array_interface__["data"][0]

    def __dlpack__(self):
        """
        DLPack not implemented in NumPy yet, so leave it out here.

        Produce DLPack capsule (see array API standard).
        Raises:
            - TypeError : if the buffer contains unsupported dtypes.
            - NotImplementedError : if DLPack support is not implemented
        Useful to have to connect to array libraries. Support optional because
        it's not completely trivial to implement for a Python-only library.
        """
        raise NotImplementedError("__dlpack__")

    def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]:
        """
        Device type and device ID for where the data in the buffer resides.
        Uses device type codes matching DLPack. Enum members are::
            - CPU = 1
            - CUDA = 2
            - CPU_PINNED = 3
            - OPENCL = 4
            - VULKAN = 7
            - METAL = 8
            - VPI = 9
            - ROCM = 10
        Note: must be implemented even if ``__dlpack__`` is not.
        """

        class Device(enum.IntEnum):
            CPU = 1

        return (Device.CPU, None)

    def __repr__(self) -> str:
        """
        Return a string representation for a particular ``Buffer``.

        Returns
        -------
        str
        """
        return (
            "Buffer("
            + str(
                {
                    "bufsize": self.bufsize,
                    "ptr": self.ptr,
                    "device": self.__dlpack_device__()[0].name,
                }
            )
            + ")"
        )


class Column:
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
        The offset of the first element.

    Notes
    -----
    This Column object can only be produced by ``__dataframe__``,
    so doesn't need its own version or ``__column__`` protocol.
    """

    def __init__(
        self, column: "DataFrame", allow_copy: bool = True, offset: int = 0
    ) -> None:
        """
        Note: doesn't deal with extension arrays yet, just assume a regular
        Series/ndarray for now.
        """
        if not isinstance(column, DataFrame):
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

        Corresponds to DataFrame.num_rows() if column is a single chunk;
        equal to size of this current chunk otherwise.

        Returns
        -------
        int
            Size of the column, in elements.
        """
        return len(self._df.index)

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
        Dtype description as a tuple ``(kind, bit-width, format string, endianness)``, where

        * Kind : DTypeKind
        * Bit-width : the number of bits as an integer
        * Format string : data type description format string in Apache Arrow C
                        Data Interface format.
        * Endianness : current only native endianness (``=``) is supported

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
        dtype = self._col.dtypes

        # For now, assume that, if the column dtype is 'O' (i.e., `object`), then we have an array of strings
        if not isinstance(dtype[0], pd.CategoricalDtype) and dtype[0].kind == "O":
            return (DTypeKind.STRING, 8, "u", "=")

        return self._dtype_from_pandasdtype(dtype)

    def _dtype_from_pandasdtype(self, dtype) -> Tuple[DTypeKind, int, str, str]:
        """
        See `self.dtype` for details.
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
        if not self.dtype[0] == DTypeKind.CATEGORICAL:
            raise TypeError(
                "`describe_categorical only works on a column with "
                "categorical dtype!"
            )

        ordered = self._col.dtype[0].ordered
        is_dictionary = True
        # NOTE: this shows the children approach is better, transforming
        # `categories` to a "mapping" dict is inefficient
        # codes = self._col.values.codes  # ndarray, length `self.size`
        # categories.values is ndarray of length n_categories
        categories = self._col.values.categories.values
        mapping = {ix: val for ix, val in enumerate(categories)}
        return ordered, is_dictionary, mapping

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

    @property
    def null_count(self) -> int:
        """
        Number of null elements, if known.
        Note: Arrow uses -1 to indicate "unknown", but None seems cleaner.
        """

        def map_func(df):
            df.isna().sum()

        return self._col.map(func=map_func).to_pandas().squeeze()

    # TODO: ``What should we return???``, remove before the changes are merged
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        The metadata for the column. See `DataFrame.metadata` for more details.
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
            for length in self._row_lengths:
                yield Column(
                    DataFrame(
                        self._df.mask(
                            row_positions=list(range(length)), col_positions=None
                        ),
                        allow_copy=self._df._allow_copy,
                        offset=offset,
                    )
                )
                offset += length
        else:
            new_row_lengths = self.num_rows() // n_chunks
            if self.num_rows() % n_chunks:
                # TODO: raise exception in this case
                new_row_lengths += 1

            new_partitions = self._df._partition_mgr_cls.map_axis_partitions(
                0,
                self._df._partitions,
                lambda df: df,
                keep_partitioning=False,
                lengths=new_row_lengths,
            )
            new_df = self._df.__constructor__(
                new_partitions,
                self._df.index,
                self._df.columns,
                new_row_lengths,
                self._df._column_widths,
            )
            for length in new_df._row_lengths:
                yield Column(
                    DataFrame(
                        self._df.mask(
                            row_positions=list(range(length)), col_positions=None
                        ),
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
        if self.dtype[0] in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
            buffer = Buffer(self._col.to_numpy().flatten(), allow_copy=self._allow_copy)
            dtype = self.dtype[0]
        elif self.dtype[0] == _k.CATEGORICAL:
            pandas_series = self._df.to_pandas().squeeze(axis=1)
            codes = pandas_series.values.codes
            buffer = Buffer(codes, allow_copy=self._allow_copy)
            dtype = self._dtype_from_pandasdtype(codes.dtype)
        elif self.dtype[0] == _k.STRING:
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


class DataFrame(object):
    """
    A data frame class, with only the methods required by the interchange protocol defined.

    Instances of this (private) class are returned from
    ``modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe.__dataframe__``
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
    offset : int, default: 0
        The offset of the first element.
    """

    def __init__(
        self,
        df: PandasDataframe,
        nan_as_null: bool = False,
        allow_copy: bool = True,
        offset: int = 0,
    ) -> None:
        self._df = df
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy
        self._offset = offset

    # TODO: ``What should we return???``, remove before the changes are merged
    @property
    def metadata(self):
        """
        The metadata for the data frame, as a dictionary with string keys.

        The contents of `metadata` may be anything, they are meant for a library
        to store information that it needs to, e.g., roundtrip losslessly or
        for two implementations to share data that is not (yet) part of the
        interchange protocol specification. For avoiding collisions with other
        entries, please add name the keys with the name of the library
        followed by a period and the desired name, e.g, ``pandas.indexcol``.
        """
        # `index` isn't a regular column, and the protocol doesn't support row
        # labels - so we export it as pandas-specific metadata here.
        return {"pandas.index": self._df.index}

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
        # copied from the initial implementation
        # TODO: not happy with Optional, but need to flag it may be expensive
        #       why include it if it may be None - what do we expect consumers
        #       to do here?
        """
        Return the number of rows in the DataFrame, if available.

        Returns
        -------
        int
            The number of rows in the DataFrame.
        """
        return len(self._df.index)

    def num_chunks(self) -> int:
        """
        Return the number of chunks the DataFrame consists of.

        Returns
        -------
        int
            The number of chunks the DataFrame consists of.
        """
        return self._df._partitions.shape[0]

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

    def get_column(self, i: int) -> Column:
        """
        Return the column at the indicated position.

        Returns
        -------
        Column
            The column at the indicated position.
        """
        return Column(
            self._df.mask(row_positions=None, col_positions=[i]),
            allow_copy=self._allow_copy,
            offset=self._offset,
        )

    def get_column_by_name(self, name: str) -> Column:
        """
        Return the column whose name is the indicated name.

        Returns
        -------
        Column
            The column whose name is the indicated name.
        """
        return Column(
            self._df.mask(row_positions=None, col_labels=[name]),
            allow_copy=self._allow_copy,
            offset=self._offset,
        )

    def get_columns(self) -> Iterable[Column]:
        """
        Return an iterator yielding the columns.

        Yields
        ------
        Column
            The ``Column`` object(s).
        """
        for name in self._df.columns:
            yield Column(
                self._df.mask(row_positions=None, col_labels=[name]),
                allow_copy=self._allow_copy,
                offset=self._offset,
            )

    def select_columns(self, indices: Sequence[int]) -> "DataFrame":
        """
        Create a new DataFrame by selecting a subset of columns by index.

        names : Sequence[int]
            Column indices to be selected out of the DataFrame.

        Returns
        -------
        DataFrame
            A new DataFrame with selected a subset of columns by index.
        """
        if not isinstance(indices, collections.Sequence):
            raise ValueError("`indices` is not a sequence")

        return DataFrame(
            self._df.mask(row_positions=None, col_positions=indices),
            allow_copy=self._allow_copy,
            offset=self._offset,
        )

    def select_columns_by_name(self, names: Sequence[str]) -> "DataFrame":
        """
        Create a new DataFrame by selecting a subset of columns by name.

        Parameters
        ----------
        names : Sequence[str]
            Column names to be selected out of the DataFrame.

        Returns
        -------
        DataFrame
            A new DataFrame with selected a subset of columns by name.
        """
        if not isinstance(names, collections.Sequence):
            raise ValueError("`names` is not a sequence")

        return DataFrame(
            self._df.mask(row_positions=None, col_labels=names),
            allow_copy=self._allow_copy,
            offset=self._offset,
        )

    def get_chunks(self, n_chunks: Optional[int] = None) -> Iterable["DataFrame"]:
        """
        Return an iterator yielding the chunks.

        By default `n_chunks=None`, yields the chunks that the data is stored as by the producer.
        If given, `n_chunks` must be a multiple of `self.num_chunks()`,
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
            for length in self._row_lengths:
                yield DataFrame(
                    self._df.mask(
                        row_positions=list(range(length)), col_positions=None
                    ),
                    allow_copy=self._allow_copy,
                    offset=offset,
                )
                offset += length
        else:
            new_row_lengths = self.num_rows() // n_chunks
            if self.num_rows() % n_chunks:
                # TODO: raise exception in this case?
                new_row_lengths += 1

            new_partitions = self._df._partition_mgr_cls.map_axis_partitions(
                0,
                self._df._partitions,
                lambda df: df,
                keep_partitioning=False,
                lengths=new_row_lengths,
            )
            new_df = self._df.__constructor__(
                new_partitions,
                self._df.index,
                self._df.columns,
                new_row_lengths,
                self._df._column_widths,
            )
            for length in new_df._row_lengths:
                yield DataFrame(
                    self._df.mask(
                        row_positions=list(range(length)), col_positions=None
                    ),
                    allow_copy=self._allow_copy,
                    offset=offset,
                )
                offset += length
