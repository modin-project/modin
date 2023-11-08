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

"""Module houses the functions building a ``pandas.DataFrame`` from a DataFrame exchange protocol object."""

import ctypes
import re
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas

from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
    ProtocolBuffer,
    ProtocolColumn,
    ProtocolDataframe,
)
from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
    ArrowCTypes,
    ColumnNullType,
    DTypeKind,
    Endianness,
)

np_types_map = {
    DTypeKind.INT: {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64},
    DTypeKind.UINT: {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64},
    DTypeKind.FLOAT: {32: np.float32, 64: np.float64},
    # Consider bitmask to be a uint8 dtype to parse the bits later
    DTypeKind.BOOL: {1: np.uint8, 8: bool},
}


def from_dataframe_to_pandas(df: ProtocolDataframe, n_chunks: Optional[int] = None):
    """
    Build a ``pandas.DataFrame`` from an object supporting the DataFrame exchange protocol, i.e. `__dataframe__` method.

    Parameters
    ----------
    df : ProtocolDataframe
        Object supporting the exchange protocol, i.e. `__dataframe__` method.
    n_chunks : int, optional
        Number of chunks to split `df`.

    Returns
    -------
    pandas.DataFrame
    """
    if not hasattr(df, "__dataframe__"):
        raise ValueError("`df` does not support __dataframe__")

    df = df.__dataframe__()
    if isinstance(df, dict):
        df = df["dataframe"]

    pandas_dfs = []
    for chunk in df.get_chunks(n_chunks):
        pandas_df = protocol_df_chunk_to_pandas(chunk)
        pandas_dfs.append(pandas_df)

    pandas_df = pandas.concat(pandas_dfs, axis=0, ignore_index=True)

    index_obj = df.metadata.get("modin.index", df.metadata.get("pandas.index", None))
    if index_obj is not None:
        pandas_df.index = index_obj

    return pandas_df


def protocol_df_chunk_to_pandas(df):
    """
    Convert exchange protocol chunk to ``pandas.DataFrame``.

    Parameters
    ----------
    df : ProtocolDataframe

    Returns
    -------
    pandas.DataFrame
    """
    # We need a dict of columns here, with each column being a NumPy array (at
    # least for now, deal with non-NumPy dtypes later).
    columns = dict()
    buffers = []  # hold on to buffers, keeps memory alive
    for name in df.column_names():
        if not isinstance(name, str):
            raise ValueError(f"Column {name} is not a string")
        if name in columns:
            raise ValueError(f"Column {name} is not unique")
        col = df.get_column_by_name(name)
        columns[name], buf = unpack_protocol_column(col)
        buffers.append(buf)

    pandas_df = pandas.DataFrame(columns)
    pandas_df._buffers = buffers
    return pandas_df


def unpack_protocol_column(
    col: ProtocolColumn,
) -> Tuple[Union[np.ndarray, pandas.Series], Any]:
    """
    Unpack an interchange protocol column to a pandas-ready column.

    Parameters
    ----------
    col : ProtocolColumn
        Column to unpack.

    Returns
    -------
    tuple
        Tuple of resulting column (either an ndarray or a series) and the object
        which keeps memory referenced by the column alive.
    """
    dtype = col.dtype[0]
    if dtype in (
        DTypeKind.INT,
        DTypeKind.UINT,
        DTypeKind.FLOAT,
        DTypeKind.BOOL,
    ):
        return primitive_column_to_ndarray(col)
    elif dtype == DTypeKind.CATEGORICAL:
        return categorical_column_to_series(col)
    elif dtype == DTypeKind.STRING:
        return string_column_to_ndarray(col)
    elif dtype == DTypeKind.DATETIME:
        return datetime_column_to_ndarray(col)
    else:
        raise NotImplementedError(f"Data type {dtype} not handled yet")


def primitive_column_to_ndarray(col: ProtocolColumn) -> Tuple[np.ndarray, Any]:
    """
    Convert a column holding one of the primitive dtypes (int, uint, float or bool) to a NumPy array.

    Parameters
    ----------
    col : ProtocolColumn

    Returns
    -------
    tuple
        Tuple of np.ndarray holding the data and the memory owner object that keeps the memory alive.
    """
    buffers = col.get_buffers()

    data_buff, data_dtype = buffers["data"]
    data = buffer_to_ndarray(data_buff, data_dtype, col.offset, col.size())

    data = set_nulls(data, col, buffers["validity"])
    return data, buffers


def categorical_column_to_series(col: ProtocolColumn) -> Tuple[pandas.Series, Any]:
    """
    Convert a column holding categorical data to a pandas Series.

    Parameters
    ----------
    col : ProtocolColumn

    Returns
    -------
    tuple
        Tuple of pandas.Series holding the data and the memory owner object that keeps the memory alive.
    """
    cat_descr = col.describe_categorical
    ordered, is_dict, categories = (
        cat_descr["is_ordered"],
        cat_descr["is_dictionary"],
        cat_descr["categories"],
    )

    if not is_dict or categories is None:
        raise NotImplementedError("Non-dictionary categoricals not supported yet")

    buffers = col.get_buffers()

    codes_buff, codes_dtype = buffers["data"]
    codes = buffer_to_ndarray(codes_buff, codes_dtype, col.offset, col.size())

    # Doing module in order to not get ``IndexError`` for out-of-bounds sentinel values in `codes`
    cat_values, categories_buf = unpack_protocol_column(categories)
    values = cat_values[codes % len(cat_values)]

    cat = pandas.Categorical(values, categories=cat_values, ordered=ordered)
    data = pandas.Series(cat)

    data = set_nulls(data, col, buffers["validity"])
    return data, [buffers, categories_buf]


def _inverse_null_buf(buf: np.ndarray, null_kind: ColumnNullType) -> np.ndarray:
    """
    Inverse the boolean value of buffer storing either bit- or bytemask.

    Parameters
    ----------
    buf : np.ndarray
        Buffer to inverse the boolean value for.
    null_kind : {ColumnNullType.USE_BYTEMASK, ColumnNullType.USE_BITMASK}
        How to treat the buffer.

    Returns
    -------
    np.ndarray
        Logically inversed buffer.
    """
    if null_kind == ColumnNullType.USE_BITMASK:
        return ~buf
    assert (
        null_kind == ColumnNullType.USE_BYTEMASK
    ), f"Unexpected null kind: {null_kind}"
    # bytemasks use 0 for `False` and anything else for `True`, so convert to bool
    # by direct comparison instead of bitwise reversal like we do for bitmasks
    return buf == 0


def string_column_to_ndarray(col: ProtocolColumn) -> Tuple[np.ndarray, Any]:
    """
    Convert a column holding string data to a NumPy array.

    Parameters
    ----------
    col : ProtocolColumn

    Returns
    -------
    tuple
        Tuple of np.ndarray holding the data and the memory owner object that keeps the memory alive.
    """
    null_kind, sentinel_val = col.describe_null

    if null_kind not in (
        ColumnNullType.NON_NULLABLE,
        ColumnNullType.USE_BITMASK,
        ColumnNullType.USE_BYTEMASK,
    ):
        raise NotImplementedError(
            f"{null_kind} null kind is not yet supported for string columns."
        )

    buffers = col.get_buffers()

    # Retrieve the data buffer containing the UTF-8 code units
    data_buff, protocol_data_dtype = buffers["data"]
    # We're going to reinterpret the buffer as uint8, so making sure we can do it safely
    assert protocol_data_dtype[1] == 8  # bitwidth == 8
    assert protocol_data_dtype[2] == ArrowCTypes.STRING  # format_str == utf-8
    # Convert the buffers to NumPy arrays, in order to go from STRING to an equivalent ndarray,
    # we claim that the buffer is uint8 (i.e., a byte array)
    data_dtype = (
        DTypeKind.UINT,
        8,
        ArrowCTypes.UINT8,
        Endianness.NATIVE,
    )
    # Specify zero offset as we don't want to chunk the string data
    data = buffer_to_ndarray(data_buff, data_dtype, offset=0, length=col.size())

    # Retrieve the offsets buffer containing the index offsets demarcating the beginning and end of each string
    offset_buff, offset_dtype = buffers["offsets"]
    # Offsets buffer contains start-stop positions of strings in the data buffer,
    # meaning that it has more elements than in the data buffer, do `col.size() + 1` here
    # to pass a proper offsets buffer size
    offsets = buffer_to_ndarray(
        offset_buff, offset_dtype, col.offset, length=col.size() + 1
    )

    null_pos = None
    if null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        valid_buff, valid_dtype = buffers["validity"]
        null_pos = buffer_to_ndarray(valid_buff, valid_dtype, col.offset, col.size())
        if sentinel_val == 0:
            null_pos = _inverse_null_buf(null_pos, null_kind)

    # Assemble the strings from the code units
    str_list = [None] * col.size()
    for i in range(col.size()):
        # Check for missing values
        if null_pos is not None and null_pos[i]:
            str_list[i] = np.nan
            continue

        # Extract a range of code units
        units = data[offsets[i] : offsets[i + 1]]

        # Convert the list of code units to bytes
        str_bytes = bytes(units)

        # Create the string
        string = str_bytes.decode(encoding="utf-8")

        # Add to our list of strings
        str_list[i] = string

    # Convert the string list to a NumPy array
    return np.asarray(str_list, dtype="object"), buffers


def datetime_column_to_ndarray(col: ProtocolColumn) -> Tuple[np.ndarray, Any]:
    """
    Convert a column holding DateTime data to a NumPy array.

    Parameters
    ----------
    col : ProtocolColumn

    Returns
    -------
    tuple
        Tuple of np.ndarray holding the data and the memory owner object that keeps the memory alive.
    """
    buffers = col.get_buffers()

    _, _, format_str, _ = col.dtype
    dbuf, dtype = buffers["data"]
    # Consider dtype being `uint` to get number of units passed since the 01.01.1970
    data = buffer_to_ndarray(
        dbuf,
        (
            DTypeKind.UINT,
            dtype[1],
            getattr(ArrowCTypes, f"UINT{dtype[1]}"),
            Endianness.NATIVE,
        ),
        col.offset,
        col.size(),
    )

    def parse_format_str(format_str, data):
        """Parse datetime `format_str` to interpret the `data`."""
        # timestamp 'ts{unit}:tz'
        timestamp_meta = re.match(r"ts([smun]):(.*)", format_str)
        if timestamp_meta:
            unit, tz = timestamp_meta.group(1), timestamp_meta.group(2)
            if tz != "":
                raise NotImplementedError("Timezones are not supported yet")
            if unit != "s":
                # the format string describes only a first letter of the unit, add one extra
                # letter to make the unit in numpy-style: 'm' -> 'ms', 'u' -> 'us', 'n' -> 'ns'
                unit += "s"
            data = data.astype(f"datetime64[{unit}]")
            return data

        # date 'td{Days/Ms}'
        date_meta = re.match(r"td([Dm])", format_str)
        if date_meta:
            unit = date_meta.group(1)
            if unit == "D":
                # NumPy doesn't support DAY unit, so converting days to seconds
                # (converting to uint64 to avoid overflow)
                data = (data.astype(np.uint64) * (24 * 60 * 60)).astype("datetime64[s]")
            elif unit == "m":
                data = data.astype("datetime64[ms]")
            else:
                raise NotImplementedError(f"Date unit is not supported: {unit}")
            return data

        raise NotImplementedError(f"DateTime kind is not supported: {format_str}")

    data = parse_format_str(format_str, data)
    data = set_nulls(data, col, buffers["validity"])
    return data, buffers


def buffer_to_ndarray(
    buffer: ProtocolBuffer,
    dtype: Tuple[DTypeKind, int, str, str],
    offset: int = 0,
    length: Optional[int] = None,
) -> np.ndarray:
    """
    Build a NumPy array from the passed buffer.

    Parameters
    ----------
    buffer : ProtocolBuffer
        Buffer to build a NumPy array from.
    dtype : tuple
        Data type of the buffer conforming protocol dtypes format.
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    length : int, optional
        If the buffer is a bit-mask, specifies a number of bits to read
        from the buffer. Has no effect otherwise.

    Returns
    -------
    np.ndarray

    Notes
    -----
    The returned array doesn't own the memory. A user of the function must keep the memory
    owner object alive as long as the returned NumPy array is being used.
    """
    kind, bit_width, _, _ = dtype

    column_dtype = np_types_map.get(kind, {}).get(bit_width, None)
    if column_dtype is None:
        raise NotImplementedError(f"Convertion for {dtype} is not yet supported.")

    # TODO: No DLPack yet, so need to construct a new ndarray from the data pointer
    # and size in the buffer plus the dtype on the column. Use DLPack as NumPy supports
    # it since https://github.com/numpy/numpy/pull/19083
    ctypes_type = np.ctypeslib.as_ctypes_type(column_dtype)
    data_pointer = ctypes.cast(
        buffer.ptr + (offset * bit_width // 8), ctypes.POINTER(ctypes_type)
    )

    if bit_width == 1:
        assert length is not None, "`length` must be specified for a bit-mask buffer."
        arr = np.ctypeslib.as_array(data_pointer, shape=(buffer.bufsize,))
        return bitmask_to_bool_ndarray(arr, length, first_byte_offset=offset % 8)
    else:
        return np.ctypeslib.as_array(
            data_pointer, shape=(buffer.bufsize // (bit_width // 8),)
        )


def bitmask_to_bool_ndarray(
    bitmask: np.ndarray, mask_length: int, first_byte_offset: int = 0
) -> np.ndarray:
    """
    Convert bit-mask to a boolean NumPy array.

    Parameters
    ----------
    bitmask : np.ndarray[uint8]
        NumPy array of uint8 dtype representing the bitmask.
    mask_length : int
        Number of elements in the mask to interpret.
    first_byte_offset : int, default: 0
        Number of elements to offset from the start of the first byte.

    Returns
    -------
    np.ndarray[bool]
    """
    bytes_to_skip = first_byte_offset // 8
    bitmask = bitmask[bytes_to_skip:]
    first_byte_offset %= 8

    bool_mask = np.zeros(mask_length, dtype=bool)

    # Proccessing the first byte separately as it has its own offset
    val = bitmask[0]
    mask_idx = 0
    bits_in_first_byte = min(8 - first_byte_offset, mask_length)
    for j in range(bits_in_first_byte):
        if val & (1 << (j + first_byte_offset)):
            bool_mask[mask_idx] = True
        mask_idx += 1

    # `mask_length // 8` describes how many full bytes to process
    for i in range((mask_length - bits_in_first_byte) // 8):
        # doing `+ 1` as we already processed the first byte
        val = bitmask[i + 1]
        for j in range(8):
            if val & (1 << j):
                bool_mask[mask_idx] = True
            mask_idx += 1

    if len(bitmask) > 1:
        # Processing reminder of last byte
        val = bitmask[-1]
        for j in range(len(bool_mask) - mask_idx):
            if val & (1 << j):
                bool_mask[mask_idx] = True
            mask_idx += 1

    return bool_mask


def set_nulls(
    data: Union[np.ndarray, pandas.Series],
    col: ProtocolColumn,
    validity: Tuple[ProtocolBuffer, Tuple[DTypeKind, int, str, str]],
    allow_modify_inplace: bool = True,
):
    """
    Set null values for the data according to the column null kind.

    Parameters
    ----------
    data : np.ndarray or pandas.Series
        Data to set nulls in.
    col : ProtocolColumn
        Column object that describes the `data`.
    validity : tuple(ProtocolBuffer, dtype) or None
        The return value of ``col.buffers()``. We do not access the ``col.buffers()``
        here to not take the ownership of the memory of buffer objects.
    allow_modify_inplace : bool, default: True
        Whether to modify the `data` inplace when zero-copy is possible (True) or always
        modify a copy of the `data` (False).

    Returns
    -------
    np.ndarray or pandas.Series
        Data with the nulls being set.
    """
    null_kind, sentinel_val = col.describe_null
    null_pos = None

    if null_kind == ColumnNullType.USE_SENTINEL:
        null_pos = data == sentinel_val
    elif null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        valid_buff, valid_dtype = validity
        null_pos = buffer_to_ndarray(valid_buff, valid_dtype, col.offset, col.size())
        if sentinel_val == 0:
            null_pos = _inverse_null_buf(null_pos, null_kind)
    elif null_kind in (ColumnNullType.NON_NULLABLE, ColumnNullType.USE_NAN):
        pass
    else:
        raise NotImplementedError(f"Null kind {null_kind} is not yet supported.")

    if null_pos is not None and np.any(null_pos):
        if not allow_modify_inplace:
            data = data.copy()
        try:
            data[null_pos] = None
        except TypeError:
            # TypeError happens if the `data` dtype appears to be non-nullable in numpy notation
            # (bool, int, uint), if such happens, cast the `data` to nullable float dtype.
            data = data.astype(float)
            data[null_pos] = None

    return data
