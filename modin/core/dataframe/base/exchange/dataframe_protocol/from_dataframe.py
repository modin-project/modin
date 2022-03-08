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

"""Module houses functions building a ``pandas.DataFrame`` from DataFrame exchange protocol object."""

import pandas
import ctypes
import numpy as np

from typing import Optional, Tuple, Any, Union
from modin.core.dataframe.base.exchange.dataframe_protocol.utils import (
    DTypeKind,
    ColumnNullType,
)
from modin.core.dataframe.base.exchange.dataframe_protocol.dataframe import (
    ProtocolDataframe,
    ProtocolColumn,
    ProtocolBuffer,
)


def from_dataframe(
    df: ProtocolDataframe, allow_copy: bool = True, nchunks: Optional[int] = None
):
    """
    Build ``pandas.DataFrame`` from an object supporting DataFrame exchange protocol (__dataframe__).

    Parameters
    ----------
    df : ProtocolDataframe
        Object supporting the exchange protocol (__dataframe__).
    allow_copy : bool, default: True
        Whether to allow for `df` providing a copy of underlying data.
    nchunks : int, optional
        Number of chunks to split `df`.

    Returns
    -------
    pandas.DataFrame
    """
    if not hasattr(df, "__dataframe__"):
        raise ValueError("`df` does not support __dataframe__")

    df = df.__dataframe__()["dataframe"]

    def get_pandas_df(df):
        # We need a dict of columns here, with each column being a numpy array (at
        # least for now, deal with non-numpy dtypes later).
        columns = dict()
        buffers = []  # hold on to buffers, keeps memory alive
        for name in df.column_names():
            if not isinstance(name, str):
                raise ValueError(f"Column {name} is not a string")
            if name in columns:
                raise ValueError(f"Column {name} is not unique")
            col = df.get_column_by_name(name)
            dtype = col.dtype[0]
            if dtype in (
                DTypeKind.INT,
                DTypeKind.UINT,
                DTypeKind.FLOAT,
                DTypeKind.BOOL,
            ):
                columns[name], buf = convert_primitive_column_to_ndarray(col)
            elif dtype == DTypeKind.CATEGORICAL:
                columns[name], buf = convert_categorical_column(col)
            elif dtype == DTypeKind.STRING:
                columns[name], buf = convert_string_column(col)
            elif dtype == DTypeKind.DATETIME:
                columns[name], buf = convert_datetime_col(col)
            else:
                raise NotImplementedError(f"Data type {dtype} not handled yet")

            buffers.append(buf)

        pandas_df = pandas.DataFrame(columns)
        pandas_df._buffers = buffers
        return pandas_df

    pandas_dfs = []
    for chunk in df.get_chunks(nchunks):
        pandas_df = get_pandas_df(chunk)
        pandas_dfs.append(pandas_df)

    pandas_df = pandas.concat(pandas_dfs, axis=0, ignore_index=True)

    if "index" in df.metadata:
        pandas_df.index = df.metadata["index"]

    return pandas_df


def convert_primitive_column_to_ndarray(col: ProtocolColumn) -> Tuple[np.ndarray, Any]:
    """
    Convert Column holding one of the primitive dtypes (int, uint, float or bool) to a NumPy array.

    Parameters
    ----------
    col : ProtocolColumn

    Returns
    -------
    tuple
        Tuple of numpy.ndarray holding the data and the memory owner object that keeps the memory alive.
    """
    buffers = col.get_buffers()

    data_buff, data_dtype = buffers["data"]
    data = buffer_to_ndarray(data_buff, data_dtype, col.offset, col.size)

    data = set_nulls(data, col, buffers["validity"])
    return data, buffers


def convert_categorical_column(col: ProtocolColumn) -> Tuple[pandas.Series, Any]:
    """
    Convert Column holding categorical data to a pandas Series.

    Parameters
    ----------
    col : ProtocolColumn

    Returns
    -------
    tuple
        Tuple of pandas.Series holding the data and the memory owner object that keeps the memory alive.
    """
    ordered, is_dict, mapping = col.describe_categorical.values()

    if not is_dict:
        raise NotImplementedError("Non-dictionary categoricals not supported yet")

    categories = np.array(list(mapping.values()))
    buffers = col.get_buffers()

    codes_buff, codes_dtype = buffers["data"]
    codes = buffer_to_ndarray(codes_buff, codes_dtype, col.offset, col.size)

    # Doing module in order to not get IndexError for out-of-bounds sentinel values in `codes`
    values = categories[codes % len(categories)]

    cat = pandas.Categorical(values, categories=categories, ordered=ordered)
    data = pandas.Series(cat)

    data = set_nulls(data, col, buffers["validity"])
    return data, buffers


def convert_string_column(col: ProtocolColumn) -> Tuple[np.ndarray, Any]:
    """
    Convert Column holding string data to a NumPy array.

    Parameters
    ----------
    col : ProtocolColumn

    Returns
    -------
    tuple
        Tuple of numpy.ndarray holding the data and the memory owner object that keeps the memory alive.
    """
    if col.describe_null[0] not in (
        ColumnNullType.NON_NULLABLE,
        ColumnNullType.USE_BITMASK,
        ColumnNullType.USE_BYTEMASK,
    ):
        raise NotImplementedError(
            f"{col.describe_null[0]} null kind is not yet supported for string columns."
        )

    buffers = col.get_buffers()

    # Retrieve the data buffer containing the UTF-8 code units
    data_buff, _ = buffers["data"]
    # Convert the buffers to NumPy arrays, in order to go from STRING to an equivalent ndarray,
    # we claim that the buffer is uint8 (i.e., a byte array)
    data_dtype = (
        DTypeKind.UINT,
        8,
        None,
        None,
    )
    # Specify zero offset as we don't want to chunk the string data
    data = buffer_to_ndarray(data_buff, data_dtype, offset=0, length=col.size)

    # Retrieve the offsets buffer containing the index offsets demarcating the beginning and end of each string
    offset_buff, offset_dtype = buffers["offsets"]
    offsets = buffer_to_ndarray(offset_buff, offset_dtype, col.offset, col.size + 1)

    null_kind, sentinel_val = col.describe_null
    null_pos = None

    if null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        valid_buff, valid_dtype = buffers["validity"]
        null_pos = buffer_to_ndarray(valid_buff, valid_dtype, col.offset, col.size)
        if sentinel_val == 0:
            null_pos = ~null_pos

    # Assemble the strings from the code units
    str_list = []
    for i in range(offsets.size - 1):
        # Check for missing values
        if null_pos is not None and null_pos[i]:
            str_list.append(np.nan)
            continue

        # Extract a range of code units
        units = data[offsets[i] : offsets[i + 1]]

        # Convert the list of code units to bytes
        str_bytes = bytes(units)

        # Create the string
        string = str_bytes.decode(encoding="utf-8")

        # Add to our list of strings
        str_list.append(string)

    # Convert the string list to a NumPy array
    return np.asarray(str_list, dtype="object"), buffers


def convert_datetime_col(col: ProtocolColumn) -> Tuple[np.ndarray, Any]:
    """
    Convert Column holding DateTime data to a NumPy array.

    Parameters
    ----------
    col : ProtocolColumn

    Returns
    -------
    tuple
        Tuple of numpy.ndarray holding the data and the memory owner object that keeps the memory alive.
    """
    buffers = col.get_buffers()

    _, _, format_str, _ = col.dtype
    dbuf, dtype = buffers["data"]
    # Consider dtype being `uint` to get number of units passed since the 01.01.1970
    data = buffer_to_ndarray(
        dbuf, (DTypeKind.UINT, dtype[1], "u", "="), col.offset, col.size
    )

    if format_str.startswith("ts"):
        # timestamp 'ts{unit}:tz'
        meta = format_str[2:].split(":")
        if len(meta) == 1:
            unit = meta[0]
            tz = ""
        else:
            unit, tz = meta
        if tz != "":
            raise NotImplementedError("Timezones are not supported yet")
        if unit != "s":
            # the format string describes only a first letter of the unit, add one extra
            # letter to make the unit in numpy-style: 'm' -> 'ms', 'u' -> 'us', 'n' -> 'ns'
            unit += "s"
        data = data.astype(f"datetime64[{unit}]")
    elif format_str.startswith("td"):
        # date 'td{Days/Ms}'
        unit = format_str[2:]
        if unit == "D":
            # numpy doesn't support DAY unit, so converting days to seconds
            # (converting to uint64 to avoid overflow)
            data = (data.astype(np.uint64) * (24 * 60 * 60)).astype("datetime64[s]")
        elif unit == "m":
            data = data.astype("datetime64[ms]")
        else:
            raise NotImplementedError(f"Date unit is not supported: {unit}")
    else:
        raise NotImplementedError(f"DateTime kind is not supported: {format_str}")

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

    if kind not in (DTypeKind.INT, DTypeKind.UINT, DTypeKind.FLOAT, DTypeKind.BOOL):
        raise RuntimeError("Not a boolean, integer or floating-point dtype")

    np_kinds = {
        DTypeKind.INT: {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64},
        DTypeKind.UINT: {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64},
        DTypeKind.FLOAT: {32: np.float32, 64: np.float64},
        # Consider bitmask to be a uint8 dtype to parse the bits later
        DTypeKind.BOOL: {1: np.uint8, 8: bool},
    }

    column_dtype = np_kinds[kind].get(bit_width, None)
    if column_dtype is None:
        raise NotImplementedError(f"Convertion for {dtype} is not yet supported.")

    # No DLPack yet, so need to construct a new ndarray from the data pointer
    # and size in the buffer plus the dtype on the column
    ctypes_type = np.ctypeslib.as_ctypes_type(column_dtype)
    data_pointer = ctypes.cast(
        buffer.ptr + (offset * bit_width // 8), ctypes.POINTER(ctypes_type)
    )

    if bit_width == 1:
        assert length is not None, "`length` must be specified for a bit-mask buffer."
        arr = np.ctypeslib.as_array(data_pointer, shape=(buffer.bufsize,))
        return bitmask_to_bool_array(arr, length, first_byte_offset=offset % 8)
    else:
        return np.ctypeslib.as_array(
            data_pointer, shape=(buffer.bufsize // (bit_width // 8),)
        )


def bitmask_to_bool_array(
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
    if first_byte_offset > 8:
        raise ValueError(
            f"First byte offset can't be more than 8, met: {first_byte_offset}"
        )

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
    data : numpy.ndarray or pandas.Series
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
    numpy.ndarray of pandas.Series
        Data with the nulls being set.
    """
    null_kind, sentinel_val = col.describe_null
    null_pos = None

    if null_kind == ColumnNullType.USE_SENTINEL:
        null_pos = data == sentinel_val
    elif null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        valid_buff, valid_dtype = validity
        null_pos = buffer_to_ndarray(valid_buff, valid_dtype, col.offset, col.size)
        if sentinel_val == 0:
            null_pos = ~null_pos
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
