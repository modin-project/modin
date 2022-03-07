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
This module contains draft implementations of the functions, converting __dataframe__
object to `pandas.DataFrame`. The location and implementations of the functions is a
subject to change, however, the contract of `from_dataframe` is supposed to stay the same.
"""

import pandas
import ctypes
import numpy as np

from typing import Optional
from modin.core.dataframe.base.exchange.dataframe_protocol.utils import DTypeKind
from modin.core.dataframe.base.exchange.dataframe_protocol.dataframe import (
    ProtocolDataframe,
    ProtocolColumn,
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
    allow_copy : bool, default True
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
            dtype = col.dtype[0]
            if dtype in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
                # Simple numerical or bool dtype, turn into numpy array
                columns[name], _buf = convert_column_to_ndarray(col)
            elif dtype == _k.CATEGORICAL:
                columns[name], _buf = convert_categorical_column(col)
            elif dtype == _k.STRING:
                columns[name], _buf = convert_string_column(col)
            elif dtype == _k.DATETIME:
                columns[name], _buf = convert_datetime_col(col)
            else:
                raise NotImplementedError(f"Data type {dtype} not handled yet")

            _buffers.append(_buf)

        pandas_df = pandas.DataFrame(columns)
        pandas_df._buffers = _buffers
        return pandas_df

    pandas_dfs = []
    for chunk in df.get_chunks(nchunks):
        pandas_df = _get_pandas_df(chunk)
        pandas_dfs.append(pandas_df)
    # Can't preserve index for now
    pandas_df = pandas.concat(pandas_dfs, axis=0, ignore_index=True)
    return pandas_df


def convert_datetime_col(col):
    if col.describe_null[0] not in (0, 3):
        raise NotImplementedError(
            "Null values represented as masks or sentinel values not handled yet"
        )

    _, _, fmt, _ = col.dtype
    dbuf, dtype = col.get_buffers()["data"]
    data = buffer_to_ndarray(dbuf, (DTypeKind.UINT, dtype[1], "u", "="), col.offset)
    if fmt.startswith("ts"):
        # timestamp ts{unit}:tz
        meta = fmt[2:].split(":")
        if len(meta) == 1:
            unit = meta[0]
            tz = ""
        else:
            unit, tz = meta
        if tz != "":
            raise NotImplementedError("Timezones are not supported yet")
        if unit != "s":
            unit += "s"
        data = data.astype(f"datetime64[{unit}]")
    elif fmt.startswith("td"):
        # date td{Days/Ms}
        unit = fmt[2:]
        if unit == "D":
            # to seconds (converting to uint64 to avoid overflow)
            data = (data.astype(np.uint64) * (24 * 60 * 60)).astype("datetime64[s]")
        elif unit == "m":
            data = data.astype("datetime64[ms]")
        else:
            raise NotImplementedError(f"Date unit is not supported: {unit}")
    else:
        raise NotImplementedError(f"Datetime is not supported: {fmt}")

    if col.describe_null[0] == 3:
        null_mask = ~bitmask_to_bool_array(
            col.get_buffers()["validity"][0], col.offset, col.size
        )
        data[null_mask] = None
    elif col.describe_null[0] in (0, 1, 2):
        pass
    else:
        raise NotImplementedError(
            "Such null kind is not supported for datetime conversion"
        )

    return data, dbuf


def convert_column_to_ndarray(col: ProtocolColumn) -> np.ndarray:
    """
    Convert an int, uint, float or bool column to a numpy array.
    """

    if col.describe_null[0] not in (0, 1, 3):
        raise NotImplementedError(
            "Null values represented as masks or sentinel values not handled yet"
        )
    # breakpoint()
    _buffer, _dtype = col.get_buffers()["data"]
    data, _bfr = buffer_to_ndarray(_buffer, _dtype, col.offset, col.size), _buffer

    if col.describe_null[0] == 3:
        null_pos = ~bitmask_to_bool_array(
            col.get_buffers()["validity"][0], col.offset, col.size
        )
        if np.any(null_pos):
            # convert to null-able type
            data = data.astype(float)
            data[null_pos] = np.nan

    return data, _bfr


def buffer_to_ndarray(
    _buffer, _dtype, offset, length=None, allow_none_buffer=False
) -> np.ndarray:
    # Handle the dtype
    if allow_none_buffer and _buffer is None:
        return None
    kind = _dtype[0]
    bitwidth = _dtype[1]
    _k = DTypeKind
    if _dtype[0] not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
        raise RuntimeError("Not a boolean, integer or floating-point dtype")

    if bitwidth == 1:
        return bitmask_to_bool_array(_buffer, offset, length)

    _ints = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
    _uints = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    _floats = {32: np.float32, 64: np.float64}
    _np_dtypes = {0: _ints, 1: _uints, 2: _floats, 20: {8: bool}}
    column_dtype = _np_dtypes[kind][bitwidth]

    # No DLPack yet, so need to construct a new ndarray from the data pointer
    # and size in the buffer plus the dtype on the column
    ctypes_type = np.ctypeslib.as_ctypes_type(column_dtype)
    data_pointer = ctypes.cast(
        _buffer.ptr + offset * (bitwidth // 8), ctypes.POINTER(ctypes_type)
    )

    # NOTE: `x` does not own its memory, so the caller of this function must
    #       either make a copy or hold on to a reference of the column or
    #       buffer! (not done yet, this is pretty awful ...)
    x = np.ctypeslib.as_array(data_pointer, shape=(_buffer.bufsize // (bitwidth // 8),))
    return x


def convert_categorical_column(col: ProtocolColumn) -> pandas.Series:
    """
    Convert a categorical column to a Series instance.
    """
    ordered, is_dict, mapping = col.describe_categorical.values()
    if not is_dict:
        raise NotImplementedError("Non-dictionary categoricals not supported yet")

    categories = np.asarray(list(mapping.values()))
    codes_buffer, codes_dtype = col.get_buffers()["data"]
    codes = buffer_to_ndarray(codes_buffer, codes_dtype, col.offset)
    # Doing module in order to not get IndexError for negative sentinel values in the `codes`
    values = categories[codes % len(categories)]

    cat = pandas.Categorical(values, categories=categories, ordered=ordered)
    series = pandas.Series(cat)
    null_kind = col.describe_null[0]
    if null_kind == 2:  # sentinel value
        sentinel = col.describe_null[1]
        series[codes == sentinel] = np.nan
    elif null_kind == 3:
        null_values = ~bitmask_to_bool_array(
            col.get_buffers()["validity"][0], col.offset, col.size
        )
        series[null_values] = np.nan
    elif null_kind == 0:
        pass
    else:
        raise NotImplementedError(
            "Only categorical columns with sentinel value supported at the moment"
        )

    return series, codes_buffer


def bitmask_to_bool_array(buffer, offset, mask_length):
    ctypes_type = np.ctypeslib.as_ctypes_type(np.uint8)
    data_pointer = ctypes.cast((buffer.ptr + offset // 8), ctypes.POINTER(ctypes_type))
    # breakpoint()
    first_byte_offset = offset % 8
    x = np.ctypeslib.as_array(data_pointer, shape=(buffer.bufsize,))

    null_mask = np.zeros(mask_length, dtype=bool)
    # Proccessing the first byte separately as it has its own offset
    val = x[0]
    mask_idx = 0
    for j in range(min(8 - first_byte_offset, mask_length)):
        if val & (1 << (j + first_byte_offset)):
            null_mask[mask_idx] = True
        mask_idx += 1

    for i in range(1, mask_length // 8):
        val = x[i]
        for j in range(8):
            if val & (1 << j):
                null_mask[mask_idx] = True
            mask_idx += 1

    if len(x) > 1:
        # Processing reminder of last byte
        val = x[-1]
        for j in range(len(null_mask) - mask_idx):
            if val & (1 << j):
                null_mask[mask_idx] = True
            mask_idx += 1

    return null_mask


def convert_string_column(col: ProtocolColumn) -> np.ndarray:
    """
    Convert a string column to a NumPy array.
    """
    # Retrieve the data buffers
    # breakpoint()
    buffers = col.get_buffers()

    # Retrieve the data buffer containing the UTF-8 code units
    dbuffer, bdtype = buffers["data"]

    # Retrieve the offsets buffer containing the index offsets demarcating the beginning and end of each string
    obuffer, odtype = buffers["offsets"]

    # Retrieve the mask buffer indicating the presence of missing values
    mbuffer, mdtype = buffers["validity"] or (None, None)
    # Retrieve the missing value encoding
    null_kind, null_value = col.describe_null

    # Convert the buffers to NumPy arrays
    dt = (
        DTypeKind.UINT,
        8,
        None,
        None,
    )  # note: in order to go from STRING to an equivalent ndarray, we claim that the buffer is uint8 (i.e., a byte array)
    dbuf = buffer_to_ndarray(dbuffer, dt, 0)
    # breakpoint()
    obuf = buffer_to_ndarray(obuffer, odtype, col.offset)
    # breakpoint()
    if null_kind == 4:
        mbuf = buffer_to_ndarray(mbuffer, mdtype, col.offset, allow_none_buffer=True)
    elif null_kind == 3:
        mbuf = ~bitmask_to_bool_array(mbuffer, col.offset, col.size)

    # Assemble the strings from the code units
    str_list = []
    for i in range(obuf.size - 1):
        # Check for missing values
        if null_kind == 3 and mbuf[i]:  # bit mask
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
    # breakpoint()
    # Convert the string list to a NumPy array
    return np.asarray(str_list, dtype="object"), buffers
