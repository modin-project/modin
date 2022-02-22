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

import ctypes
import enum
import numpy as np
import pandas
from typing import Any

import modin.pandas as pd
from modin.pandas.utils import from_pandas

DataFrameObject = Any
ColumnObject = Any


class DTypeKind(enum.IntEnum):
    """Enum for data types."""

    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


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
    # Since a pandas DataFrame doesn't support __dataframe__ for now,
    # we just create a Modin Dataframe to get __dataframe__ from it.
    if isinstance(df, pandas.DataFrame):
        df = pd.DataFrame(df)._query_compiler._modin_frame

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
                columns[name], _buf = _convert_column_to_ndarray(col)
            elif dtype == _k.CATEGORICAL:
                columns[name], _buf = _convert_categorical_column(col)
            elif dtype == _k.STRING:
                columns[name], _buf = _convert_string_column(col)
            else:
                raise NotImplementedError(f"Data type {dtype} not handled yet")

            _buffers.append(_buf)

        pandas_df = pandas.DataFrame(columns)
        pandas_df._buffers = _buffers
        return pandas_df

    pandas_dfs = []
    for chunk in df.get_chunks():
        pandas_df = _get_pandas_df(chunk)
        pandas_dfs.append(pandas_df)
    pandas_df = pandas.concat(pandas_dfs, axis=0)
    modin_frame = from_pandas(pandas_df)._query_compiler._modin_frame
    return modin_frame


def _convert_column_to_ndarray(col: ColumnObject) -> np.ndarray:
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
    return _buffer_to_ndarray(_buffer, _dtype), _buffer


def _buffer_to_ndarray(_buffer, _dtype) -> np.ndarray:
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
    if kind not in (_k.INT, _k.UINT, _k.FLOAT, _k.BOOL):
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


def _convert_categorical_column(col: ColumnObject) -> pandas.Series:
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
    codes = _buffer_to_ndarray(codes_buffer, codes_dtype)
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


def _convert_string_column(col: ColumnObject) -> np.ndarray:
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
    dbuf = _buffer_to_ndarray(dbuffer, dt)

    obuf = _buffer_to_ndarray(obuffer, odtype)
    mbuf = _buffer_to_ndarray(mbuffer, mdtype)

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
