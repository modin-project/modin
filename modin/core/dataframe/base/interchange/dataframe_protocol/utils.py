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
"""

import enum
import re
from typing import Optional, Union

import numpy as np
import pandas
from pandas.api.types import is_datetime64_dtype


class DTypeKind(enum.IntEnum):  # noqa PR01
    """
    Integer enum for data types.
    Attributes
    ----------
    INT : int
        Matches to signed integer data type.
    UINT : int
        Matches to unsigned integer data type.
    FLOAT : int
        Matches to floating point data type.
    BOOL : int
        Matches to boolean data type.
    STRING : int
        Matches to string data type (UTF-8 encoded).
    DATETIME : int
        Matches to datetime data type.
    CATEGORICAL : int
        Matches to categorical data type.
    """

    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


class ColumnNullType(enum.IntEnum):  # noqa PR01
    """
    Integer enum for null type representation.
    Attributes
    ----------
    NON_NULLABLE : int
        Non-nullable column.
    USE_NAN : int
        Use explicit float NaN value.
    USE_SENTINEL : int
        Sentinel value besides NaN.
    USE_BITMASK : int
        The bit is set/unset representing a null on a certain position.
    USE_BYTEMASK : int
        The byte is set/unset representing a null on a certain position.
    """

    NON_NULLABLE = 0
    USE_NAN = 1
    USE_SENTINEL = 2
    USE_BITMASK = 3
    USE_BYTEMASK = 4


class DlpackDeviceType(enum.IntEnum):  # noqa PR01
    """Integer enum for device type codes matching DLPack."""

    CPU = 1
    CUDA = 2
    CPU_PINNED = 3
    OPENCL = 4
    VULKAN = 7
    METAL = 8
    VPI = 9
    ROCM = 10


class ArrowCTypes:
    """
    Enum for Apache Arrow C type format strings.

    The Arrow C data interface:
    https://arrow.apache.org/docs/format/CDataInterface.html#data-type-description-format-strings
    """

    NULL = "n"
    BOOL = "b"
    INT8 = "c"
    UINT8 = "C"
    INT16 = "s"
    UINT16 = "S"
    INT32 = "i"
    UINT32 = "I"
    INT64 = "l"
    UINT64 = "L"
    FLOAT16 = "e"
    FLOAT32 = "f"
    FLOAT64 = "g"
    STRING = "u"  # utf-8
    DATE32 = "tdD"
    DATE64 = "tdm"
    # Resoulution:
    #   - seconds -> 's'
    #   - miliseconds -> 'm'
    #   - microseconds -> 'u'
    #   - nanoseconds -> 'n'
    TIMESTAMP = "ts{resolution}:{tz}"
    TIME = "tt{resolution}"


class Endianness:
    """Enum indicating the byte-order of a data-type."""

    LITTLE = "<"
    BIG = ">"
    NATIVE = "="
    NA = "|"


def pandas_dtype_to_arrow_c(dtype: Union[np.dtype, pandas.CategoricalDtype]) -> str:
    """
    Represent pandas `dtype` as a format string in Apache Arrow C notation.

    Parameters
    ----------
    dtype : np.dtype
        Datatype of pandas DataFrame to represent.

    Returns
    -------
    str
        Format string in Apache Arrow C notation of the given `dtype`.
    """
    if isinstance(dtype, pandas.CategoricalDtype):
        return ArrowCTypes.INT64
    elif dtype == pandas.api.types.pandas_dtype("O"):
        return ArrowCTypes.STRING

    format_str = getattr(ArrowCTypes, dtype.name.upper(), None)
    if format_str is not None:
        return format_str

    if is_datetime64_dtype(dtype):
        # Selecting the first char of resolution string:
        # dtype.str -> '<M8[ns]'
        resolution = re.findall(r"\[(.*)\]", dtype.str)[0][:1]
        return ArrowCTypes.TIMESTAMP.format(resolution=resolution, tz="")

    raise NotImplementedError(
        f"Convertion of {dtype} to Arrow C format string is not implemented."
    )


def raise_copy_alert(copy_reason: Optional[str] = None) -> None:
    """
    Raise a ``RuntimeError`` mentioning that there's a copy required.

    Parameters
    ----------
    copy_reason : str, optional
        The reason of making a copy. Should fit to the following format:
        'The copy occured due to {copy_reason}.'.
    """
    msg = "Copy required but 'allow_copy=False' is set."
    if copy_reason:
        msg += f" The copy occured due to {copy_reason}."
    raise RuntimeError(msg)
