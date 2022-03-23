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

"""Utility functions for the DataFrame exchange protocol implementation for ``OmnisciOnNative`` execution."""

import pyarrow as pa
import numpy as np
import functools

from modin.core.dataframe.base.exchange.dataframe_protocol.utils import (
    ArrowCTypes,
    pandas_dtype_to_arrow_c,
    raise_copy_alert,
    DTypeKind,
)


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


def arrow_dtype_to_arrow_c(dtype: pa.DataType) -> str:
    """
    Represent PyArrow `dtype` as a format string in Apache Arrow C notation.

    Parameters
    ----------
    dtype : pa.DataType
        Datatype of PyArrow table to represent.

    Returns
    -------
    str
        Format string in Apache Arrow C notation of the given `dtype`.
    """
    if pa.types.is_timestamp(dtype):
        return ArrowCTypes.TIMESTAMP.format(
            resolution=dtype.unit[:1], tz=dtype.tz or ""
        )
    elif pa.types.is_date(dtype):
        return getattr(ArrowCTypes, f"DATE{dtype.bit_width}", "DATE64")
    elif pa.types.is_time(dtype):
        # TODO: for some reason `time32` type doesn't have a `unit` attribute,
        # always return "s" for now.
        # return ArrowCTypes.TIME.format(resolution=dtype.unit[:1])
        return ArrowCTypes.TIME.format(resolution=getattr(dtype, "unit", "s")[:1])
    elif pa.types.is_dictionary(dtype):
        return arrow_dtype_to_arrow_c(dtype.index_type)
    else:
        return pandas_dtype_to_arrow_c(np.dtype(dtype.to_pandas_dtype()))


def raise_copy_alert_if_materialize(fn):
    """
    Decorate ``OmnisciProtocolDataframe`` method with a check raising a copy-alert if it's impossible to retrieve the data in zero-copy way.

    Parameters
    ----------
    fn : callable
        ``OmnisciProtocolDataframe`` method.

    Returns
    -------
    callable
    """

    @functools.wraps(fn)
    def method(self, *args, **kwargs):
        if not self._allow_copy and not self._is_zero_copy_possible:
            raise_copy_alert()
        return fn(self, *args, **kwargs)

    return method
