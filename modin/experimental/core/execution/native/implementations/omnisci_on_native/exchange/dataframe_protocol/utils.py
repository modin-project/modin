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
import numpy as np

from modin.core.dataframe.base.exchange.dataframe_protocol.utils import (
    ArrowCTypes,
    pandas_dtype_to_arrow_c,
)


def arrow_dtype_to_arrow_c(dtype):
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
        return ArrowCTypes.TIME.format(resolution="s")
    elif pa.types.is_dictionary(dtype):
        return arrow_dtype_to_arrow_c(dtype.index_type)
    else:
        return pandas_dtype_to_arrow_c(np.dtype(dtype.to_pandas_dtype()))
