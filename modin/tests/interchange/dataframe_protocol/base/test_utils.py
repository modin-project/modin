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

"""Tests for common utility functions of the DataFrame exchange protocol."""

import numpy as np
import pandas
import pytest

from modin.core.dataframe.base.interchange.dataframe_protocol.utils import (
    pandas_dtype_to_arrow_c,
)


# TODO: use ArrowSchema to get reference C-string.
# At the time, there is no way to access ArrowSchema holding a type format string from python.
# The only way to 'touch' it is to export the structure to a C-pointer:
# https://github.com/apache/arrow/blob/5680d209fd870f99134e2d7299b47acd90fabb8e/python/pyarrow/types.pxi#L230-L239
@pytest.mark.parametrize(
    "pandas_dtype, c_string",
    [
        (np.dtype("bool"), "b"),
        (np.dtype("int8"), "c"),
        (np.dtype("uint8"), "C"),
        (np.dtype("int16"), "s"),
        (np.dtype("uint16"), "S"),
        (np.dtype("int32"), "i"),
        (np.dtype("uint32"), "I"),
        (np.dtype("int64"), "l"),
        (np.dtype("uint64"), "L"),
        (np.dtype("float16"), "e"),
        (np.dtype("float32"), "f"),
        (np.dtype("float64"), "g"),
        (pandas.Series(["a"]).dtype, "u"),
        (
            pandas.Series([0]).astype("datetime64[ns]").dtype,
            "tsn:",
        ),
    ],
)
def test_dtype_to_arrow_c(pandas_dtype, c_string):  # noqa PR01
    """Test ``pandas_dtype_to_arrow_c`` utility function."""
    assert pandas_dtype_to_arrow_c(pandas_dtype) == c_string
