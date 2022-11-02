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

"""Module that houses compat functions and objects for `pandas.io.common`."""

from modin._compat import PandasCompatVersion

if PandasCompatVersion.CURRENT == PandasCompatVersion.PY36:
    from .py36.pandas_common import (
        get_handle,
        pandas_pivot_table,
        pandas_convert_dtypes,
        pandas_compare,
        pandas_dataframe_join,
        reconstruct_func,
        pandas_reset_index,
        pandas_to_csv,
        DataError,
        SpecificationError,
    )


elif PandasCompatVersion.CURRENT == PandasCompatVersion.LATEST:
    from .latest.pandas_common import (
        get_handle,
        pandas_pivot_table,
        pandas_convert_dtypes,
        pandas_compare,
        pandas_dataframe_join,
        reconstruct_func,
        pandas_reset_index,
        pandas_to_csv,
        DataError,
        SpecificationError,
    )

__all__ = [
    "get_handle",
    "pandas_pivot_table",
    "pandas_convert_dtypes",
    "pandas_compare",
    "pandas_dataframe_join",
    "reconstruct_func",
    "pandas_reset_index",
    "pandas_to_csv",
    "DataError",
    "SpecificationError",
]
