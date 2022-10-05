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

from pandas.io.common import get_handle
from pandas.core.apply import reconstruct_func
from pandas import DataFrame as pandas_DataFrame
from pandas.errors import DataError, SpecificationError


def pandas_pivot_table(df, **kwargs):  # noqa: PR01, RT01
    """Perform pandas pivot_table against a dataframe."""
    return df.pivot_table(**kwargs)


def pandas_convert_dtypes(df, **kwargs):  # noqa: PR01, RT01
    """Perform pandas convert_dtypes against a dataframe or series."""
    return df.convert_dtypes(**kwargs)


def pandas_compare(df, **kwargs):  # noqa: PR01, RT01
    """Perform pandas compare against a dataframe or series."""
    return df.compare(**kwargs)


def pandas_dataframe_join(df, other, **kwargs):  # noqa: PR01, RT01
    """Perform pandas DataFrame.join against a dataframe or series."""
    return pandas_DataFrame.join(df, other, **kwargs)


def pandas_reset_index(df, **kwargs):  # noqa: PR01, RT01
    """Perform pandas reset_index against a dataframe or series."""
    return pandas_DataFrame.reset_index(df, **kwargs)


def pandas_to_csv(df, **kwargs):  # noqa: PR01, RT01
    """Perform pandas to_csv against a dataframe or series."""
    return df.to_csv(**kwargs)


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
