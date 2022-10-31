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

from contextlib import contextmanager
from collections import namedtuple

from pandas import DataFrame as pandas_DataFrame
from pandas.io.common import get_handle as pandas_get_handle
from pandas.core.aggregation import reconstruct_func
from pandas.core.base import DataError, SpecificationError

from modin.utils import _inherit_docstrings

_HandleWrapper = namedtuple("_HandleWrapper", ["handle"])


@contextmanager
@_inherit_docstrings(pandas_get_handle)
def get_handle(
    path_or_buf,
    mode: str,
    encoding=None,
    compression=None,
    memory_map=False,
    is_text=True,
    errors=None,
    storage_options=None,
):
    assert storage_options is None
    f, handles = pandas_get_handle(
        path_or_buf,
        mode=mode,
        encoding=encoding,
        compression=compression,
        memory_map=memory_map,
        is_text=is_text,
        errors=errors,
    )
    try:
        yield _HandleWrapper(handle=f)
    finally:
        for handle in handles:
            try:
                handle.close()
            except (OSError, ValueError):
                pass


def pandas_pivot_table(df, **kwargs):  # noqa: PR01, RT01
    """Perform pandas pivot_table against a dataframe ignoring unsupported args."""
    unsupported_sort = kwargs.pop("sort", None)
    assert (
        unsupported_sort is None
    ), "Unsupported argument passed to DataFrame.pivot_table: sort"
    return df.pivot_table(**kwargs)


def pandas_convert_dtypes(df, **kwargs):  # noqa: PR01, RT01
    """Perform pandas convert_dtypes against a dataframe or series ignoring unsupported args."""
    unsupported_convert = kwargs.pop("convert_floating", None)
    assert (
        unsupported_convert is None
    ), "Unsupported argument passed to DataFrame.convert_dtypes: convert_floating"
    return df.convert_dtypes(**kwargs)


_RESULT_NAMES_DEFAULT = ("self", "other")


def pandas_compare(df, **kwargs):
    result_names = kwargs.pop("result_names", _RESULT_NAMES_DEFAULT)
    assert (
        result_names == _RESULT_NAMES_DEFAULT
    ), "Unsupported argument passed to DataFrame.compare: result_names"
    return df.compare(**kwargs)


def pandas_dataframe_join(df, other, **kwargs):
    validate = kwargs.pop("validate", None)
    assert validate is None, "Unsupported argument passed to DataFrame.join: validate"
    return pandas_DataFrame.join(df, other, **kwargs)


def pandas_reset_index(df, **kwargs):
    allow_duplicates = kwargs.pop("allow_duplicates", None)
    names = kwargs.pop("names", None)
    assert (
        allow_duplicates is None
    ), "Unsupported argument passed to reset_index: allow_duplicates"
    assert names is None, "Unsupported argument passed to reset_index: name"
    return pandas_DataFrame.reset_index(df, **kwargs)


def pandas_to_csv(df, **kwargs):
    kwargs["line_terminator"] = kwargs.pop("lineterminator", None)
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
